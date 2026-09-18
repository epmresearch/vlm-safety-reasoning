#!/usr/bin/env python3
"""
Runs the teacher VLM over a MOCS image selection and writes one annotation proposal
per image. Resumable, crash-safe, and does not touch anything the training pipeline
reads.

WHAT THIS IS NOT. These are PROPOSALS, not labels. Nothing here is fit to train on
until a human has reviewed it. Two specific reasons, both measured:

  1. A zero-shot VLM's violation-region IoU on this exact task tops out around 23%
     in the dataset paper's Table 8, while this repo's own fine-tuned 8B reaches
     45.6%. Training on teacher boxes would actively degrade the one axis where
     these models already beat every published result on all four rules. For the
     rule_4 bucket, prefer the human-annotated MOCS geometry carried on the
     selection record (`worker_machine_pairs[].union_box`) over anything the teacher
     draws.
  2. Verification is one-sided: a reviewer removes false positives but can never
     recover a violation the teacher never proposed. The prompt is calibrated toward
     recall for that reason (see mocs_annotation/prompts.py), which means the
     proposal stream is deliberately over-inclusive.

WHY THIS RESUMES, WHEN run_inference.py DELIBERATELY DOES NOT. CLAUDE.md is emphatic
that the training-path inference has no auto-resume, because a partial re-run there
corrupts the metric denominator (two records for one image made
structural_json_validity_rate read 0.500 over a denominator of 2). Neither hazard
exists here: there is no denominator, and partial progress on a multi-hour
annotation job is genuinely valuable. Safety comes from the same place instead --
exactly one line per attempted image, keyed by id, and ids already present are
skipped rather than re-run and appended.

HPC only (needs torch + transformers + a GPU). Everything else in this package runs
on a laptop.

Usage (see scripts/hpc_annotate_mocs.sh for the real invocation):
    python -m mocs_annotation.annotate \
        --selection  $OUT/selection.json \
        --fewshot    $OUT/fewshot.json \
        --out-dir    $OUT \
        --batch-size 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.logging import get_logger
from mocs_annotation.prompts import (
    build_messages,
    build_system_prompt,
    prompt_fingerprint,
)
from mocs_annotation.schema import (
    STATUS_GENERATION_ERROR,
    STATUS_OK,
    failure_record,
    parse_proposal,
    proposal_to_record,
)

logger = get_logger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

# Same 1.2 MP ceiling configs/sft.yaml applies to every training and inference call
# in this repo. Non-negotiable here: MOCS val contains a 14.63 MP image, and an
# uncapped image of that size expands to ~14,000 vision tokens. That is the exact
# shape of the 92.97 GiB OOM recorded in models/model_loader.py's own comment.
DEFAULT_MAX_PIXELS = 1204224
DEFAULT_MIN_PIXELS = 200704


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_done_ids(path: Path) -> set:
    """Ids already present in proposals.jsonl, so a resume never double-writes.

    Tolerates a truncated final line, which is what a walltime kill or a node
    failure mid-write leaves behind.
    """
    done: set = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["new_image_id"])
            except Exception:
                logger.warning(f"{path.name}:{lineno} is not parseable -- ignoring that line")
    return done


def _write_json_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_annotator(
    model_id: str,
    max_pixels: int,
    min_pixels: int,
    load_4bit: bool = False,
    attn_implementation: str = "sdpa",
):
    """Loads the teacher VLM and its processor, with the project's pixel cap applied.

    Reuses models/model_loader.py::apply_pixel_bounds rather than reimplementing it.
    That function is importable without unsloth (unsloth is imported inside the
    training functions, not at module scope) and it carries two hard-won details:
    the Qwen image processor stores pixel AREAS under the misleadingly named
    "shortest_edge"/"longest_edge" keys, and min_pixels/max_pixels are read-only
    properties on transformers 5.4.0 so writing them must be best-effort.
    """
    import torch
    from transformers import AutoProcessor

    from models.model_loader import apply_pixel_bounds

    logger.info(f"Loading annotator model: {model_id} (4bit={load_4bit}, attn={attn_implementation})")

    kwargs: Dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": "cuda:0",
        "attn_implementation": attn_implementation,
    }
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs.pop("dtype")

    model = None
    errors: List[str] = []
    # Qwen3-VL's own class first; the generic multimodal class as a fallback so a
    # transformers version bump that renames or relocates it does not break the job.
    for loader_name in ("Qwen3VLForConditionalGeneration", "AutoModelForImageTextToText"):
        try:
            import transformers

            cls = getattr(transformers, loader_name)
        except AttributeError:
            errors.append(f"{loader_name}: not present in this transformers build")
            continue
        try:
            model = cls.from_pretrained(model_id, **kwargs)
            logger.info(f"Loaded via {loader_name}")
            break
        except Exception as e:  # noqa: BLE001 - we want the next loader to get a turn
            errors.append(f"{loader_name}: {type(e).__name__}: {e}")

    if model is None:
        raise RuntimeError("Could not load the annotator model.\n  " + "\n  ".join(errors))

    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)
    apply_pixel_bounds(processor, min_pixels=min_pixels, max_pixels=max_pixels)

    # Left padding is required for batched decoder-only generation, same as
    # models/inference.py::generate_batch.
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(
            f"GPU: {props.name} | total {props.total_memory / 1e9:.1f} GB | "
            f"allocated after load {torch.cuda.memory_allocated() / 1e9:.1f} GB"
        )
    return model, processor


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def load_fewshot_images(fewshot: List[Dict[str, Any]], fewshot_path: str) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Loads the five example images ONCE, next to fewshot.json.

    Loaded up front and reused for every request -- decoding them per image would be
    2,000 x 5 redundant decodes. A block whose image will not load is DROPPED from the
    example set rather than sent text-only: an unanchored box list in front of the
    model is worse than one fewer example, because those coordinates then refer to
    nothing.
    """
    from PIL import Image

    base = Path(fewshot_path).parent
    kept: List[Dict[str, Any]] = []
    images: List[Any] = []
    for ex in fewshot:
        rel = ex.get("image_file")
        if not rel:
            logger.warning(
                f"few-shot block {ex.get('source')} has no 'image_file' -- dropping it. "
                "Re-run mocs_annotation.build_fewshot to regenerate the images."
            )
            continue
        path = base / rel
        try:
            images.append(Image.open(path).convert("RGB"))
            kept.append(ex)
        except Exception as e:
            logger.warning(f"could not load few-shot image {path}: {e} -- dropping that block")
    return kept, images


def generate_batch(
    model,
    processor,
    records: List[Dict[str, Any]],
    query_images: List[Any],
    fewshot: List[Dict[str, Any]],
    fewshot_images: List[Any],
    include_box_hints: bool,
    max_new_tokens: int,
    repetition_penalty: float,
    do_sample: bool,
    temperature: float,
) -> List[str]:
    """Greedy (by default) batched generation. Returns one completion per input.

    Each conversation carries the five example images plus the query image, so the
    per-request vision cost is 6 x ~1,176 tokens. That is the term that sets the batch
    size -- see --batch-size.
    """
    import torch
    from qwen_vl_utils import process_vision_info

    conversations = [
        build_messages(
            query_image=img,
            record=rec,
            fewshot=fewshot,
            fewshot_images=fewshot_images,
            include_box_hints=include_box_hints,
        )
        for rec, img in zip(records, query_images)
    ]
    texts = [
        processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
        for c in conversations
    ]

    image_inputs: List[Any] = []
    for c in conversations:
        imgs, _ = process_vision_info(c)
        if imgs:
            image_inputs.extend(imgs if isinstance(imgs, list) else [imgs])

    # NO truncation. models/inference.py truncates the prompt on purpose, because its
    # sequence is one short user turn and the risk is a prompt that overruns the
    # window. Here the sequence is a system turn + five demonstration turns + the
    # query, so the LAST thing in it is the query image and its box hints --
    # right-truncation would silently drop the very image being annotated and the
    # model would answer about a few-shot example instead. Qwen3-VL's context is far
    # larger than the ~8,600 tokens this builds, so the ceiling is GPU memory (see
    # --batch-size), not the window.
    inputs = processor(
        text=texts,
        images=image_inputs or None,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "use_cache": True,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    return processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    from PIL import Image

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proposals_path = out_dir / "proposals.jsonl"
    progress_path = out_dir / "progress.json"
    manifest_path = out_dir / "annotate_manifest.json"

    with open(args.selection, "r", encoding="utf-8") as f:
        selection = json.load(f)
    records: List[Dict[str, Any]] = selection["records"]
    images_root = Path(args.images_root or selection["manifest"]["images_root"])
    if not images_root.is_dir():
        raise SystemExit(f"images root does not exist: {images_root}")

    fewshot: List[Dict[str, Any]] = []
    fewshot_images: List[Any] = []
    if args.fewshot:
        with open(args.fewshot, "r", encoding="utf-8") as f:
            fewshot = json.load(f)
        fewshot, fewshot_images = load_fewshot_images(fewshot, args.fewshot)
        logger.info(
            f"Loaded {len(fewshot)} few-shot block(s) with images from {args.fewshot}: "
            f"{[e.get('label') for e in fewshot]}"
        )
        vis_per_image = args.max_pixels // 1024
        logger.info(
            f"Vision budget: {len(fewshot)} example image(s) + 1 query = "
            f"{(len(fewshot) + 1) * vis_per_image} tokens/request at batch 1, "
            f"{(len(fewshot) + 1) * vis_per_image * args.batch_size} at batch "
            f"{args.batch_size}"
        )
    else:
        logger.warning(
            "No --fewshot given. The teacher will not be format- or register-"
            "conditioned, which raises the parse-failure rate and produces reasons in "
            "the wrong voice. Run mocs_annotation.build_fewshot first."
        )

    done = _read_done_ids(proposals_path)
    if done:
        logger.info(f"Resuming: {len(done)} image(s) already in {proposals_path.name}, skipping them")

    todo = [r for r in records if r["new_image_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    logger.info(f"Selection holds {len(records)} image(s); {len(todo)} to annotate this run")

    if not todo:
        logger.info("Nothing to do.")
        return

    model, processor = load_annotator(
        args.model, args.max_pixels, args.min_pixels, args.load_4bit, args.attn_implementation
    )

    # Provenance. Same role as the LLM judge's llm_judge_status.json: a later yield
    # comparison between two batches is meaningless unless you can prove they ran
    # under the same instructions and the same decoding settings.
    _write_json_atomic({
        "model_id": args.model,
        "load_4bit": args.load_4bit,
        "attn_implementation": args.attn_implementation,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "batch_size": args.batch_size,
        "box_hints": args.box_hints,
        "fewshot_path": args.fewshot,
        "fewshot_count": len(fewshot),
        "fewshot_blocks": [
            {"label": e.get("label"), "source": e.get("source"), "image_file": e.get("image_file")}
            for e in fewshot
        ],
        "selection_path": str(args.selection),
        "selection_manifest": selection.get("manifest", {}),
        "images_root": str(images_root),
        "system_prompt": build_system_prompt(),
        # Fingerprints the CONTRACT -- system prompt + the five example answers --
        # not one request, so two batches can be compared for having run under
        # identical instructions even though their box hints differ per image.
        "prompt_sha256": prompt_fingerprint(fewshot),
        "n_selected": len(records),
        "n_this_run": len(todo),
    }, manifest_path)
    logger.info(f"Wrote run manifest to {manifest_path}")

    status_counts: Counter = Counter()
    bucket_hits: Counter = Counter()
    started = time.time()
    processed = 0

    # Append mode: the file already holds whatever a previous attempt finished, and
    # each line is one attempted image.
    out_f = open(proposals_path, "a", encoding="utf-8")
    try:
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start : start + args.batch_size]

            images: List[Any] = []
            usable: List[Dict[str, Any]] = []
            for rec in chunk:
                path = images_root / rec["file_name"]
                try:
                    img = Image.open(path).convert("RGB")
                except Exception as e:
                    line = failure_record(rec, STATUS_GENERATION_ERROR, f"image load failed: {e}")
                    out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    status_counts[STATUS_GENERATION_ERROR] += 1
                    continue
                images.append(img)
                usable.append(rec)

            if not usable:
                out_f.flush()
                continue

            # Batch first; on any failure fall back to one image at a time so a
            # single bad image cannot cost the whole batch. Unlike the training
            # path, a per-image failure here is RECORDED as a failure and never
            # written out as an empty success.
            try:
                completions = generate_batch(
                    model, processor, usable, images, fewshot, fewshot_images,
                    args.box_hints,
                    args.max_new_tokens, args.repetition_penalty,
                    args.do_sample, args.temperature,
                )
            except Exception as e:
                logger.warning(
                    f"Batch of {len(usable)} failed ({type(e).__name__}: {e}); retrying one by one"
                )
                # A failed batch is very often OOM, which leaves the allocator
                # fragmented; without this the one-by-one retry tends to OOM too and
                # the whole batch is lost for a reason that was recoverable.
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                completions = []
                for rec, img in zip(usable, images):
                    try:
                        completions.append(generate_batch(
                            model, processor, [rec], [img], fewshot, fewshot_images,
                            args.box_hints,
                            args.max_new_tokens, args.repetition_penalty,
                            args.do_sample, args.temperature,
                        )[0])
                    except Exception as inner:
                        completions.append(None)
                        logger.error(f"single-image generation failed: {inner}")

            for rec, raw in zip(usable, completions):
                if raw is None:
                    line = failure_record(
                        rec, STATUS_GENERATION_ERROR, "generation raised for this image"
                    )
                else:
                    proposal, status, error = parse_proposal(raw)
                    if proposal is None:
                        line = failure_record(rec, status, error or "unknown", raw_output=raw)
                    else:
                        line = proposal_to_record(proposal, rec)
                        if args.keep_raw:
                            line["raw_output"] = raw
                        for r in line["flagged_rules"]:
                            bucket_hits[f"{rec['selection_bucket']}::{r}"] += 1
                out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
                status_counts[line["status"]] += 1
                processed += 1

            out_f.flush()
            os.fsync(out_f.fileno())

            # EARLY ABORT. This job runs unattended for hours; without this, a
            # SYSTEMATIC failure -- the processor rejecting the six-image
            # conversation, a bad images root, an OOM that recurs on every batch --
            # would quietly burn the whole allocation writing 2,017 error records and
            # be discovered the next morning. Stop as soon as the evidence is in, and
            # say what the errors actually were. Whatever succeeded is already on
            # disk and a resubmit picks up from there.
            ok_so_far = status_counts[STATUS_OK]
            if processed >= args.abort_after and ok_so_far / max(processed, 1) < args.min_success_rate:
                recent = [
                    f"{k}={v}" for k, v in status_counts.most_common() if k != STATUS_OK
                ]
                raise RuntimeError(
                    f"Aborting: only {ok_so_far}/{processed} images produced a usable "
                    f"proposal (floor is {args.min_success_rate:.0%} after "
                    f"{args.abort_after} attempts). Failure mix: {', '.join(recent)}. "
                    f"Read the `error` and `raw_output` fields in {proposals_path} to see "
                    "why, fix it, then resubmit -- the successful records are kept and "
                    "will be skipped. Pass --abort-after 0 to disable this guard."
                )

            elapsed = time.time() - started
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = (len(todo) - processed) / rate if rate > 0 else float("nan")
            _write_json_atomic({
                "n_selected": len(records),
                "n_attempted_this_run": processed,
                "n_remaining_this_run": len(todo) - processed,
                "status_counts": dict(status_counts),
                "rule_hits_by_bucket": dict(bucket_hits),
                "elapsed_seconds": round(elapsed, 1),
                "images_per_second": round(rate, 3),
                "eta_seconds": None if rate <= 0 else round(remaining, 1),
            }, progress_path)

            if (start // max(args.batch_size, 1)) % args.log_every == 0:
                logger.info(
                    f"{processed}/{len(todo)}  ok={status_counts[STATUS_OK]}  "
                    f"{rate:.2f} img/s  eta {remaining / 60:.0f} min"
                )
    finally:
        out_f.close()

    logger.info("=" * 70)
    logger.info(f"Annotation pass complete in {(time.time() - started) / 60:.1f} min")
    for status, n in status_counts.most_common():
        logger.info(f"  {status:<20}{n:>6}")
    if bucket_hits:
        logger.info("Rule hits by selection bucket (proposals, NOT verified labels):")
        for key, n in sorted(bucket_hits.items()):
            logger.info(f"  {key:<44}{n:>5}")
    logger.info(f"Proposals -> {proposals_path}")
    logger.info(f"Progress  -> {progress_path}")
    logger.info("Next: python -m mocs_annotation.export_review --proposals ... --out review.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", required=True, help="selection.json from select_images.py")
    ap.add_argument("--fewshot", default=None, help="fewshot.json from build_fewshot.py")
    ap.add_argument("--out-dir", required=True, help="Directory for proposals.jsonl / progress.json")
    ap.add_argument("--images-root", default=None,
                     help="Override the images root recorded in selection.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=1,
                     help="1 is the safe default now that the few-shot carries five "
                          "IMAGES: six images per request is ~7,000 vision tokens, so a "
                          "sequence is ~9,000 tokens and its KV cache alone is ~2.3 GB. "
                          "On an 80 GB H100 (66 GB of bf16 weights, ~14 GB free) batch 2 "
                          "sits right at the edge and batch 4 will OOM. On a 141 GB H200, "
                          "4 is comfortable and 8 fits. Prove it with --limit 8 first")
    ap.add_argument("--box-hints", action="store_true",
                     help="OFF by default. Injects MOCS's human-annotated worker/machine "
                          "boxes for the QUERY image as text in the final turn. Off because "
                          "(a) the five demonstrations are image->json with no hint block, so "
                          "hinting only the query makes the one turn the model must answer "
                          "the one turn it has never been shown; (b) the geometry it would "
                          "improve is the teacher's rule_4 box, which export_review discards "
                          "in favour of mocs_suggested_rule4_box_1000 from the same MOCS "
                          "annotations; and (c) rule_4_geometric images always carry the "
                          "pair note while random_control ones mostly do not, which would "
                          "confound the bucket yield comparison the control exists to enable. "
                          "Turn on only to measure how much yield the priming is worth")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                     help="A ~50-word caption plus four violation objects is ~300 tokens, "
                          "so 1024 is generous headroom for an image with many instances "
                          "per rule. It also matches configs/tasks/violations_only.yaml's "
                          "own max_new_tokens. Costs nothing on a well-behaved completion "
                          "(greedy stops at EOS) and ~131 MB of KV cache per sequence; the "
                          "only real cost is that a degenerate decode loop burns twice as "
                          "long before being cut, which repetition_penalty 1.05 guards")
    ap.add_argument("--repetition-penalty", type=float, default=1.05,
                     help="1.05, NOT the pipeline's 1.0. That 1.0 is a deliberate, pinned "
                          "decision for the training/inference path (CLAUDE.md's "
                          "ghost-variable table) so results stay comparable. This is an "
                          "offline tool with no comparability constraint, and a teacher "
                          "generating free-form caption prose is exactly where a decode "
                          "loop appears -- the 2B baseline looped to the token cap on 80%% "
                          "of images. Set 1.0 to match the pipeline exactly")
    ap.add_argument("--do-sample", action="store_true",
                     help="Off by default: greedy makes the annotation pass reproducible, "
                          "matching the pipeline's do_sample=False inference")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    ap.add_argument("--load-4bit", action="store_true",
                     help="NF4 via bitsandbytes (~20 GB instead of ~66 GB). Escape hatch if "
                          "80 GB proves tight; costs some box precision")
    ap.add_argument("--attn-implementation", default="sdpa",
                     help="sdpa is always available; flash_attention_2 is faster if installed")
    ap.add_argument("--limit", type=int, default=None,
                     help="Annotate only the first N un-done images. USE THIS FIRST: "
                          "--limit 8 is the smoke test that proves the model loads, the "
                          "images resolve and the output parses")
    ap.add_argument("--keep-raw", action="store_true",
                     help="Also store the raw completion on successful records (bigger file, "
                          "useful while tuning the prompt)")
    ap.add_argument("--abort-after", type=int, default=16,
                     help="Stop the job once this many images have been attempted if the "
                          "success rate is below --min-success-rate. Exists because this "
                          "runs unattended: a systematic failure would otherwise spend the "
                          "whole allocation writing error records. 0 disables the guard")
    ap.add_argument("--min-success-rate", type=float, default=0.2,
                     help="Success floor for the --abort-after guard. Deliberately low -- "
                          "it is there to catch TOTAL failure, not a mediocre teacher")
    ap.add_argument("--log-every", type=int, default=10, help="Log every N batches")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
