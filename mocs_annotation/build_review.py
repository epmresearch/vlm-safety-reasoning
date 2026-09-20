#!/usr/bin/env python3
"""
Turns the combined corpus into work packages a human can actually get through.

Takes proposals_all.jsonl (from combine.py) and writes a self-contained review/
directory: several CSVs split by what is worth reviewing first, plus a copy of every
queued image with the PROPOSED BOXES DRAWN ON IT.

WHY TIERS RATHER THAN ONE CSV. Measured on the real corpus, rule_3 is about half of
all proposals while rule_2 and rule_4 together are under a fifth -- and rule_2/rule_4
are the rules with only 53 and 42 training images today, so a verified row there is
worth several times one from rule_3. One flat CSV spends the reviewer's first ten
hours on the cheapest rows.

    review_sample.csv      ~150 rows drawn at random from everything flagged. DO THIS
                           FIRST. It is the only unbiased estimate of the teacher's
                           precision, and it tells you whether the rest of the queue
                           is worth opening at all.
    review_tier1.csv       rule_2 + rule_4. Scarcest rules, highest value per minute.
    review_tier2.csv       rule_3. Largest and cheapest; sample it before committing.
    review_tier3.csv       rule_1 only (optional, --include-rule1).
    review_negatives.csv   ~100 images where the teacher proposed NOTHING. One-sided
                           verification cannot see a miss, so this sample is the only
                           estimate of the false-negative rate -- and teacher recall is
                           the hard ceiling on the whole harvest.

WHY THE BOXES ARE DRAWN. Judging "is this box on the right worker" from
`[[402,338,498,742]]` in a spreadsheet cell is slow and error-prone. Seeing it on the
photograph is instant. At a few thousand rows that is the difference between forty
hours of review and roughly half that, for a few minutes of CPU.

Where MOCS geometry exists (val-sourced images only), the human-annotated
worker+machine union box is drawn too, in a different colour, because it is a better
rule_4 box than anything the teacher produced -- the literature's best zero-shot
violation IoU is ~23% against this repo's own 45.6%.

export_review.py still exists and is unchanged: it is the quick single-run look with
the bucket yield table. This is the multi-run, reviewer-facing builder.

Runs anywhere -- pure file I/O plus Pillow. No GPU, no model, no dataset.

Usage:
    python -m mocs_annotation.build_review \
        --corpus $VLM_DATA_ROOT/datasets/mocs_annotation_combined/proposals_all.jsonl \
        --out    $VLM_DATA_ROOT/datasets/mocs_annotation_combined/review
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.constants import RULES
from core.logging import get_logger
from mocs_annotation.schema import STATUS_OK

logger = get_logger(__name__)

# High-saturation hues chosen to stay legible against construction-site tones (grey
# concrete, brown earth, orange hi-vis), and distinct from each other.
RULE_COLOURS = {
    "rule_1": "#00E5FF",   # cyan
    "rule_2": "#FFEA00",   # yellow
    "rule_3": "#FF3D00",   # orange-red
    "rule_4": "#D500F9",   # magenta
}
MOCS_COLOUR = "#00E676"    # green -- the human-annotated suggestion, not the model's

TIERS = {
    "tier1": ("rule_2", "rule_4"),
    "tier2": ("rule_3",),
    "tier3": ("rule_1",),
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                logger.warning(f"{path.name}:{lineno} unparseable -- skipped")
    return out


def _boxes_to_1000(boxes: Optional[Sequence[Sequence[float]]]) -> str:
    """The corpus stores [0,1]; the reviewer sees 0-1000, the scale the model was
    asked to produce and the scale any correction should be written back in."""
    if not boxes:
        return ""
    return "; ".join(
        "[" + ", ".join(str(int(round(c * 1000))) for c in box) + "]" for box in boxes
    )


def _suggested_rule4_boxes(rec: Dict[str, Any]) -> List[List[float]]:
    return [p["union_box"] for p in (rec.get("worker_machine_pairs") or [])[:4] if "union_box" in p]


def _font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except Exception:
        return ImageFont.load_default()


def render(rec: Dict[str, Any], dest: Path, max_px: int, draw_mocs: bool,
           boxes: bool = True) -> bool:
    """Stage the image, optionally with every proposed box drawn on. True on success.

    `boxes=False` is the UI path: the app draws boxes on a canvas from the [0,1]
    coords instead, which is what makes per-rule layer toggles and box CORRECTION
    possible. Burning them in is only for the spreadsheet path, where the reviewer
    has no canvas.
    """
    from PIL import Image, ImageDraw

    src = rec.get("image_path")
    if not src or not Path(src).exists():
        return False
    try:
        img = Image.open(src)
        img = img.convert("RGB")
    except Exception as e:
        logger.warning(f"{rec['new_image_id']}: cannot open image ({e})")
        return False

    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        w, h = img.size

    if not boxes:
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, "JPEG", quality=88)
        return True

    draw = ImageDraw.Draw(img)
    line_w = max(2, round(min(w, h) / 300))
    font = _font(max(13, round(min(w, h) / 45)))

    # A rule_1 and a rule_4 box on the SAME worker is the common case (it is what the
    # few-shot rule_4 example itself shows), and two labels at the same corner render
    # on top of each other -- unreadable exactly where the reviewer needs both. Track
    # what has been placed and slide a colliding label down until it is clear.
    placed: List[tuple] = []

    def _free_slot(x: float, y: float, tw: float, th: float) -> tuple:
        for step in range(8):
            cand = (x, y + step * (th + 2), x + tw, y + step * (th + 2) + th)
            if not any(
                cand[0] < p[2] and p[0] < cand[2] and cand[1] < p[3] and p[1] < cand[3]
                for p in placed
            ):
                return cand
        return (x, y, x + tw, y + th)

    def box_and_label(box: Sequence[float], colour: str, label: str) -> None:
        # Corpus boxes are [0,1] xyxy; scale to this (possibly resized) image.
        x1, y1, x2, y2 = (box[0] * w, box[1] * h, box[2] * w, box[3] * h)
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=line_w)
        try:
            tw = draw.textlength(label, font=font) + 8
            th = font.size + 4
        except Exception:
            tw, th = 8 * len(label) + 8, 16
        lx = min(max(0.0, x1), max(0.0, w - tw))       # never clip off the right edge
        ly = y1 - th if y1 - th >= 0 else y1           # below the top edge if no room above
        slot = _free_slot(lx, ly, tw, th)
        placed.append(slot)
        draw.rectangle(list(slot), fill=colour)
        draw.text((slot[0] + 4, slot[1] + 2), label, fill="#000000", font=font)

    for r in RULES:
        v = rec.get(f"{r}_violation")
        if not isinstance(v, dict):
            continue
        for box in v.get("bounding_box") or []:
            if len(box) == 4:
                box_and_label(box, RULE_COLOURS[r], r)

    if draw_mocs:
        for box in _suggested_rule4_boxes(rec):
            box_and_label(box, MOCS_COLOUR, "MOCS r4")

    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=88)
    return True


def row_for(rec: Dict[str, Any], image_file: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "new_image_id": rec.get("new_image_id"),
        "image_file": image_file,
        "mocs_image_id": rec.get("mocs_image_id"),
        "file_name": rec.get("file_name"),
        "source": rec.get("source"),
        "run": rec.get("run"),
        "proposed_rules": " ".join(rec.get("flagged_rules") or []),
        "n_flagged": rec.get("n_flagged", 0),
        "caption": rec.get("image_caption", ""),
        "mocs_categories": " ".join(rec.get("mocs_categories") or []),
        "mocs_suggested_rule4_box_1000": _boxes_to_1000(_suggested_rule4_boxes(rec)),
        "image_path_original": rec.get("image_path", ""),
    }
    for r in RULES:
        v = rec.get(f"{r}_violation")
        row[f"{r}_proposed"] = 1 if isinstance(v, dict) else 0
        row[f"{r}_reason"] = v.get("reason", "") if isinstance(v, dict) else ""
        row[f"{r}_boxes_1000"] = _boxes_to_1000(v.get("bounding_box")) if isinstance(v, dict) else ""
    # Reviewer fills these. Deliberately blank. The column set is IDENTICAL to what
    # the UI's export produces (review_ui.py::doExport), so whichever way the review
    # happens the file coming back downstream has the same shape.
    row.update({
        "verify_decision": "", "verify_rule_1": "", "verify_rule_2": "",
        "verify_rule_3": "", "verify_rule_4": "", "verify_caption_ok": "",
        "verify_corrected_reason": "", "verify_corrected_reason_json": "",
        "verify_corrected_boxes_1000": "",
        "verify_corrected_boxes_json": "", "verify_difficulty": "", "verify_notes": "",
    })
    return row


FIELDNAMES = (
    ["new_image_id", "image_file", "mocs_image_id", "file_name", "source", "run",
     "proposed_rules", "n_flagged", "caption"]
    + [f"{r}_{s}" for r in RULES for s in ("proposed", "reason", "boxes_1000")]
    + ["mocs_categories", "mocs_suggested_rule4_box_1000", "image_path_original",
       "verify_decision", "verify_rule_1", "verify_rule_2", "verify_rule_3",
       "verify_rule_4", "verify_caption_ok", "verify_corrected_reason",
       "verify_corrected_reason_json", "verify_corrected_boxes_1000",
       "verify_corrected_boxes_json", "verify_difficulty", "verify_notes"]
)


def write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    # utf-8-sig so Excel opens it without mangling accents.
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            n += 1
    return n


REVIEWER_README = """# MOCS annotation review

Every row here is a **proposal from a model**, not a label. Your acceptance is what
makes it data.

## How to work

1. Open a `review_*.csv` in Excel or LibreOffice.
2. For each row, open `images/<image_file>` — the proposed boxes are already drawn on
   it, colour-coded per rule.
3. Fill the `verify_*` columns. Leave everything else untouched.

## The colours

| colour | meaning |
|---|---|
| cyan | proposed rule_1 — basic PPE |
| yellow | proposed rule_2 — safety harness at height |
| orange-red | proposed rule_3 — edge / excavation protection |
| magenta | proposed rule_4 — person in a machine's operating radius |
| **green — "MOCS r4"** | **human-annotated** worker+machine box from the MOCS dataset |

## Two things that are easy to get wrong

**1. On every row you accept, fill ALL FOUR rules AND the caption — not just the
proposed rule.** About one in ten construction images violates rule_1 (missing hard hat
/ uncovered shoulders or legs). If you accept a row and leave `verify_rule_1` blank when
the image really does show a rule_1 violation, you inject a false negative into the
strongest rule in the project. Mark each of `verify_rule_1` … `verify_rule_4` as `y` or
`n`, and `verify_caption_ok` as `y` or `n` — a rule left blank is read as "not
violated", and a caption left blank cannot be used at all.

**2. Do not trust the model's boxes.** Where a green "MOCS r4" box exists, prefer it —
it comes from human annotation. Elsewhere, if the box is wrong but the finding is
right, write a corrected box into `verify_corrected_boxes_1000` using the same
`[xmin, ymin, xmax, ymax]` 0–1000 scale (0,0 = top-left).

## Column meanings

| column | fill with |
|---|---|
| `verify_decision` | `accept` / `reject` / `unsure` |
| `verify_rule_1..4` | `y` if that rule really is violated, `n` if not |
| `verify_caption_ok` | `y` / `n` — is the caption factually true of the image? |
| `verify_corrected_reason` | a better one-sentence reason, if the model's is wrong |
| `verify_corrected_boxes_1000` | `[x1, y1, x2, y2]`, 0–1000 scale; `;` between boxes |
| `verify_notes` | anything else |

`mocs_suggested_rule4_box_1000` is **empty for most rows on purpose** — MOCS's test
split ships no human annotations, so the suggestion only exists for val-sourced images.

## Order

Do `review_sample.csv` first. It is a random sample and it tells us how accurate the
model is overall. Then `review_tier1.csv` — those are the two rarest, most valuable
rules. `review_negatives.csv` is images where the model proposed *nothing*: check
whether it missed a real violation.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="proposals_all.jsonl from combine.py")
    ap.add_argument("--out", required=True, help="Review directory to create")
    ap.add_argument("--sample-size", type=int, default=150,
                    help="Rows in review_sample.csv -- the unbiased precision estimate")
    ap.add_argument("--negatives-size", type=int, default=100,
                    help="Rows in review_negatives.csv -- the false-negative estimate")
    ap.add_argument("--include-rule1", action="store_true",
                    help="Also write review_tier3.csv (rule_1 only). Off by default: "
                         "rule_1 already has 609 training images, the rare rules have 42-98")
    ap.add_argument("--max-rows-per-tier", type=int, default=None,
                    help="Cap each tier, e.g. to hand out a week of work at a time")
    ap.add_argument("--ui", action="store_true",
                    help="Also write a standalone offline review app (index.html) and "
                         "stage CLEAN images for it -- the app draws the boxes on a canvas, "
                         "so layers can be toggled and the reviewer can DRAW CORRECTED "
                         "BOXES. Without --ui the boxes are burnt into the jpgs for the "
                         "spreadsheet workflow instead")
    ap.add_argument("--reuse-images", action="store_true",
                    help="Do not stage images; use whatever is already in <out>/images/. "
                         "This is how you regenerate index.html LOCALLY after a UI tweak: "
                         "the corpus's `image_path` points at ARC, which does not exist on "
                         "a laptop, so a normal run would fail to render all 5,056 and "
                         "produce an app with no pictures")
    ap.add_argument("--no-render", action="store_true",
                    help="Skip staging images entirely (CSV only, much faster)")
    ap.add_argument("--max-render-px", type=int, default=1600,
                    help="Longest edge of a staged image. 1600 is plenty to judge a box "
                         "and keeps the folder small enough to hand over")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    corpus = _load_jsonl(Path(args.corpus))
    logger.info(f"Loaded {len(corpus)} record(s) from {Path(args.corpus).name}")

    ok = [r for r in corpus if r.get("status") == STATUS_OK]
    flagged = [r for r in ok if r.get("n_flagged", 0) > 0]
    empty = [r for r in ok if r.get("n_flagged", 0) == 0]
    logger.info(f"  ok={len(ok)}  with a proposal={len(flagged)}  nothing proposed={len(empty)}")

    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def pick(rules: Sequence[str]) -> List[Dict[str, Any]]:
        sel = [r for r in flagged if any(x in (r.get("flagged_rules") or []) for x in rules)]
        # Richer rows first -- more proposed rules means more to confirm per image
        # opened. No model confidence exists to sort on, and inventing one from an
        # unvalidated teacher would read meaningful without being so.
        sel.sort(key=lambda r: (-r.get("n_flagged", 0), r.get("new_image_id") or ""))
        return sel[: args.max_rows_per_tier] if args.max_rows_per_tier else sel

    queues: Dict[str, List[Dict[str, Any]]] = {}
    for name, rules in TIERS.items():
        if name == "tier3" and not args.include_rule1:
            continue
        if name == "tier2":
            # rule_3 rows already covered by tier1 would be reviewed twice.
            queues[name] = [r for r in pick(rules)
                            if not any(x in (r.get("flagged_rules") or []) for x in TIERS["tier1"])]
        else:
            queues[name] = pick(rules)

    sample_pool = list(flagged)
    rng.shuffle(sample_pool)
    queues["sample"] = sample_pool[: args.sample_size]

    neg_pool = list(empty)
    rng.shuffle(neg_pool)
    queues["negatives"] = neg_pool[: args.negatives_size]

    # ------------------------------------------------------------ render once
    needed: Dict[str, Dict[str, Any]] = {}
    for rows in queues.values():
        for r in rows:
            needed[r["new_image_id"]] = r

    images_dir = out_dir / "images"
    rendered: Dict[str, str] = {}
    if args.reuse_images:
        for rid in needed:
            if (images_dir / f"{rid}.jpg").exists():
                rendered[rid] = f"{rid}.jpg"
        logger.info(f"--reuse-images: found {len(rendered)}/{len(needed)} already staged "
                    f"in {images_dir}")
        if len(rendered) < len(needed):
            logger.warning(f"  {len(needed) - len(rendered)} image(s) missing -- those rows "
                           "will have no picture in the app")
    elif args.no_render:
        logger.info("--no-render: CSVs only, image_file column left empty")
    else:
        kind = "clean (the app draws boxes)" if args.ui else "with boxes drawn on"
        logger.info(f"Staging {len(needed)} unique image(s) {kind} -> {images_dir}")
        failures = 0
        for i, (rid, rec) in enumerate(sorted(needed.items()), start=1):
            name = f"{rid}.jpg"
            if render(rec, images_dir / name, args.max_render_px,
                      draw_mocs=bool(rec.get("worker_machine_pairs")),
                      boxes=not args.ui):
                rendered[rid] = name
            else:
                failures += 1
            if i % 500 == 0:
                logger.info(f"  {i}/{len(needed)}")
        logger.info(f"  done: {len(rendered)} staged, {failures} failed")

    # ------------------------------------------------------------ write CSVs
    logger.info("")
    written = {}
    for name, rows in queues.items():
        path = out_dir / f"review_{name}.csv"
        n = write_csv((row_for(r, rendered.get(r["new_image_id"], "")) for r in rows), path)
        written[name] = n
        logger.info(f"  review_{name + '.csv':<22} {n:>6} row(s)")

    (out_dir / "README_REVIEWER.md").write_text(REVIEWER_README, encoding="utf-8")

    # ------------------------------------------------------------ the app
    if args.ui:
        from mocs_annotation.review_ui import build_html

        in_queue: Dict[str, List[str]] = {}
        for name, rows in queues.items():
            for r in rows:
                in_queue.setdefault(r["new_image_id"], []).append(name)

        # Short keys: this JSON is INLINED into the html, so every byte is paid
        # ~5,000 times. Only what the app actually renders goes in -- no raw_output,
        # no prompt hash, no per-record provenance the reviewer cannot act on.
        payload = []
        for rid, rec in sorted(needed.items()):
            rules_out = {}
            for r in RULES:
                v = rec.get(f"{r}_violation")
                rules_out[r] = ({"p": 1, "reason": v.get("reason", ""),
                                 "boxes": [[round(c, 4) for c in b]
                                           for b in (v.get("bounding_box") or [])]}
                                if isinstance(v, dict) else {"p": 0})
            payload.append({
                "id": rid,
                "img": f"images/{rendered.get(rid, '')}" if rendered.get(rid) else "",
                "orig": rec.get("image_path", ""),
                "fn": rec.get("file_name", ""), "mid": rec.get("mocs_image_id"),
                "src": rec.get("source", ""), "run": rec.get("run", ""),
                "w": rec.get("width"), "h": rec.get("height"),
                "cap": rec.get("image_caption", ""),
                "r": rules_out,
                "cats": rec.get("mocs_categories") or [],
                "r4": [[round(c, 4) for c in b] for b in _suggested_rule4_boxes(rec)],
                "q": in_queue.get(rid, []),
            })

        # Keys the localStorage autosave by corpus, so two different review packages
        # on one machine cannot overwrite each other's work.
        corpus_key = f"{Path(args.corpus).stem}_{len(payload)}"
        html = build_html(
            title="MOCS safety-rule review",
            data_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            meta_json=json.dumps({"corpus_key": corpus_key, "n": len(payload)}),
        )
        (out_dir / "index.html").write_text(html, encoding="utf-8")
        logger.info(f"  {'index.html':<22} {len(html) / 1e6:>6.1f} MB "
                    f"({len(payload)} records inlined)")

    rule_counts = Counter()
    for r in flagged:
        for x in r.get("flagged_rules") or []:
            rule_counts[x] += 1

    summary = {
        "corpus": str(Path(args.corpus).resolve()),
        "records_total": len(corpus),
        "records_ok": len(ok),
        "records_with_a_proposal": len(flagged),
        "records_with_nothing_proposed": len(empty),
        "proposals_by_rule": {r: rule_counts[r] for r in RULES},
        "queues": written,
        "unique_images_staged": len(rendered),
        "rendered": not args.no_render,
        "max_render_px": args.max_render_px,
        "seed": args.seed,
        "tiers": {k: list(v) for k, v in TIERS.items()},
    }
    with open(out_dir / "review_manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info(f"Review package ready: {out_dir}")
    logger.info("  Hand over the WHOLE directory -- the CSVs reference images/ relatively.")
    logger.info("  Tell the reviewer to start with review_sample.csv (README_REVIEWER.md says so too).")


if __name__ == "__main__":
    main()
