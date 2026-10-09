#!/usr/bin/env python3
"""
Builds ``datasets/augmented_v3`` (SFT) and ``datasets/grpo_pool_v3`` (GRPO) from
ConstructionSite plus the human-verified MOCS harvest.

This is the one thing the ``violations_think`` arm was blocked on. The contract it
satisfies is ``V3_THINK_IMPLEMENTATION.md`` §8; the decisions behind every column are
in ``data/mocs_rows.py``'s docstring and in ``V3_DATA_COMBINE.md``.

TWO OUTPUTS, ONE SCRIPT, ON PURPOSE. The expensive and error-prone part is the
combine: load ConstructionSite, convert and repair the review rows, bake a ``thinking``
block for every row, and drop the ones that cannot carry one. Both outputs must be
built from *that exact* base -- if the SFT set and the GRPO pool disagreed about which
MOCS rows survived, or about which boxes rule_4 kept, nothing downstream would notice
and the arm's conclusions would be built on two different datasets. Running the
combine once and fanning out makes that impossible rather than merely unlikely.

ORDER OF OPERATIONS, and why it is this order::

    datasets/processed  (6308 train / 701 val / 3004 test, NOT augmented)
            |
            +-- + provenance column
            |
    review_results.json --> data/mocs_rows.py --> 265 rows + image refs
            |
            v
    base_train = CS train (6308) + MOCS (265)             <- un-augmented
            |
            +-- bake `thinking` for every train/val row; DROP the unbuildable
            |
            +----------------------------> grpo_pool_v3   <- from the UN-augmented base
            |                                                (duplicates would make
            |                                                 correlated reward groups)
            v
    augment rules 2/3/4 (x2 by default) --> augmented_v3

The GRPO pool branches off BEFORE augmentation for the reason
``data/build_grpo_pool.py``'s own docstring gives: a pixel-jittered near-duplicate
sitting in the same rollout pool produces a redundant, correlated reward group instead
of independent signal.

WHAT IS NOT TOUCHED, and why you can prove it:

  * ``datasets/augmented``, ``datasets/processed`` and ``datasets/grpo_pool`` are
    opened read-only and are REFUSED as output paths (:data:`PROTECTED_SUBDIRS`), so
    ``vo-*-v2``'s inputs, ``unified``, ``object_only`` and ``caption_only`` cannot be
    affected by anything here. MOCS rows reaching ``datasets/augmented`` would teach
    ``unified`` that excavators/rebar/hard-hats are absent on thousands of images
    where they were simply never annotated (CLAUDE.md; PLAN_V3_THINK.md trap 9).
  * ``data/augment_rare_classes.py`` is IMPORTED, never modified. Its
    ``RULE_MULTIPLIERS = {4: 16, 2: 12, 3: 6}`` is pinned by
    ``tests/test_core/test_blocker_fixes.py::test_v2_augmentation_multipliers_unchanged``,
    which regex-reads that line out of the source file, and ``datasets/augmented``
    must stay reproducible. The v3 counts are passed in as :option:`--rule-copies`
    and recorded in the manifest.

  * The ``test`` split is carried over from ``datasets/processed`` verbatim. It is what
    every arm is scored on, and ``experiments/run_inference.py:189`` /
    ``experiments/run_evaluation.py:247`` both load the DEFAULT root
    (``datasets/augmented``) regardless of task -- so comparability with v2 is
    automatic, and this split exists here only so that
    ``scripts/validate_think_dataset.py``'s identity check has something to compare.
    No MOCS row ever enters it.

``--rule-copies`` SEMANTICS DIFFER FROM ``RULE_MULTIPLIERS``, deliberately.
``RULE_MULTIPLIERS[4] = 16`` means *sixteen EXTRA copies* -- 17 rows per rule_4 image.
``--rule-copies 4=2`` means *two rows in TOTAL*, i.e. one extra. The v3 decision is
written as "augmentation drops to 2x", and "2x" meaning "appears twice" is the reading
that matches its own arithmetic (+452 duplicates on 452 rare images). Spelling it as a
total removes the off-by-one entirely; the manifest records both forms.

WHERE THIS RUNS. **ARC**, not the laptop. ``$VLM_DATA_ROOT`` is empty locally, and the
MOCS pixels live on the cluster. It needs no GPU, no model and no SLURM allocation --
it is CPU and disk only -- but it reads ~10k images and writes ~12k, so run it from a
login node with patience or as a small CPU batch job.

Usage::

    python data/build_v3_datasets.py --review $VLM_DATA_ROOT/datasets/review_results.json
    python data/build_v3_datasets.py --review ... --dry-run          # counts only, no writes
    python data/build_v3_datasets.py --review ... --smoke 200        # tiny rehearsal
    python data/build_v3_datasets.py --review ... --only pool        # rebuild just the pool
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datasets as _hf_datasets
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

# MUST come before any load_from_disk. `Dataset.add_column` on a `.select()`ed dataset
# calls `flatten_indices()`, which is a `map()`, which caches to a file NEXT TO THE
# SOURCE SHARDS -- i.e. it would write a multi-GB `cache-<fingerprint>.arrow` into
# `$VLM_DATA_ROOT/datasets/processed/train/`. That directory is on the ARC HOME quota
# and this script documents it as opened read-only. `.shuffle()` does the same with its
# indices file. Disabling the cache sends both to an auto-deleted temp dir instead.
_hf_datasets.disable_caching()

from core.constants import RULES
from core.io import ensure_dir, get_drive_path
from core.logging import get_logger
from core.think_format import THINK_FIELD, build_think_body, think_row_problems, violations_from_row
from data.mocs_rows import (
    PROVENANCE_CS,
    PROVENANCE_MOCS,
    MocsImageRef,
    build_mocs_rows,
    load_review_file,
)

logger = get_logger(__name__)

SEED = 42

# Writing any of these would damage a finished experiment or another task's inputs.
# `datasets/augmented` is the one that matters most: `unified` reads it, every v2
# number was produced from it, and MOCS rows in it would be silent poison.
PROTECTED_SUBDIRS = frozenset({
    "datasets/processed",
    "datasets/augmented",
    "datasets/grpo_pool",
    "datasets/raw",
    "datasets/raw_cleaned",
})

DEFAULT_RULE_COPIES = {4: 2, 2: 2, 3: 2}

# A separate, tighter ceiling for the harvest alone -- see the second drop-rate arm in
# main(). The corpus-wide --max-drop-rate is far too loose to notice a MOCS-specific
# problem, because MOCS is ~3.6% of the rows.
_MOCS_MAX_DROP_RATE = 0.02

# Only these are read for the string-only passes. Selecting them explicitly keeps the
# `image` column out of the view: iterating an HF dataset with an Image() feature
# decodes a fresh PIL object per row, which on 12k rows is minutes of work and
# gigabytes of RAM for what is pure string manipulation.
_TEXT_COLUMNS = ["image_id", "image_caption", "provenance"] + [f"{r}_violation" for r in RULES]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _git_metadata() -> Dict[str, Any]:
    """Commit and dirty flag, matching what the SFT/GRPO trainers record.

    A dataset built from an uncommitted tree is exactly as unreproducible as a run
    trained from one -- `README_v2.md` §13 P0-4 exists because nine manifests said
    `git_is_dirty: true`. Recorded here so the same question is answerable about the
    data, not only about the runs.
    """
    commit, dirty = "unknown", None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except Exception:  # not a checkout, or no git on PATH -- never fatal
        pass
    return {"git_commit": commit, "git_is_dirty": dirty}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _text_view(ds: Dataset) -> Dataset:
    """A view carrying only the string/struct columns, with `image` dropped."""
    keep = [c for c in _TEXT_COLUMNS if c in ds.column_names]
    return ds.select_columns(keep)


def _parse_rule_copies(spec: Optional[str]) -> Dict[int, int]:
    """``"4=2,2=2,3=2"`` -> ``{4: 2, 2: 2, 3: 2}``. TOTAL copies, not extra ones."""
    if not spec:
        return dict(DEFAULT_RULE_COPIES)
    out: Dict[int, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            rule_s, n_s = part.split("=")
            rule, n = int(rule_s), int(n_s)
        except ValueError:
            raise SystemExit(f"--rule-copies: cannot parse {part!r}; expected e.g. 4=2,2=2,3=2")
        if rule not in (2, 3, 4):
            raise SystemExit(f"--rule-copies: rule {rule} is not augmentable (only 2, 3, 4)")
        if n < 1:
            raise SystemExit(f"--rule-copies: rule {rule} wants {n} copies; the minimum is 1 "
                             "(1 = the original only, i.e. no augmentation)")
        out[rule] = n
    # An unnamed rule keeps its DEFAULT, not 1. Defaulting to 1 would make
    # `--rule-copies 4=3` silently stop augmenting rules 2 and 3 -- a one-flag way to
    # ship a dataset nobody meant to build. Disabling a rule stays possible, it just
    # has to be said: `--rule-copies 2=1`.
    for rule, default in DEFAULT_RULE_COPIES.items():
        out.setdefault(rule, default)
    return out


def _guard_outputs(out_sft: str, out_pool: str) -> None:
    """Refuses an output path that RESOLVES onto a protected directory.

    The comparison is on the resolved filesystem path, not on the string. A literal
    check (``subdir in PROTECTED_SUBDIRS``) is defeated by every other spelling of the
    same place -- ``./datasets/augmented``, ``datasets//augmented``,
    ``datasets/augmented/.``, ``datasets/../datasets/augmented`` -- all of which
    ``get_drive_path`` happily resolves onto the real directory. This is the single
    highest-value guard in the file: MOCS rows in ``datasets/augmented`` would teach
    ``unified``/``object_only`` that excavators, rebar and hard hats are *absent* on
    thousands of images where they were merely never annotated, and nothing downstream
    would notice.
    """
    protected = {}
    for sub in PROTECTED_SUBDIRS:
        try:
            protected[Path(get_drive_path(sub)).resolve()] = sub
        except OSError:                     # unresolvable root; nothing to protect
            continue
    for name, subdir in (("--out-sft-subdir", out_sft), ("--out-pool-subdir", out_pool)):
        try:
            resolved = Path(get_drive_path(subdir)).resolve()
        except OSError:
            continue
        if resolved in protected:
            raise SystemExit(
                f"{name}={subdir!r} resolves to {resolved}, a protected directory "
                f"({protected[resolved]}). Writing it would damage an existing "
                "experiment's inputs (CLAUDE.md: MOCS rows must never reach "
                "datasets/augmented)."
            )
    if Path(get_drive_path(out_sft)).resolve() == Path(get_drive_path(out_pool)).resolve():
        raise SystemExit(
            f"--out-sft-subdir and --out-pool-subdir both resolve to "
            f"{Path(get_drive_path(out_sft)).resolve()}. The flat pool would clobber the "
            "DatasetDict."
        )


def _warn_on_sibling_drift(manifest: Dict[str, Any], sibling_subdir: str) -> None:
    """Loudly compares this build's inputs/policies against the other output's manifest.

    The SFT set and the GRPO pool must describe the same combined base: if they
    disagree about which review file, which rule_4 box policy or which seed, nothing
    downstream notices and the arm's conclusions rest on two different datasets. Within
    one ``--only both`` run that is impossible; across two ``--only`` runs it is not.

    A warning rather than a refusal: rebuilding one half deliberately after changing a
    policy is a legitimate thing to do, and a hard stop would need its own override
    flag. The mismatch is also recorded, since both manifests are kept.
    """
    path = Path(get_drive_path(sibling_subdir)) / "build_manifest.json"
    try:
        other = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return                                  # no sibling yet, or not ours
    diffs = []
    for key in ("review_sha256", "source_subdir"):
        if other.get("inputs", {}).get(key) != manifest["inputs"].get(key):
            diffs.append(f"inputs.{key}: {other.get('inputs', {}).get(key)!r} vs "
                         f"{manifest['inputs'].get(key)!r}")
    if other.get("policies") != manifest["policies"]:
        diffs.append("policies differ")
    if other.get("seed") != manifest["seed"]:
        diffs.append(f"seed: {other.get('seed')} vs {manifest['seed']}")
    if diffs:
        logger.warning(
            f"The sibling output {sibling_subdir} was built from DIFFERENT inputs or "
            f"policies: {'; '.join(diffs)}. The SFT set and the GRPO pool would describe "
            "two different combined bases. Rebuild both with --only both unless this is "
            "deliberate."
        )


def _rule_counts(view: Dataset) -> Dict[str, int]:
    """Per-rule violation counts + safe + total, over a text-only view.

    Same shape as ``data/build_grpo_pool.py::count_by_rule``'s output, and like it the
    per-rule counts are NOT mutually exclusive -- an image can trip several rules, so
    they can sum past ``total - safe``.
    """
    out = {r: 0 for r in RULES}
    safe = 0
    for row in view:
        hit = False
        for r in RULES:
            if row.get(f"{r}_violation") is not None:
                out[r] += 1
                hit = True
        safe += not hit
    out["safe"] = safe
    out["total"] = len(view)
    return out


# ---------------------------------------------------------------------------
# step 1 -- the MOCS rows, with their pixels
# ---------------------------------------------------------------------------

def resolve_mocs_images(
    refs: Sequence[MocsImageRef],
    images_root: Optional[str],
    review_images: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    """Reads the bytes for each MOCS image. Returns (image values, tally, missing ids).

    Three candidate locations, tried in this order, because they are in decreasing
    order of fidelity:

      1. ``image_path_original`` -- the absolute path the image had on ARC when it was
         annotated (``.../datasets/filtered/instances_{val,test}/<file_name>``). These
         are the untouched MOCS JPEGs.
      2. ``<images_root>/instances_<source>/<file_name>`` -- the same files, if the
         data root has moved since annotation.
      3. ``<review_images>/<review_image>`` -- the staged render the review UI showed.
         Verified to be the same pixel dimensions as the original for all 266 rows
         (nothing exceeded the 1600 px render cap) and carries no burned-in boxes, but
         it is a q88 JPEG re-encode, so it is the fallback rather than the default.

    The bytes are passed through verbatim as ``{"bytes", "path"}``, which is the HF
    ``Image()`` feature's own storage form -- no decode, no re-encode, no resize.
    """
    values: List[Dict[str, Any]] = []
    tally: Counter = Counter()
    missing: List[str] = []

    for ref in refs:
        candidates: List[Tuple[str, Path]] = []
        if ref.path_original:
            candidates.append(("image_path_original", Path(ref.path_original)))
        if images_root and ref.file_name:
            candidates.append(("images_root",
                               Path(images_root) / f"instances_{ref.source}" / ref.file_name))
            candidates.append(("images_root_flat", Path(images_root) / ref.file_name))
        if review_images:
            candidates.append(("review_render", Path(review_images) / ref.review_image))

        for label, path in candidates:
            try:
                data = path.read_bytes()
            except OSError:
                continue
            values.append({"bytes": data, "path": path.name})
            tally[label] += 1
            break
        else:
            missing.append(ref.image_id)
            values.append(None)

    return values, dict(tally), missing


# ---------------------------------------------------------------------------
# step 2 -- the thinking column
# ---------------------------------------------------------------------------

def bake_thinking(ds: Dataset, split_name: str) -> Tuple[Dataset, List[Dict[str, str]]]:
    """Adds the ``thinking`` column, dropping rows whose block cannot be built.

    Returns ``(dataset_with_thinking, offenders)``.

    EVERY row goes through ``core/think_format.py::build_think_body`` -- the executable
    definition of the format -- and then through ``think_row_problems``, the same
    validator ``data/preprocessor.py`` and ``scripts/validate_think_dataset.py`` use.
    Baking and validating in one pass means a block that this build emits cannot be one
    the SFT target builder later refuses; the standalone validator is still the gate to
    run before training, but it should have nothing left to find.

    DROPPING is the only coherent policy for a row whose block is unbuildable (a blank
    caption, or a violation asserted with no reason):

      * keeping it with an empty block would make ``violations_think``'s SFT job raise
        on that row -- ``_build_violations_think_target_json`` refuses, by design,
        because a bad block is invisible to every downstream metric;
      * keeping it only for ``violations_only`` would put the v3 and v4 arms on
        different data and turn "does the block help?" into a two-variable question.

    So both arms see the identical dataset, and the drops are listed in the manifest.
    """
    view = _text_view(ds)
    keep: List[int] = []
    bodies: List[str] = []
    offenders: List[Dict[str, str]] = []

    for i in range(len(view)):
        row = view[i]
        try:
            body = build_think_body(row.get("image_caption"), violations_from_row(row))
        except ValueError as exc:
            offenders.append({"split": split_name,
                              "image_id": str(row.get("image_id")),
                              "provenance": str(row.get("provenance")),
                              "problem": str(exc)})
            continue
        # Belt and braces: build_think_body cannot by construction emit a body its own
        # validator rejects, but this build is the only place both sides exist at once,
        # and the check is pure string work.
        problems = think_row_problems({**row, THINK_FIELD: body})
        if problems:
            offenders.append({"split": split_name,
                              "image_id": str(row.get("image_id")),
                              "provenance": str(row.get("provenance")),
                              "problem": "; ".join(problems)})
            continue
        keep.append(i)
        bodies.append(body)

    kept = ds.select(keep) if len(keep) != len(ds) else ds
    return kept.add_column(THINK_FIELD, bodies), offenders


# ---------------------------------------------------------------------------
# step 3 -- augmentation
# ---------------------------------------------------------------------------

def augment_rare(train: Dataset, rule_copies: Dict[int, int], seed: int) -> Tuple[Dataset, Dict[str, Any]]:
    """Pixel-only duplication of rules 2/3/4, at the v3 copy counts.

    Reuses ``data/augment_rare_classes.py``'s own pipeline and ``augment_sample``
    verbatim rather than re-implementing them, so the transforms, the ``_aug{n}``
    ``image_id`` suffix and the "all other columns carried through" behaviour are
    identical to what produced ``datasets/augmented``. Only the COUNTS differ, and
    they are a parameter here instead of the module constant.

    Precedence is unchanged: rule_4 > rule_2 > rule_3, so an image tripping several
    rare rules is duplicated once, under its rarest rule.

    The import is deferred because the module hard-requires ``albumentations`` at
    import time; ``--only pool`` must work on a node that does not have it.
    """
    from data.augment_rare_classes import augment_sample, get_pixel_augmentation_pipeline

    view = _text_view(train)
    plan: List[Tuple[int, int, str]] = []      # (row index, extra copies, rule label)
    for i in range(len(view)):
        row = view[i]
        has = {r: row.get(f"rule_{r}_violation") is not None for r in (2, 3, 4)}
        for r in (4, 2, 3):                     # rarest first -- the existing precedence
            if has[r]:
                extra = rule_copies.get(r, 1) - 1
                if extra > 0:
                    plan.append((i, extra, f"rule_{r}"))
                break

    if not plan:
        # Still shuffled, so the written row order does not depend on whether any rare
        # image happened to exist -- otherwise "two builds produced different row
        # order" has two possible explanations instead of one.
        return train.shuffle(seed=seed), {
            "rows_generated": 0, "rare_images": 0, "by_rule": {}, "rng_seeded_via": "n/a",
        }

    # Reproducible pixel jitter, which the v2 build did not have. Seeding happens
    # BEFORE the pipeline is constructed: albumentations < 1.4.21 draws some parameters
    # at construction time from the globals, and albumentations is pinned nowhere in
    # this repo (not requirements.txt, not setup_arc.sh), so the ARC version is not
    # knowable from here. >= 1.4.21 carries its own RNG, which `set_random_seed`
    # reaches; which path was taken is recorded, so a rebuild that does not reproduce
    # byte-for-byte has a visible explanation rather than a mystery.
    seeded = "globals"
    import random as _random
    _random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed % (2 ** 32))
    except Exception:
        pass
    pipeline = get_pixel_augmentation_pipeline()
    try:
        pipeline.set_random_seed(seed)
        seeded = "albumentations.set_random_seed"
    except AttributeError:
        pass

    generated: List[Dict[str, Any]] = []
    by_rule: Counter = Counter()
    for idx, extra, label in plan:
        sample = train[idx]                     # decodes this one image
        by_rule[label] += 1
        for k in range(1, extra + 1):
            generated.append(augment_sample(sample, pipeline, k))

    aug_ds = Dataset.from_list(generated, features=train.features)
    combined = concatenate_datasets([train, aug_ds]).shuffle(seed=seed)
    return combined, {
        "rows_generated": len(aug_ds),
        "rare_images": len(plan),
        "by_rule": dict(by_rule),
        "rng_seeded_via": seeded,
    }


# ---------------------------------------------------------------------------
# step 4 -- the GRPO pool
# ---------------------------------------------------------------------------

def build_pool(train: Dataset, val: Dataset, seed: int) -> Tuple[Dataset, Dict[str, Any]]:
    """The v3 GRPO pool: the whole val split + every train violation + safe to ~50/50.

    The composition rule is ``data/build_grpo_pool.py``'s, unchanged -- its predicates
    are imported rather than restated so the two builders cannot drift -- applied to
    the COMBINED, un-augmented base. The ratio is what matters: ``p* = 0.298`` (the
    assert/abstain break-even that ``violation_tn_constant: 0.30`` sets) is solved
    against ~50/50, so a different ratio moves the model's operating point with nothing
    in the logs to show for it.

    MOCS rows need no admission filter here. The contract says "admit a MOCS row only
    if it carries a reason", and ``data/mocs_rows.py`` has already guaranteed it:
    an assertion with neither reason nor box was nulled, and one with a box but no
    reason does not exist in the file (measured: 0).
    """
    from data.build_grpo_pool import split_violation_safe

    val_view, train_view = _text_view(val), _text_view(train)
    val_viol, val_safe = split_violation_safe(val_view)
    train_viol, train_safe = split_violation_safe(train_view)

    total_violations = len(val_viol) + len(train_viol)
    safe_needed = max(0, total_violations - len(val_safe))
    if safe_needed > len(train_safe):
        logger.warning(
            f"Only {len(train_safe)} safe train images available but {safe_needed} are "
            "needed for 50/50; the pool will lean violation-heavy. Re-run "
            "scripts/validate_rewards.py --probe before training."
        )
        safe_needed = len(train_safe)

    import random
    rng = random.Random(seed)
    sampled_safe = sorted(rng.sample(train_safe, safe_needed))

    pool = concatenate_datasets([
        val, train.select(train_viol), train.select(sampled_safe)
    ]).shuffle(seed=seed)

    # The contract is explicit that the pool carries no `thinking` column: GRPO has no
    # target text, so there is nowhere for it to go (V3_THINK_IMPLEMENTATION.md §7).
    # Dropping it keeps the pool honest about what it is rather than shipping a column
    # that silently never trains anything.
    if THINK_FIELD in pool.column_names:
        pool = pool.remove_columns([THINK_FIELD])

    total_safe = len(val_safe) + safe_needed

    # MEASURED off the written pool, not re-derived from the same locals that built it.
    # The arithmetic above and the dataset can only disagree if `select()` took the
    # wrong indices -- which is exactly the bug a manifest computed from the locals
    # would hide, while still printing a perfect 50/50.
    measured = _rule_counts(_text_view(pool))
    measured_violation = measured["total"] - measured["safe"]
    if measured_violation != total_violations or measured["safe"] != total_safe:
        raise SystemExit(
            f"Pool composition does not match the arithmetic: counted "
            f"{measured_violation} violation / {measured['safe']} safe, expected "
            f"{total_violations} / {total_safe}. The row selection is wrong; do not train "
            "on this pool."
        )
    if abs(measured_violation - measured["safe"]) > 0.02 * max(measured["total"], 1):
        # Covers the branch build_grpo_pool.py never warned about: if val alone carries
        # more safe rows than there are violations, no top-up happens and the pool comes
        # out SAFE-heavy. p* = 0.298 is solved against ~50/50 either way.
        logger.warning(
            f"Pool is {100 * measured_violation / max(measured['total'], 1):.1f}% violation, "
            "not ~50%. violation_tn_constant's break-even p* is solved against 50/50 -- "
            "re-run scripts/validate_rewards.py --probe before training, and do NOT fix "
            "this by editing the constant."
        )

    stats = {
        "seed": seed,
        "val_total": len(val),
        "val_violation": len(val_viol),
        "val_safe": len(val_safe),
        "train_violation_taken": len(train_viol),
        "train_safe_available": len(train_safe),
        "train_safe_taken": safe_needed,
        "pool_total": len(pool),
        "pool_violation": total_violations,
        "pool_safe": total_safe,
        "pool_violation_pct": round(100 * total_violations / max(len(pool), 1), 2),
        "pool_safe_pct": round(100 * total_safe / max(len(pool), 1), 2),
        "pool_violation_measured": measured_violation,
        "pool_safe_measured": measured["safe"],
        "pool_by_rule": measured,
        "pool_by_provenance": dict(Counter(_text_view(pool)["provenance"])),
    }
    return pool, stats


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _steps(rows: int, unique_per_update: int = 32, epochs: int = 2) -> int:
    """``floor(rows / 32) * 2`` -- the repo's step arithmetic, for the manifest.

    Effective SFT batch is 32 (``configs/model_registry.yaml``: 32 x 1 at every tier)
    with ``dataloader_drop_last: true`` and ``num_train_epochs: 2``. GRPO reaches the
    same 32 by a different route (16 x 16 / num_generations 8).
    """
    return (rows // unique_per_update) * epochs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", required=True,
                    help="Path to review_results.json (schema mocs_review/2).")
    ap.add_argument("--source-subdir", default="datasets/processed",
                    help="Un-augmented ConstructionSite base, relative to VLM_DATA_ROOT "
                         "(default: datasets/processed).")
    ap.add_argument("--out-sft-subdir", default="datasets/augmented_v3",
                    help="Where to write the SFT DatasetDict (default: datasets/augmented_v3).")
    ap.add_argument("--out-pool-subdir", default="datasets/grpo_pool_v3",
                    help="Where to write the flat GRPO pool (default: datasets/grpo_pool_v3).")
    ap.add_argument("--mocs-images-root", default=None,
                    help="Directory holding instances_val/ and instances_test/ "
                         "(default: <data root>/datasets/filtered).")
    ap.add_argument("--mocs-review-images", default=None,
                    help="Fallback directory of staged review renders "
                         "(default: <data root>/datasets/mocs_annotation_combined/review/images).")
    ap.add_argument("--rule-copies", default=None, metavar="4=2,2=2,3=2",
                    help="TOTAL copies of each rare-rule image, original included. "
                         "Default 4=2,2=2,3=2. NOTE this is one more than the legacy "
                         "RULE_MULTIPLIERS convention, which counts EXTRA copies.")
    ap.add_argument("--rule4-box-policy", choices=("union", "keep"), default="union",
                    help="union (default): collapse multi-box rule_4 violations to one "
                         "enclosing box, per REVIEW_RUBRIC.md's measured GT convention "
                         "(69 of 70 GT rule_4 violations carry exactly one box). "
                         "keep: leave the reviewer's boxes untouched.")
    ap.add_argument("--only", choices=("both", "sft", "pool"), default="both",
                    help="Which output to write (default: both).")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-drop-rate", type=float, default=0.01,
                    help="Abort if more than this fraction of train+val rows cannot "
                         "carry a think block (default: 0.01). A large number means a "
                         "ground-truth problem to look at, not a dataset to shrink.")
    ap.add_argument("--smoke", type=int, default=None, metavar="N",
                    help="Rehearsal: truncate every ConstructionSite split to N rows. "
                         "Output subdirs get a '_smoke' suffix so a rehearsal can never "
                         "overwrite the real thing.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report every count and write nothing.")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.smoke is not None and args.smoke < 1:
        raise SystemExit(f"--smoke must be >= 1, got {args.smoke}. (0 would be read as "
                         "'no smoke mode' and write the REAL output directories.)")

    out_sft = args.out_sft_subdir
    out_pool = args.out_pool_subdir
    if args.smoke is not None:
        out_sft, out_pool = f"{out_sft}_smoke", f"{out_pool}_smoke"
    _guard_outputs(out_sft, out_pool)

    rule_copies = _parse_rule_copies(args.rule_copies)
    review_path = Path(args.review)

    print("=" * 78)
    print("Building the v3 datasets")
    print("=" * 78)
    logger.info(f"review file      : {review_path}")
    logger.info(f"source           : {args.source_subdir}")
    logger.info(f"outputs          : {out_sft} | {out_pool}  (only={args.only})")
    logger.info(f"rule copies      : {rule_copies}  (TOTAL per rare image, original included)")
    logger.info(f"rule_4 box policy: {args.rule4_box_policy}")

    # --- ConstructionSite base -------------------------------------------------
    source_path = get_drive_path(args.source_subdir)
    if not Path(source_path).exists():
        raise SystemExit(
            f"No dataset at {source_path}. This build runs on ARC: $VLM_DATA_ROOT must "
            "point at the real data root."
        )
    ds = load_from_disk(str(source_path))
    for split in ("train", "val", "test"):
        if split not in ds:
            raise SystemExit(f"{args.source_subdir} has no {split!r} split (found {list(ds)}).")
    if args.smoke is not None:
        ds = DatasetDict({k: v.select(range(min(args.smoke, len(v)))) for k, v in ds.items()})
    logger.info(f"ConstructionSite : " + "  ".join(f"{k}={len(v)}" for k, v in ds.items()))

    cs = DatasetDict({
        k: v.add_column("provenance", [PROVENANCE_CS] * len(v)) for k, v in ds.items()
    })

    # --- MOCS rows -------------------------------------------------------------
    try:
        review = load_review_file(review_path)
    except (OSError, ValueError) as exc:
        # A bad review file is an operator error like any other here, so it gets a
        # one-line message rather than a traceback.
        raise SystemExit(str(exc))
    mocs_dicts, refs, mocs_report = build_mocs_rows(review, rule4_box_policy=args.rule4_box_policy)
    logger.info(f"MOCS rows        : {mocs_report.rows_out} of {mocs_report.rows_in} accepted "
                f"({len(mocs_report.dropped)} dropped in conversion)")
    for d in mocs_report.dropped:
        logger.warning(f"  dropped {d.get('image_id')} [{d.get('cause')}]: {d.get('reason')}")
    if mocs_report.dropped:
        logger.warning(f"  drops by cause  : {mocs_report.dropped_by_cause}")
    if mocs_report.rule4_boxes_unioned:
        logger.info(f"  unioned rule_4 boxes on {len(mocs_report.rule4_boxes_unioned)} row(s)")

    if not mocs_dicts:
        # Without this the failure surfaces as `Dataset.from_list([], features=...)`
        # raising "Keys mismatch: ... and {}", which names the ConstructionSite
        # dataset_info and says nothing about the review file that is actually wrong.
        raise SystemExit(
            f"No usable MOCS rows from {review_path} ({mocs_report.rows_in} entries in, "
            f"{len(mocs_report.dropped)} dropped). Nothing to combine -- check that the "
            "review file is the one you meant to stage."
        )

    images_root = args.mocs_images_root or str(get_drive_path("datasets", "filtered"))
    review_images = args.mocs_review_images or str(
        get_drive_path("datasets", "mocs_annotation_combined", "review", "images"))
    image_values, image_tally, missing = resolve_mocs_images(refs, images_root, review_images)
    if missing:
        raise SystemExit(
            f"{len(missing)} MOCS image(s) could not be located, e.g. {missing[:5]}. "
            f"Looked at image_path_original, {images_root}/instances_<source>/ and "
            f"{review_images}. Pass --mocs-images-root / --mocs-review-images."
        )
    logger.info(f"MOCS images from : {image_tally}")

    # Features are taken from the ConstructionSite split rather than declared here. The
    # inner Arrow types of the violation struct (float64 vs float32, nullable or not)
    # are whatever `datasets/processed` was written with, and a column that is None in
    # every MOCS row would otherwise infer as Value('null') and refuse to concatenate.
    mocs_rows = [dict(r, image=v) for r, v in zip(mocs_dicts, image_values)]
    mocs_ds = Dataset.from_list(mocs_rows, features=cs["train"].features)

    base_train = concatenate_datasets([cs["train"], mocs_ds])
    logger.info(f"combined train   : {len(base_train)} "
                f"({len(cs['train'])} CS + {len(mocs_ds)} MOCS)")

    # --- the thinking column ---------------------------------------------------
    train_t, off_train = bake_thinking(base_train, "train")
    val_t, off_val = bake_thinking(cs["val"], "val")
    offenders = off_train + off_val
    checked = len(base_train) + len(cs["val"])
    drop_rate = len(offenders) / max(checked, 1)
    dropped_mocs = sum(1 for o in offenders if o["provenance"] == PROVENANCE_MOCS)
    dropped_cs = len(offenders) - dropped_mocs
    mocs_drop_rate = dropped_mocs / max(len(mocs_ds), 1)
    for o in offenders[:25]:
        logger.warning(f"  no block for {o['split']}/{o['image_id']} ({o['provenance']}): "
                       f"{o['problem']}")
    logger.info(f"think blocks     : {len(train_t)} train + {len(val_t)} val baked; "
                f"{len(offenders)} row(s) dropped ({drop_rate:.3%}) = "
                f"{dropped_cs} constructionsite + {dropped_mocs} mocs "
                f"({mocs_drop_rate:.2%} of the harvest)")
    # Three arms, MOST SPECIFIC FIRST, so the message names the real problem rather
    # than the aggregate it also happens to trip.
    #
    # 1. val is load-bearing in a way train is not: it must stay identical to
    #    ConstructionSite's so eval_loss remains comparable across v2 / v4 / v3, and
    #    scripts/validate_think_dataset.py only enforces the TEST id set -- a shrunken
    #    val would be flagged nowhere else.
    if len(val_t) != len(cs["val"]):
        raise SystemExit(
            f"The val split lost {len(cs['val']) - len(val_t)} row(s) to unbuildable think "
            "blocks. val must stay identical to ConstructionSite's or eval_loss is no "
            f"longer comparable with v2. Offenders: {off_val[:10]}"
        )
    # 2. The harvest alone. The corpus-wide rate cannot see a MOCS-specific failure:
    #    1% of ~7,274 rows is 72 rows of slack and the whole harvest is only ~265, so a
    #    reviewer who hit Enter mid-caption could lose a quarter of the thing this build
    #    exists to acquire without tripping anything, leaving only some warning lines
    #    scrolling past on a login node.
    if mocs_drop_rate > _MOCS_MAX_DROP_RATE:
        raise SystemExit(
            f"{dropped_mocs} of {len(mocs_ds)} verified MOCS rows ({mocs_drop_rate:.1%}) "
            f"cannot carry a think block, above {_MOCS_MAX_DROP_RATE:.0%}. The harvest is "
            "the entire point of this build -- fix the offending captions/reasons in the "
            f"review file rather than losing them. Offenders: "
            f"{[o for o in offenders if o['provenance'] == PROVENANCE_MOCS][:10]}"
        )
    # 3. The corpus as a whole, for a systemic ground-truth problem.
    if drop_rate > args.max_drop_rate:
        raise SystemExit(
            f"{len(offenders)} of {checked} rows ({drop_rate:.2%}) cannot carry a think "
            f"block, above --max-drop-rate {args.max_drop_rate:.2%}. That is a ground-truth "
            "problem (blank captions, or violations asserted with no reason), not something "
            "to silently drop. Inspect the list above before raising the threshold."
        )

    # The test split is carried verbatim and gets an EMPTY block: nothing is ever
    # trained on it (run_sft reads train+val only) and inference loads the default root
    # anyway, so a block here could only drift from the one the model is taught.
    test_t = cs["test"].add_column(THINK_FIELD, [""] * len(cs["test"]))

    # --- counts ----------------------------------------------------------------
    train_counts = _rule_counts(_text_view(train_t))
    val_counts = _rule_counts(_text_view(val_t))
    prov_counts = dict(Counter(_text_view(train_t)["provenance"]))

    pool = pool_stats = None
    if args.only in ("both", "pool"):
        pool, pool_stats = build_pool(train_t, val_t, args.seed)
        logger.info(f"GRPO pool        : {pool_stats['pool_total']} rows "
                    f"({pool_stats['pool_violation_pct']}% violation / "
                    f"{pool_stats['pool_safe_pct']}% safe) -> {_steps(pool_stats['pool_total'])} steps")

    aug_train = aug_stats = None
    if args.only in ("both", "sft"):
        aug_train, aug_stats = augment_rare(train_t, rule_copies, args.seed)
        logger.info(f"augmented train  : {len(aug_train)} rows "
                    f"(+{aug_stats['rows_generated']} from {aug_stats['rare_images']} rare images) "
                    f"-> {_steps(len(aug_train))} SFT steps")

    # --- manifest --------------------------------------------------------------
    manifest: Dict[str, Any] = {
        "built_by": "data/build_v3_datasets.py",
        **_git_metadata(),
        "seed": args.seed,
        "smoke": args.smoke,
        "inputs": {
            "review_file": str(review_path),
            "review_sha256": _sha256(review_path),
            "review_schema": review.get("schema"),
            "review_box_scale": review.get("box_scale"),
            "review_reviewer": review.get("reviewer"),
            "review_saved_at": review.get("saved_at"),
            "source_subdir": args.source_subdir,
            "source_rows": {k: len(v) for k, v in ds.items()},
            "mocs_images_root": images_root,
            "mocs_review_images": review_images,
            "mocs_image_sources": image_tally,
        },
        "policies": {
            "rule_copies_total": {f"rule_{k}": v for k, v in sorted(rule_copies.items())},
            "rule_copies_extra_legacy_form": {
                f"rule_{k}": v - 1 for k, v in sorted(rule_copies.items())},
            "rule4_box_policy": args.rule4_box_policy,
            "mocs_rows_go_to": "train only (val and test are ConstructionSite-only)",
            "unbuildable_block_policy": "drop the row, both arms, recorded below",
            "max_drop_rate": args.max_drop_rate,
        },
        "mocs_conversion": mocs_report.as_dict(),
        "think_block_drops": offenders,
        "think_block_drops_by_provenance": {PROVENANCE_CS: dropped_cs,
                                            PROVENANCE_MOCS: dropped_mocs},
        # `mocs_conversion.rows_out` is the count BEFORE the think-block pass; this is
        # how many MOCS rows actually reached the dataset.
        "mocs_rows_in_dataset": prov_counts.get(PROVENANCE_MOCS, 0),
        "splits": {
            "train": {"rows_unaugmented": len(train_t), "by_rule": train_counts,
                      "by_provenance_unaugmented": prov_counts},
            "val": {"rows": len(val_t), "by_rule": val_counts},
            "test": {"rows": len(test_t), "by_provenance": {PROVENANCE_CS: len(test_t)}},
        },
    }
    if aug_stats is not None:
        manifest["augmentation"] = {
            **aug_stats,
            "train_rows_after": len(aug_train),
            "sft_steps_expected": _steps(len(aug_train)),
            "transforms": "pixel-only (brightness/contrast, JPEG compression, gamma) via "
                          "data/augment_rare_classes.py::get_pixel_augmentation_pipeline; "
                          "no spatial transforms, so boxes and directional caption phrases "
                          "stay valid",
            "train_by_rule_after": _rule_counts(_text_view(aug_train)),
            # Augmented copies inherit their source row's provenance, so this is NOT
            # the unaugmented tally above -- and it IS what
            # scripts/validate_think_dataset.py's provenance histogram will print.
            "train_by_provenance_after": dict(Counter(_text_view(aug_train)["provenance"])),
        }
    if pool_stats is not None:
        manifest["grpo_pool"] = {**pool_stats, "grpo_steps_expected": _steps(pool_stats["pool_total"])}

    # A summary, not a truncated dump. Slicing json.dumps() at 4000 chars printed
    # syntactically invalid JSON, and this is the only thing a login-node operator
    # actually reads before deciding to let the build write.
    print("\n" + "-" * 78)
    print(json.dumps({k: manifest[k] for k in
                      ("git_commit", "git_is_dirty", "seed", "smoke", "policies",
                       "splits", "mocs_rows_in_dataset", "think_block_drops_by_provenance")
                      if k in manifest}, indent=2))
    for key in ("augmentation", "grpo_pool"):
        if key in manifest:
            print(f'"{key}": ' + json.dumps(
                {k: v for k, v in manifest[key].items() if not isinstance(v, str)}, indent=2))
    print("-" * 78)

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    # --- write -----------------------------------------------------------------
    # One process fanning out to both outputs makes them consistent by construction.
    # Two separate `--only` invocations do not, and the usage block advertises them,
    # so check the sibling's recorded inputs and policies before adding to the pair.
    if args.only == "sft":
        _warn_on_sibling_drift(manifest, out_pool)
    elif args.only == "pool":
        _warn_on_sibling_drift(manifest, out_sft)

    if aug_train is not None:
        out = get_drive_path(out_sft)
        ensure_dir(out)
        logger.info(f"Saving SFT dataset to {out}")
        DatasetDict({"train": aug_train, "val": val_t, "test": test_t}).save_to_disk(
            str(out), max_shard_size="100MB")
        (Path(out) / "build_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        logger.info(f"Wrote {out}/build_manifest.json")

    if pool is not None:
        out = get_drive_path(out_pool)
        ensure_dir(out)
        logger.info(f"Saving GRPO pool to {out}")
        pool.save_to_disk(str(out), max_shard_size="100MB")
        (Path(out) / "build_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        logger.info(f"Wrote {out}/build_manifest.json")

    print("\n" + "=" * 78)
    print("Done. Next, BEFORE any GPU time:")
    print(f"  python scripts/validate_think_dataset.py --subdir {out_sft} --strict")
    print(f"  python scripts/validate_rewards.py --task violations_think --probe --census "
          f"--pool-stats")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
