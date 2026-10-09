"""
End-to-end tests for data/build_v3_datasets.py against a synthetic data root.

The real build only runs on ARC -- $VLM_DATA_ROOT is empty locally and the MOCS pixels
live on the cluster -- so these drive the whole script over a tiny fabricated
ConstructionSite dataset and a fabricated review file. That is enough to pin every
property the real build has to have, because none of them depend on scale:

  * no MOCS row reaches `val` or `test`
  * the `test` split's image_id set is unchanged (the comparability guarantee)
  * the image column stays named `image` and stays an Image() feature (invariant 1)
  * boxes stay in [0,1]
  * every train/val row's `thinking` passes `think_row_problems`
  * a contradictory / unbuildable row is refused, not silently written
  * `datasets/augmented` and friends are refused as output paths
  * the GRPO pool is ~50/50, flat, and carries no `thinking` column
  * the augmentation copy counts are what --rule-copies said, and the module-level
    RULE_MULTIPLIERS is untouched
"""
import io
import json
import os
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value, load_from_disk
from PIL import Image

from core.constants import GROUNDING_CLASSES, METADATA_FIELDS, RULES
from core.think_format import THINK_FIELD, think_row_problems
from data.mocs_rows import PROVENANCE_CS, PROVENANCE_MOCS, REVIEW_BOX_SCALE, REVIEW_SCHEMA

build_v3 = pytest.importorskip("data.build_v3_datasets")


# ---------------------------------------------------------------------------
# a synthetic data root
# ---------------------------------------------------------------------------

_BOX_FEATURE = [[Value("float64")]]
_VIOLATION_FEATURE = {"bounding_box": _BOX_FEATURE, "reason": Value("string")}


def _features():
    f = {"image": HFImage(), "image_id": Value("string"), "image_caption": Value("string")}
    for r in RULES:
        f[f"{r}_violation"] = dict(_VIOLATION_FEATURE)
    for c in GROUNDING_CLASSES:
        f[c] = _BOX_FEATURE
    for m in METADATA_FIELDS:
        f[m] = Value("string")
    f["resolution"] = Value("int64")
    return Features(f)


def _jpeg_bytes(w=16, h=12, colour=(20, 40, 60)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="JPEG")
    return buf.getvalue()


def _cs_row(image_id, *, caption=None, rule=None, boxes=None):
    row = {
        "image": {"bytes": _jpeg_bytes(), "path": f"{image_id}.jpg"},
        "image_id": image_id,
        "image_caption": caption if caption is not None else f"Scene {image_id}.",
        "resolution": 16 * 12,
    }
    for r in RULES:
        row[f"{r}_violation"] = None
    if rule:
        row[f"{rule}_violation"] = {
            "bounding_box": boxes if boxes is not None else [[0.1, 0.1, 0.5, 0.5]],
            "reason": f"A {rule} breach is visible.",
        }
    for c in GROUNDING_CLASSES:
        row[c] = []
    for m in METADATA_FIELDS:
        row[m] = "normal lighting"
    return row


def _make_processed(root: Path, *, n_train=40, n_val=8, n_test=6, bad_caption_rows=0,
                    bad_val_caption_rows=0):
    """A miniature `datasets/processed`: a few rare-rule rows, the rest safe."""
    train = []
    # 3 rule_4, 2 rule_2, 2 rule_3, 5 rule_1, rest safe -- mirrors the real skew.
    for i in range(3):
        train.append(_cs_row(f"cs_r4_{i}", rule="rule_4"))
    for i in range(2):
        train.append(_cs_row(f"cs_r2_{i}", rule="rule_2"))
    for i in range(2):
        train.append(_cs_row(f"cs_r3_{i}", rule="rule_3"))
    for i in range(5):
        train.append(_cs_row(f"cs_r1_{i}", rule="rule_1"))
    for i in range(bad_caption_rows):
        train.append(_cs_row(f"cs_bad_{i}", caption="   ", rule="rule_1"))
    while len(train) < n_train:
        train.append(_cs_row(f"cs_safe_{len(train)}"))

    val = [_cs_row(f"cs_val_{i}", rule="rule_1" if i < 2 else None) for i in range(n_val)]
    for i in range(bad_val_caption_rows):
        val.append(_cs_row(f"cs_val_bad_{i}", caption="  "))
    test = [_cs_row(f"cs_test_{i}") for i in range(n_test)]

    feats = _features()
    out = root / "datasets" / "processed"
    DatasetDict({
        "train": Dataset.from_list(train, features=feats),
        "val": Dataset.from_list(val, features=feats),
        "test": Dataset.from_list(test, features=feats),
    }).save_to_disk(str(out))
    return out


def _review_entry(image_id, images_dir: Path, *, caption=None, rules=None, width=16, height=12):
    (images_dir / f"{image_id}.jpg").write_bytes(_jpeg_bytes(width, height, (90, 10, 10)))
    row = {
        "new_image_id": image_id,
        "mocs_image_id": 1,
        "file_name": f"{image_id[5:]}.jpg",
        "source": "val",
        "run": "mocs_annotation",
        "image_path_original": f"/nonexistent/instances_val/{image_id[5:]}.jpg",
        "review_image": f"{image_id}.jpg",
        "width": width,
        "height": height,
        "image_caption": caption if caption is not None else f"A MOCS scene {image_id}.",
        "caption_usable": True,
        "caption_ok": "y",
        "caption_edited": False,
        "caption_model": "model caption",
        "mocs_categories": ["Worker"],
        "flagged_rules": sorted(rules or {}),
        "hard": False,
        "notes": "",
        "reviewed_at": "2026-10-05T00:00:00.000Z",
    }
    for r in RULES:
        v = (rules or {}).get(r)
        row[f"{r}_violation"] = None if v is None else {
            "bounding_box": v["boxes"], "reason": v["reason"],
            "provenance": {"proposed_by_model": True, "reason_edited": False,
                           "model_boxes_kept": len(v["boxes"]), "model_boxes_deleted": 0,
                           "reviewer_boxes": 0},
        }
    return row


def _make_review(root: Path, entries):
    path = root / "review_results.json"
    path.write_text(json.dumps({
        "schema": REVIEW_SCHEMA, "box_scale": REVIEW_BOX_SCALE, "reviewer": "T",
        "saved_at": "2026-10-07T00:00:00.000Z", "dataset_rows": entries,
    }), encoding="utf-8")
    return path


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("VLM_DATA_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def built(data_root):
    """Runs the real `main()` once; returns (sft DatasetDict, pool Dataset, manifest)."""
    _make_processed(data_root)
    images = data_root / "renders"
    images.mkdir()
    entries = [
        _review_entry("mocs_0000001", images,
                      rules={"rule_2": {"boxes": [[0.1, 0.1, 0.4, 0.8]],
                                        "reason": "No harness on the scaffold."}}),
        _review_entry("mocs_0000002", images,
                      rules={"rule_4": {"boxes": [[0.1, 0.2, 0.3, 0.6], [0.5, 0.1, 0.7, 0.8]],
                                        "reason": "Both workers are within the operation radius."}}),
        _review_entry("mocs_0000003", images,
                      rules={"rule_3": {"boxes": [[0.0, 0.0, 1.0, 0.9]],
                                        "reason": "The trench edge is unguarded."}}),
        _review_entry("mocs_0000004", images),                       # confirmed safe
        _review_entry("mocs_0000005", images, caption=""),           # must be dropped
    ]
    review = _make_review(data_root, entries)

    rc = build_v3.main([
        "--review", str(review),
        "--mocs-review-images", str(images),
        "--mocs-images-root", str(data_root / "nope"),
    ])
    assert rc == 0
    sft = load_from_disk(str(data_root / "datasets" / "augmented_v3"))
    pool = load_from_disk(str(data_root / "datasets" / "grpo_pool_v3"))
    manifest = json.loads((data_root / "datasets" / "augmented_v3" / "build_manifest.json")
                          .read_text(encoding="utf-8"))
    return sft, pool, manifest


# ---------------------------------------------------------------------------
# shape and isolation
# ---------------------------------------------------------------------------

def test_sft_dataset_has_the_three_splits_and_the_two_new_columns(built):
    sft, _, _ = built
    assert set(sft) == {"train", "val", "test"}
    for split in sft.values():
        assert THINK_FIELD in split.column_names
        assert "provenance" in split.column_names


def test_image_column_is_named_image_and_still_decodes(built):
    """Invariant 1: TRL's rollout code looks for exactly this key."""
    sft, pool, _ = built
    assert "image" in sft["train"].column_names and "images" not in sft["train"].column_names
    assert isinstance(sft["train"].features["image"], HFImage)
    assert isinstance(pool.features["image"], HFImage)
    assert sft["train"][0]["image"].size[0] > 0


def test_no_mocs_row_reaches_val_or_test(built):
    sft, _, _ = built
    for split in ("val", "test"):
        assert set(sft[split]["provenance"]) == {PROVENANCE_CS}
        assert not any(i.startswith("mocs_") for i in sft[split]["image_id"])


def test_test_split_image_ids_are_unchanged(built, data_root):
    """The comparability guarantee -- every arm is scored on the same images."""
    sft, _, _ = built
    source = load_from_disk(str(data_root / "datasets" / "processed"))
    assert set(sft["test"]["image_id"]) == set(source["test"]["image_id"])


def test_val_split_is_unchanged_so_eval_loss_stays_comparable(built, data_root):
    sft, _, _ = built
    source = load_from_disk(str(data_root / "datasets" / "processed"))
    assert set(sft["val"]["image_id"]) == set(source["val"]["image_id"])


def test_mocs_rows_are_in_train_and_tagged(built):
    sft, _, _ = built
    prov = dict(zip(sft["train"]["image_id"], sft["train"]["provenance"]))
    mocs = [i for i, p in prov.items() if p == PROVENANCE_MOCS]
    assert all(i.startswith("mocs_") for i in mocs)
    # Augmented copies inherit the provenance (augment_sample copies every column),
    # so count the originals: 5 accepted, 1 dropped for a blank caption.
    originals = [i for i in mocs if "_aug" not in i]
    assert len(originals) == 4, originals


def test_the_blank_caption_row_is_dropped_and_recorded(built):
    sft, _, manifest = built
    assert "mocs_0000005" not in set(sft["train"]["image_id"])
    dropped = {d["image_id"] for d in manifest["mocs_conversion"]["dropped"]}
    assert "mocs_0000005" in dropped


# ---------------------------------------------------------------------------
# the thinking column
# ---------------------------------------------------------------------------

def test_every_train_and_val_row_passes_the_row_validator(built):
    """The same check scripts/validate_think_dataset.py runs, over the real output."""
    sft, _, _ = built
    cols = ["image_id", "image_caption", THINK_FIELD] + [f"{r}_violation" for r in RULES]
    for split in ("train", "val"):
        view = sft[split].select_columns(cols)
        for row in view:
            assert think_row_problems(row) == [], f"{split}/{row['image_id']}"


def test_block_body_is_byte_identical_to_build_think_body(built):
    from core.think_format import build_think_body, violations_from_row
    sft, _, _ = built
    cols = ["image_id", "image_caption", THINK_FIELD] + [f"{r}_violation" for r in RULES]
    for row in sft["train"].select_columns(cols):
        assert row[THINK_FIELD] == build_think_body(row["image_caption"],
                                                    violations_from_row(row))


def test_block_has_no_coordinates_and_no_braces(built):
    """Coordinates ARE the answer; emitting them in the block as well as the JSON
    doubles the exposure to the malformed-box-array failure mode for no benefit."""
    sft, _, _ = built
    cols = ["image_id", THINK_FIELD] + [f"{r}_violation" for r in RULES]
    for row in sft["train"].select_columns(cols):
        body = row[THINK_FIELD]
        assert "{" not in body and "}" not in body and "```" not in body
        for r in RULES:
            v = row[f"{r}_violation"]
            if v is None:
                continue
            for box in v["bounding_box"]:
                for coord in box:
                    assert str(coord) not in body, f"{row['image_id']} leaks {coord}"


def test_test_split_block_is_empty_by_design(built):
    sft, _, _ = built
    assert set(sft["test"][THINK_FIELD]) == {""}


# ---------------------------------------------------------------------------
# boxes
# ---------------------------------------------------------------------------

def test_every_box_stays_in_0_1(built):
    sft, pool, _ = built
    for ds in (sft["train"], sft["val"], pool):
        for row in ds.select_columns([f"{r}_violation" for r in RULES]):
            for r in RULES:
                v = row[f"{r}_violation"]
                if v is None:
                    continue
                for box in v["bounding_box"]:
                    assert all(0.0 <= c <= 1.0 for c in box), box


def test_rule4_multibox_was_unioned(built):
    sft, _, manifest = built
    assert manifest["mocs_conversion"]["rule4_boxes_unioned_count"] == 1
    row = next(r for r in sft["train"].select_columns(["image_id", "rule_4_violation"])
               if r["image_id"] == "mocs_0000002")
    assert row["rule_4_violation"]["bounding_box"] == [[0.1, 0.1, 0.7, 0.8]]


# ---------------------------------------------------------------------------
# augmentation
# ---------------------------------------------------------------------------

def test_augmentation_doubles_rules_234_only(built):
    sft, _, manifest = built
    ids = sft["train"]["image_id"]
    aug = [i for i in ids if "_aug" in i]
    # CS: 3 rule_4 + 2 rule_2 + 2 rule_3 = 7.  MOCS: rule_2, rule_4, rule_3 = 3.
    assert len(aug) == 10, aug
    assert manifest["augmentation"]["rows_generated"] == 10
    # rule_1 and safe images are never duplicated.
    assert not any(a.startswith("cs_r1_") or a.startswith("cs_safe_") for a in aug)
    assert all(a.endswith("_aug1") for a in aug), "2 copies total means exactly 1 extra"


def test_rule_copies_is_honoured(data_root):
    _make_processed(data_root)
    images = data_root / "renders"
    images.mkdir()
    review = _make_review(data_root, [_review_entry("mocs_0000001", images)])
    assert build_v3.main([
        "--review", str(review), "--mocs-review-images", str(images),
        "--only", "sft", "--rule-copies", "4=3,2=1,3=1",
    ]) == 0
    train = load_from_disk(str(data_root / "datasets" / "augmented_v3"))["train"]
    aug = [i for i in train["image_id"] if "_aug" in i]
    assert len(aug) == 6, "3 rule_4 images x 2 extra copies; rule_2/rule_3 at 1 = no copies"
    assert sorted({a.rsplit("_aug", 1)[1] for a in aug}) == ["1", "2"]


def test_the_pinned_module_constant_is_untouched_on_disk():
    """tests/test_core/test_blocker_fixes.py pins this by reading the source; the v3
    build must parameterise rather than edit it."""
    import re
    src = (Path(__file__).resolve().parents[2] / "data" / "augment_rare_classes.py").read_text(
        encoding="utf-8")
    assert re.search(r"^RULE_MULTIPLIERS\s*=\s*\{4: 16, 2: 12, 3: 6\}", src, re.M)


def test_the_pinned_module_constant_is_untouched_at_runtime(built):
    """The source check above would still pass if the build imported the dict and
    mutated it in place, which would silently change `datasets/augmented`'s next
    rebuild."""
    from data.augment_rare_classes import RULE_MULTIPLIERS
    assert RULE_MULTIPLIERS == {4: 16, 2: 12, 3: 6}


# ---------------------------------------------------------------------------
# the GRPO pool
# ---------------------------------------------------------------------------

def test_pool_is_flat_and_carries_no_thinking_column(built):
    _, pool, _ = built
    assert isinstance(pool, Dataset)
    assert THINK_FIELD not in pool.column_names
    assert "provenance" in pool.column_names


def test_pool_is_about_half_violations_counted_from_the_pool_itself(built):
    """`pool_violation`/`pool_safe` are arithmetic over the same locals that built the
    pool, so asserting they are equal is an identity. Count the written rows instead:
    if `select()` ever took the wrong indices, only this catches it."""
    _, pool, manifest = built
    stats = manifest["grpo_pool"]
    viol = sum(1 for r in pool.select_columns([f"{x}_violation" for x in RULES])
               if any(r[f"{x}_violation"] is not None for x in RULES))
    safe = len(pool) - viol
    assert viol == safe, (viol, safe)
    assert viol == stats["pool_violation"] and safe == stats["pool_safe"]
    assert 45.0 <= 100 * viol / len(pool) <= 55.0


def test_pool_holds_no_augmented_duplicates(built):
    """Near-duplicates in one rollout pool make correlated reward groups, not signal."""
    _, pool, _ = built
    assert not any("_aug" in i for i in pool["image_id"])


def test_pool_contains_the_mocs_violations(built):
    _, pool, _ = built
    assert {"mocs_0000001", "mocs_0000002", "mocs_0000003"} <= set(pool["image_id"])


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("protected", [
    "datasets/augmented", "datasets/processed", "datasets/grpo_pool",
    # Every other spelling of the same place. A literal set-membership check lets all
    # of these through, and `get_drive_path` resolves them onto the real directory --
    # so the guard has to compare resolved paths, not strings.
    "./datasets/augmented", "datasets//augmented", "datasets/augmented/",
    "datasets/augmented/.", "datasets/../datasets/augmented",
])
def test_protected_output_paths_are_refused(data_root, protected):
    _make_processed(data_root)
    images = data_root / "renders"
    images.mkdir()
    review = _make_review(data_root, [_review_entry("mocs_0000001", images)])
    with pytest.raises(SystemExit, match="protected"):
        build_v3.main(["--review", str(review), "--mocs-review-images", str(images),
                       "--out-sft-subdir", protected])
    # and nothing was written on the way to refusing
    assert not (data_root / "datasets" / "augmented").exists()


def test_the_two_outputs_may_not_be_the_same_directory(data_root):
    _make_processed(data_root)
    images = data_root / "renders"
    images.mkdir()
    review = _make_review(data_root, [_review_entry("mocs_0000001", images)])
    with pytest.raises(SystemExit, match="clobber"):
        build_v3.main(["--review", str(review), "--mocs-review-images", str(images),
                       "--out-sft-subdir", "datasets/x", "--out-pool-subdir", "datasets/x"])


def test_zero_usable_mocs_rows_fails_with_a_readable_message(data_root):
    """Otherwise it surfaces as an Arrow 'Keys mismatch' naming the WRONG dataset."""
    _make_processed(data_root)
    images = data_root / "renders"
    images.mkdir()
    review = _make_review(data_root, [_review_entry("mocs_0000001", images, caption="")])
    with pytest.raises(SystemExit, match="No usable MOCS rows"):
        build_v3.main(["--review", str(review), "--mocs-review-images", str(images)])


def test_a_malformed_review_file_is_an_operator_error_not_a_traceback(data_root):
    _make_processed(data_root)
    images = data_root / "renders"
    images.mkdir()
    bad = data_root / "bad.json"
    bad.write_text(json.dumps({"schema": REVIEW_SCHEMA, "box_scale": "xyxy_0_1000",
                               "dataset_rows": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="box_scale"):
        build_v3.main(["--review", str(bad), "--mocs-review-images", str(images)])


def test_losing_a_val_row_aborts_the_build(data_root):
    """val must stay identical to ConstructionSite's or eval_loss stops being
    comparable with v2, and the think validator only enforces the TEST id set."""
    _make_processed(data_root, bad_val_caption_rows=1)
    images = data_root / "renders"
    images.mkdir()
    review = _make_review(data_root, [_review_entry("mocs_0000001", images)])
    with pytest.raises(SystemExit, match="val split lost"):
        build_v3.main(["--review", str(review), "--mocs-review-images", str(images)])


def test_losing_too_much_of_the_harvest_aborts_even_below_the_corpus_threshold(data_root):
    """1% of ~7k rows is 72 rows of slack against a ~265-row harvest; the harvest needs
    its own arm or a quarter of it could vanish into warning lines."""
    _make_processed(data_root)
    images = data_root / "renders"
    images.mkdir()
    entries = [_review_entry(f"mocs_000000{i}", images) for i in range(1, 9)]
    # 1 of 8 = 12.5% of the harvest, but only ~2% of the corpus -- under --max-drop-rate.
    entries[0]["image_caption"] = "a caption\nwith a newline"
    review = _make_review(data_root, entries)
    with pytest.raises(SystemExit, match="verified MOCS rows"):
        build_v3.main(["--review", str(review), "--mocs-review-images", str(images)])


def test_smoke_zero_is_rejected_rather_than_treated_as_off(data_root):
    """`if args.smoke:` would read 0 as 'no smoke mode' and write the REAL outputs."""
    _make_processed(data_root)
    images = data_root / "renders"
    images.mkdir()
    review = _make_review(data_root, [_review_entry("mocs_0000001", images)])
    with pytest.raises(SystemExit, match="--smoke must be"):
        build_v3.main(["--review", str(review), "--mocs-review-images", str(images),
                       "--smoke", "0"])


def test_a_partial_rule_copies_keeps_the_defaults_for_the_rules_you_did_not_name(data_root):
    _make_processed(data_root)
    images = data_root / "renders"
    images.mkdir()
    review = _make_review(data_root, [_review_entry("mocs_0000001", images)])
    assert build_v3.main(["--review", str(review), "--mocs-review-images", str(images),
                          "--only", "sft", "--rule-copies", "4=3"]) == 0
    m = json.loads((data_root / "datasets" / "augmented_v3" / "build_manifest.json")
                   .read_text(encoding="utf-8"))
    assert m["policies"]["rule_copies_total"] == {"rule_2": 2, "rule_3": 2, "rule_4": 3}
    train = load_from_disk(str(data_root / "datasets" / "augmented_v3"))["train"]
    aug = [i for i in train["image_id"] if "_aug" in i]
    # 3 rule_4 x 2 extra + 2 rule_2 x 1 + 2 rule_3 x 1 = 10
    assert len(aug) == 10, aug


def test_sub_threshold_drops_are_recorded_not_just_tolerated(data_root):
    """The production path is 'drop quietly and record it' -- the record is the only
    thing standing between a silent shrink and a noticed one."""
    _make_processed(data_root, n_train=400, bad_caption_rows=2)
    images = data_root / "renders"
    images.mkdir()
    review = _make_review(data_root, [_review_entry("mocs_0000001", images)])
    assert build_v3.main(["--review", str(review), "--mocs-review-images", str(images),
                          "--only", "sft"]) == 0
    m = json.loads((data_root / "datasets" / "augmented_v3" / "build_manifest.json")
                   .read_text(encoding="utf-8"))
    assert len(m["think_block_drops"]) == 2
    assert {d["image_id"] for d in m["think_block_drops"]} == {"cs_bad_0", "cs_bad_1"}
    assert m["think_block_drops_by_provenance"] == {PROVENANCE_CS: 2, PROVENANCE_MOCS: 0}
    ids = set(load_from_disk(str(data_root / "datasets" / "augmented_v3"))["train"]["image_id"])
    assert not any(i.startswith("cs_bad_") for i in ids)


def test_unbuildable_constructionsite_rows_abort_the_build(data_root):
    """A blank-caption CS row has no block; above the threshold the build must stop
    rather than quietly shrink the dataset."""
    _make_processed(data_root, bad_caption_rows=4)
    images = data_root / "renders"
    images.mkdir()
    review = _make_review(data_root, [_review_entry("mocs_0000001", images)])
    with pytest.raises(SystemExit, match="max-drop-rate"):
        build_v3.main(["--review", str(review), "--mocs-review-images", str(images)])


def test_missing_mocs_images_abort_the_build(data_root):
    _make_processed(data_root)
    images = data_root / "renders"
    images.mkdir()
    review = _make_review(data_root, [_review_entry("mocs_0000001", images)])
    (images / "mocs_0000001.jpg").unlink()
    with pytest.raises(SystemExit, match="could not be located"):
        build_v3.main(["--review", str(review), "--mocs-review-images", str(images)])


def test_dry_run_writes_nothing(data_root):
    _make_processed(data_root)
    images = data_root / "renders"
    images.mkdir()
    review = _make_review(data_root, [_review_entry("mocs_0000001", images)])
    assert build_v3.main(["--review", str(review), "--mocs-review-images", str(images),
                          "--dry-run"]) == 0
    assert not (data_root / "datasets" / "augmented_v3").exists()
    assert not (data_root / "datasets" / "grpo_pool_v3").exists()


def test_smoke_mode_cannot_overwrite_the_real_outputs(data_root):
    _make_processed(data_root)
    images = data_root / "renders"
    images.mkdir()
    review = _make_review(data_root, [_review_entry("mocs_0000001", images)])
    assert build_v3.main(["--review", str(review), "--mocs-review-images", str(images),
                          "--smoke", "12"]) == 0
    assert (data_root / "datasets" / "augmented_v3_smoke").exists()
    assert not (data_root / "datasets" / "augmented_v3").exists()


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------

def test_manifest_records_inputs_policies_and_provenance(built):
    _, _, m = built
    assert m["inputs"]["review_sha256"]
    assert m["inputs"]["review_schema"] == REVIEW_SCHEMA
    assert m["policies"]["rule_copies_total"] == {"rule_2": 2, "rule_3": 2, "rule_4": 2}
    assert m["policies"]["rule_copies_extra_legacy_form"] == {"rule_2": 1, "rule_3": 1,
                                                              "rule_4": 1}
    assert m["policies"]["rule4_box_policy"] == "union"
    assert m["splits"]["train"]["by_provenance_unaugmented"][PROVENANCE_MOCS] == 4
    assert m["mocs_rows_in_dataset"] == 4
    assert "git_commit" in m and "git_is_dirty" in m
    assert m["augmentation"]["sft_steps_expected"] == (
        m["augmentation"]["train_rows_after"] // 32) * 2


def test_manifest_counts_match_the_written_dataset(built):
    sft, pool, m = built
    assert m["augmentation"]["train_rows_after"] == len(sft["train"])
    assert m["grpo_pool"]["pool_total"] == len(pool)
    assert m["splits"]["val"]["rows"] == len(sft["val"])
    assert m["splits"]["test"]["rows"] == len(sft["test"])
    # The post-augmentation provenance tally is the one validate_think_dataset.py
    # prints; the unaugmented one above it is a different number by design.
    from collections import Counter
    assert m["augmentation"]["train_by_provenance_after"] == dict(
        Counter(sft["train"]["provenance"]))
    assert m["grpo_pool"]["pool_by_provenance"] == dict(Counter(pool["provenance"]))


# ---------------------------------------------------------------------------
# the two-outputs-one-combine guarantee
# ---------------------------------------------------------------------------

def test_only_sft_then_only_pool_reproduces_only_both(data_root, tmp_path, monkeypatch):
    """The whole fan-out design rests on this: two separate invocations must describe
    the same combined base as one."""
    _make_processed(data_root)
    images = data_root / "renders"
    images.mkdir()
    review = _make_review(data_root, [
        _review_entry("mocs_0000001", images,
                      rules={"rule_2": {"boxes": [[0.1, 0.1, 0.4, 0.8]], "reason": "No harness."}}),
        _review_entry("mocs_0000002", images),
    ])
    common = ["--review", str(review), "--mocs-review-images", str(images)]
    assert build_v3.main(common) == 0
    both_train = load_from_disk(str(data_root / "datasets" / "augmented_v3"))["train"]["image_id"]
    both_pool = load_from_disk(str(data_root / "datasets" / "grpo_pool_v3"))["image_id"]

    assert build_v3.main(common + ["--only", "sft", "--out-sft-subdir", "datasets/a"]) == 0
    assert build_v3.main(common + ["--only", "pool", "--out-pool-subdir", "datasets/b"]) == 0
    assert load_from_disk(str(data_root / "datasets" / "a"))["train"]["image_id"] == both_train
    assert load_from_disk(str(data_root / "datasets" / "b"))["image_id"] == both_pool
