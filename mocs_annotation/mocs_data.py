"""
MOCS (Moving Objects in Construction Sites) reader + candidate mining.

MOCS is a COCO-format object-detection dataset: 41,668 images from 174 construction
sites, 13 categories, 222,861 box+mask instances. It carries NO safety-rule labels
and NO captions -- it is a source of candidate IMAGES, not of annotations. See
docs/others_extra/dataset_paper.md Table 2.

What is actually on disk (measured 2026-09-18, data/filtering/):

    annotation_val.json    4,000 images   18,965 annotations   13 categories
    image_info_test.json  18,264 images   NO 'annotations' key  13 categories

So only the val file can be mined geometrically; the test file gives image
metadata only. Both are handled -- `has_annotations` tells you which you have.

Record shapes, verified against the real files:

    images[i]      {"height": int, "width": int, "id": int, "file_name": "0019406.jpg"}
    annotations[i] {"segmentation": [...], "iscrowd": 0, "image_id": int,
                    "bbox": [x, y, w, h], "area": float, "category_id": int, "id": int}
    categories[i]  {"supercategory": "Construction", "id": 1, "name": "Worker"}

IMPORTANT: `bbox` is COCO xywh in ABSOLUTE PIXELS. Everything this module returns
is converted to [xmin, ymin, xmax, ymax] normalised to [0, 1] -- the dataset-native
scale used by ConstructionSite ground truth (see data/box_utils.py). Predicted boxes
elsewhere in this repo are [0, 1000]; ground truth is [0, 1]. Do not mix them up.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Category groups
#
# MOCS's 13 categories, verbatim from the file:
#   Worker, Static crane, Hanging head, Crane, Roller, Bulldozer, Excavator,
#   Truck, Loader, Pump truck, Concrete mixer, Pile driving, Other vehicle
#
# Note what is NOT there: no scaffold, no harness, no trench/excavation, no guard
# rail, no hard hat. That is why only rule_4 is geometrically minable, and rule_2 /
# rule_3 have to rely on weak priors plus the VLM's own judgement.
# ---------------------------------------------------------------------------
CAT_WORKER = "Worker"

# Machines with an operating radius a person can stand inside -- the rule_4 shape
# ("a person is standing within the operating radius or blind spot of an excavator
# or other heavy machine"). Trucks and mixers are excluded by default: they move on
# roads rather than slewing through an arc, so worker-near-truck is a much weaker
# signal. Override with --machine-categories if you disagree.
MACHINE_CATEGORIES = (
    "Excavator",
    "Bulldozer",
    "Loader",
    "Roller",
    "Pile driving",
    "Crane",
    "Pump truck",
)

# Weak prior for rule_2 ("a person working at height ... is not wearing a safety
# harness"). MOCS cannot label height work, but tower/vehicle cranes and hanging
# hooks indicate vertical construction, where height work happens.
HEIGHT_HINT_CATEGORIES = ("Static crane", "Crane", "Hanging head", "Pile driving")

# Weak prior for rule_3 ("an open excavation, trench, pit or floor edge has no
# guard rail"). An excavator on site implies digging, hence an open excavation.
EXCAVATION_HINT_CATEGORIES = ("Excavator", "Bulldozer")


@dataclass
class MocsImage:
    """One MOCS image plus its annotations, boxes already normalised to [0, 1]."""

    image_id: int
    file_name: str
    width: int
    height: int
    # Which annotation file this came from ("val" / "test"), and where its jpg lives.
    # Both are carried per-image because a selection may draw from SEVERAL sources at
    # once, and their images sit in different directories.
    source: str = ""
    images_root: str = ""
    # (category_name, [xmin, ymin, xmax, ymax]) in [0, 1]
    boxes: List[Tuple[str, List[float]]] = field(default_factory=list)

    @property
    def stem(self) -> str:
        """Filename without extension -- the ONLY globally unique key.

        MOCS restarts its COCO `id` at 1 in every split, so val and test share all
        4,000 of val's ids. Filenames do not collide (val 19406-23406, test
        23407-41672), so everything here keys on the stem and `image_id` is kept only
        as provenance.
        """
        return self.file_name.rsplit(".", 1)[0]

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    @property
    def aspect_ratio(self) -> float:
        lo = min(self.width, self.height)
        return max(self.width, self.height) / lo if lo else 0.0

    def categories(self) -> set:
        return {name for name, _ in self.boxes}

    def boxes_of(self, names: Iterable[str]) -> List[List[float]]:
        wanted = set(names)
        return [b for name, b in self.boxes if name in wanted]


@dataclass
class MocsSplit:
    """One or more parsed MOCS annotation files, keyed by filename stem."""

    path: str
    images: Dict[str, MocsImage]
    category_names: List[str]
    has_annotations: bool

    def __len__(self) -> int:
        return len(self.images)


def merge_splits(*splits: "MocsSplit") -> "MocsSplit":
    """Combines several sources into one selectable pool.

    Safe because the key is the filename stem, not the COCO id. Merging on `id` would
    silently overwrite 4,000 val images with test images of the same id.

    `has_annotations` becomes True if ANY source had them; per-image presence is what
    the buckets actually test, so a mixed val+test pool mines the val images
    geometrically and still carries the test ones in the random bucket.
    """
    merged: Dict[str, MocsImage] = {}
    cats: List[str] = []
    any_ann = False
    paths: List[str] = []
    for sp in splits:
        collisions = set(merged) & set(sp.images)
        if collisions:
            raise ValueError(
                f"{len(collisions)} filename collision(s) between sources, e.g. "
                f"{sorted(collisions)[:3]}. Filenames are the unique key -- two sources "
                "sharing one cannot be merged safely."
            )
        merged.update(sp.images)
        any_ann = any_ann or sp.has_annotations
        paths.append(str(sp.path))
        for c in sp.category_names:
            if c not in cats:
                cats.append(c)
    return MocsSplit(path=" + ".join(paths), images=merged,
                      category_names=cats, has_annotations=any_ann)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def coco_xywh_to_xyxy_norm(bbox: Sequence[float], width: int, height: int) -> Optional[List[float]]:
    """COCO [x, y, w, h] in absolute pixels -> [xmin, ymin, xmax, ymax] in [0, 1].

    Returns None for a degenerate or unusable box rather than emitting one that
    data/box_utils.py::is_valid_box would silently drop later.
    """
    if not bbox or len(bbox) != 4 or not width or not height:
        return None
    try:
        x, y, w, h = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    box = [x / width, y / height, (x + w) / width, (y + h) / height]
    # Clamp: a few MOCS boxes run a pixel or two past the frame edge.
    box = [min(1.0, max(0.0, c)) for c in box]
    if box[2] - box[0] <= 1e-4 or box[3] - box[1] <= 1e-4:
        return None
    return box


def _expand(box: Sequence[float], margin_frac: float) -> List[float]:
    """Grow a box outward by `margin_frac` of its longest side, clamped to [0, 1].

    This is the operating-radius proxy. A slewing excavator's danger zone extends
    well past its own bounding box, so a worker standing just outside the box is
    still the rule_4 shape; margin 0 would only catch workers already overlapping
    the machine's pixels.
    """
    x1, y1, x2, y2 = box
    m = margin_frac * max(x2 - x1, y2 - y1)
    return [max(0.0, x1 - m), max(0.0, y1 - m), min(1.0, x2 + m), min(1.0, y2 + m)]


def _intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    return (min(a[2], b[2]) - max(a[0], b[0])) > 0 and (min(a[3], b[3]) - max(a[1], b[1])) > 0


def _union_box(boxes: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def worker_machine_pairs(
    img: MocsImage,
    machine_categories: Sequence[str] = MACHINE_CATEGORIES,
    margin_frac: float = 0.25,
) -> List[Dict[str, Any]]:
    """Every (worker, machine) pair where the worker sits inside the machine's
    expanded box. These are rule_4 CANDIDATES -- a geometric shortlist, not labels.

    Measured on the real annotation_val.json (4,000 images):
        margin 0.00 -> 1,064 images (26.6%),  2,044 pairs
        margin 0.15 -> 1,202 images (30.1%),  2,726 pairs
        margin 0.25 -> 1,253 images (31.3%),  3,087 pairs   <- default
        margin 0.40 -> 1,320 images (33.0%),  3,563 pairs

    Each pair carries the union of the worker and machine boxes, which is the
    natural rule_4 violation region ("the worker and the machine they are too close
    to") and is human-annotated on both sides -- far better than any box a
    zero-shot VLM would draw.
    """
    workers = [(n, b) for n, b in img.boxes if n == CAT_WORKER]
    machines = [(n, b) for n, b in img.boxes if n in set(machine_categories)]
    pairs: List[Dict[str, Any]] = []
    for m_name, m_box in machines:
        expanded = _expand(m_box, margin_frac)
        for _, w_box in workers:
            if _intersects(expanded, w_box):
                pairs.append({
                    "machine_category": m_name,
                    "machine_box": [round(c, 5) for c in m_box],
                    "worker_box": [round(c, 5) for c in w_box],
                    "union_box": [round(c, 5) for c in _union_box([m_box, w_box])],
                })
    return pairs


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_mocs(annotation_path: str | Path, images_root: str = "", source: str = "") -> MocsSplit:
    """Parses a MOCS COCO json (either an annotation_*.json or an image_info_*.json).

    Tolerates the image_info flavour, which has no 'annotations' key at all -- that
    is the COCO convention for an unlabelled split, and MOCS's test file is exactly
    that (18,264 images, categories only).
    """
    path = Path(annotation_path)
    if not path.exists():
        raise FileNotFoundError(f"MOCS annotation file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if "images" not in raw:
        raise ValueError(
            f"{path} has no 'images' key -- is this really a MOCS/COCO annotation file? "
            f"Top-level keys: {sorted(raw)}"
        )

    cat_by_id = {c["id"]: c["name"] for c in raw.get("categories", [])}
    source = source or path.stem.replace("annotation_", "").replace("image_info_", "")

    # Keyed by filename STEM, not COCO id -- ids restart at 1 in every MOCS split, so
    # val and test share all 4,000 of val's ids and an id-keyed dict would silently
    # lose half a merged pool. A second index maps id -> stem for the annotation pass.
    images: Dict[str, MocsImage] = {}
    by_coco_id: Dict[int, str] = {}
    for im in raw["images"]:
        obj = MocsImage(
            image_id=im["id"],
            file_name=im["file_name"],
            width=int(im.get("width") or 0),
            height=int(im.get("height") or 0),
            source=source,
            images_root=str(images_root),
        )
        images[obj.stem] = obj
        by_coco_id[im["id"]] = obj.stem

    anns = raw.get("annotations") or []
    dropped = 0
    for a in anns:
        stem = by_coco_id.get(a.get("image_id"))
        img = images.get(stem) if stem else None
        if img is None:
            dropped += 1
            continue
        name = cat_by_id.get(a.get("category_id"))
        if name is None:
            dropped += 1
            continue
        box = coco_xywh_to_xyxy_norm(a.get("bbox"), img.width, img.height)
        if box is None:
            dropped += 1
            continue
        img.boxes.append((name, box))

    return MocsSplit(
        path=str(path),
        images=images,
        category_names=[c["name"] for c in raw.get("categories", [])],
        has_annotations=bool(anns),
    )


# ---------------------------------------------------------------------------
# Candidate buckets
#
# One bucket per target rule, plus a random control. Every selected image records
# WHICH bucket put it in the pool, so a later analysis can ask "did the geometric
# rule_4 candidates actually yield more rule_4 than the random control?" -- which is
# the only way to know whether the mining is worth anything.
# ---------------------------------------------------------------------------

BUCKETS = ("rule_4_geometric", "rule_2_height_prior", "rule_3_excavation_prior", "random_control")


def bucket_images(
    split: MocsSplit,
    machine_categories: Sequence[str] = MACHINE_CATEGORIES,
    margin_frac: float = 0.25,
    max_megapixels: Optional[float] = None,
    max_aspect_ratio: Optional[float] = None,
) -> Dict[str, List[int]]:
    """Assigns every image to zero or more candidate buckets.

    Buckets are NOT mutually exclusive -- an image with a worker near an excavator
    and a crane overhead is a candidate for both rule_4 and rule_2. Selection
    (select_images.py) resolves overlap by filling quotas in bucket order and
    skipping already-picked ids.

    `max_megapixels` / `max_aspect_ratio` drop pathological images up front. MOCS
    val has one 14.63 MP image and one at aspect ratio 4.55; those are not worth
    spending a VLM call on and the extreme panorama shapes tokenize badly.
    """
    out: Dict[str, List[str]] = {b: [] for b in BUCKETS}
    machines = set(machine_categories)
    height_hints = set(HEIGHT_HINT_CATEGORIES)
    excavation_hints = set(EXCAVATION_HINT_CATEGORIES)

    for key, img in split.images.items():
        if max_megapixels is not None and img.megapixels > max_megapixels:
            continue
        if max_aspect_ratio is not None and img.aspect_ratio > max_aspect_ratio:
            continue

        out["random_control"].append(key)

        # Per-IMAGE, not per-split: a merged val+test pool has boxes on the val
        # images and none on the test ones, and the mined buckets must apply to
        # exactly the images that actually carry geometry.
        if not img.boxes:
            continue

        cats = img.categories()
        has_worker = CAT_WORKER in cats

        if has_worker and worker_machine_pairs(img, machine_categories, margin_frac):
            out["rule_4_geometric"].append(key)
        if has_worker and (cats & height_hints):
            out["rule_2_height_prior"].append(key)
        if cats & excavation_hints:
            out["rule_3_excavation_prior"].append(key)

    return out


def select_candidates(
    split: MocsSplit,
    quotas: Dict[str, int],
    seed: int = 42,
    machine_categories: Sequence[str] = MACHINE_CATEGORIES,
    margin_frac: float = 0.25,
    max_megapixels: Optional[float] = None,
    max_aspect_ratio: Optional[float] = None,
    exclude_ids: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Picks images per bucket quota without repeats, and returns (records, manifest).

    Deterministic given `seed`: the same quotas always yield the same image set, so
    a re-run of the annotation job covers the same images and `annotate.py`'s resume
    stays meaningful.
    """
    buckets = bucket_images(
        split, machine_categories, margin_frac, max_megapixels, max_aspect_ratio
    )

    # Images already annotated by an earlier run. Dropped from EVERY bucket before
    # any quota is filled, so a follow-up run cannot re-spend GPU time on an image
    # that already has a proposal -- and cannot produce a second, conflicting record
    # for it either.
    excluded = {str(x) for x in (exclude_ids or ())}
    n_excluded = 0
    if excluded:
        for bucket, ids in buckets.items():
            keep = [i for i in ids if f"mocs_{i}" not in excluded and i not in excluded]
            n_excluded += len(ids) - len(keep)
            buckets[bucket] = keep

    rng = random.Random(seed)

    chosen: Dict[str, str] = {}
    per_bucket_taken: Dict[str, int] = {}
    # Rarest-first so a scarce bucket is not drained by a broader one that happens
    # to contain the same images.
    order = sorted(quotas, key=lambda b: len(buckets.get(b, [])))
    for bucket in order:
        want = int(quotas.get(bucket, 0))
        pool = [i for i in buckets.get(bucket, []) if i not in chosen]
        rng.shuffle(pool)
        picked = pool[:want]
        for i in picked:
            chosen[i] = bucket
        per_bucket_taken[bucket] = len(picked)

    records: List[Dict[str, Any]] = []
    for key, bucket in sorted(chosen.items()):
        img = split.images[key]
        pairs = (
            worker_machine_pairs(img, machine_categories, margin_frac)
            if img.boxes
            else []
        )
        records.append({
            # `mocs_` prefix so a combined dataset can always be split back apart,
            # mirroring the `_aug{N}` convention data/augment_rare_classes.py uses.
            "new_image_id": f"mocs_{img.file_name.rsplit('.', 1)[0]}",
            "mocs_image_id": img.image_id,
            "file_name": img.file_name,
            "width": img.width,
            "height": img.height,
            # Carried PER RECORD because a pool may mix val and test, whose jpgs live
            # in different directories. annotate.py prefers this over the manifest's
            # global images_root, falling back for selections made before it existed.
            "source": img.source,
            "images_root": img.images_root,
            "selection_bucket": bucket,
            "mocs_categories": sorted(img.categories()),
            # Human-annotated boxes, [0,1] xyxy. Kept as a SIDECAR, never written
            # into the excavator/rebar/worker_with_white_hard_hat training columns:
            # MOCS's category extents are not defined the same way (our prompt
            # includes an excavator's arm and bucket) and rebar / white-hard-hat
            # have no MOCS equivalent at all, so a partially-filled object row
            # would be unverified label noise.
            "mocs_boxes": [{"category": n, "box": [round(c, 5) for c in b]} for n, b in img.boxes],
            "worker_machine_pairs": pairs,
        })

    manifest = {
        "source_file": str(split.path),
        "excluded_ids_supplied": len(excluded),
        "excluded_from_pools": n_excluded,
        "source_has_annotations": split.has_annotations,
        "source_total_images": len(split),
        "seed": seed,
        "margin_frac": margin_frac,
        "machine_categories": list(machine_categories),
        "max_megapixels": max_megapixels,
        "max_aspect_ratio": max_aspect_ratio,
        "quotas_requested": dict(quotas),
        "quotas_filled": per_bucket_taken,
        "bucket_pool_sizes": {b: len(v) for b, v in buckets.items()},
        "selected_total": len(records),
    }
    return records, manifest
