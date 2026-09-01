"""run_inference_batched runs start to finish: no auto-resume, no per-batch retry.

Auto-resume and the batch try/except were written for Google Colab, where the
runtime disconnects unpredictably. On SLURM they combined into a silent
metric-corruption bug (BUG-07 in the pre-flight audit):

  1. a failing batch was caught and written as ``raw_output: ""`` for every image
     in it, so a hard failure looked like a model that emitted nothing;
  2. auto-resume re-ran exactly those images, because it only counted a non-empty
     output as complete;
  3. the file was opened in append mode, so the retry was ADDED rather than
     replacing the blank -- leaving two records for one image_id;
  4. nothing downstream de-duplicates, so every metric was computed over an
     inflated denominator padded with spurious failures.

These tests pin all four of those away.
"""
import json

import pytest

from models.inference import run_inference_batched


class _FakeBatch:
    """Minimal stand-in for an HF Dataset slice."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self._rows]
        return self._rows[key]

    def select(self, idx):
        return _FakeBatch([self._rows[i] for i in idx])


def _dataset(n=6):
    return _FakeBatch([
        {"image_id": f"img_{i}", "image": f"PIL_{i}", "image_caption": f"caption {i}"}
        for i in range(n)
    ])


def _read(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture
def good_generate(monkeypatch):
    """generate_batch that echoes a distinct output per image."""
    def _gen(model, tokenizer, pil_images, **kwargs):
        return [f"out::{img}" for img in pil_images]
    monkeypatch.setattr("models.inference.generate_batch", _gen)
    return _gen


# ---------------------------------------------------------------------------
# No retry: a batch failure must crash the job
# ---------------------------------------------------------------------------

def test_batch_failure_propagates(monkeypatch, tmp_path):
    """A failing batch must raise, not be laundered into blank predictions."""
    def _boom(model, tokenizer, pil_images, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("models.inference.generate_batch", _boom)
    out = tmp_path / "predictions.jsonl"

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run_inference_batched(None, None, _dataset(4), batch_size=2,
                              output_path=str(out), show_progress=False)


def test_no_blank_records_are_ever_written(monkeypatch, tmp_path):
    """Even when the failure happens mid-run, no record may carry raw_output ""."""
    calls = {"n": 0}

    def _fail_second(model, tokenizer, pil_images, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("bad image")
        return [f"out::{img}" for img in pil_images]

    monkeypatch.setattr("models.inference.generate_batch", _fail_second)
    out = tmp_path / "predictions.jsonl"

    with pytest.raises(RuntimeError):
        run_inference_batched(None, None, _dataset(6), batch_size=2,
                              output_path=str(out), show_progress=False)

    # The first batch's real output is flushed and kept for debugging, but nothing
    # blank and nothing carrying an "error" field was written.
    records = _read(out)
    assert all(r["raw_output"].strip() for r in records), "a blank record was written"
    assert all("error" not in r for r in records)

    # The .json twin is only written on a clean finish, so it must not exist.
    assert not (tmp_path / "predictions.json").exists()


# ---------------------------------------------------------------------------
# No resume: the file is truncated, never appended to
# ---------------------------------------------------------------------------

def test_output_file_is_truncated_not_appended(good_generate, tmp_path):
    """A re-run must produce a clean file, not accumulate onto the last one."""
    out = tmp_path / "predictions.jsonl"
    out.write_text(json.dumps({"image_id": "stale", "raw_output": "old junk"}) + "\n",
                   encoding="utf-8")

    run_inference_batched(None, None, _dataset(4), batch_size=2,
                          output_path=str(out), show_progress=False)

    records = _read(out)
    assert len(records) == 4, "stale records survived the re-run"
    assert not any(r["image_id"] == "stale" for r in records)


def test_rerun_is_idempotent_and_never_duplicates(good_generate, tmp_path):
    """Running twice must give exactly one record per image, not two."""
    out = tmp_path / "predictions.jsonl"
    for _ in range(2):
        run_inference_batched(None, None, _dataset(5), batch_size=2,
                              output_path=str(out), show_progress=False)

    records = _read(out)
    ids = [r["image_id"] for r in records]
    assert len(records) == 5
    assert len(set(ids)) == len(ids), f"duplicate image_ids: {ids}"


def test_previously_completed_images_are_not_skipped(good_generate, tmp_path):
    """The old code filtered out anything already present. It must re-run
    everything now, or a re-run would silently produce a short file."""
    out = tmp_path / "predictions.jsonl"
    existing = [{"image_id": f"img_{i}", "raw_output": "already done"} for i in range(3)]
    out.write_text("\n".join(json.dumps(r) for r in existing) + "\n", encoding="utf-8")

    results = run_inference_batched(None, None, _dataset(3), batch_size=3,
                                    output_path=str(out), show_progress=False)

    assert len(results) == 3
    assert all(r["raw_output"].startswith("out::") for r in _read(out))


# ---------------------------------------------------------------------------
# Record count must equal input count
# ---------------------------------------------------------------------------

def test_record_count_matches_input_count(good_generate, tmp_path):
    """Every downstream metric divides by the record count."""
    out = tmp_path / "predictions.jsonl"
    results = run_inference_batched(None, None, _dataset(7), batch_size=3,
                                    output_path=str(out), show_progress=False)
    assert len(results) == 7
    assert len(_read(out)) == 7
    assert len(json.loads((tmp_path / "predictions.json").read_text(encoding="utf-8"))) == 7


def test_short_generate_output_is_caught(monkeypatch, tmp_path):
    """A generate_batch that silently returns fewer outputs than inputs would
    desynchronise the image_id -> output pairing. It must not pass quietly."""
    def _short(model, tokenizer, pil_images, **kwargs):
        return [f"out::{img}" for img in pil_images][:-1]

    monkeypatch.setattr("models.inference.generate_batch", _short)
    out = tmp_path / "predictions.jsonl"
    with pytest.raises((IndexError, RuntimeError)):
        run_inference_batched(None, None, _dataset(4), batch_size=4,
                              output_path=str(out), show_progress=False)


def test_max_samples_caps_without_resume_arithmetic(good_generate, tmp_path):
    """max_samples used to be adjusted by the number of already-completed images,
    which meant a re-run with --max_samples produced a different subset."""
    out = tmp_path / "predictions.jsonl"
    for _ in range(2):
        results = run_inference_batched(None, None, _dataset(10), batch_size=3,
                                        max_samples=4, output_path=str(out),
                                        show_progress=False)
        assert len(results) == 4
        assert [r["image_id"] for r in _read(out)] == ["img_0", "img_1", "img_2", "img_3"]


def test_empty_dataset_returns_empty(good_generate, tmp_path):
    results = run_inference_batched(None, None, _dataset(0), batch_size=2,
                                    output_path=str(tmp_path / "p.jsonl"),
                                    show_progress=False)
    assert results == []


def test_works_without_an_output_path(good_generate):
    results = run_inference_batched(None, None, _dataset(3), batch_size=2,
                                    output_path=None, show_progress=False)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Static guard: the resume machinery must not come back
# ---------------------------------------------------------------------------

def test_no_resume_machinery_in_source():
    import inspect
    from models import inference

    src = inspect.getsource(inference.run_inference_batched)
    body = src[src.index('"""', src.index('"""') + 3):]  # skip the docstring

    assert "completed_ids" not in body, "auto-resume machinery is back"
    assert 'open(output_path, "a"' not in body, "append-mode write is back"
    assert '"raw_output": ""' not in body, "blank-record fallback is back"
    assert "dataset.filter" not in body, "resume-based dataset filtering is back"
