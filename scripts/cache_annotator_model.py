#!/usr/bin/env python3
"""
Caches the MOCS annotator VLM on the ARC LOGIN NODE, then verifies the cache.

Compute nodes have no internet, so the model must be downloaded before any job is
submitted -- the same constraint that applies to the LLM judge (OPERATIONS.md §2.3).
Unlike the judge, Qwen3-VL is not a gated repo, so no licence acceptance or token is
needed.

HF_HOME IS THE WHOLE POINT. Every scripts/hpc_*.sh sets HF_HOME to
$HOME/scratch/hf_cache. A login-node download without it lands in
~/.cache/huggingface, where no job will ever look -- and burns home-directory quota,
which on this account has far less headroom than /scratch's 15 TB. This script
refuses to run unless HF_HOME is set, rather than silently downloading 66 GB to the
wrong place. That failure mode is already documented for submit_pipeline.py's
preload step, which has no HF_HOME override of its own.

Usage (login node, after the every-session preamble):
    export HF_HOME="$HOME/scratch/hf_cache"
    python scripts/cache_annotator_model.py
    python scripts/cache_annotator_model.py --model Qwen/Qwen3-VL-30B-A3B-Instruct
    python scripts/cache_annotator_model.py --verify-only
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

# Approximate bf16 download sizes, for the disk-space check below.
KNOWN_SIZES_GB = {
    "Qwen/Qwen3-VL-32B-Instruct": 66,
    "Qwen/Qwen3-VL-30B-A3B-Instruct": 60,
    "Qwen/Qwen3-VL-8B-Instruct": 17,
}


def _require_hf_home() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise SystemExit(
            "HF_HOME is not set.\n\n"
            "Every scripts/hpc_*.sh reads $HOME/scratch/hf_cache, so a download without\n"
            "HF_HOME lands somewhere no job will look and eats home-directory quota.\n\n"
            "  export HF_HOME=\"$HOME/scratch/hf_cache\"\n"
        )
    path = Path(hf_home)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _check_space(hf_home: Path, model: str) -> None:
    need_gb = KNOWN_SIZES_GB.get(model, 70)
    usage = shutil.disk_usage(hf_home)
    free_gb = usage.free / 1e9
    print(f"Cache dir : {hf_home}")
    print(f"Free space: {free_gb:.0f} GB  |  model needs ~{need_gb} GB")
    if free_gb < need_gb * 1.2:
        raise SystemExit(
            f"Not enough headroom: {free_gb:.0f} GB free, ~{need_gb} GB needed "
            f"(plus margin for the incomplete-download staging area). "
            f"Point HF_HOME at /scratch, or clear space."
        )


def download(model: str) -> None:
    """Delegates to the `hf` CLI, which resumes a partial download."""
    cmd = ["hf", "download", model]
    print(f"\n$ {' '.join(cmd)}")
    print("(this is 60+ GB -- expect a while; the CLI resumes if interrupted)\n")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit(
            "`hf` is not on PATH. Activate the environment first:\n"
            "  module load gcc/13.3.0 python/3.12.5\n"
            "  source $HOME/envs/vlm_grpo/bin/activate"
        )
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Download failed with exit code {e.returncode}.")


def verify(model: str) -> None:
    """Confirms the cache is usable WITHOUT loading 66 GB of weights.

    Deliberately checks the config + processor only. Loading the full model needs a
    GPU (or 66 GB of host RAM), which is not what a login node is for -- and a
    `snapshot_download(local_files_only=True)` check is worse than useless here: it
    raises on any file the download skipped, which is a false alarm. The real
    end-to-end check is the srun command printed at the end, and the honest smoke
    test is `annotate.py --limit 8`.
    """
    print("\nVerifying the cache (config + processor only, no weights)...")
    from transformers import AutoConfig, AutoProcessor

    cfg = AutoConfig.from_pretrained(model, local_files_only=True)
    print(f"  config  OK  model_type={getattr(cfg, 'model_type', '?')}")

    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    print(f"  processor OK  {type(processor).__name__}")

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        print(
            "  WARNING: the processor has no image_processor. That is the Unsloth/Qwen3-VL\n"
            "  degradation models/model_loader.py warns about -- apply_pixel_bounds would\n"
            "  have nothing to cap and the 1.2 MP ceiling would not bind. Do not proceed."
        )
    else:
        print(f"  image_processor OK  current size={getattr(image_processor, 'size', None)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--verify-only", action="store_true", help="Skip the download")
    ap.add_argument("--skip-space-check", action="store_true")
    args = ap.parse_args()

    hf_home = _require_hf_home()
    if not args.verify_only:
        if not args.skip_space_check:
            _check_space(hf_home, args.model)
        download(args.model)
    verify(args.model)

    print("\n" + "=" * 70)
    print("Cached and verified. Optional end-to-end GPU check (2 minutes on a GPU node):")
    print(f"""
srun --partition=gpu-h100 --gres=gpu:h100:1 --mem=100G --time=00:20:00 --pty bash -lc '
  module load gcc/13.3.0 python/3.12.5 && source $HOME/envs/vlm_grpo/bin/activate
  export HF_HOME=$HOME/scratch/hf_cache
  python -c "
import torch, transformers
from transformers import AutoProcessor
m = \\"{args.model}\\"
cls = getattr(transformers, \\"Qwen3VLForConditionalGeneration\\", None) or transformers.AutoModelForImageTextToText
mod = cls.from_pretrained(m, dtype=torch.bfloat16, device_map=\\"cuda:0\\", local_files_only=True)
AutoProcessor.from_pretrained(m, local_files_only=True)
print(\\"annotator OK, weights on GPU:\\", torch.cuda.memory_allocated()/1e9, \\"GB\\")"'
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
