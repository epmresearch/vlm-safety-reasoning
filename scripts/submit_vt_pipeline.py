#!/usr/bin/env python3
"""
Submits the Violations-Think pipeline. Thin wrapper around scripts/submit_pipeline.py,
kept as a stable entry point for symmetry with the other four; all the orchestration
logic lives in one place.

Usage:
    python scripts/submit_vt_pipeline.py --tiers 2b 4b 8b --version v3
    python scripts/submit_vt_pipeline.py --tiers 2b --version v3      # one tier
    python scripts/submit_vt_pipeline.py --tiers 4b 8b --version v3   # any subset

Equivalent to:
    python scripts/submit_pipeline.py --task violations_think --tiers 2b 4b 8b --version v3

This task needs no --sft-dataset / --grpo-pool flags: configs/tasks/violations_think.yaml
already points at datasets/augmented_v3 and datasets/grpo_pool_v3. Those flags are for
running a DIFFERENT task against this data -- notably the v4 arm:

    python scripts/submit_pipeline.py --task violations_only --version v4 \\
        --tiers 2b 4b 8b --sft-dataset datasets/augmented_v3 \\
        --grpo-pool datasets/grpo_pool_v3

which is violations_only's exact v2 settings on the new data with no think block, so
v3 - v4 isolates the block and v4 - v2 isolates the data.

--version is independent per pipeline. Every name this produces is namespaced by the
'vt' task prefix, so it can run concurrently with all four other pipelines at the same
--version and tier.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.submit_pipeline import main

if __name__ == "__main__":
    main(task_default="violations_think")
