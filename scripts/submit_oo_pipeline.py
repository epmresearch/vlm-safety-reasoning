#!/usr/bin/env python3
"""
Submits the Object-Only (object grounding) pipeline. Thin wrapper around scripts/submit_pipeline.py,
kept as a stable entry point so the commands documented in CLAUDE.md keep
working; all the orchestration logic lives in one place.

Usage:
    python scripts/submit_oo_pipeline.py --tiers 2b 4b 8b --version v1

Equivalent to:
    python scripts/submit_pipeline.py --task object_only --tiers 2b 4b 8b --version v1

--version is independent per pipeline. Every name this produces is namespaced by
the 'oo' task prefix, so this can run concurrently with the other three
pipelines at the same --version and tier.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.submit_pipeline import main

if __name__ == "__main__":
    main(task_default="object_only")
