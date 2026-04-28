#!/usr/bin/env python3
"""CLI shim for the rubric mining pipeline.

The implementation lives in ``rubric_miner`` so each pipeline stage stays
small and readable.
"""

from rubric_miner.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
