from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = Path("data/cache_pair_data")
DEFAULT_PAIR_RUBRICS = Path("data/pair_rubric/pair_rubrics.json")
DEFAULT_OUTPUT_DIR = Path("data/rubric_merge")

