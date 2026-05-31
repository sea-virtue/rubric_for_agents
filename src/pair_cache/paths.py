from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = Path("data/cache_data/agent-reward-bench/trajectories/cleaned")
DEFAULT_OUTPUT_ROOT = Path("data/cache_pair_data")
