from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .trace import stable_record_id


TASK_KEYS = ("task", "question", "prompt", "instruction", "goal", "user_request", "query")
OUTCOME_KEYS = ("outcome", "label", "status", "result", "success", "passed", "is_success", "correct", "score")
TRACE_KEYS = ("trace", "trajectory", "messages", "steps", "events", "conversation", "log")
ID_KEYS = ("__record_id__", "record_id", "trace_id", "sample_id", "id", "case_id", "episode_id")
STEP_KEYS = ("step", "step_index", "event_index", "turn", "turn_index", "timestep")


class TraceDataLoader:
    """Load messy trace datasets into the miner's canonical record format.

    Canonical output:
      {
        "__record_id__": "...",
        "task": "...",
        "outcome": "success|failure|unknown",
        "trace": [...],
        "metadata": {...}
      }
    """

    def __init__(
        self,
        *,
        input_format: Optional[str] = None,
        field_map: Optional[Mapping[str, str]] = None,
        csv_group_by: Optional[str] = None,
        max_records: Optional[int] = None,
        agent_reward_observation_chars: int = 1200,
        agent_reward_observation_policy: str = "last",
    ) -> None:
        self.input_format = input_format
        self.field_map = dict(field_map or {})
        self.csv_group_by = csv_group_by
        self.max_records = max_records
        self.agent_reward_observation_chars = agent_reward_observation_chars
        self.agent_reward_observation_policy = agent_reward_observation_policy

    def load(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            if self.input_format == "agent_reward_bench" or (path / "cleaned").exists():
                return self._load_agent_reward_bench(path)
            return self._load_directory(path)

        fmt = (self.input_format or path.suffix.lower().lstrip(".")).lower()
        if fmt == "agent_reward_bench":
            root = path if path.is_dir() else path.parent
            return self._load_agent_reward_bench(root)
        if fmt == "jsonl":
            raw_records = self._read_jsonl(path)
        elif fmt == "json":
            raw_records = self._read_json(path)
        elif fmt in {"yaml", "yml"}:
            raw_records = self._read_yaml(path)
        elif fmt == "csv":
            raw_records = self._read_csv(path)
        else:
            raise ValueError(f"unsupported input format: {fmt}")
        if self.max_records is not None:
            raw_records = raw_records[: self.max_records]
        return [self.normalize_record(item, idx) for idx, item in enumerate(raw_records)]

    def normalize_record(self, item: Any, idx: int) -> Dict[str, Any]:
        if not isinstance(item, dict):
            item = {"trace": item}

        field_map = self.field_map
        record_id = self._pick(item, [field_map.get("__record_id__", ""), *ID_KEYS])
        task = self._pick(item, [field_map.get("task", ""), *TASK_KEYS])
        outcome = self._pick(item, [field_map.get("outcome", ""), *OUTCOME_KEYS])
        trace_value = self._pick_raw(item, [field_map.get("trace", ""), *TRACE_KEYS])
        if trace_value is None:
            trace_value = self._record_to_event(item)

        canonical = {
            "__record_id__": str(record_id) if record_id not in (None, "") else stable_record_id(item, idx),
            "task": str(task or ""),
            "outcome": self._normalize_outcome(outcome),
            "trace": self._normalize_trace(trace_value),
            "metadata": {
                key: value
                for key, value in item.items()
                if key not in set(ID_KEYS + TASK_KEYS + OUTCOME_KEYS + TRACE_KEYS + STEP_KEYS)
            },
            "raw_input": {"source_keys": sorted(map(str, item.keys()))},
        }
        return canonical

    def _load_directory(self, path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for child in sorted(path.iterdir()):
            if child.is_file() and child.suffix.lower() in {".json", ".jsonl", ".yaml", ".yml", ".csv"}:
                records.extend(self.load(child))
        return records

    def _load_agent_reward_bench(self, root: Path) -> List[Dict[str, Any]]:
        """Load AgentRewardBench trajectory snapshots.

        Expected layout:
          root/
            cleaned/<benchmark>/<agent>/<experiment>/*.json
            data/annotations.csv
            data/<benchmark>.csv
        """

        cleaned_root = root / "cleaned"
        if not cleaned_root.exists():
            cleaned_root = root
        annotations = self._load_agent_reward_annotations(root / "data" / "annotations.csv")
        benchmark_meta = self._load_agent_reward_task_metadata(root / "data")

        records: List[Dict[str, Any]] = []
        for file_path in sorted(cleaned_root.rglob("*.json")):
            if self.max_records is not None and len(records) >= self.max_records:
                break
            try:
                with file_path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                if not isinstance(raw, dict) or "steps" not in raw:
                    continue
                records.append(
                    self._normalize_agent_reward_record(
                        raw,
                        file_path=file_path,
                        root=root,
                        annotations=annotations,
                        benchmark_meta=benchmark_meta,
                    )
                )
            except Exception as exc:
                records.append(
                    {
                        "__record_id__": str(file_path.relative_to(root)).replace("\\", "/"),
                        "task": "",
                        "outcome": "unknown",
                        "trace": [],
                        "metadata": {"source_path": str(file_path), "load_error": str(exc)},
                        "raw_input": {},
                    }
                )
        return records

    def _load_agent_reward_annotations(self, path: Path) -> Dict[tuple[str, str, str], Dict[str, Any]]:
        if not path.exists():
            return {}
        rows = self._read_csv_rows(path)
        index: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        grouped: Dict[tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (
                str(row.get("benchmark", "")),
                str(row.get("task_id", "")),
                str(row.get("model_name", "")),
            )
            grouped[key].append(row)
        for key, group in grouped.items():
            successes = [str(row.get("trajectory_success", "")).lower() for row in group]
            side_effects = [str(row.get("trajectory_side_effect", "")).lower() for row in group]
            looping = [str(row.get("trajectory_looping", "")).lower() for row in group]
            index[key] = {
                "annotation_count": len(group),
                "trajectory_success_votes": successes,
                "trajectory_side_effect_votes": side_effects,
                "trajectory_looping_votes": looping,
                "success_vote_count": sum("successful" in value and "unsuccessful" not in value for value in successes),
                "failure_vote_count": sum("unsuccessful" in value for value in successes),
            }
        return index

    def _load_agent_reward_task_metadata(self, data_root: Path) -> Dict[str, Dict[str, Any]]:
        if not data_root.exists():
            return {}
        meta: Dict[str, Dict[str, Any]] = {}
        for path in data_root.glob("*.csv"):
            if path.name in {"annotations.csv", "complete_task_ids.csv", "splits.csv"}:
                continue
            for row in self._read_csv_rows(path):
                task_name = str(row.get("task_name", "") or row.get("task_id", ""))
                if task_name:
                    meta[task_name] = dict(row)
        splits_path = data_root / "splits.csv"
        if splits_path.exists():
            for row in self._read_csv_rows(splits_path):
                task_id = str(row.get("task_id", ""))
                if task_id:
                    meta.setdefault(task_id, {}).update({"split": row.get("split"), "benchmark": row.get("benchmark")})
        return meta

    def _normalize_agent_reward_record(
        self,
        raw: Mapping[str, Any],
        *,
        file_path: Path,
        root: Path,
        annotations: Mapping[tuple[str, str, str], Mapping[str, Any]],
        benchmark_meta: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        benchmark = str(raw.get("benchmark", ""))
        agent = str(raw.get("agent", ""))
        model = str(raw.get("model", ""))
        experiment = str(raw.get("experiment", ""))
        task_id = file_path.stem
        record_id = f"{benchmark}/{agent}/{experiment}/{task_id}"
        summary = raw.get("summary_info", {}) if isinstance(raw.get("summary_info"), dict) else {}
        annotation = annotations.get((benchmark, task_id, agent), {})
        task_meta = benchmark_meta.get(task_id, {})

        return {
            "__record_id__": record_id,
            "task": str(raw.get("goal", "")).strip(),
            "outcome": self._agent_reward_outcome(summary, annotation),
            "trace": self._agent_reward_steps(raw.get("steps", [])),
            "metadata": {
                "dataset": "agent-reward-bench",
                "benchmark": benchmark,
                "agent": agent,
                "model": model,
                "experiment": experiment,
                "task_id": task_id,
                "seed": raw.get("seed"),
                "valid": raw.get("valid"),
                "summary_info": summary,
                "annotation": annotation,
                "task_metadata": task_meta,
                "source_path": str(file_path),
                "relative_source_path": str(file_path.relative_to(root)).replace("\\", "/"),
            },
            "raw_input": {
                "benchmark": benchmark,
                "agent": agent,
                "model": model,
                "experiment": experiment,
                "goal": raw.get("goal"),
                "summary_info": summary,
            },
        }

    def _agent_reward_steps(self, steps: Any) -> List[Dict[str, Any]]:
        if not isinstance(steps, list):
            return []
        normalized = []
        last_step_pos = len(steps) - 1
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                normalized.append({"index": idx, "content": str(step)})
                continue
            step_num = step.get("num", idx)
            last_action_error = step.get("last_action_error")
            normalized.append(
                {
                    "index": step_num,
                    "thought": step.get("reasoning", ""),
                    "action": step.get("action", ""),
                    "url": step.get("url", ""),
                    "focused_element": step.get("focused_element", ""),
                    "screenshot_path": step.get("screenshot_path", ""),
                }
            )
            observation = self._short_observation(step) if self._keep_agent_reward_observation(idx, last_step_pos, last_action_error) else ""
            if observation:
                normalized.append(
                    {
                        "index": f"{step_num}.obs",
                        "observation": observation,
                        "url": step.get("url", ""),
                        "screenshot_path": step.get("screenshot_path", ""),
                    }
                )
            if last_action_error:
                normalized.append(
                    {
                        "index": f"{step_num}.error",
                        "error": last_action_error,
                        "action": step.get("action", ""),
                    }
                )
        return normalized

    def _keep_agent_reward_observation(self, idx: int, last_step_pos: int, last_action_error: Any) -> bool:
        policy = str(self.agent_reward_observation_policy or "last").lower()
        if policy == "none":
            return False
        if policy == "all":
            return True
        if policy == "last_and_errors":
            return idx == last_step_pos or bool(last_action_error)
        return idx == last_step_pos

    def _short_observation(self, step: Mapping[str, Any]) -> str:
        for key in ("axtree_pruned", "axtree"):
            value = step.get(key)
            if isinstance(value, str) and value.strip():
                return value[: self.agent_reward_observation_chars]
        return ""

    def _agent_reward_outcome(
        self,
        summary: Mapping[str, Any],
        annotation: Mapping[str, Any],
    ) -> str:
        success_votes = int(annotation.get("success_vote_count", 0) or 0)
        failure_votes = int(annotation.get("failure_vote_count", 0) or 0)
        if success_votes or failure_votes:
            return "success" if success_votes >= failure_votes else "failure"
        reward = summary.get("cum_reward", summary.get("cum_raw_reward"))
        if isinstance(reward, (int, float)):
            return "success" if reward > 0 else "failure"
        if summary.get("err_msg") or summary.get("stack_trace"):
            return "failure"
        return "unknown"

    def _read_jsonl(self, path: Path) -> List[Any]:
        rows: List[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _read_json(self, path: Path) -> List[Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return self._records_from_container(data)

    def _read_yaml(self, path: Path) -> List[Any]:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML input requires PyYAML. Use JSON/CSV or install pyyaml.") from exc
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return self._records_from_container(data)

    def _read_csv(self, path: Path) -> List[Dict[str, Any]]:
        rows = self._read_csv_rows(path)
        if not rows:
            return []
        group_key = self.csv_group_by or self._detect_csv_group_key(rows)
        if group_key:
            return self._group_csv_events(rows, group_key)
        return [self._coerce_csv_row(row) for row in rows]

    def _read_csv_rows(self, path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _records_from_container(self, data: Any) -> List[Any]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("traces", "records", "items", "data", "examples", "episodes"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
        return [{"trace": data}]

    def _detect_csv_group_key(self, rows: Sequence[Mapping[str, Any]]) -> Optional[str]:
        headers = set(rows[0])
        id_key = next((key for key in ID_KEYS if key in headers), None)
        has_step = any(key in headers for key in STEP_KEYS)
        if id_key and has_step:
            values = [str(row.get(id_key, "")) for row in rows]
            if len(set(values)) < len(values):
                return id_key
        return None

    def _group_csv_events(self, rows: Sequence[Mapping[str, Any]], group_key: str) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(group_key, ""))].append(row)

        records: List[Dict[str, Any]] = []
        for record_id, group_rows in grouped.items():
            sorted_rows = sorted(group_rows, key=self._step_sort_key)
            first = sorted_rows[0]
            record = self._coerce_csv_row(first)
            record["__record_id__"] = record_id
            record["trace"] = [self._record_to_event(self._coerce_csv_row(row)) for row in sorted_rows]
            record["raw_rows"] = [dict(row) for row in sorted_rows]
            records.append(record)
        return records

    def _coerce_csv_row(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        coerced: Dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                coerced[key] = value
            else:
                coerced[key] = self._parse_cell(value)
        return coerced

    def _record_to_event(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        event: Dict[str, Any] = {}
        for source, target in (
            ("role", "role"),
            ("content", "content"),
            ("message", "content"),
            ("thought", "thought"),
            ("action", "action"),
            ("observation", "observation"),
            ("obs", "observation"),
            ("tool", "tool"),
            ("tool_name", "tool"),
            ("tool_input", "tool_input"),
            ("tool_output", "tool_output"),
            ("error", "error"),
            ("state", "state"),
            ("state_change", "state_change"),
        ):
            if source in row and row[source] not in (None, ""):
                event[target] = row[source]
        return event or dict(row)

    def _normalize_trace(self, trace_value: Any) -> Any:
        parsed = self._parse_cell(trace_value) if isinstance(trace_value, str) else trace_value
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in TRACE_KEYS:
                if isinstance(parsed.get(key), list):
                    return parsed[key]
            return [parsed]
        if parsed in (None, ""):
            return []
        return [{"content": str(parsed)}]

    def _normalize_outcome(self, value: Any) -> str:
        if isinstance(value, bool):
            return "success" if value else "failure"
        if isinstance(value, (int, float)):
            return "success" if value >= 0.5 else "failure"
        lowered = str(value or "").strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return "success"
        if lowered in {"0", "false", "no", "n"}:
            return "failure"
        if any(token in lowered for token in ("fail", "failed", "error", "incorrect", "wrong")):
            return "failure"
        if any(token in lowered for token in ("success", "succeed", "pass", "passed", "correct")):
            return "success"
        return "unknown"

    def _pick(self, item: Mapping[str, Any], keys: Iterable[str]) -> Any:
        for key in keys:
            if key and key in item and item[key] not in (None, ""):
                return item[key]
        return None

    def _pick_raw(self, item: Mapping[str, Any], keys: Iterable[str]) -> Any:
        for key in keys:
            if key and key in item:
                return item[key]
        return None

    def _parse_cell(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            return ""
        if stripped[:1] in {"[", "{"}:
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
        if stripped.lower() in {"true", "false"}:
            return stripped.lower() == "true"
        try:
            if "." in stripped:
                return float(stripped)
            return int(stripped)
        except ValueError:
            return value

    def _step_sort_key(self, row: Mapping[str, Any]) -> tuple[int, str]:
        for key in STEP_KEYS:
            value = row.get(key)
            if value not in (None, ""):
                try:
                    return int(value), str(value)
                except ValueError:
                    return 0, str(value)
        return 0, ""
