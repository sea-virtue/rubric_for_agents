from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence

from pair_rubric_extraction.io import load_selected_cache_record
from pair_rubric_extraction.prompting import limit_response_payload, state_cards_from_record


def satisfaction_prompt_messages(
    pair_record: Mapping[str, Any],
    bank_items: Sequence[Mapping[str, Any]],
    *,
    max_chars_per_response: int | None,
    card_order: str,
) -> List[Dict[str, str]]:
    rubrics = []
    for idx, item in enumerate(bank_items, start=1):
        rubric = item.get("rubric", {}) if isinstance(item.get("rubric"), Mapping) else item
        rubrics.append(
            {
                "rubric_index": idx,
                "rubric_id": item.get("rubric_id") or item.get("__record_id__"),
                "source_pair_id": item.get("source_pair_id"),
                "dimension": rubric.get("dimension", ""),
                "criterion": rubric.get("criterion", ""),
                "severity": rubric.get("severity", "medium"),
                "positive_evidence": rubric.get("positive_evidence", []),
                "negative_evidence": rubric.get("negative_evidence", []),
                "verification_guide": rubric.get("verification_guide", {}),
            }
        )
    responses = pair_eval_responses(pair_record, max_chars_per_response=max_chars_per_response, card_order=card_order)
    schema = {
        "response_id": "response_0",
        "rubric_scores": [
            {
                "rubric_index": 1,
                "score": 0.0,
                "evidence": "short evidence from state_cards",
            }
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You evaluate two unlabeled agent trajectories against fixed rubrics. "
                "Return only a strict JSON array with one object per response."
            ),
        },
        {
            "role": "user",
            "content": (
                "Score each response against every rubric using only the provided state_cards. "
                "Do not infer from response order, hidden labels, validation, model identity, or agent identity. "
                "Terminal state and final-answer evidence are primary for task completion. If intermediate evidence "
                "conflicts with the terminal state or final answer, the terminal/final evidence wins. "
                "Use score 1.0 for clearly satisfied, 0.5 for partial/ambiguous, and 0.0 for absent or violated. "
                "Do not compute a total_score; only return rubric_scores.\n\n"
                f"JSON item schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"query: {pair_record.get('query', pair_record.get('task_instruction', ''))}\n\n"
                f"RUBRICS:\n{json.dumps(rubrics, ensure_ascii=False, indent=2)}\n\n"
                f"RESPONSES:\n{json.dumps(responses, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def pair_eval_responses(pair_record: Mapping[str, Any], *, max_chars_per_response: int | None, card_order: str) -> List[Dict[str, Any]]:
    responses = []
    selected_records = pair_record.get("selected_records", [])
    if isinstance(selected_records, list) and len(selected_records) >= 2:
        for idx, selected in enumerate(selected_records[:2]):
            if not isinstance(selected, Mapping):
                continue
            record = load_selected_cache_record(selected)
            response = limit_response_payload(
                {"state_cards": state_cards_from_record(record)},
                max_chars=max_chars_per_response,
                card_order=card_order,
            )
            responses.append({"response_id": f"response_{idx}", **response})
        return responses
    raw_responses = pair_record.get("responses", [])
    if isinstance(raw_responses, list):
        for idx, response in enumerate(raw_responses[:2]):
            if isinstance(response, Mapping):
                responses.append(
                    {
                        "response_id": f"response_{idx}",
                        **limit_response_payload(response, max_chars=max_chars_per_response, card_order=card_order),
                    }
                )
    return responses

