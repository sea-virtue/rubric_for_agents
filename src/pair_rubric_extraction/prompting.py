from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

from .io import load_selected_cache_record
from .sanitize import clean_prompt_payload


def prompt_messages(pair: Mapping[str, Any], *, max_chars_per_response: int | None) -> List[Dict[str, str]]:
    schema = {
        "dimension": "short capability or failure-mode name",
        "criterion": "observable criterion that explains why the successful trajectory is better",
        "positive_evidence": ["state/action/final-answer pattern from the successful response"],
        "negative_evidence": ["state/action/final-answer pattern from the failed response"],
        "severity": "low|medium|high",
        "rationale": "why this criterion separates success from failure for this task",
        "verification_guide": {
            "primary_evidence_stage": "terminal_state|final_answer|process|side_effect",
            "what_to_extract": ["observable facts to inspect from future trajectories"],
            "checks": ["general checks to apply without model/agent/source metadata"],
            "evidence_pattern": "compact success/failure pattern using placeholders where needed",
        },
    }
    pair_payload = pair_prompt_payload(pair, max_chars_per_response=max_chars_per_response)
    return [
        {
            "role": "system",
            "content": (
                "You extract pair-level evaluation rubrics from one task and two agent trajectory summaries. "
                "Return only a strict JSON array. Do not use markdown, prose, or a wrapper object."
            ),
        },
        {
            "role": "user",
            "content": (
                "Extract 2-5 concise pair-level rubrics that explain why the validated successful trajectory "
                "is better than the failed trajectory for this exact task. These are candidate rubrics that "
                "will later be embedded, deduplicated, and merged across tasks.\n\n"
                "Requirements:\n"
                "- Use only the query, responses[].state_cards, and validation shown below.\n"
                "- Treat validation as supervision for which response succeeded or failed; do not turn reward, "
                "label, response index, model identity, or agent identity into rubric content.\n"
                "- Criteria must be observable from trajectory evidence, not generic advice.\n"
                "- Terminal state and final-answer evidence are primary when judging task completion. "
                "If intermediate evidence conflicts with the terminal state or final answer, the terminal/final evidence wins.\n"
                "- Do not create criteria that can be satisfied only by a transient intermediate state unless the task "
                "explicitly evaluates process behavior. For most web/navigation tasks, require the final visible state "
                "or final user-facing answer to support completion.\n"
                "- Prefer discriminative criteria visible in the state cards: correct target identification, "
                "required field/value completion, final answer correctness, avoided side effects, or failure omissions.\n"
                "- Do not hard-code instance-specific step numbers, element IDs, UUIDs, request numbers, prices, "
                "names, dates, or record values unless they are core task parameters. Use placeholders such as "
                "<target_record>, <required_field>, <expected_answer>, <date_range>, <threshold>, or <target_url>.\n"
                "- Evidence fields should summarize reusable patterns, not copy long raw observations.\n"
                "- Include verification_guide for each item, including primary_evidence_stage.\n\n"
                f"JSON item schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"PAIR_INPUT:\n{json.dumps(pair_payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def pair_prompt_payload(pair: Mapping[str, Any], *, max_chars_per_response: int | None) -> Dict[str, Any]:
    clean_responses = []
    validation = []
    selected_records = pair.get("selected_records", [])
    if isinstance(selected_records, list) and len(selected_records) >= 2:
        for idx, selected in enumerate(selected_records[:2]):
            if not isinstance(selected, Mapping):
                continue
            record = load_selected_cache_record(selected)
            clean_responses.append(
                limit_response_payload(
                    {"state_cards": state_cards_from_record(record)},
                    max_chars=max_chars_per_response,
                )
            )
            validation.append(validation_from_record(record, selected=selected, response_index=idx))
    else:
        responses = pair.get("responses", [])
        if isinstance(responses, list):
            for response in responses[:2]:
                response_payload = clean_prompt_payload(response if isinstance(response, Mapping) else {})
                clean_responses.append(limit_response_payload(response_payload, max_chars=max_chars_per_response))
        pair_validation = pair.get("validation", [])
        validation = clean_prompt_payload(pair_validation if isinstance(pair_validation, list) else [])

    return {
        "query": str(pair.get("query") or "").strip(),
        "responses": clean_responses,
        "validation": clean_prompt_payload(validation if isinstance(validation, list) else []),
    }


def state_cards_from_record(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    runtime = record.get("runtime_summary")
    cards = runtime.get("state_cards", []) if isinstance(runtime, Mapping) else []
    if not isinstance(cards, list):
        return []
    cleaned = []
    for card in cards:
        if isinstance(card, Mapping):
            cleaned_card = clean_prompt_payload(card)
            if cleaned_card:
                cleaned.append(cleaned_card)
    return cleaned


def validation_from_record(record: Mapping[str, Any], *, selected: Mapping[str, Any], response_index: int) -> Dict[str, Any]:
    runtime = record.get("runtime_summary")
    validation = runtime.get("validation") if isinstance(runtime, Mapping) else None
    if not isinstance(validation, Mapping):
        validation = record.get("validation")
    output = compact_validation(validation if isinstance(validation, Mapping) else {})
    outcome = selected.get("outcome") or (runtime.get("outcome") if isinstance(runtime, Mapping) else record.get("outcome"))
    if outcome not in (None, ""):
        output["outcome"] = outcome
    output["pair_role"] = selected.get("pair_role", "")
    output["response_index"] = response_index
    return output


def compact_validation(validation: Mapping[str, Any]) -> Dict[str, Any]:
    keep = ("reward", "raw_reward", "n_steps", "terminated", "truncated", "has_error", "outcome")
    return {key: validation.get(key) for key in keep if validation.get(key) not in (None, "")}


def limit_response_payload(response: Mapping[str, Any], *, max_chars: int | None, card_order: str = "priority") -> Dict[str, Any]:
    state_cards = response.get("state_cards", [])
    output: Dict[str, Any] = {"state_cards": []}
    if not isinstance(state_cards, list):
        return output
    ordered_cards = prioritized_state_cards(state_cards) if card_order == "priority" else state_cards
    if max_chars is None:
        output["state_cards"] = [card for card in ordered_cards if isinstance(card, Mapping)]
        return output
    if max_chars <= 0:
        return output
    for card in ordered_cards:
        if not isinstance(card, Mapping):
            continue
        candidate_cards = [*output["state_cards"], card]
        candidate = {"state_cards": candidate_cards}
        if len(json.dumps(candidate, ensure_ascii=False)) > max_chars and output["state_cards"]:
            break
        output["state_cards"].append(card)
    if len(json.dumps(output, ensure_ascii=False)) > max_chars:
        output["state_cards"] = [shrink_card(card) for card in output["state_cards"]]
    return output


def prioritized_state_cards(cards: List[Any]) -> List[Any]:
    indexed = list(enumerate(cards))
    indexed.sort(key=lambda item: (state_card_priority(item[1]), item[0]))
    return [card for _, card in indexed]


def state_card_priority(card: Any) -> int:
    if not isinstance(card, Mapping):
        return 99
    state_id = str(card.get("state_id", "")).lower()
    role = str(card.get("rubric_role", card.get("evidence_role", ""))).lower()
    if state_id == "task_context" or role == "task_definition":
        return 0
    if state_id == "final_observation" or role == "terminal_state_evidence":
        return 1
    if state_id == "output_or_answer" or role == "final_response_evidence":
        return 2
    if state_id == "risk_signals" or role == "negative_or_side_effect_evidence":
        return 3
    if state_id == "initial_observation" or role == "initial_context":
        return 4
    if state_id.startswith("action_transition"):
        return 5
    if state_id.startswith("evidence_observation"):
        return 6
    return 7


def shrink_card(card: Mapping[str, Any]) -> Dict[str, Any]:
    keep = ("state_id", "card_type", "stage", "evidence_role", "rubric_role", "state_summary", "matched_goal_terms", "evidence_lines", "facets")
    output: Dict[str, Any] = {}
    for key in keep:
        value = card.get(key)
        if value in (None, "", [], {}):
            continue
        if key == "evidence_lines" and isinstance(value, list):
            output[key] = [str(item)[:500] for item in value[:10] if str(item).strip()]
        else:
            output[key] = value
    return output
