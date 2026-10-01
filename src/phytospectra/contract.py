"""Symbolic evidence-reference output contract and fail-closed verifier."""

from __future__ import annotations

import json
from typing import Any

from .data import TARGETS


def expected_plan(intent: str, targets: list[str], evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    items = []
    for target in targets:
        item: dict[str, Any] = {"target": target, "value_ref": f"{target}.value"}
        if target in {"Fertilizer", "Cultivar"} and evidence is not None:
            value = evidence["predictions"][target]["value"]
            item["class_value"] = f"{float(value):g}" if target == "Fertilizer" else str(value)
        if intent == "EVIDENCE_REQUEST":
            item["roi_ref"] = f"{target}.roi"
        items.append(item)
    return {"version": 1, "intent": intent, "items": items}


def parse_plan(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            candidates.append(value)
    plans = [
        value for value in candidates
        if {"version", "intent", "items"}.issubset(value)
        and isinstance(value.get("items"), list)
    ]
    return plans[-1] if plans else (candidates[-1] if candidates else None)


def verify_plan(text: str, intent: str, targets: list[str], evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    parsed = parse_plan(text)
    expected = expected_plan(intent, targets, evidence)
    accepted = parsed == expected
    return {"accepted": accepted, "parsed": parsed, "expected": expected, "fallback_used": not accepted}


def resolve_references(plan: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    targets = [item.get("target") for item in plan.get("items", []) if isinstance(item, dict)]
    if any(target not in TARGETS for target in targets):
        raise ValueError("Plan contains an unsupported target")
    return {target: evidence["predictions"][target] for target in targets}
