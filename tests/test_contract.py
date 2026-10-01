import json

from phytospectra.contract import expected_plan, parse_plan, resolve_references, verify_plan


def _evidence():
    return {
        "predictions": {
            "GrainWeight": {"value": 7000.0, "task": "regression"},
            "Fertilizer": {"value": 0.5, "task": "classification"},
            "Cultivar": {"value": "Sheriff", "task": "classification"},
        }
    }


def test_nested_multi_target_plan_round_trip():
    evidence = _evidence()
    plan = expected_plan("TARGET_VALUE", ["GrainWeight", "Fertilizer", "Cultivar"], evidence)
    text = "prefix ignored\n" + json.dumps(plan, separators=(",", ":"))
    assert parse_plan(text) == plan
    assert verify_plan(text, "TARGET_VALUE", ["GrainWeight", "Fertilizer", "Cultivar"], evidence)["accepted"]
    resolved = resolve_references(plan, evidence)
    assert list(resolved) == ["GrainWeight", "Fertilizer", "Cultivar"]


def test_class_value_and_roi_are_contract_bound():
    evidence = _evidence()
    plan = expected_plan("EVIDENCE_REQUEST", ["Cultivar"], evidence)
    assert plan["items"][0]["class_value"] == "Sheriff"
    assert plan["items"][0]["roi_ref"] == "Cultivar.roi"
    altered = json.loads(json.dumps(plan))
    altered["items"][0]["class_value"] = "Kvium"
    assert not verify_plan(json.dumps(altered), "EVIDENCE_REQUEST", ["Cultivar"], evidence)["accepted"]
