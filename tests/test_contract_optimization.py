import json

from phytospectra.contract_optimization import build_curriculum, legal_reference_plans


def test_finite_plan_space_is_complete_and_deterministic():
    assert len(legal_reference_plans("TARGET_VALUE", ["GrainWeight"])) == 1
    assert len(legal_reference_plans("TARGET_VALUE", ["Fertilizer"])) == 3
    assert len(legal_reference_plans("EVIDENCE_REQUEST", ["Cultivar"])) == 4
    summaries = legal_reference_plans(
        "IMAGE_SUMMARY",
        ["GrainWeight", "Gsw", "PhiPS2", "Fertilizer", "Cultivar"],
    )
    assert len(summaries) == 12
    assert all(len(plan["items"]) == 5 for plan in summaries)


def test_unsupported_has_one_exact_empty_plan():
    assert legal_reference_plans("UNSUPPORTED", []) == [
        {"version": 1, "intent": "UNSUPPORTED", "items": []}
    ]


def test_summary_is_not_misclassified_as_single_target_regression():
    image_id = "00001"

    def record(key, intent, targets, plan):
        return {
            "record_id": key,
            "image_id": image_id,
            "intent": intent,
            "targets": targets,
            "question": key,
            "answer": json.dumps(plan, separators=(",", ":")),
        }

    rows = [
        record("reg", "TARGET_VALUE", ["GrainWeight"], legal_reference_plans("TARGET_VALUE", ["GrainWeight"])[0]),
        record("summary", "IMAGE_SUMMARY", ["GrainWeight", "Gsw", "PhiPS2", "Fertilizer", "Cultivar"], legal_reference_plans("IMAGE_SUMMARY", ["GrainWeight", "Gsw", "PhiPS2", "Fertilizer", "Cultivar"])[0]),
        record("unsupported", "UNSUPPORTED", [], legal_reference_plans("UNSUPPORTED", [])[0]),
    ]
    for index, plan in enumerate(legal_reference_plans("TARGET_VALUE", ["Fertilizer"])):
        rows.append(record(f"fert-{index}", "TARGET_VALUE", ["Fertilizer"], plan))
    for index, plan in enumerate(legal_reference_plans("TARGET_VALUE", ["Cultivar"])):
        rows.append(record(f"cult-{index}", "TARGET_VALUE", ["Cultivar"], plan))
    single, joint, report = build_curriculum(rows, [image_id], {}, 3)
    assert all(row["intent"] != "IMAGE_SUMMARY" for row in single)
    assert report["regression_records"] == 1
    assert report["summary_source_records"] == 1
    assert len(joint) - len(single) == 3
