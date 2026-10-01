import numpy as np
import pandas as pd
import json

from phytospectra.data import apply_stage42d_train_augmentation
from phytospectra.evidence import split_stage42d_alignment_images, stage42d_candidate_boxes
from phytospectra.refinement import _balanced_cultivar_donor_map
from phytospectra.vlm_evaluation import _local_only_eligible_records


def test_stage42d_augmentation_is_epoch_seed_deterministic_and_masks_background():
    cube = np.zeros((48, 352, 204), dtype=np.float32)
    mask = np.zeros((48, 352), dtype=bool)
    mask[8:40, 32:320] = True
    cube[mask] = 0.5
    first, first_mask = apply_stage42d_train_augmentation(cube, mask, np.random.default_rng(123))
    second, second_mask = apply_stage42d_train_augmentation(cube, mask, np.random.default_rng(123))
    assert np.array_equal(first_mask, second_mask)
    assert np.array_equal(first, second)
    assert np.count_nonzero(first[~first_mask]) == 0
    assert first.dtype == np.float32


def test_stage42d_alignment_split_is_model_seed_independent_and_disjoint():
    rows = []
    labels = ("Heerup", "Kvium", "Rembrandt", "Sheriff")
    fertilizer = (0.0, 0.5, 1.0)
    for index in range(200):
        cultivar = labels[index % len(labels)]
        fert = fertilizer[index % len(fertilizer)]
        rows.append({"ImageId": f"{index:05d}", "Fold": 1 if index >= 180 else 0, "Cultivar": cultivar, "Fertilizer": fert, "Stratum": f"{cultivar}__F{fert}"})
    frame = pd.DataFrame(rows)
    train, dev = split_stage42d_alignment_images(frame, fold=1, train_images=128, dev_images=16)
    again_train, again_dev = split_stage42d_alignment_images(frame, fold=1, train_images=128, dev_images=16)
    assert (train, dev) == (again_train, again_dev)
    assert len(train) == 128 and len(dev) == 16
    assert not set(train) & set(dev)


def test_stage42d_candidate_boxes_are_four_equal_unordered_windows():
    mask = np.ones((48, 352), dtype=bool)
    first = stage42d_candidate_boxes(mask, image_id="00042", seed=403026)
    second = stage42d_candidate_boxes(mask, image_id="00042", seed=403026)
    assert np.array_equal(first, second)
    assert first.shape == (4, 4)
    assert np.all(first[:, 2] - first[:, 0] == 64)
    assert np.all(first[:, 3] - first[:, 1] == 16)
    assert len({tuple(row) for row in first.tolist()}) == 4


def test_stage42d_cultivar_donors_produce_exact_four_class_balance(tmp_path):
    labels = ("Heerup", "Kvium", "Rembrandt", "Sheriff")
    image_ids = [f"{index:05d}" for index in range(128)]
    rows_by_image = {}
    observed = {}
    for index, image_id in enumerate(image_ids):
        label = labels[index % 4]
        evidence = tmp_path / f"{image_id}.json"
        evidence.write_text(json.dumps({"predictions": {"Cultivar": {"value": label}}}), encoding="utf-8")
        rows_by_image[image_id] = [{"evidence_path": str(evidence)}]
        observed[image_id] = label
    donors, audit = _balanced_cultivar_donor_map(image_ids, rows_by_image)
    combined = [observed[image_id] for image_id in image_ids] * 2 + [observed[donors[image_id]] for image_id in image_ids] * 2
    assert {label: combined.count(label) for label in labels} == {label: 128 for label in labels}
    assert all(observed[recipient] != observed[donor] for recipient, donor in donors.items())
    assert audit["method"] == "deterministic_forbidden_diagonal_max_flow"


def test_local_only_records_require_an_informative_frozen_roi(tmp_path):
    active = tmp_path / "active.json"
    active.write_text(json.dumps({
        "selections": {
            "Fertilizer": {"status": "informative", "selected_box": [0, 0, 64, 16]},
        }
    }), encoding="utf-8")
    inactive = tmp_path / "inactive.json"
    inactive.write_text(json.dumps({
        "selections": {
            "Cultivar": {"status": "uninformative", "selected_box": None},
        }
    }), encoding="utf-8")
    records = [
        {"record_id": "active", "intent": "TARGET_VALUE", "targets": ["Fertilizer"], "evidence_path": str(active)},
        {"record_id": "inactive", "intent": "EVIDENCE_REQUEST", "targets": ["Cultivar"], "evidence_path": str(inactive)},
        {"record_id": "regression", "intent": "TARGET_VALUE", "targets": ["Gsw"], "evidence_path": str(active)},
        {"record_id": "summary", "intent": "IMAGE_SUMMARY", "targets": ["Fertilizer"], "evidence_path": str(active)},
    ]
    assert [row["record_id"] for row in _local_only_eligible_records(records)] == ["active"]
