"""The v3 upload endpoint should retain the reference chatbot's upload UX."""

from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import torch
from fastapi.testclient import TestClient

from app import main as chatbot_main
from app import runtime as chatbot_runtime
from app.runtime import ImageState
from app.query import plan_query
from phytospectra.contract import expected_plan


def _tiff_bytes(shape=(204, 48, 352)) -> bytes:
    stream = BytesIO()
    tifffile.imwrite(stream, np.zeros(shape, dtype=np.uint16))
    return stream.getvalue()


def test_upload_without_session_creates_one_and_cleans_temporary_file(monkeypatch, tmp_path):
    cube = np.zeros((48, 352, 204), dtype=np.float32)
    mask = np.zeros((48, 352), dtype=bool)
    mask[10:20, 20:40] = True
    cube[mask] = 0.1
    state = ImageState("test-image", cube, mask, None, {}, {})
    monkeypatch.setattr(chatbot_main, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(chatbot_main, "MEDIA", tmp_path / "media")
    chatbot_main.UPLOADS.mkdir()
    chatbot_main.MEDIA.mkdir()
    monkeypatch.setattr(chatbot_main.runtime, "prepare_image", lambda path: state)

    response = TestClient(chatbot_main.app).post(
        "/api/v1/upload",
        files={"file": ("leaf.tiff", _tiff_bytes(), "image/tiff")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"]
    assert payload["original_shape"] == [204, 48, 352]
    assert payload["canonical_shape"] == [48, 352, 204]
    assert payload["mask_statistics"]["final_foreground_pixels"] == 200
    assert not list(chatbot_main.UPLOADS.iterdir())


def test_upload_rejects_oversized_or_wrong_shape_before_model_load(monkeypatch, tmp_path):
    monkeypatch.setattr(chatbot_main, "UPLOADS", tmp_path)
    def should_not_run(path):
        raise AssertionError("model must not load for invalid input")
    monkeypatch.setattr(chatbot_main.runtime, "prepare_image", should_not_run)
    client = TestClient(chatbot_main.app)

    monkeypatch.setattr(chatbot_main, "MAX_UPLOAD_BYTES", 4)
    too_large = client.post(
        "/api/v1/upload", files={"file": ("leaf.tiff", b"12345", "image/tiff")}
    )
    assert too_large.status_code == 413

    monkeypatch.setattr(chatbot_main, "MAX_UPLOAD_BYTES", 100 * 1024 * 1024)
    wrong_shape = client.post(
        "/api/v1/upload", files={"file": ("leaf.tiff", _tiff_bytes((2, 2)), "image/tiff")}
    )
    assert wrong_shape.status_code == 400
    assert not list(tmp_path.iterdir())


def test_live_upload_uses_four_candidate_selection(monkeypatch, tmp_path):
    cube = np.zeros((48, 352, 204), dtype=np.float32)
    mask = np.zeros((48, 352), dtype=bool)
    mask[8:24, 32:160] = True
    cube[mask] = 0.1
    model = chatbot_runtime.Runtime()
    model.model = object()
    model.normalization = {
        "mean": np.zeros(204, dtype=np.float32),
        "std": np.ones(204, dtype=np.float32),
        "targets": {},
    }
    monkeypatch.setattr(model, "_load", lambda: None)
    monkeypatch.setattr(chatbot_runtime, "load_cube", lambda path: cube)
    monkeypatch.setattr(chatbot_runtime, "summarize_predictions", lambda *args: {})
    from phytospectra import data
    monkeypatch.setattr(data, "create_leaf_mask", lambda value: mask)
    boxes = np.asarray([(32, 8, 96, 24), (48, 8, 112, 24), (64, 8, 128, 24), (80, 8, 144, 24)])
    scores = np.asarray([
        [0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.1],
        [0.4, 0.3, 0.2, 0.1], [0.2, 0.3, 0.4, 0.1], [0.3, 0.2, 0.1, 0.4],
    ])
    observed = {}
    def four_candidate_evidence(*args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(spatial_scores=scores), boxes
    monkeypatch.setattr(chatbot_runtime, "_stage42d_evidence", four_candidate_evidence)
    upload = tmp_path / "leaf.tiff"
    upload.write_bytes(b"stable image content")

    state = model.prepare_image(upload)

    assert observed["seed"] == 403026 + model.fold
    assert len(observed["image_id"]) == 64
    assert state.selections["GrainWeight"]["selected_box"] == [80, 8, 144, 24]
    assert state.selections["Gsw"]["selected_box"] is None


def test_chat_generation_receives_selected_roi_geometry(monkeypatch):
    model = chatbot_runtime.Runtime()
    model.normalization = {
        "mean": np.zeros(204, dtype=np.float32),
        "std": np.ones(204, dtype=np.float32),
    }
    state = ImageState(
        "test-image", np.zeros((48, 352, 204), dtype=np.float32),
        np.ones((48, 352), dtype=bool), None,
        {"Gsw": {"selected_box": [32, 8, 96, 24]}},
        {"Gsw": {"value": 0.1}},
    )
    observed = {}
    def fake_visual(*args, **kwargs):
        observed["boxes"] = args[4]
        return {name: torch.zeros(1, 1) for name in (
            "global_hsi", "global_leaf_mask", "local_hsi", "local_leaf_masks",
            "local_view_mask", "local_target_indices", "primary_target_index",
        )} | {"roi_boxes_xyxy": torch.tensor([[[32, 8, 96, 24], [0, 0, 0, 0]]])}
    monkeypatch.setattr(chatbot_runtime, "prepare_visual_inputs", fake_visual)

    class FakeCollator:
        def __init__(self, *args):
            pass
        def for_generation(self, samples):
            visual = {name: value.unsqueeze(0) for name, value in samples[0].items() if isinstance(value, torch.Tensor)}
            return visual | {name: torch.ones(1, 1, dtype=torch.long) for name in (
                "input_ids", "attention_mask", "question_input_ids", "question_attention_mask",
            )}
    class FakeVLM:
        def generate(self, **kwargs):
            assert kwargs["roi_boxes_xyxy"].shape == (1, 2, 4)
            return torch.tensor([[1]])
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        def decode(self, *args, **kwargs):
            return json.dumps(expected_plan("TARGET_VALUE", ["Gsw"], {"predictions": state.predictions}))

    monkeypatch.setattr(chatbot_runtime, "HyperLeafVQACollator", FakeCollator)
    monkeypatch.setattr(model, "_load_vlm", lambda: None)
    model.vlm = FakeVLM()
    model.tokenizer = FakeTokenizer()
    result = model._generate_plan(state, "What is the predicted Gsw?", plan_query("What is the predicted Gsw?"))

    assert observed["boxes"] == {"Gsw": [32, 8, 96, 24]}
    assert result["accepted"] is True


@pytest.mark.skipif(
    os.environ.get("PHYTOSPECTRA_LIVE_UPLOAD_TEST") != "1",
    reason="Requires local trained artifacts and one HyperLeaf TIFF",
)
def test_trained_model_accepts_real_tiff(monkeypatch, tmp_path):
    workspace = Path(__file__).resolve().parents[2]
    research = workspace / "research_v3"
    sample = workspace / "data" / "raw" / "images" / "00000.tiff"
    checkpoint = research / "artifacts" / "stage42d_exact_v1" / "runs" / "seed_2026_fold_0" / "phytocore" / "phytocore_final.pt"
    if not sample.is_file() or not checkpoint.is_file():
        pytest.skip("Local TIFF or trained checkpoint is unavailable")
    monkeypatch.setenv("PHYTOSPECTRA_ROOT", str(research))
    monkeypatch.setenv("PHYTOSPECTRA_RUN_ID", "stage42d_exact_v1")
    monkeypatch.setenv("PHYTOSPECTRA_SEED", "2026")
    monkeypatch.setenv("PHYTOSPECTRA_FOLD", "0")
    live_runtime = chatbot_runtime.Runtime()
    monkeypatch.setattr(chatbot_main, "runtime", live_runtime)
    monkeypatch.setattr(chatbot_main, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(chatbot_main, "MEDIA", tmp_path / "media")
    chatbot_main.UPLOADS.mkdir()
    chatbot_main.MEDIA.mkdir()

    with sample.open("rb") as source:
        response = TestClient(chatbot_main.app).post(
            "/api/v1/upload", files={"file": (sample.name, source, "image/tiff")}
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["canonical_shape"] == [48, 352, 204]
    assert payload["image_id"] in live_runtime.images
    assert (chatbot_main.MEDIA / f"{payload['image_id']}.png").is_file()
    assert not list(chatbot_main.UPLOADS.iterdir())
