"""
PhytoSpectra-VLM / WheatSpectraChat
TIFF upload, leaf-mask preprocessing, and live inference caching API endpoint.
"""

import io
import uuid
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
import numpy as np
import tifffile

from app.config import settings
from app.core.constants import PHENOTYPES
from app.core.gpu_lock import get_gpu_lock
from app.core.registry import get_model_registry
from app.core.session_cache import session_cache
from app.inference.preprocessing import (
    validate_and_canonicalize_cube,
    create_leaf_mask,
)
from app.schemas.api import MaskStatistics, UploadResponse

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_tiff(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    # Validate extension
    filename = file.filename or ""
    if not (filename.lower().endswith(".tif") or filename.lower().endswith(".tiff")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only TIFF files (.tif, .tiff) are accepted for hyperspectral inference.",
        )

    # Validate or create session
    if not session_id:
        session_id = session_cache.create_session()
    elif not session_cache.get_session(session_id):
        # A persisted browser conversation can outlive the backend TTL. Return a
        # fresh, valid session instead of recreating an already-expired ID.
        session_id = session_cache.create_session()

    # Enforce the limit while reading so oversized uploads are not fully buffered.
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    chunks = []
    total_bytes = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded file exceeds maximum limit of {settings.MAX_UPLOAD_SIZE_MB}MB.",
            )
        chunks.append(chunk)
    contents = b"".join(chunks)

    try:
        raw_cube = tifffile.imread(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode the TIFF file. Confirm that it is a valid, uncompressed hyperspectral TIFF.",
        )

    try:
        cube_unit, orig_shape = validate_and_canonicalize_cube(raw_cube)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Generate leaf mask on the fly (Notebook 01B algorithm)
    mask, support, mask_stats = create_leaf_mask(cube_unit)
    if mask_stats["final_foreground_pixels"] == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Leaf masking produced 0 foreground pixels. Sample is invalid or empty.",
        )

    image_id = str(uuid.uuid4())[:12]
    registry = get_model_registry()

    # Reject a broken explicitly configured runtime before doing legacy GPU
    # inference or writing preview artifacts.
    image_vlm_enabled = bool(
        settings.RESEARCH_V3_VLM_ENABLED
        or settings.FROZEN_STAGE42_VLM_ENABLED
        or settings.FROZEN_STAGE40_VLM_ENABLED
        or settings.IDCV_DEVELOPMENT_VLM_ENABLED
    )
    if image_vlm_enabled and registry.idcv_development_vlm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The configured ID-CV vision-language runtime is unavailable. "
                "No legacy prediction was substituted."
            ),
        )
    if registry.idcv_development_vlm is not None:
        runtime_status = registry.idcv_development_vlm.status()
        if not bool(runtime_status.get("verified", False)):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The configured ID-CV vision-language bundle did not pass "
                    "verification. No legacy prediction was substituted."
                ),
            )

    # Pre-render composite preview
    preview_url, _ = registry.ver_renderer.render_composite(cube_unit, mask, image_id)

    # A frozen Stage-40/42 runtime computes its own hash-bound PhytoCore output
    # and direct evidence. Do not run the obsolete ensemble/PCTG stack beside it.
    if (
        (
            settings.RESEARCH_V3_VLM_ENABLED
            and settings.RESEARCH_V3_EXCLUSIVE_RUNTIME
        )
        or (
            settings.FROZEN_STAGE42_VLM_ENABLED
            and settings.FROZEN_STAGE42_EXCLUSIVE_RUNTIME
        )
        or (
            settings.FROZEN_STAGE40_VLM_ENABLED
            and settings.FROZEN_STAGE40_EXCLUSIVE_RUNTIME
        )
    ):
        ensemble_predictions = {}
        pctg_result = {}
    else:
        async with get_gpu_lock():
            ensemble_predictions = registry.phytocore_ensemble.predict_live_ensemble(
                cube_unit, mask
            )
            pctg_result = registry.pctg_manager.compute_pctg_for_image(
                registry.phytocore_ensemble.reference_model, cube_unit, mask
            )

    # Optionally run Stage-30 IDCV development preparation. Once explicitly
    # enabled this path is fail-closed: a request must not silently cache only
    # the legacy representation and later return an answer from another model.
    idcv_prepared = None
    if registry.idcv_development_vlm is not None:
        try:
            async with get_gpu_lock():
                idcv_prepared = registry.idcv_development_vlm.prepare_image(
                    cube_unit_hwc=cube_unit,
                    leaf_mask=mask,
                    image_id=image_id,
                    media_dir=settings.MEDIA_DIR,
                )
        except Exception as exc:
            print(
                f"IDCV development preparation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The ID-CV vision-language runtime could not prepare this "
                    "image. No legacy prediction was substituted."
                ),
            ) from exc

    # Cache image data in session
    cache_data = {
        "cube_unit": cube_unit,
        "mask": mask,
        "mask_stats": mask_stats,
        "original_shape": list(orig_shape),
        "canonical_shape": list(cube_unit.shape),
        "preview_url": preview_url,
        "ensemble_predictions": ensemble_predictions,
        "pctg_result": pctg_result,
        "idcv_prepared": (
            idcv_prepared.to_cache_dict() if idcv_prepared else None
        ),
    }
    try:
        session_cache.store_image_data(
            session_id=session_id,
            image_id=image_id,
            data=cache_data,
        )
    except Exception:
        # Remove partial cache entry on failure
        session_cache._images.pop(image_id, None)
        raise

    mask_schema = MaskStatistics(
        raw_foreground_pixels=mask_stats["raw_foreground_pixels"],
        final_foreground_pixels=mask_stats["final_foreground_pixels"],
        raw_components=mask_stats["raw_components"],
        kept_components=mask_stats["kept_components"],
        retained_foreground_ratio=mask_stats.get("retained_foreground_ratio"),
        foreground_fraction=mask_stats["foreground_fraction"],
        sat_fraction=mask_stats.get("sat_fraction"),
        near_sat_fraction=mask_stats.get("near_sat_fraction"),
        bbox=mask_stats.get("bbox"),
    )

    return UploadResponse(
        session_id=session_id,
        image_id=image_id,
        original_shape=list(orig_shape),
        canonical_shape=list(cube_unit.shape),
        mask_statistics=mask_schema,
        preview_url=preview_url,
        supported_targets=PHENOTYPES,
    )
