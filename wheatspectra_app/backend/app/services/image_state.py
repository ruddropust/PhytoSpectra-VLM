"""
PhytoSpectra-VLM / WheatSpectraChat
ImageStateBuilder: Adapts cached session image data into structured scientific state and LLM grounding context.
"""

from typing import Any, Dict, List, Optional


class ImageStateBuilder:
    @staticmethod
    def build(image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds a structured dictionary from cached session image data.
        """
        if not image_data:
            return {}

        cube = image_data.get("cube_unit")
        original_shape = image_data.get("original_shape", [])
        canonical_shape = image_data.get("canonical_shape", [])

        # Dimensions
        if canonical_shape and len(canonical_shape) == 3:
            height = int(canonical_shape[0])
            width = int(canonical_shape[1])
            bands = int(canonical_shape[2])
        elif cube is not None and hasattr(cube, "shape") and len(cube.shape) == 3:
            height = int(cube.shape[0])
            width = int(cube.shape[1])
            bands = int(cube.shape[2])
        else:
            height = 48
            width = 352
            bands = 204

        # Predicted targets
        raw_preds = image_data.get("ensemble_predictions", {})
        predicted_targets: Dict[str, Any] = {}

        for target in ["GrainWeight", "Gsw", "PhiPS2"]:
            if target in raw_preds and isinstance(raw_preds[target], dict):
                p = raw_preds[target]
                predicted_targets[target] = {
                    "value": p.get("value"),
                    "ensemble_std": p.get("ensemble_std"),
                }

        if "Fertilizer" in raw_preds and isinstance(raw_preds["Fertilizer"], dict):
            p = raw_preds["Fertilizer"]
            predicted_targets["Fertilizer"] = {
                "value": p.get("value"),
                "ensemble_std": p.get("ensemble_std"),
            }

        if "Cultivar" in raw_preds and isinstance(raw_preds["Cultivar"], dict):
            p = raw_preds["Cultivar"]
            predicted_targets["Cultivar"] = {
                "value": p.get("value"),
                "confidence": p.get("vote_confidence"),
                "cultivar_probabilities": p.get("cultivar_probabilities"),
            }

        # Mask statistics
        mask_stats = image_data.get("mask_stats", {})
        mask_statistics = {
            "foreground_fraction": mask_stats.get("foreground_fraction"),
            "bbox": mask_stats.get("bbox"),
            "raw_foreground_pixels": mask_stats.get("raw_foreground_pixels"),
            "final_foreground_pixels": mask_stats.get("final_foreground_pixels"),
        }

        # Evidence tokens
        pctg_result = image_data.get("pctg_result", {})
        pctg_evidence = pctg_result.get("evidence", {}) if isinstance(pctg_result, dict) else {}

        spatial_tokens = {}
        spectral_tokens = {}
        for pheno, ev in pctg_evidence.items():
            if isinstance(ev, dict):
                sp = ev.get("spatial", {})
                sl = ev.get("spectral", {})
                spatial_tokens[pheno] = sp.get("selected_token_indices", [])
                spectral_tokens[pheno] = sl.get("selected_token_indices", [])

        evidence = {
            "spatial_tokens": spatial_tokens,
            "spectral_tokens": spectral_tokens,
        }

        return {
            "image_type": "Hyperspectral wheat image",
            "height": height,
            "width": width,
            "bands": bands,
            "original_shape": original_shape or [bands, height, width],
            "canonical_shape": canonical_shape or [height, width, bands],
            "predicted_targets": predicted_targets,
            "mask_statistics": mask_statistics,
            "evidence": evidence,
            "preview_url": image_data.get("preview_url"),
        }

    @staticmethod
    def build_text_context(image_data: Dict[str, Any]) -> str:
        """
        Formats image state into exact grounding context for the LLM.
        """
        state = ImageStateBuilder.build(image_data)
        if not state:
            return "No image data available in session."

        h = state.get("height", 48)
        w = state.get("width", 352)
        b = state.get("bands", 204)
        targets = state.get("predicted_targets", {})

        lines = [
            "This is a hyperspectral wheat image.",
            f"Image dimension: {h} × {w} × {b}",
            f"Spectral bands: {b}",
            "",
            "Predicted phenotype measurements:",
        ]

        if "GrainWeight" in targets and targets["GrainWeight"].get("value") is not None:
            gw = targets["GrainWeight"]
            val = gw["value"]
            std = gw.get("ensemble_std")
            std_str = f" (std: {std:.2f})" if std is not None else ""
            lines.append(f"- GrainWeight = {val:.2f}{std_str}" if isinstance(val, float) else f"- GrainWeight = {val}")

        if "Gsw" in targets and targets["Gsw"].get("value") is not None:
            gsw = targets["Gsw"]
            val = gsw["value"]
            std = gsw.get("ensemble_std")
            std_str = f" (std: {std:.4f})" if std is not None else ""
            lines.append(f"- Gsw = {val:.4f}{std_str}" if isinstance(val, float) else f"- Gsw = {val}")

        if "PhiPS2" in targets and targets["PhiPS2"].get("value") is not None:
            phi = targets["PhiPS2"]
            val = phi["value"]
            std = phi.get("ensemble_std")
            std_str = f" (std: {std:.4f})" if std is not None else ""
            lines.append(f"- PhiPS2 = {val:.4f}{std_str}" if isinstance(val, float) else f"- PhiPS2 = {val}")

        if "Fertilizer" in targets and targets["Fertilizer"].get("value") is not None:
            fert = targets["Fertilizer"]
            val = fert["value"]
            lines.append(f"- Fertilizer = {val:.4f}" if isinstance(val, float) else f"- Fertilizer = {val}")

        if "Cultivar" in targets and targets["Cultivar"].get("value") is not None:
            cult = targets["Cultivar"]
            cult_val = cult["value"]
            conf = cult.get("confidence")
            conf_str = f" (confidence: {conf:.2f})" if conf is not None else ""
            lines.append(f"- Cultivar = {cult_val}{conf_str}")

        mask_stats = state.get("mask_statistics", {})
        fg_frac = mask_stats.get("foreground_fraction")
        if fg_frac is not None:
            lines.append(f"\nLeaf foreground fraction: {fg_frac:.2%}")

        bbox = mask_stats.get("bbox")
        if bbox is not None:
            lines.append(f"Leaf bounding box [x0, y0, x1, y1]: {bbox}")

        return "\n".join(lines)


# Backwards compatibility alias
ImageStateManager = ImageStateBuilder