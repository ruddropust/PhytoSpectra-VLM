"""Deterministic, phenotype-specific visual evidence rendering.

The renderer converts PCTG spatial/spectral evidence into a false-color HSI
composite, attribution heatmap, selected-patch view, focused crop, spectral-token
plot, and a text-labelled multi-panel bundle. It never treats attribution as a
biological or causal explanation.
"""

from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.config import settings
from app.inference.preprocessing import create_false_color_image
from app.inference.visual_encoder import PhenotypeConditionedVisualEncoder


class VERRenderer:
    def __init__(self, media_dir: Path = settings.MEDIA_DIR):
        self.media_dir = media_dir
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.visual_encoder = PhenotypeConditionedVisualEncoder()

    @staticmethod
    def _font(size: int = 18):
        try:
            return ImageFont.truetype("arial.ttf", size)
        except OSError:
            return ImageFont.load_default()

    def render_composite(
        self,
        cube_unit: np.ndarray,
        mask: np.ndarray,
        image_id: str,
    ) -> Tuple[str, Image.Image]:
        img = create_false_color_image(cube_unit, mask)
        filename = f"{image_id}_composite.png"
        img.save(self.media_dir / filename)
        return f"/media/{filename}", img

    def draw_spatial_boxes(
        self,
        image: Image.Image,
        boxes: List[List[int]],
        focus_boxes: Optional[List[List[int]]] = None,
    ) -> Image.Image:
        """Draw all selected patches in cyan and the focused component in yellow."""
        marked = image.copy()
        draw = ImageDraw.Draw(marked)
        for box in boxes:
            x0, y0, x1, y1 = box
            draw.rectangle(
                [x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)],
                outline=(0, 235, 255),
                width=1,
            )
        for box in focus_boxes or []:
            x0, y0, x1, y1 = box
            draw.rectangle(
                [x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)],
                outline=(255, 220, 0),
                width=2,
            )
        return marked

    def render_attention_heatmap(
        self,
        composite: Image.Image,
        normalized_spatial_map: np.ndarray,
        mask: np.ndarray,
    ) -> Image.Image:
        """Overlay normalized model-space spatial-token sensitivity on the HSI view."""
        grid = np.asarray(normalized_spatial_map, dtype=np.float32)
        if grid.ndim != 2:
            grid = np.zeros((6, 22), dtype=np.float32)
        grid = np.clip(np.where(np.isfinite(grid), grid, 0.0), 0.0, 1.0)

        heat_small = (plt.get_cmap("inferno")(grid)[..., :3] * 255.0).astype(np.uint8)
        heat = Image.fromarray(heat_small).resize(composite.size, Image.Resampling.BILINEAR)
        heat_array = np.asarray(heat, dtype=np.float32)
        base_array = np.asarray(composite.convert("RGB"), dtype=np.float32)
        alpha_small = Image.fromarray((grid * 170.0).astype(np.uint8))
        alpha = np.asarray(
            alpha_small.resize(composite.size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        alpha *= mask.astype(np.float32)
        blended = base_array * (1.0 - alpha[..., None]) + heat_array * alpha[..., None]
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    def crop_and_zoom(
        self,
        image: Image.Image,
        crop_box: Optional[List[int]],
        zoom_factor: int = 4,
    ) -> Image.Image:
        if not crop_box:
            crop_box = [0, 0, image.width, image.height]
        cropped = image.crop(tuple(crop_box))
        return cropped.resize(
            (
                max(1, cropped.width * zoom_factor),
                max(1, cropped.height * zoom_factor),
            ),
            Image.Resampling.NEAREST,
        )

    def render_spectral_plot(
        self,
        scores: np.ndarray,
        selected_indices: List[int],
        phenotype: str,
        filename: str,
    ) -> str:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        selected_indices = [
            int(index)
            for index in selected_indices
            if 0 <= int(index) < len(scores)
        ]
        fig = plt.figure(figsize=(7.5, 3.2), dpi=150)
        ax = fig.add_subplot(111)
        x = np.arange(len(scores))
        ax.plot(x, scores, color="#2563eb", linewidth=1.5, label="Token sensitivity")
        if selected_indices:
            indices = np.asarray(selected_indices, dtype=int)
            ax.scatter(
                indices,
                scores[indices],
                color="#dc2626",
                s=35,
                zorder=5,
                label="Selected tokens",
            )
        ax.set_xlabel("Spectral token index (model space, not nm)")
        ax.set_ylabel("Counterfactual sensitivity")
        ax.set_title(f"PCTG spectral evidence — {phenotype}")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(self.media_dir / filename, bbox_inches="tight")
        plt.close(fig)
        return f"/media/{filename}"

    @staticmethod
    def label_panel(image: Image.Image, label: str) -> Image.Image:
        top_margin = 24
        canvas = Image.new("RGB", (image.width, image.height + top_margin), "black")
        canvas.paste(image, (0, top_margin))
        ImageDraw.Draw(canvas).text((6, 4), label[:100], fill="white", font=VERRenderer._font(14))
        return canvas

    def compose_bundle(
        self,
        panels: List[Tuple[str, Image.Image]],
        filename: str,
    ) -> str:
        columns = 2
        cell_width = 700
        cell_height = 330
        label_height = 30
        rows = (len(panels) + columns - 1) // columns
        bundle = Image.new("RGB", (columns * cell_width, rows * cell_height), "black")
        bundle_draw = ImageDraw.Draw(bundle)

        for index, (label, image) in enumerate(panels):
            row = index // columns
            column = index % columns
            x_origin = column * cell_width
            y_origin = row * cell_height
            available_height = cell_height - label_height
            scale = min(
                cell_width / max(1, image.width),
                available_height / max(1, image.height),
            )
            resized = image.resize(
                (
                    max(1, int(round(image.width * scale))),
                    max(1, int(round(image.height * scale))),
                ),
                Image.Resampling.NEAREST,
            )
            paste_x = x_origin + (cell_width - resized.width) // 2
            paste_y = y_origin + label_height + (available_height - resized.height) // 2
            bundle.paste(resized, (paste_x, paste_y))
            bundle_draw.text(
                (x_origin + 10, y_origin + 6),
                label[:100],
                fill="white",
                font=self._font(18),
            )

        bundle.save(self.media_dir / filename)
        return f"/media/{filename}"

    @staticmethod
    def _prediction_text(phenotype: str, prediction: Optional[Dict[str, Any]]) -> str:
        prediction = prediction or {}
        value = prediction.get("value")
        if value is None:
            return f"{phenotype}: output unavailable"
        if phenotype == "Cultivar":
            text = f"Cultivar = {value}"
            vote = prediction.get("vote_confidence", prediction.get("confidence"))
            if isinstance(vote, (int, float)):
                text += f" (ensemble vote {vote:.0%})"
            return text
        rendered = f"{float(value):.2f}" if isinstance(value, (int, float)) and abs(value) >= 100 else (
            f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)
        )
        unit = prediction.get("unit")
        return f"{phenotype} = {rendered}" + (f" {unit}" if unit else "")

    def render_phenotype_visuals(
        self,
        cube_unit: np.ndarray,
        mask: np.ndarray,
        pctg_result: Dict[str, Any],
        phenotype: str,
        session_id: str,
        prediction: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Render text-linked visual evidence for one requested phenotype."""
        render_id = str(uuid.uuid4())[:8]
        composite = create_false_color_image(cube_unit, mask)
        encoding = self.visual_encoder.encode(pctg_result, phenotype, mask)
        all_boxes = [token["pixel_box"] for token in encoding.selected_tokens]
        focus_boxes = [token["pixel_box"] for token in encoding.focus_tokens]

        heatmap = self.render_attention_heatmap(composite, encoding.normalized_spatial_map, mask)
        marked = self.draw_spatial_boxes(composite, all_boxes, focus_boxes)
        focused_overlay = self.draw_spatial_boxes(heatmap, focus_boxes, focus_boxes)
        zoom = self.crop_and_zoom(focused_overlay, encoding.focus_box)

        prefix = f"{session_id}_{phenotype}_{render_id}"
        heatmap_filename = f"{prefix}_heatmap.png"
        marked_filename = f"{prefix}_marked.png"
        zoom_filename = f"{prefix}_zoom.png"
        heatmap.save(self.media_dir / heatmap_filename)
        marked.save(self.media_dir / marked_filename)
        zoom.save(self.media_dir / zoom_filename)

        spectral_url: Optional[str] = None
        spectral_image: Optional[Image.Image] = None
        if encoding.spectral_scores is not None:
            spectral_filename = f"{prefix}_spectral.png"
            spectral_url = self.render_spectral_plot(
                encoding.spectral_scores,
                encoding.selected_spectral_indices,
                phenotype,
                spectral_filename,
            )
            spectral_image = Image.open(self.media_dir / spectral_filename).convert("RGB")

        prediction_text = self._prediction_text(phenotype, prediction)
        bundle_panels: List[Tuple[str, Image.Image]] = [
            ("Input: HSI false-color composite", composite),
            (f"{phenotype}: attribution heatmap", heatmap),
            ("Cyan: selected patches; yellow: focus", marked),
            (f"Focused crop — {prediction_text}", zoom),
        ]
        if spectral_image is not None:
            bundle_panels.append(("Spectral-token sensitivity (not nm)", spectral_image))
        bundle_filename = f"{prefix}_bundle.png"
        bundle_url = self.compose_bundle(bundle_panels, bundle_filename)

        selected_indices = [
            int(token.get("token_index"))
            for token in encoding.selected_tokens
            if isinstance(token, dict) and token.get("token_index") is not None
        ]
        if encoding.evidence_available and encoding.focus_box:
            spatial_summary = (
                f"For {phenotype}, PCTG selected {len(selected_indices)} spatial patches. "
                f"The strongest connected evidence region is shown at pixel box {encoding.focus_box}. "
                "Yellow boxes identify the focused region; cyan boxes show other selected patches."
            )
        else:
            spatial_summary = (
                f"Target-specific spatial evidence was unavailable for {phenotype}; "
                "the crop falls back to the segmented leaf region."
            )

        return {
            "target": phenotype,
            "prediction_text": prediction_text,
            "visual_method": "PCTG_COUNTERFACTUAL_PATCH_ATTRIBUTION",
            "evidence_available": encoding.evidence_available,
            "focus_box": encoding.focus_box,
            "selected_token_indices": selected_indices,
            "selected_token_count": len(selected_indices),
            "spatial_summary": spatial_summary,
            "heatmap_url": f"/media/{heatmap_filename}",
            "marked_url": f"/media/{marked_filename}",
            "zoom_url": f"/media/{zoom_filename}",
            "spectral_plot_url": spectral_url,
            "bundle_url": bundle_url,
        }
