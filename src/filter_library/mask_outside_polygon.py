"""
Filtro: MaskOutsidePolygon
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any
from .base_filter import BaseFilter, FILTER_REGISTRY


class MaskOutsidePolygon(BaseFilter):
    """Pinta de blanco todo lo que cae fuera del polígono detectado"""

    FILTER_NAME = "MaskOutsidePolygon"
    DESCRIPTION = (
        "Pinta de blanco el área exterior al polígono de la página. "
        "Recibe las esquinas del polígono (en espacio pre-crop) y el crop_rect "
        "para ajustar las coordenadas al espacio de la imagen recortada. "
        "Aplica blur al borde de la máscara para una transición suave."
    )

    INPUTS = {
        "input_image": "image",
        "corners": "quad_points",
        "crop_rect": "rect"
    }

    OUTPUTS = {
        "masked_image": "image",
        "sample_image": "image"
    }

    PARAMS = {
        "blur_radius": {
            "default": 10,
            "min": 0,
            "max": 100,
            "step": 2,
            "description": "Radio del blur en el borde de la máscara. 0=borde duro, mayor=transición más suave"
        },
        "show_comparison": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "0=solo resultado, 1=comparación lado a lado (original vs enmascarado)"
        }
    }

    # Orden de las esquinas para construir el polígono (sentido horario)
    _CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]

    def _get_polygon_points(self, corners: dict, offset_x: int, offset_y: int) -> np.ndarray | None:
        """Extrae y ajusta los puntos del polígono al espacio de la imagen recortada."""
        pts = []
        for name in self._CORNER_ORDER:
            c = corners.get(name)
            if c is None or "x" not in c or "y" not in c:
                return None
            pts.append([int(c["x"]) - offset_x, int(c["y"]) - offset_y])
        return np.array(pts, dtype=np.int32)

    def _load_from_crop_json(self) -> dict | None:
        """Carga el polígono desde el .crop.json compañero de current_image_path."""
        if not self.current_image_path:
            return None
        crop_path = Path(self.current_image_path).with_suffix(".crop.json")
        if not crop_path.exists():
            return None
        try:
            with open(crop_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("polygon", {})
        except Exception:
            return None

    def _create_mask(self, h: int, w: int, pts: np.ndarray, blur_radius: int) -> np.ndarray:
        """Genera la máscara: 255 dentro del polígono, 0 fuera, con blur en el borde."""
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        if blur_radius > 0:
            ksize = int(blur_radius) * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), blur_radius / 3.0)

        return mask

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compone imagen con fondo blanco usando la máscara como alpha."""
        alpha = mask.astype(np.float32) / 255.0

        if len(image.shape) == 3:
            alpha3 = alpha[:, :, np.newaxis]
            white = np.ones_like(image, dtype=np.float32) * 255.0
            result = alpha3 * image.astype(np.float32) + (1.0 - alpha3) * white
        else:
            white = np.full_like(image, 255, dtype=np.float32)
            result = alpha * image.astype(np.float32) + (1.0 - alpha) * white

        return np.clip(result, 0, 255).astype(np.uint8)

    def _create_comparison(self, original: np.ndarray, result: np.ndarray) -> np.ndarray:
        orig = original if len(original.shape) == 3 else cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        res = result if len(result.shape) == 3 else cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        h, w = orig.shape[:2]
        sep = 20
        canvas = np.zeros((h, w * 2 + sep, 3), dtype=np.uint8)
        canvas[:, :w] = orig
        canvas[:, w + sep:] = res
        canvas[:, w:w + sep] = (100, 100, 100)
        cv2.putText(canvas, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "ENMASCARADO", (w + sep + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return canvas

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        corners = inputs.get("corners", {})
        crop_rect = inputs.get("crop_rect", {})

        blur_radius = int(self.params["blur_radius"])
        show_comparison = int(self.params["show_comparison"])

        h, w = input_img.shape[:2]

        # Si no hay corners en los inputs, intentar cargar desde .crop.json
        if not corners:
            corners = self._load_from_crop_json() or {}

        # Offset del crop (0 si el polígono ya viene en espacio de imagen, e.g. desde .crop.json)
        offset_x = int(crop_rect.get("x1", 0)) if crop_rect else 0
        offset_y = int(crop_rect.get("y1", 0)) if crop_rect else 0

        pts = self._get_polygon_points(corners, offset_x, offset_y)

        if pts is None:
            # Sin polígono válido: devolver imagen sin cambios
            sample = input_img.copy() if len(input_img.shape) == 3 \
                else cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
            return {"masked_image": input_img, "sample_image": sample}

        mask = self._create_mask(h, w, pts, blur_radius)
        masked_image = self._apply_mask(input_img, mask)

        if self.without_preview:
            sample_image = masked_image.copy() if len(masked_image.shape) == 3 \
                else cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)
        elif show_comparison:
            sample_image = self._create_comparison(input_img, masked_image)
        else:
            sample_image = masked_image.copy() if len(masked_image.shape) == 3 \
                else cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)

        return {
            "masked_image": masked_image,
            "sample_image": sample_image
        }
