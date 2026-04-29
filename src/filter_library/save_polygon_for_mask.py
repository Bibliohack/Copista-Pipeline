"""
Filtro: SavePolygonForMask
"""

import numpy as np
from typing import Dict, Any
from .base_filter import BaseFilter, FILTER_REGISTRY


class SavePolygonForMask(BaseFilter):
    """Transforma el polígono al espacio de la imagen recortada+orientada y lo emite como JSON"""

    FILTER_NAME = "SavePolygonForMask"
    DESCRIPTION = (
        "Toma las corners rotadas (pre-crop) y el crop_rect, ajusta las coordenadas al espacio "
        "de la imagen recortada y aplica la rotación ortogonal final. Emite el polígono como "
        "crop_polygon (metadata) para que batch_processor lo guarde como .crop.json compañero. "
        "Pasa la imagen de entrada sin cambios."
    )

    INPUTS = {
        "input_image": "image",
        "corners": "quad_points",
        "crop_rect": "rect"
    }

    OUTPUTS = {
        "output_image": "image",
        "crop_polygon": "metadata",
        "sample_image": "image"
    }

    PARAMS = {
        "rotation": {
            "default": 0,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": (
                "Rotación ortogonal aplicada por orient_page: "
                "0=ninguna, 1=90° CCW, 2=180°, 3=90° CW (Izquierda=3, Derecha=1)"
            )
        }
    }

    _CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]

    def _transform_point(self, x: float, y: float,
                         crop_x1: int, crop_y1: int,
                         crop_w: int, crop_h: int,
                         rotation: int) -> tuple[int, int]:
        """Ajusta un punto al espacio crop y aplica la rotación ortogonal."""
        ax = x - crop_x1
        ay = y - crop_y1

        if rotation == 0:
            return int(round(ax)), int(round(ay))
        elif rotation == 1:
            # 90° CCW: new_x = ay, new_y = crop_w - 1 - ax
            return int(round(ay)), int(round(crop_w - 1 - ax))
        elif rotation == 2:
            # 180°: new_x = crop_w - 1 - ax, new_y = crop_h - 1 - ay
            return int(round(crop_w - 1 - ax)), int(round(crop_h - 1 - ay))
        else:
            # 90° CW: new_x = crop_h - 1 - ay, new_y = ax
            return int(round(crop_h - 1 - ay)), int(round(ax))

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        corners = inputs.get("corners", {})
        crop_rect = inputs.get("crop_rect", {})
        rotation = int(self.params["rotation"])

        crop_x1 = int(crop_rect.get("x1", 0))
        crop_y1 = int(crop_rect.get("y1", 0))
        crop_x2 = int(crop_rect.get("x2", input_img.shape[1]))
        crop_y2 = int(crop_rect.get("y2", input_img.shape[0]))
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        # Dimensiones de la imagen final (tras orient_page)
        h, w = input_img.shape[:2]

        polygon = {}
        for name in self._CORNER_ORDER:
            c = corners.get(name)
            if c and "x" in c and "y" in c:
                nx, ny = self._transform_point(
                    float(c["x"]), float(c["y"]),
                    crop_x1, crop_y1, crop_w, crop_h, rotation
                )
                polygon[name] = {"x": nx, "y": ny}

        crop_polygon = {
            "polygon": polygon,
            "image_width": w,
            "image_height": h,
            "rotation_applied": rotation
        }

        sample = input_img.copy() if len(input_img.shape) == 3 else input_img.copy()

        return {
            "output_image": input_img,
            "crop_polygon": crop_polygon,
            "sample_image": sample
        }
