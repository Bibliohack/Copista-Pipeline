"""
Filtro: CalculatePolygonProportion

Calcula la proporción de un polígono de 4 esquinas.
La proporción se define como: avg(lado_izquierdo, lado_derecho) / lado_superior

Diseñado para usarse con la salida de PolygonToGTFormat o archivos .det.json/.gt.json.
"""

import cv2
import math
import numpy as np
from typing import Dict, Any, Optional
from .base_filter import BaseFilter, FILTER_REGISTRY


# Orden canónico de esquinas (igual que polygon_to_gt_format)
CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]

SIDE_COLORS = {
    "top":    (0, 255, 255),    # amarillo
    "right":  (255, 128, 0),    # naranja
    "bottom": (0, 128, 255),    # naranja claro
    "left":   (255, 0, 255),    # magenta
}
POLYGON_COLOR = (0, 230, 118)


def _dist(p1: Dict, p2: Dict) -> float:
    """Distancia euclidiana entre dos puntos {x, y}."""
    return math.sqrt((p2["x"] - p1["x"]) ** 2 + (p2["y"] - p1["y"]) ** 2)


class CalculatePolygonProportion(BaseFilter):
    """
    Calcula la proporción de un polígono de página.

    A partir de las 4 esquinas en orden horario (TL, TR, BR, BL) calcula:
        proporcion = avg(lado_izquierdo, lado_derecho) / lado_superior

    Esta proporción es invariante al zoom y característica del tamaño de papel
    del corpus. Utilizada por el pipeline de calibración para determinar la
    proporción verdadera del corpus mediante análisis de pico (FindPeakProportion).
    """

    FILTER_NAME = "CalculatePolygonProportion"
    DESCRIPTION = (
        "Calcula la proporción avg(lado_izq, lado_der) / lado_superior de un polígono "
        "de 4 esquinas. Diseñado para el pipeline de calibración de proporción del corpus."
    )

    INPUTS = {
        "gt_data": "metadata",   # dict con clave 'polygon' (de PolygonToGTFormat)
    }

    OUTPUTS = {
        "proportion_data": "metadata",
        "sample_image":    "image",
    }

    PARAMS = {
        "visualization_size": {
            "default": 900,
            "min": 400,
            "max": 1920,
            "step": 100,
            "description": "Ancho de la imagen de muestra.",
        },
        "show_side_lengths": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar longitud de cada lado en la visualización (0=No, 1=Sí).",
        },
    }

    # ── Proceso principal ──────────────────────────────────────────────────────

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        gt_data = inputs.get("gt_data") or {}
        polygon  = gt_data.get("polygon", [])

        proportion_data = self._calculate(polygon, gt_data)

        result = {"proportion_data": proportion_data}

        if not self.without_preview:
            result["sample_image"] = self._make_sample(original_image, polygon, proportion_data)

        return result

    # ── Cálculo ────────────────────────────────────────────────────────────────

    def _calculate(self, polygon, gt_data: Dict) -> Dict:
        """Calcula la proporción y las longitudes de cada lado."""
        valid = (
            polygon
            and len(polygon) == 4
            and all(p is not None and "x" in p and "y" in p for p in polygon)
        )

        if not valid:
            return {
                "proportion":    None,
                "top_side":      None,
                "right_side":    None,
                "bottom_side":   None,
                "left_side":     None,
                "valid":         False,
                "image_width":   gt_data.get("image_width"),
                "image_height":  gt_data.get("image_height"),
            }

        tl, tr, br, bl = polygon

        top_side    = _dist(tl, tr)
        right_side  = _dist(tr, br)
        bottom_side = _dist(bl, br)
        left_side   = _dist(tl, bl)

        if top_side == 0:
            proportion = None
            valid_prop = False
        else:
            proportion  = (left_side + right_side) / (2.0 * top_side)
            valid_prop  = True

        return {
            "proportion":   round(proportion, 6) if valid_prop else None,
            "top_side":     round(top_side, 2),
            "right_side":   round(right_side, 2),
            "bottom_side":  round(bottom_side, 2),
            "left_side":    round(left_side, 2),
            "valid":        valid_prop,
            "image_width":  gt_data.get("image_width"),
            "image_height": gt_data.get("image_height"),
        }

    # ── Visualización ──────────────────────────────────────────────────────────

    def _make_sample(self, original_image: np.ndarray,
                     polygon, proportion_data: Dict) -> np.ndarray:
        vis_w         = self.params["visualization_size"]
        show_lengths  = bool(self.params["show_side_lengths"])

        orig_h, orig_w = original_image.shape[:2]
        vis_h = int(vis_w * orig_h / orig_w)
        sx = vis_w / orig_w
        sy = vis_h / orig_h

        # Fondo
        if len(original_image.shape) == 2:
            vis = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis = original_image.copy()
        vis = cv2.resize(vis, (vis_w, vis_h))

        valid = proportion_data.get("valid", False)

        if valid and len(polygon) == 4 and all(p is not None for p in polygon):
            pts = [(int(p["x"] * sx), int(p["y"] * sy)) for p in polygon]
            tl_v, tr_v, br_v, bl_v = pts

            # Dibujar polígono
            arr = np.array(pts, dtype=np.int32)
            cv2.polylines(vis, [arr], isClosed=True, color=POLYGON_COLOR, thickness=3)

            if show_lengths:
                sides = [
                    ("top",    tl_v, tr_v, proportion_data["top_side"]),
                    ("right",  tr_v, br_v, proportion_data["right_side"]),
                    ("bottom", bl_v, br_v, proportion_data["bottom_side"]),
                    ("left",   tl_v, bl_v, proportion_data["left_side"]),
                ]
                for name, p1, p2, length in sides:
                    color = SIDE_COLORS[name]
                    mx = (p1[0] + p2[0]) // 2
                    my = (p1[1] + p2[1]) // 2
                    label = f"{name}: {length:.0f}px"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis, (mx - 2, my - th - 4), (mx + tw + 2, my + 2), (0, 0, 0), -1)
                    cv2.putText(vis, label, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Dibujar esquinas
            for pt in pts:
                cv2.circle(vis, pt, 7, (0, 0, 0), -1)
                cv2.circle(vis, pt, 5, POLYGON_COLOR, -1)

        # Resultado en esquina superior
        proportion = proportion_data.get("proportion")
        if proportion is not None:
            status      = f"Proporcion: {proportion:.4f}"
            status_color = (0, 230, 118)
        else:
            status      = "INVALIDO - poligono incompleto"
            status_color = (0, 0, 255)

        cv2.putText(vis, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
        cv2.putText(vis, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

        return vis
