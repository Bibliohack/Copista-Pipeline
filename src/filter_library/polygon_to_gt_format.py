"""
Filtro: PolygonToGTFormat

Convierte las esquinas detectadas por el pipeline (escaladas a la resolución
de trabajo) a coordenadas de la imagen original y las empaqueta en el mismo
formato JSON que genera ground_truth_annotator.py.

El dict resultante puede guardarse directamente como .gt.json via batch_processor.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
from .base_filter import BaseFilter, FILTER_REGISTRY


# Orden canónico: sentido horario desde sup-izq (igual que ground_truth_annotator)
CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]

# Colores para visualización
CORNER_COLORS = {
    "top_left":     (0,   255, 255),  # amarillo
    "top_right":    (255, 255, 0),    # cyan
    "bottom_right": (0,   165, 255),  # naranja
    "bottom_left":  (255, 0,   255),  # magenta
}
POLYGON_COLOR = (0, 230, 118)
FALLBACK_COLOR = (0, 0, 255)          # rojo = esquina de imagen (fallback)


class PolygonToGTFormat(BaseFilter):
    """
    Convierte quad_corners escalados a formato ground truth para comparación con IoU.

    Recibe las esquinas en el espacio de resolución de trabajo (main_resize) y las
    re-escala a las dimensiones originales de la imagen, devolviendo un diccionario
    con la misma estructura que los archivos .gt.json generados manualmente.
    """

    FILTER_NAME = "PolygonToGTFormat"
    DESCRIPTION = (
        "Convierte las esquinas detectadas (en resolución de trabajo) a coordenadas "
        "de imagen original y las empaqueta en formato ground truth compatible con "
        "ground_truth_annotator.py."
    )

    INPUTS = {
        "scaled_corners":  "quad_points",
        "scaled_metadata": "metadata",
    }

    OUTPUTS = {
        "gt_data":     "metadata",   # dict listo para guardar como .gt.json
        "sample_image": "image",
    }

    PARAMS = {
        "visualization_size": {
            "default": 900,
            "min": 400,
            "max": 1920,
            "step": 100,
            "description": "Ancho de la imagen de muestra.",
        },
        "show_types": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar el tipo de cada esquina en la visualización (0=No, 1=Sí).",
        },
    }

    # ── Proceso principal ─────────────────────────────────────────────────────

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        scaled_corners  = inputs.get("scaled_corners",  {}) or {}
        scaled_metadata = inputs.get("scaled_metadata", {}) or {}

        orig_h, orig_w = original_image.shape[:2]

        # Dimensiones del espacio de trabajo (resolución de main_resize)
        target_w = scaled_metadata.get("target_width")
        target_h = scaled_metadata.get("target_height")

        if target_w and target_h:
            sx = orig_w / target_w
            sy = orig_h / target_h
        else:
            # Sin metadata, asumir que las coordenadas ya son originales
            sx, sy = 1.0, 1.0

        # Construir polígono en coords originales
        polygon = []
        all_found = True

        for name in CORNER_ORDER:
            corner = scaled_corners.get(name)
            if corner and "x" in corner and "y" in corner:
                polygon.append({
                    "x": round(corner["x"] * sx),
                    "y": round(corner["y"] * sy),
                    "type": corner.get("type", "unknown"),
                })
            else:
                polygon.append(None)
                all_found = False

        gt_data = {
            "image_file":   "",           # el batch_processor lo nombra por la imagen
            "image_width":  orig_w,
            "image_height": orig_h,
            "polygon":      polygon,
            "all_corners_found": all_found,
        }

        sample = self._make_sample(original_image, polygon)

        return {
            "gt_data":      gt_data,
            "sample_image": sample,
        }

    # ── Visualización ─────────────────────────────────────────────────────────

    def _make_sample(self, original_image: np.ndarray,
                     polygon) -> np.ndarray:
        vis_w = self.params["visualization_size"]
        show_types = bool(self.params["show_types"])

        orig_h, orig_w = original_image.shape[:2]
        vis_h = int(vis_w * orig_h / orig_w)
        sx = vis_w / orig_w
        sy = vis_h / orig_h

        # Imagen de fondo
        if len(original_image.shape) == 2:
            vis = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis = original_image.copy()
        vis = cv2.resize(vis, (vis_w, vis_h))

        # Proyectar puntos a espacio de visualización
        vis_pts = []
        for p in polygon:
            if p is not None:
                vis_pts.append((int(p["x"] * sx), int(p["y"] * sy)))
            else:
                vis_pts.append(None)

        # Dibujar polígono si están las 4 esquinas
        valid_pts = [pt for pt in vis_pts if pt is not None]
        if len(valid_pts) == 4:
            arr = np.array(valid_pts, dtype=np.int32)
            cv2.polylines(vis, [arr], isClosed=True, color=POLYGON_COLOR, thickness=3)

        # Dibujar cada esquina
        r = 8
        for i, (name, pt) in enumerate(zip(CORNER_ORDER, vis_pts)):
            if pt is None:
                continue
            p = polygon[i]
            is_fallback = p.get("type", "") in ("image_corner", "mixed_h_border", "mixed_v_border")
            color = FALLBACK_COLOR if is_fallback else CORNER_COLORS[name]

            cv2.circle(vis, pt, r, color, -1)
            cv2.circle(vis, pt, r + 2, (0, 0, 0), 2)

            label = name
            if show_types and p.get("type"):
                label += f" [{p['type']}]"

            tx = pt[0] + r + 4
            ty = pt[1] - r
            # Fondo para texto
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
            cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Info en esquina superior
        all_found = all(p is not None for p in polygon)
        status = "OK - 4 esquinas" if all_found else f"INCOMPLETO - {sum(p is not None for p in polygon)}/4"
        status_color = (0, 230, 118) if all_found else (0, 0, 255)
        cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return vis
