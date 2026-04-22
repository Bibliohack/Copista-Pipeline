"""
Filtro: FilterLinesByPPBZone
Filtra líneas Hough clasificadas usando las posiciones del PPB como zona de confianza.
"""

import cv2
import numpy as np
from typing import Dict, Any, List
from .base_filter import BaseFilter, FILTER_REGISTRY


class FilterLinesByPPBZone(BaseFilter):
    """Filtra líneas Hough descartando las que estén lejos de la zona predicha por PPB."""

    FILTER_NAME = "FilterLinesByPPBZone"
    DESCRIPTION = (
        "Recibe las líneas Hough clasificadas (horizontales y verticales) y la metadata del PPB. "
        "Filtra las líneas descartando las que estén fuera de la tolerancia respecto a las posiciones "
        "top/bottom/left/right predichas por el PPB. Elimina líneas fantasma espurias antes de "
        "pasarlas a SelectBorderLines."
    )

    INPUTS = {
        "horizontal_lines": "lines",
        "vertical_lines": "lines",
        "ppb_metadata": "metadata",   # selection_metadata del PPB
        "base_image": "image"
    }

    OUTPUTS = {
        "horizontal_lines": "lines",   # líneas H filtradas
        "vertical_lines": "lines",     # líneas V filtradas
        "lines_metadata": "metadata",
        "sample_image": "image"
    }

    PARAMS = {
        "tolerance_h": {
            "default": 15,
            "min": 0,
            "max": 50,
            "step": 1,
            "description": "Tolerancia para bordes horizontales (top/bottom) como % del alto de imagen."
        },
        "tolerance_v": {
            "default": 15,
            "min": 0,
            "max": 50,
            "step": 1,
            "description": "Tolerancia para bordes verticales (left/right) como % del ancho de imagen."
        }
    }

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        horizontal_lines_in = inputs.get("horizontal_lines", [])
        vertical_lines_in = inputs.get("vertical_lines", [])
        ppb_metadata = inputs.get("ppb_metadata", {})
        base_img = inputs.get("base_image", original_image)

        if base_img is None:
            base_img = original_image

        h, w = base_img.shape[:2]

        # Obtener posiciones PPB
        ppb_top    = int(ppb_metadata.get("top",    0))
        ppb_bottom = int(ppb_metadata.get("bottom", h - 1))
        ppb_left   = int(ppb_metadata.get("left",   0))
        ppb_right  = int(ppb_metadata.get("right",  w - 1))

        # Calcular tolerancias en píxeles
        tol_h_pct = float(self.params["tolerance_h"])
        tol_v_pct = float(self.params["tolerance_v"])
        tol_h_px = tol_h_pct / 100.0 * h
        tol_v_px = tol_v_pct / 100.0 * w

        # --- Filtrar líneas horizontales ---
        # Posición Y aproximada = promedio de y1 e y2
        h_lines_kept = []
        h_lines_discarded = []
        for line in horizontal_lines_in:
            y_line = (line["y1"] + line["y2"]) / 2.0
            near_top    = abs(y_line - ppb_top)    <= tol_h_px
            near_bottom = abs(y_line - ppb_bottom) <= tol_h_px
            if near_top or near_bottom:
                h_lines_kept.append(line)
            else:
                h_lines_discarded.append(line)

        # Fallback: si no quedó ninguna, devolver todas las originales
        h_fallback_used = False
        if len(horizontal_lines_in) > 0 and len(h_lines_kept) == 0:
            h_lines_kept = list(horizontal_lines_in)
            h_lines_discarded = []
            h_fallback_used = True

        # --- Filtrar líneas verticales ---
        # Posición X aproximada = promedio de x1 e x2
        v_lines_kept = []
        v_lines_discarded = []
        for line in vertical_lines_in:
            x_line = (line["x1"] + line["x2"]) / 2.0
            near_left  = abs(x_line - ppb_left)  <= tol_v_px
            near_right = abs(x_line - ppb_right) <= tol_v_px
            if near_left or near_right:
                v_lines_kept.append(line)
            else:
                v_lines_discarded.append(line)

        # Fallback: si no quedó ninguna, devolver todas las originales
        v_fallback_used = False
        if len(vertical_lines_in) > 0 and len(v_lines_kept) == 0:
            v_lines_kept = list(vertical_lines_in)
            v_lines_discarded = []
            v_fallback_used = True

        # --- Construir lines_metadata ---
        lines_metadata = {
            "image_width":  int(w),
            "image_height": int(h),
            "h_lines_in":   len(horizontal_lines_in),
            "h_lines_out":  len(h_lines_kept),
            "v_lines_in":   len(vertical_lines_in),
            "v_lines_out":  len(v_lines_kept),
            "ppb_top":      ppb_top,
            "ppb_bottom":   ppb_bottom,
            "ppb_left":     ppb_left,
            "ppb_right":    ppb_right,
            "h_fallback_used": h_fallback_used,
            "v_fallback_used": v_fallback_used,
        }

        result = {
            "horizontal_lines": h_lines_kept,
            "vertical_lines":   v_lines_kept,
            "lines_metadata":   lines_metadata,
        }

        # --- Visualización ---
        if not self.without_preview:
            if len(base_img.shape) == 2:
                vis_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
            else:
                vis_img = base_img.copy()

            COLOR_DISCARDED = (0, 0, 255)    # rojo  — líneas descartadas
            COLOR_KEPT      = (0, 255, 0)    # verde — líneas mantenidas
            COLOR_PPB       = (0, 255, 255)  # amarillo — referencia PPB

            # Dibujar líneas horizontales descartadas (rojo)
            for line in h_lines_discarded:
                cv2.line(vis_img, (line["x1"], line["y1"]), (line["x2"], line["y2"]),
                         COLOR_DISCARDED, 1)

            # Dibujar líneas verticales descartadas (rojo)
            for line in v_lines_discarded:
                cv2.line(vis_img, (line["x1"], line["y1"]), (line["x2"], line["y2"]),
                         COLOR_DISCARDED, 1)

            # Dibujar líneas horizontales mantenidas (verde)
            for line in h_lines_kept:
                cv2.line(vis_img, (line["x1"], line["y1"]), (line["x2"], line["y2"]),
                         COLOR_KEPT, 2)

            # Dibujar líneas verticales mantenidas (verde)
            for line in v_lines_kept:
                cv2.line(vis_img, (line["x1"], line["y1"]), (line["x2"], line["y2"]),
                         COLOR_KEPT, 2)

            # Dibujar líneas de referencia PPB (amarillo, grosor 2)
            cv2.line(vis_img, (0, ppb_top),    (w - 1, ppb_top),    COLOR_PPB, 2)
            cv2.line(vis_img, (0, ppb_bottom), (w - 1, ppb_bottom), COLOR_PPB, 2)
            cv2.line(vis_img, (ppb_left,  0),  (ppb_left,  h - 1),  COLOR_PPB, 2)
            cv2.line(vis_img, (ppb_right, 0),  (ppb_right, h - 1),  COLOR_PPB, 2)

            result["sample_image"] = vis_img

        return result
