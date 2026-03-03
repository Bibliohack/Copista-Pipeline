"""
Filtro: RefinePolygonByArea

Busca exhaustivamente la combinación de 4 líneas de Hough (del conjunto completo
horizontal/vertical) cuyo cuadrilátero resultante tiene la proporción más cercana
al objetivo. Si se provee el área objetivo (mismo zoom en todo el corpus), también
penaliza desviaciones de área, lo que hace la búsqueda mucho más precisa.

Reemplaza o complementa a SelectBorderLines: produce el mismo formato de salida
(selected_lines: border_lines) y es compatible con CalculateQuadCorners.

Estrategia de score:
  Con target_area > 0:  score = w_prop * (1-Δp/p)   + w_area * (1-ΔA/A)
  Sin target_area (=0): score = w_prop * (1-Δp/p)   + w_area * (area/img_area)
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional

from .base_filter import BaseFilter, FILTER_REGISTRY
from ._polygon_geometry import (
    add_positions, limit_lines, exhaustive_search,
    candidate_to_border_lines, extract_target_proportion,
    make_refinement_sample,
)


class RefinePolygonByArea(BaseFilter):
    """
    Refinamiento de polígono por proporción (+ área opcional).

    Recibe TODAS las líneas de ClassifyLinesByAngle (no solo las 4 de SelectBorderLines)
    y busca la combinación óptima por búsqueda exhaustiva.
    """

    FILTER_NAME = "RefinePolygonByArea"
    DESCRIPTION = (
        "Busca entre todas las líneas Hough clasificadas la combinación de 4 líneas "
        "(top/bottom/left/right) cuyo polígono tiene la proporción más cercana al objetivo. "
        "Con target_area (mismo zoom del corpus) añade restricción de área para mayor precisión. "
        "Salida compatible con CalculateQuadCorners."
    )

    INPUTS = {
        "horizontal_lines": "lines",     # de ClassifyLinesByAngle
        "vertical_lines":   "lines",     # de ClassifyLinesByAngle
        "lines_metadata":   "metadata",  # dimensiones de imagen
        "proportion_data":  "metadata",  # opcional: de FindPeakProportion / CalculatePolygonProportion
    }

    OUTPUTS = {
        "selected_lines":     "border_lines",  # compatible con CalculateQuadCorners
        "selection_metadata": "metadata",
        "sample_image":       "image",
    }

    PARAMS = {
        "target_proportion": {
            "default": 0.0,
            "min":     0.0,
            "max":     5.0,
            "step":    0.01,
            "description": (
                "Proporción objetivo avg(izq,der)/sup. "
                "Si es 0, se lee de la entrada proportion_data."
            ),
        },
        "target_area": {
            "default": 0.0,
            "min":     0.0,
            "max":     100000000.0,
            "step":    10000.0,
            "description": (
                "Área objetivo del polígono en píxeles² (resolución de trabajo). "
                "0 = no restringir área, maximizar tamaño."
            ),
        },
        "max_lines": {
            "default": 20,
            "min":     4,
            "max":     60,
            "step":    2,
            "description": (
                "Máximo de líneas por orientación a considerar. "
                "Limita el espacio de búsqueda para evitar cuelgues con pools grandes. "
                "Las líneas se muestrean uniformemente preservando los extremos."
            ),
        },
        "proportion_weight": {
            "default": 0.6,
            "min":     0.0,
            "max":     1.0,
            "step":    0.05,
            "description": "Peso del término de proporción en el score (0-1).",
        },
        "area_weight": {
            "default": 0.4,
            "min":     0.0,
            "max":     1.0,
            "step":    0.05,
            "description": "Peso del término de área en el score (0-1).",
        },
        "top_k": {
            "default": 3,
            "min":     1,
            "max":     10,
            "step":    1,
            "description": "Número de candidatos a conservar (se usa el primero como resultado).",
        },
        "visualization_size": {
            "default": 900,
            "min":     400,
            "max":     1920,
            "step":    100,
            "description": "Ancho de la imagen de muestra.",
        },
    }

    # ── Proceso principal ──────────────────────────────────────────────────────

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        h_lines        = inputs.get("horizontal_lines", []) or []
        v_lines        = inputs.get("vertical_lines",   []) or []
        lines_meta     = inputs.get("lines_metadata",   {}) or {}
        proportion_data = inputs.get("proportion_data", None)

        img_w = int(lines_meta.get("image_width",  original_image.shape[1]))
        img_h = int(lines_meta.get("image_height", original_image.shape[0]))

        target_proportion = extract_target_proportion(
            self.params["target_proportion"], proportion_data
        )
        target_area  = float(self.params["target_area"])
        max_lines    = int(self.params["max_lines"])
        prop_weight  = float(self.params["proportion_weight"])
        area_weight  = float(self.params["area_weight"])
        top_k        = int(self.params["top_k"])

        # Añadir posiciones y limitar pool
        h_with_pos = add_positions(h_lines, img_w, img_h, is_horizontal=True)
        v_with_pos = add_positions(v_lines, img_w, img_h, is_horizontal=False)
        h_limited  = limit_lines(h_with_pos, max_lines)
        v_limited  = limit_lines(v_with_pos, max_lines)

        # Metadatos base
        meta = {
            "image_width":           img_w,
            "image_height":          img_h,
            "h_lines_total":         len(h_lines),
            "v_lines_total":         len(v_lines),
            "h_lines_used":          len(h_limited),
            "v_lines_used":          len(v_limited),
            "combinations_max":      len(h_limited) ** 2 * len(v_limited) ** 2,
            "target_proportion":     target_proportion,
            "target_area":           target_area,
            "top_is_image_border":   False,
            "bottom_is_image_border": False,
            "left_is_image_border":  False,
            "right_is_image_border": False,
        }

        # Sin proporción objetivo → no se puede buscar
        if target_proportion is None:
            return self._empty_result(original_image, meta, h_limited, v_limited,
                                      "Sin target_proportion: configura el parámetro "
                                      "o conecta proportion_data")

        if len(h_limited) < 2 or len(v_limited) < 2:
            return self._empty_result(original_image, meta, h_limited, v_limited,
                                      "Pool insuficiente: se necesitan ≥2 líneas horizontales y ≥2 verticales")

        # Búsqueda exhaustiva
        candidates = exhaustive_search(
            h_limited, v_limited,
            target_proportion, target_area,
            img_w, img_h,
            prop_weight, area_weight,
            top_k,
        )

        if not candidates:
            return self._empty_result(original_image, meta, h_limited, v_limited,
                                      "Búsqueda sin candidatos válidos")

        best = candidates[0]
        selected_lines = candidate_to_border_lines(best)

        meta.update({
            "best_score":       best["score"],
            "best_proportion":  best["proportion"],
            "best_area":        best["area"],
            "candidates_found": len(candidates),
            "valid":            True,
        })

        result = {
            "selected_lines":     selected_lines,
            "selection_metadata": meta,
        }

        if not self.without_preview:
            result["sample_image"] = make_refinement_sample(
                original_image, h_limited, v_limited, candidates,
                self.params["visualization_size"],
            )

        return result

    # ── Resultado vacío ────────────────────────────────────────────────────────

    def _empty_result(self, original_image, meta, h_lines, v_lines, reason):
        meta["valid"]  = False
        meta["reason"] = reason

        empty_lines = {"top": None, "bottom": None, "left": None, "right": None}
        result = {
            "selected_lines":     empty_lines,
            "selection_metadata": meta,
        }
        if not self.without_preview:
            vis_w = self.params["visualization_size"]
            sample = make_refinement_sample(original_image, h_lines, v_lines, [], vis_w)
            orig_h, orig_w = original_image.shape[:2]
            vis_h = int(vis_w * orig_h / orig_w)
            import cv2
            sample = cv2.resize(sample, (vis_w, vis_h)) if sample.shape[1] != vis_w else sample
            cv2.putText(sample, reason, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            result["sample_image"] = sample
        return result
