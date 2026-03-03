"""
Filtro: RefinePolygonByCanny

Variante del refinamiento de polígono que pre-filtra las líneas candidatas
según su soporte de bordes en la imagen Canny antes de la búsqueda exhaustiva.

Flujo:
  1. Calcular soporte Canny para cada línea (fracción de puntos con borde cercano).
  2. Descartar líneas por debajo de min_canny_support.
  3. Sobre las supervivientes, búsqueda exhaustiva por proporción + área (opcional).

Con target_area > 0 (corpus de zoom fijo):
  score = w_prop * (1-Δp/p) + w_area * (1-ΔA/A)

Sin target_area:
  score = w_prop * (1-Δp/p) + w_area * (area/img_area)

Salida compatible con CalculateQuadCorners (formato border_lines).
"""

import math
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from .base_filter import BaseFilter, FILTER_REGISTRY
from ._polygon_geometry import (
    add_positions, limit_lines, exhaustive_search,
    candidate_to_border_lines, extract_target_proportion,
    make_refinement_sample,
)


# ── Soporte Canny ──────────────────────────────────────────────────────────────

def _canny_support(line: Dict, canny: np.ndarray,
                   band_px: int, sample_step: int) -> float:
    """
    Fracción de puntos muestreados a lo largo de la línea que tienen al menos
    un píxel Canny activo dentro de una banda de `band_px` píxeles.

    Retorna 0.0 si la línea está completamente fuera de la imagen.
    """
    x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
    h, w = canny.shape[:2]

    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    n_steps = max(1, int(length / max(sample_step, 1)))

    hits  = 0
    valid = 0

    for i in range(n_steps + 1):
        t  = i / n_steps
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        xi = int(round(px))
        yi = int(round(py))

        # Ventana de búsqueda
        x0 = max(0, xi - band_px);  x1_ = min(w - 1, xi + band_px)
        y0 = max(0, yi - band_px);  y1_ = min(h - 1, yi + band_px)

        if x0 > x1_ or y0 > y1_:
            continue  # fuera de imagen

        valid += 1
        if canny[y0:y1_ + 1, x0:x1_ + 1].any():
            hits += 1

    return hits / valid if valid > 0 else 0.0


def _score_lines_canny(lines: List[Dict], canny: np.ndarray,
                       band_px: int, sample_step: int) -> List[Dict]:
    """Añade clave 'canny_support' (0-1) a cada línea."""
    result = []
    for line in lines:
        support = _canny_support(line, canny, band_px, sample_step)
        result.append({**line, "canny_support": round(support, 4)})
    return result


def _filter_by_support(lines: List[Dict], min_support: float) -> List[Dict]:
    return [l for l in lines if l.get("canny_support", 0.0) >= min_support]


# ── Filtro ─────────────────────────────────────────────────────────────────────

class RefinePolygonByCanny(BaseFilter):
    """
    Refinamiento de polígono usando soporte de bordes Canny + proporción (+ área opcional).

    Recibe TODAS las líneas de ClassifyLinesByAngle. Descarta las que tienen
    poco soporte en la imagen Canny, luego hace búsqueda exhaustiva entre las
    supervivientes buscando la proporción objetivo.
    """

    FILTER_NAME = "RefinePolygonByCanny"
    DESCRIPTION = (
        "Pre-filtra líneas Hough por soporte de bordes Canny y busca exhaustivamente "
        "la combinación de 4 líneas con proporción más cercana al objetivo. "
        "Con target_area (corpus de zoom fijo) añade restricción de área. "
        "Salida compatible con CalculateQuadCorners."
    )

    INPUTS = {
        "horizontal_lines": "lines",     # de ClassifyLinesByAngle
        "vertical_lines":   "lines",     # de ClassifyLinesByAngle
        "lines_metadata":   "metadata",  # dimensiones de imagen
        "canny_image":      "image",     # imagen de bordes para soporte
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
                "Área objetivo en píxeles² (resolución de trabajo). "
                "0 = no restringir área, maximizar tamaño."
            ),
        },
        "min_canny_support": {
            "default": 0.25,
            "min":     0.0,
            "max":     1.0,
            "step":    0.05,
            "description": (
                "Soporte Canny mínimo para conservar una línea (0-1). "
                "Fracción de puntos de la línea con borde detectado en la banda. "
                "0 = no filtrar, 1 = solo líneas perfectamente sobre un borde."
            ),
        },
        "canny_band_px": {
            "default": 5,
            "min":     1,
            "max":     20,
            "step":    1,
            "description": "Radio de la banda (en píxeles) alrededor de cada punto para buscar borde.",
        },
        "canny_sample_step": {
            "default": 8,
            "min":     1,
            "max":     30,
            "step":    1,
            "description": "Distancia en píxeles entre puntos de muestreo a lo largo de la línea.",
        },
        "max_lines": {
            "default": 20,
            "min":     4,
            "max":     60,
            "step":    2,
            "description": (
                "Máximo de líneas por orientación tras el filtrado Canny. "
                "Si el filtrado Canny ya reduce suficiente el pool, este límite no actúa."
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
        h_lines         = inputs.get("horizontal_lines", []) or []
        v_lines         = inputs.get("vertical_lines",   []) or []
        lines_meta      = inputs.get("lines_metadata",   {}) or {}
        canny_image     = inputs.get("canny_image")
        proportion_data = inputs.get("proportion_data", None)

        img_w = int(lines_meta.get("image_width",  original_image.shape[1]))
        img_h = int(lines_meta.get("image_height", original_image.shape[0]))

        target_proportion = extract_target_proportion(
            self.params["target_proportion"], proportion_data
        )
        target_area    = float(self.params["target_area"])
        min_support    = float(self.params["min_canny_support"])
        band_px        = int(self.params["canny_band_px"])
        sample_step    = int(self.params["canny_sample_step"])
        max_lines      = int(self.params["max_lines"])
        prop_weight    = float(self.params["proportion_weight"])
        area_weight    = float(self.params["area_weight"])
        top_k          = int(self.params["top_k"])

        # Preparar Canny (convertir a escala de grises binaria si es BGR)
        if canny_image is not None:
            if len(canny_image.shape) == 3:
                canny_gray = cv2.cvtColor(canny_image, cv2.COLOR_BGR2GRAY)
            else:
                canny_gray = canny_image
        else:
            canny_gray = None

        # ── Calcular soporte Canny ─────────────────────────────────────────────
        if canny_gray is not None:
            h_scored = _score_lines_canny(h_lines, canny_gray, band_px, sample_step)
            v_scored = _score_lines_canny(v_lines, canny_gray, band_px, sample_step)
            h_filtered = _filter_by_support(h_scored, min_support)
            v_filtered = _filter_by_support(v_scored, min_support)
        else:
            # Sin Canny: tratar todas las líneas con soporte 1.0
            h_scored   = [{**l, "canny_support": 1.0} for l in h_lines]
            v_scored   = [{**l, "canny_support": 1.0} for l in v_lines]
            h_filtered = h_scored
            v_filtered = v_scored

        # Añadir posiciones y limitar pool
        h_with_pos = add_positions(h_filtered, img_w, img_h, is_horizontal=True)
        v_with_pos = add_positions(v_filtered, img_w, img_h, is_horizontal=False)
        h_limited  = limit_lines(h_with_pos, max_lines)
        v_limited  = limit_lines(v_with_pos, max_lines)

        # Metadatos base
        meta = {
            "image_width":             img_w,
            "image_height":            img_h,
            "h_lines_total":           len(h_lines),
            "v_lines_total":           len(v_lines),
            "h_lines_after_canny":     len(h_filtered),
            "v_lines_after_canny":     len(v_filtered),
            "h_lines_used":            len(h_limited),
            "v_lines_used":            len(v_limited),
            "combinations_max":        len(h_limited) ** 2 * len(v_limited) ** 2,
            "target_proportion":       target_proportion,
            "target_area":             target_area,
            "min_canny_support":       min_support,
            "top_is_image_border":     False,
            "bottom_is_image_border":  False,
            "left_is_image_border":    False,
            "right_is_image_border":   False,
        }

        # Guardar soporte medio de cada orientación para diagnóstico
        if h_scored:
            meta["h_mean_support"] = round(
                sum(l["canny_support"] for l in h_scored) / len(h_scored), 3)
        if v_scored:
            meta["v_mean_support"] = round(
                sum(l["canny_support"] for l in v_scored) / len(v_scored), 3)

        # Sin proporción objetivo → no se puede buscar
        if target_proportion is None:
            return self._empty_result(original_image, meta, h_limited, v_limited,
                                      "Sin target_proportion: configura el parámetro "
                                      "o conecta proportion_data")

        if len(h_limited) < 2 or len(v_limited) < 2:
            needed_h = 2 - len(h_limited)
            needed_v = 2 - len(v_limited)
            reason = (f"Pool insuficiente tras filtro Canny "
                      f"(h={len(h_limited)}, v={len(v_limited)}). "
                      f"Reducir min_canny_support.")
            return self._empty_result(original_image, meta, h_limited, v_limited, reason)

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
            # Para visualización usamos las líneas filtradas (con soporte)
            result["sample_image"] = self._make_sample(
                original_image, h_scored, v_scored,
                h_limited, v_limited, candidates,
                min_support,
            )

        return result

    # ── Visualización ──────────────────────────────────────────────────────────

    def _make_sample(self, original_image, h_scored, v_scored,
                     h_limited, v_limited, candidates, min_support):
        """
        Visualización con código de colores según soporte Canny:
        - Rojo tenue: líneas descartadas (soporte < umbral)
        - Azul/verde tenue: líneas supervivientes no seleccionadas
        - Amarillo/naranja prominente: las 4 líneas del mejor candidato
        """
        from ._polygon_geometry import polygon_from_lines

        vis_w = self.params["visualization_size"]
        orig_h, orig_w = original_image.shape[:2]
        vis_h = int(vis_w * orig_h / orig_w)
        sx = vis_w / orig_w
        sy = vis_h / orig_h

        if len(original_image.shape) == 2:
            vis = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis = original_image.copy()
        vis = cv2.resize(vis, (vis_w, vis_h))

        def draw_line(line, color, thickness):
            p1 = (int(line["x1"] * sx), int(line["y1"] * sy))
            p2 = (int(line["x2"] * sx), int(line["y2"] * sy))
            cv2.line(vis, p1, p2, color, thickness)

        # Todas las líneas originales (con color por soporte)
        for line in h_scored:
            s = line.get("canny_support", 0.0)
            if s < min_support:
                draw_line(line, (0, 0, 80), 1)       # rojo muy tenue
            else:
                draw_line(line, (60, 60, 120), 1)    # azul tenue

        for line in v_scored:
            s = line.get("canny_support", 0.0)
            if s < min_support:
                draw_line(line, (0, 80, 0), 1)       # verde muy tenue
            else:
                draw_line(line, (60, 120, 60), 1)    # verde tenue

        if not candidates:
            cv2.putText(vis, "Sin candidatos", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return vis

        # Candidatos alternativos
        for alt in candidates[1:]:
            poly = polygon_from_lines(alt["top"], alt["bottom"],
                                      alt["left"], alt["right"])
            if poly:
                pts = np.array(
                    [(int(x * sx), int(y * sy)) for x, y in poly], dtype=np.int32)
                cv2.polylines(vis, [pts], isClosed=True, color=(60, 60, 60), thickness=1)

        # Mejor candidato
        best = candidates[0]
        draw_line(best["top"],    (0, 255, 255), 2)
        draw_line(best["bottom"], (0, 200, 200), 2)
        draw_line(best["left"],   (255, 128, 0), 2)
        draw_line(best["right"],  (200, 100, 0), 2)

        poly = polygon_from_lines(best["top"], best["bottom"],
                                   best["left"], best["right"])
        if poly:
            pts = np.array(
                [(int(x * sx), int(y * sy)) for x, y in poly], dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 230, 118), thickness=3)
            for pt in pts:
                cv2.circle(vis, tuple(pt), 6, (0, 230, 118), -1)

        # Encabezado
        h_surv = sum(1 for l in h_scored if l.get("canny_support", 0) >= min_support)
        v_surv = sum(1 for l in v_scored if l.get("canny_support", 0) >= min_support)
        label = (f"score={best['score']:.3f}  prop={best['proportion']:.4f}  "
                 f"h:{h_surv}/{len(h_scored)}  v:{v_surv}/{len(v_scored)}")
        cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 118), 1)

        return vis

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
            vis_w  = self.params["visualization_size"]
            orig_h, orig_w = original_image.shape[:2]
            vis_h  = int(vis_w * orig_h / orig_w)
            if len(original_image.shape) == 2:
                vis = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            else:
                vis = original_image.copy()
            vis = cv2.resize(vis, (vis_w, vis_h))
            cv2.putText(vis, reason, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            result["sample_image"] = vis
        return result
