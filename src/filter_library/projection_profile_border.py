"""
Filtro: ProjectionProfileBorder
Detecta los 4 bordes del papel usando perfiles de proyeccion de bordes (Shamqoli 2013).
"""

import cv2
import numpy as np
from typing import Dict, Any
from .base_filter import BaseFilter, FILTER_REGISTRY


class ProjectionProfileBorder(BaseFilter):
    """Detecta los 4 bordes del papel usando perfiles de proyeccion de bordes (Shamqoli 2013)."""

    FILTER_NAME = "ProjectionProfileBorder"
    DESCRIPTION = (
        "Detecta los 4 bordes del papel usando perfiles de proyeccion de bordes (Shamqoli 2013). "
        "Fase 1: encuentra los picos del histograma horizontal/vertical de la imagen de bordes. "
        "Fase 2: refinamiento por cuartiles para corregir contaminacion de texto de pagina adyacente. "
        "Salida compatible con CalculateQuadCorners (formato border_lines)."
    )

    INPUTS = {
        "edge_image": "image",   # imagen de bordes (salida de CannyEdge)
        "base_image": "image",   # imagen base para visualizacion
    }

    OUTPUTS = {
        "selected_lines":     "border_lines",  # compatible con CalculateQuadCorners
        "selection_metadata": "metadata",
        "sample_image":       "image",
    }

    PARAMS = {
        "search_zone_h": {
            "default": 0.45, "min": 0.1, "max": 0.7, "step": 0.05,
            "description": "Fraccion de altura donde buscar los bordes top/bottom (0.45 = 45% superior/inferior).",
        },
        "search_zone_v": {
            "default": 0.45, "min": 0.1, "max": 0.7, "step": 0.05,
            "description": "Fraccion de ancho donde buscar los bordes left/right (0.45 = 45% izq/derecha).",
        },
        "use_phase2": {
            "default": 1, "min": 0, "max": 1, "step": 1,
            "description": "Activar Fase 2 de refinamiento por cuartiles (0=no, 1=si).",
        },
        "onset_threshold": {
            "default": 0.05, "min": 0.01, "max": 0.3, "step": 0.01,
            "description": "Fraccion del maximo del perfil para considerar 'primera senal significativa' (onset). Ej: 0.05 = 5%.",
        },
        "onset_neighborhood": {
            "default": 15, "min": 3, "max": 60, "step": 1,
            "description": "Pixels desde el onset donde buscar el pico local extremo.",
        },
        "use_prewitt": {
            "default": 0, "min": 0, "max": 1, "step": 1,
            "description": "Calcular bordes Prewitt internamente en lugar de usar edge_image (0=usar input, 1=Prewitt interno).",
        },
        "visualization_size": {
            "default": 900, "min": 400, "max": 1920, "step": 100,
            "description": "Ancho de la imagen de muestra.",
        },
    }

    def _find_onset_peak(self, profile_slice: np.ndarray, from_start: bool,
                         threshold_fraction: float, neighborhood: int) -> int:
        """
        Encuentra el borde como el pico local mas cercano al extremo del perfil.

        Logica:
          1. Escanea desde el extremo (from_start=True: desde indice 0;
             from_start=False: desde el ultimo indice hacia atras).
          2. Encuentra el 'onset': primera posicion con valor >= max * threshold_fraction.
          3. Dentro del vecindario [onset, onset+neighborhood) devuelve la posicion
             del maximo local (el pico extremo).
          4. Si no hay senal en el slice, devuelve 0 (el extremo de la zona).

        Devuelve: indice en profile_slice (no en el perfil completo).
        """
        n = len(profile_slice)
        if n == 0:
            return 0

        p = profile_slice if from_start else profile_slice[::-1]

        pmax = float(p.max())
        if pmax == 0:
            return 0

        threshold = pmax * threshold_fraction

        # Onset: primera posicion con senal significativa
        onset = n - 1  # fallback: extremo opuesto de la zona
        for i in range(n):
            if p[i] >= threshold:
                onset = i
                break

        # Pico local en vecindario desde onset
        end = min(n, onset + neighborhood)
        pos_in_p = onset + int(np.argmax(p[onset:end]))

        # Convertir de vuelta al indice en profile_slice
        return pos_in_p if from_start else (n - 1 - pos_in_p)

    def _compute_prewitt(self, gray: np.ndarray) -> np.ndarray:
        """Calcula el mapa de bordes Prewitt sobre una imagen en grises."""
        k_h = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        k_v = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        eh = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, k_h)
        ev = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, k_v)
        edge_map = np.sqrt(eh ** 2 + ev ** 2)
        edge_map = np.clip(edge_map, 0, 255).astype(np.uint8)
        return edge_map

    def _phase1(self, edge_map: np.ndarray, search_zone_h: float, search_zone_v: float,
                onset_threshold: float, onset_neighborhood: int):
        """
        Fase 1: detectar top, bottom, left, right mediante onset + pico extremo.

        Para cada borde: busca el primer valor significativo del perfil viniendo
        desde el extremo (onset) y toma el pico maximo en un vecindario chico
        desde ese onset. Esto evita que picos internos grandes (texto) ganen
        sobre el borde fisico del papel, que siempre es el primer bloque del perfil.
        """
        h, w = edge_map.shape[:2]

        # Histograma horizontal: suma de pixeles activos por fila
        H = edge_map.astype(np.float32).sum(axis=1)  # shape (h,)

        top_zone_end    = max(1, int(h * search_zone_h))
        bottom_zone_start = min(h - 1, h - int(h * search_zone_h))

        top = self._find_onset_peak(
            H[:top_zone_end], from_start=True,
            threshold_fraction=onset_threshold, neighborhood=onset_neighborhood,
        )
        bottom_local = self._find_onset_peak(
            H[bottom_zone_start:], from_start=False,
            threshold_fraction=onset_threshold, neighborhood=onset_neighborhood,
        )
        bottom = bottom_zone_start + bottom_local

        # Histograma vertical limitado entre filas [top, bottom]
        strip = edge_map[top:bottom + 1, :]
        V = strip.astype(np.float32).sum(axis=0)  # shape (w,)

        left_zone_end    = max(1, int(w * search_zone_v))
        right_zone_start = min(w - 1, w - int(w * search_zone_v))

        left = self._find_onset_peak(
            V[:left_zone_end], from_start=True,
            threshold_fraction=onset_threshold, neighborhood=onset_neighborhood,
        )
        right_local = self._find_onset_peak(
            V[right_zone_start:], from_start=False,
            threshold_fraction=onset_threshold, neighborhood=onset_neighborhood,
        )
        right = right_zone_start + right_local

        return top, bottom, left, right

    def _phase2(self, edge_map: np.ndarray, top: int, bottom: int, left: int, right: int):
        """
        Fase 2: refinamiento por cuartiles.
        Devuelve (left, right, phase2_side) donde phase2_side puede ser 'left', 'right' o None.
        """
        if bottom <= top or right <= left:
            return left, right, None

        # Histograma vertical limitado dentro del rectangulo detectado
        strip = edge_map[top:bottom + 1, left:right + 1]
        V_lim = strip.astype(np.float32).sum(axis=0)  # longitud = right - left + 1
        total = float(V_lim.sum())

        if total == 0:
            return left, right, None

        center = (left + right) / 2.0

        # Mediana ponderada (50%)
        cumsum = 0.0
        median_pos = left
        for j, v in enumerate(V_lim):
            cumsum += float(v)
            if cumsum >= total * 0.5:
                median_pos = left + j
                break

        phase2_side = None

        if median_pos > center:
            # Texto contaminando por la derecha -> recalcular left con cuartil inferior (25%)
            cumsum = 0.0
            for j, v in enumerate(V_lim):
                cumsum += float(v)
                if cumsum >= total * 0.25:
                    left = left + j
                    phase2_side = "left"
                    break
        elif median_pos < center:
            # Texto contaminando por la izquierda -> recalcular right con cuartil superior (75%)
            cumsum = 0.0
            for j, v in enumerate(V_lim):
                cumsum += float(v)
                if cumsum >= total * 0.75:
                    right = left + j  # left original se uso al construir V_lim
                    phase2_side = "right"
                    break

        return left, right, phase2_side

    def _fallback_lines(self, h: int, w: int):
        """Devuelve las lineas de borde de imagen como fallback."""
        return {
            "top":    {"x1": 0,     "y1": 0,     "x2": w - 1, "y2": 0,     "angle": 0.0},
            "bottom": {"x1": 0,     "y1": h - 1, "x2": w - 1, "y2": h - 1, "angle": 0.0},
            "left":   {"x1": 0,     "y1": 0,     "x2": 0,     "y2": h - 1, "angle": 90.0},
            "right":  {"x1": w - 1, "y1": 0,     "x2": w - 1, "y2": h - 1, "angle": 90.0},
        }

    def _draw_sample(self, base_img: np.ndarray, edge_map: np.ndarray,
                     top: int, bottom: int, left: int, right: int,
                     vis_width: int) -> np.ndarray:
        """
        Crea la imagen de visualizacion:
        - imagen base redimensionada al ancho vis_width
        - rectangulo verde con los bordes detectados
        - histogramas H (lateral izquierdo) y V (parte inferior) como barras
        """
        h, w = base_img.shape[:2]

        # Convertir a BGR si es grises
        if len(base_img.shape) == 2:
            vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            vis = base_img.copy()

        # Margenes para histogramas
        margin_left_hist = 60   # para histograma H (horizontal)
        margin_bottom_hist = 60  # para histograma V (vertical)

        # Calcular escala para la imagen principal
        available_w = vis_width - margin_left_hist
        available_h = int(available_w * h / w)
        scale = available_w / w

        vis_resized = cv2.resize(vis, (available_w, available_h))

        # Escalar coordenadas
        t = int(top * scale)
        b = int(bottom * scale)
        l = int(left * scale)
        r = int(right * scale)

        # Dibujar rectangulo detectado en verde
        cv2.rectangle(vis_resized, (l, t), (r, b), (0, 220, 0), 2)

        # Dibujar lineas individuales mas tenues
        cv2.line(vis_resized, (0, t), (available_w - 1, t), (0, 255, 100), 1)
        cv2.line(vis_resized, (0, b), (available_w - 1, b), (0, 255, 100), 1)
        cv2.line(vis_resized, (l, 0), (l, available_h - 1), (100, 255, 0), 1)
        cv2.line(vis_resized, (r, 0), (r, available_h - 1), (100, 255, 0), 1)

        # Canvas final: margen izquierdo + imagen + margen inferior
        canvas_w = margin_left_hist + available_w
        canvas_h = available_h + margin_bottom_hist
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:available_h, margin_left_hist:] = vis_resized

        # Histograma H (filas) — barra horizontal en el margen izquierdo
        H = edge_map.astype(np.float32).sum(axis=1)
        if H.max() > 0:
            H_norm = (H / H.max() * (margin_left_hist - 4)).astype(np.int32)
            # Redimensionar al alto disponible
            H_disp = np.interp(
                np.linspace(0, len(H) - 1, available_h),
                np.arange(len(H)), H_norm
            ).astype(np.int32)
            for row_idx in range(available_h):
                bar_len = int(H_disp[row_idx])
                if bar_len > 0:
                    cv2.line(canvas, (margin_left_hist - bar_len - 2, row_idx),
                             (margin_left_hist - 2, row_idx), (180, 180, 60), 1)
            # Marcar top y bottom en el histograma lateral
            t_disp = int(top * scale)
            b_disp = int(bottom * scale)
            cv2.line(canvas, (0, t_disp), (margin_left_hist - 1, t_disp), (0, 220, 0), 1)
            cv2.line(canvas, (0, b_disp), (margin_left_hist - 1, b_disp), (0, 220, 0), 1)

        # Histograma V (columnas) — barra vertical en el margen inferior
        strip = edge_map[top:bottom + 1, :]
        V = strip.astype(np.float32).sum(axis=0)
        if V.max() > 0:
            V_norm = (V / V.max() * (margin_bottom_hist - 4)).astype(np.int32)
            # Redimensionar al ancho disponible
            V_disp = np.interp(
                np.linspace(0, len(V) - 1, available_w),
                np.arange(len(V)), V_norm
            ).astype(np.int32)
            for col_idx in range(available_w):
                bar_len = int(V_disp[col_idx])
                if bar_len > 0:
                    cv2.line(canvas,
                             (margin_left_hist + col_idx, available_h + margin_bottom_hist - bar_len - 2),
                             (margin_left_hist + col_idx, available_h + margin_bottom_hist - 2),
                             (60, 180, 180), 1)
            # Marcar left y right en el histograma inferior
            l_disp = margin_left_hist + int(left * scale)
            r_disp = margin_left_hist + int(right * scale)
            cv2.line(canvas, (l_disp, available_h),
                     (l_disp, available_h + margin_bottom_hist - 1), (0, 220, 0), 1)
            cv2.line(canvas, (r_disp, available_h),
                     (r_disp, available_h + margin_bottom_hist - 1), (0, 220, 0), 1)

        return canvas

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        base_img = inputs.get("base_image", original_image)
        edge_input = inputs.get("edge_image", None)

        # Asegurar imagen base BGR o grises
        if base_img is None:
            base_img = original_image

        h, w = base_img.shape[:2]

        # --- Obtener mapa de bordes ---
        use_prewitt = int(self.params.get("use_prewitt", 0))

        if use_prewitt == 1:
            # Calcular Prewitt internamente
            if len(base_img.shape) == 3:
                gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = base_img.copy()
            edge_map = self._compute_prewitt(gray)
        else:
            # Usar edge_image del input
            if edge_input is not None:
                if len(edge_input.shape) == 3:
                    edge_map = cv2.cvtColor(edge_input, cv2.COLOR_BGR2GRAY)
                else:
                    edge_map = edge_input.copy()
            else:
                # Fallback: Prewitt sobre base_image
                if len(base_img.shape) == 3:
                    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = base_img.copy()
                edge_map = self._compute_prewitt(gray)

        # Comprobar que el mapa no este vacio
        if edge_map.max() == 0:
            # Fallback: bordes de imagen
            selected_lines = self._fallback_lines(h, w)
            selection_metadata = {
                "image_width": int(w), "image_height": int(h),
                "top": 0, "bottom": h - 1, "left": 0, "right": w - 1,
                "phase2_applied": False, "phase2_side": None,
                "top_is_image_border": False, "bottom_is_image_border": False,
                "left_is_image_border": False, "right_is_image_border": False,
                "valid": False,
                "fallback": True,
            }
            result = {
                "selected_lines": selected_lines,
                "selection_metadata": selection_metadata,
            }
            if not self.without_preview:
                result["sample_image"] = base_img.copy() if len(base_img.shape) == 3 else cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
            return result

        # --- Fase 1 ---
        search_zone_h      = float(self.params.get("search_zone_h", 0.45))
        search_zone_v      = float(self.params.get("search_zone_v", 0.45))
        onset_threshold    = float(self.params.get("onset_threshold", 0.05))
        onset_neighborhood = int(self.params.get("onset_neighborhood", 15))

        top, bottom, left, right = self._phase1(
            edge_map, search_zone_h, search_zone_v, onset_threshold, onset_neighborhood
        )

        # --- Fase 2 (opcional) ---
        use_phase2 = int(self.params.get("use_phase2", 1))
        phase2_applied = False
        phase2_side = None

        if use_phase2 == 1:
            left_orig = left
            right_orig = right
            left, right, phase2_side = self._phase2(edge_map, top, bottom, left, right)
            if phase2_side is not None:
                phase2_applied = True

        # --- Construir selected_lines en formato border_lines ---
        selected_lines = {
            "top":    {"x1": 0,     "y1": int(top),    "x2": w - 1, "y2": int(top),    "angle": 0.0},
            "bottom": {"x1": 0,     "y1": int(bottom), "x2": w - 1, "y2": int(bottom), "angle": 0.0},
            "left":   {"x1": int(left),  "y1": 0,      "x2": int(left),  "y2": h - 1,  "angle": 90.0},
            "right":  {"x1": int(right), "y1": 0,      "x2": int(right), "y2": h - 1,  "angle": 90.0},
        }

        selection_metadata = {
            "image_width":  int(w),
            "image_height": int(h),
            "top":    int(top),
            "bottom": int(bottom),
            "left":   int(left),
            "right":  int(right),
            "phase2_applied": bool(phase2_applied),
            "phase2_side":    phase2_side,
            "top_is_image_border":    False,
            "bottom_is_image_border": False,
            "left_is_image_border":   False,
            "right_is_image_border":  False,
            "valid": True,
        }

        result = {
            "selected_lines":     selected_lines,
            "selection_metadata": selection_metadata,
        }

        # --- Visualizacion ---
        if not self.without_preview:
            vis_width = int(self.params.get("visualization_size", 900))
            sample = self._draw_sample(base_img, edge_map, top, bottom, left, right, vis_width)
            result["sample_image"] = sample

        return result
