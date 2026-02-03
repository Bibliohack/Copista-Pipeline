"""
Filtro: NormalizeFromHistogram
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class NormalizeFromHistogram(BaseFilter):
    """Normaliza la imagen usando datos de HistogramVisualize mapeando 3 puntos de referencia"""
    
    FILTER_NAME = "NormalizeFromHistogram"
    DESCRIPTION = "Normaliza la imagen mapeando tres puntos del histograma (dark_peak, mínimo, light_peak) a nuevas posiciones objetivo. Usa datos de HistogramVisualize."
    
    INPUTS = {
        "input_image": "image",
        "histogram_data": "histogram"  # De HistogramVisualize
    }
    
    OUTPUTS = {
        "normalized_image": "image",
        "normalization_metadata": "metadata",
        "sample_image": "image"
    }
    
    PARAMS = {
        "dark_target": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Valor objetivo para el dark_peak (típicamente 0 = negro)."
        },
        "min_target": {
            "default": 128,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Valor objetivo para el mínimo entre picos (típicamente 128 = gris medio)."
        },
        "light_target": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Valor objetivo para el light_peak (típicamente 255 = blanco)."
        },
        "use_piecewise": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Usar mapeo por tramos (1) o lineal simple (0). Por tramos da mejor control."
        },
        "show_comparison": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar comparación antes/después (1=Sí, 0=No)."
        },
        "show_mapping_info": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar información de mapeo en la visualización (1=Sí, 0=No)."
        }
    }
    
    def _calculate_min_from_histogram(self, hist: np.ndarray, dark_peak: int, light_peak: int) -> int:
        """
        Calcula el mínimo entre picos a partir del histograma.
        """
        if dark_peak < light_peak:
            between_zone = hist[dark_peak:light_peak+1]
            min_between = int(dark_peak + np.argmin(between_zone))
        else:
            min_between = (dark_peak + light_peak) // 2
        
        return min_between
    
    def _normalize_piecewise(self, image: np.ndarray,
                            dark_src: int, min_src: int, light_src: int,
                            dark_tgt: int, min_tgt: int, light_tgt: int) -> np.ndarray:
        """
        Normaliza usando mapeo por tramos (piecewise linear).
        
        Mapea:
        - [0, dark_src] → [0, dark_tgt]
        - [dark_src, min_src] → [dark_tgt, min_tgt]
        - [min_src, light_src] → [min_tgt, light_tgt]
        - [light_src, 255] → [light_tgt, 255]
        """
        img_float = image.astype(np.float32)
        result = np.zeros_like(img_float)
        
        # Tramo 1: [0, dark_src] → [0, dark_tgt]
        mask1 = img_float <= dark_src
        if dark_src > 0:
            result[mask1] = (img_float[mask1] / dark_src) * dark_tgt
        else:
            result[mask1] = dark_tgt
        
        # Tramo 2: [dark_src, min_src] → [dark_tgt, min_tgt]
        mask2 = (img_float > dark_src) & (img_float <= min_src)
        if min_src > dark_src:
            t = (img_float[mask2] - dark_src) / (min_src - dark_src)
            result[mask2] = dark_tgt + t * (min_tgt - dark_tgt)
        else:
            result[mask2] = min_tgt
        
        # Tramo 3: [min_src, light_src] → [min_tgt, light_tgt]
        mask3 = (img_float > min_src) & (img_float <= light_src)
        if light_src > min_src:
            t = (img_float[mask3] - min_src) / (light_src - min_src)
            result[mask3] = min_tgt + t * (light_tgt - min_tgt)
        else:
            result[mask3] = light_tgt
        
        # Tramo 4: [light_src, 255] → [light_tgt, 255]
        mask4 = img_float > light_src
        if light_src < 255:
            t = (img_float[mask4] - light_src) / (255 - light_src)
            result[mask4] = light_tgt + t * (255 - light_tgt)
        else:
            result[mask4] = 255
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _normalize_linear(self, image: np.ndarray,
                         dark_src: int, light_src: int,
                         dark_tgt: int, light_tgt: int) -> np.ndarray:
        """
        Normaliza usando mapeo lineal simple (ignora el mínimo).
        
        Mapea [dark_src, light_src] → [dark_tgt, light_tgt]
        """
        img_float = image.astype(np.float32)
        
        # Clipping
        img_clipped = np.clip(img_float, dark_src, light_src)
        
        # Normalizar a [0, 1]
        if light_src > dark_src:
            img_norm = (img_clipped - dark_src) / (light_src - dark_src)
        else:
            img_norm = np.zeros_like(img_float)
        
        # Escalar a [dark_tgt, light_tgt]
        img_scaled = img_norm * (light_tgt - dark_tgt) + dark_tgt
        
        return np.clip(img_scaled, 0, 255).astype(np.uint8)
    
    def _create_comparison_with_info(self, original: np.ndarray, normalized: np.ndarray,
                                     dark_src: int, min_src: int, light_src: int,
                                     dark_tgt: int, min_tgt: int, light_tgt: int,
                                     method: str, show_info: bool) -> np.ndarray:
        """Crea visualización comparando antes/después con información de mapeo"""
        
        # Asegurar que ambas son BGR
        if len(original.shape) == 2:
            orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            orig_bgr = original
        
        if len(normalized.shape) == 2:
            norm_bgr = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        else:
            norm_bgr = normalized
        
        # Redimensionar si es necesario para que coincidan
        h1, w1 = orig_bgr.shape[:2]
        h2, w2 = norm_bgr.shape[:2]
        
        if (h1, w1) != (h2, w2):
            norm_bgr = cv2.resize(norm_bgr, (w1, h1))
        
        # Agregar labels
        orig_labeled = orig_bgr.copy()
        norm_labeled = norm_bgr.copy()
        
        # Label "ORIGINAL"
        cv2.rectangle(orig_labeled, (5, 5), (155, 40), (0, 0, 0), -1)
        cv2.putText(orig_labeled, "ORIGINAL", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Label "NORMALIZADO"
        cv2.rectangle(norm_labeled, (5, 5), (205, 40), (0, 0, 0), -1)
        cv2.putText(norm_labeled, "NORMALIZADO", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combinar horizontalmente
        comparison = np.hstack([orig_labeled, norm_labeled])
        
        # Agregar información de mapeo debajo
        if show_info:
            h_comp, w_comp = comparison.shape[:2]
            info_height = 120
            
            # Crear panel de info
            info_panel = np.zeros((info_height, w_comp, 3), dtype=np.uint8)
            info_panel[:] = (40, 40, 40)
            
            y = 25
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Título
            cv2.putText(info_panel, "MAPEO DE VALORES:", (10, y), font, 0.6, (255, 255, 0), 1)
            y += 30
            
            # Info de mapeo
            if method == "piecewise":
                # Mostrar los 3 puntos
                mapping_lines = [
                    f"Dark Peak:  {dark_src:3d} -> {dark_tgt:3d}",
                    f"Minimo:     {min_src:3d} -> {min_tgt:3d}",
                    f"Light Peak: {light_src:3d} -> {light_tgt:3d}",
                    f"Metodo: Piecewise (por tramos)"
                ]
            else:
                # Solo dark y light
                mapping_lines = [
                    f"Dark Peak:  {dark_src:3d} -> {dark_tgt:3d}",
                    f"Light Peak: {light_src:3d} -> {light_tgt:3d}",
                    f"Metodo: Lineal (ignora minimo)"
                ]
            
            for line in mapping_lines:
                cv2.putText(info_panel, line, (10, y), font, 0.5, (200, 200, 200), 1)
                y += 22
            
            # Combinar verticalmente
            final = np.vstack([comparison, info_panel])
        else:
            final = comparison
        
        return final
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        histogram_data = inputs.get("histogram_data", {})
        
        # Convertir a grayscale si es necesario
        if len(input_img.shape) == 3:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_img.copy()
        
        h, w = gray.shape[:2]
        
        # Obtener parámetros
        use_piecewise = bool(self.params["use_piecewise"])
        show_comparison = bool(self.params["show_comparison"])
        show_info = bool(self.params["show_mapping_info"])
        
        dark_tgt = self.params["dark_target"]
        min_tgt = self.params["min_target"]
        light_tgt = self.params["light_target"]
        
        # Obtener valores source del histogram_data
        if not isinstance(histogram_data, dict):
            # Sin datos de histograma, retornar error
            error_img = np.ones((400, 600, 3), dtype=np.uint8) * 240
            cv2.putText(error_img, "ERROR: No histogram_data", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(error_img, "Conecta HistogramVisualize", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            return {
                "normalized_image": gray,
                "normalization_metadata": {
                    "error": "No histogram_data provided",
                    "image_width": int(w),
                    "image_height": int(h)
                },
                "sample_image": error_img
            }
        
        # Obtener dark_marker y light_marker
        dark_src = histogram_data.get("dark_marker")
        light_src = histogram_data.get("light_marker")
        
        if dark_src is None or light_src is None:
            # Sin markers, retornar error
            error_img = np.ones((400, 600, 3), dtype=np.uint8) * 240
            cv2.putText(error_img, "ERROR: Missing markers", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(error_img, "HistogramVisualize needs", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(error_img, "dark_marker and light_marker", (50, 270),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            return {
                "normalized_image": gray,
                "normalization_metadata": {
                    "error": "Missing dark_marker or light_marker in histogram_data",
                    "image_width": int(w),
                    "image_height": int(h)
                },
                "sample_image": error_img
            }
        
        # Calcular el mínimo a partir del histograma
        hist_list = histogram_data.get("histogram")
        if hist_list is not None:
            hist = np.array(hist_list, dtype=np.float32)
            min_src = self._calculate_min_from_histogram(hist, dark_src, light_src)
        else:
            # Fallback si no hay histograma
            min_src = (dark_src + light_src) // 2
        
        # Validar valores source
        dark_src = max(0, min(254, int(dark_src)))
        light_src = max(dark_src + 1, min(255, int(light_src)))
        min_src = max(dark_src, min(light_src, int(min_src)))
        
        # Validar valores target
        dark_tgt = max(0, min(255, dark_tgt))
        light_tgt = max(0, min(255, light_tgt))
        min_tgt = max(0, min(255, min_tgt))
        
        # Aplicar normalización
        if use_piecewise:
            normalized = self._normalize_piecewise(
                gray, dark_src, min_src, light_src,
                dark_tgt, min_tgt, light_tgt
            )
            method = "piecewise"
        else:
            normalized = self._normalize_linear(
                gray, dark_src, light_src,
                dark_tgt, light_tgt
            )
            method = "linear"
        
        # Crear metadata
        normalization_metadata = {
            "image_width": int(w),
            "image_height": int(h),
            "source_values": {
                "dark_peak": int(dark_src),
                "min_between": int(min_src),
                "light_peak": int(light_src)
            },
            "target_values": {
                "dark_target": int(dark_tgt),
                "min_target": int(min_tgt),
                "light_target": int(light_tgt)
            },
            "method": method
        }
        
        result = {
            "normalized_image": normalized,
            "normalization_metadata": normalization_metadata
        }
        
        # Solo generar visualización si no estamos en modo without_preview
        if not self.without_preview:
            if show_comparison:
                sample = self._create_comparison_with_info(
                    gray, normalized,
                    dark_src, min_src, light_src,
                    dark_tgt, min_tgt, light_tgt,
                    method, show_info
                )
            else:
                # Solo la imagen normalizada en BGR para visualización
                sample = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            
            result["sample_image"] = sample
        
        return result

