"""
Filtro: NormalizePeaks
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class NormalizePeaks(BaseFilter):
    """Normaliza la imagen mapeando rangos de intensidad basados en picos"""
    
    FILTER_NAME = "NormalizePeaks"
    DESCRIPTION = "Normaliza la imagen mapeando [dark_peak, light_peak] a [dark_target, light_target]. Útil para ajustar contraste basado en histograma."
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "normalized_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "dark_peak": {
            "default": 30,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Valor de gris del pico oscuro (fondo). Píxeles <= a este valor se mapean al target oscuro."
        },
        "light_peak": {
            "default": 220,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Valor de gris del pico claro (papel). Píxeles >= a este valor se mapean al target claro."
        },
        "dark_target": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Valor objetivo para píxeles oscuros (típicamente 0 = negro)."
        },
        "light_target": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Valor objetivo para píxeles claros (típicamente 255 = blanco)."
        },
        "auto_detect": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Auto-detectar picos del histograma (0=No, 1=Sí). Si es 1, ignora dark_peak y light_peak."
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        # Convertir a grayscale si es necesario
        if len(input_img.shape) == 3:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_img.copy()
        
        # Auto-detectar picos si está habilitado
        if self.params["auto_detect"] == 1:
            dark_peak, light_peak = self._auto_detect_peaks(gray)
        else:
            dark_peak = self.params["dark_peak"]
            light_peak = self.params["light_peak"]
        
        dark_target = self.params["dark_target"]
        light_target = self.params["light_target"]
        
        # Asegurar que los valores son válidos
        dark_peak = max(0, min(254, dark_peak))
        light_peak = max(dark_peak + 1, min(255, light_peak))
        
        # Normalizar: mapear [dark_peak, light_peak] -> [dark_target, light_target]
        img = gray.astype(np.float32)
        
        # Clipping y escalado
        img = np.clip(img, dark_peak, light_peak)
        img = (img - dark_peak) / (light_peak - dark_peak)  # Normalizar a [0, 1]
        img = img * (light_target - dark_target) + dark_target  # Escalar a targets
        
        result = np.clip(img, 0, 255).astype(np.uint8)
        
        # sample_image en BGR para visualización
        sample = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return {
            "normalized_image": result,
            "sample_image": sample
        }
    
    def _auto_detect_peaks(self, gray: np.ndarray) -> tuple:
        """Auto-detecta picos oscuro y claro del histograma"""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        # Buscar pico en zona oscura (0-100) y zona clara (150-255)
        dark_zone = hist[:100]
        light_zone = hist[150:]
        
        dark_peak = int(np.argmax(dark_zone)) if dark_zone.max() > 0 else 30
        light_peak = int(150 + np.argmax(light_zone)) if light_zone.max() > 0 else 220
        
        return dark_peak, light_peak
