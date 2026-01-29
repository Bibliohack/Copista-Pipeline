"""
Filtro: ResizeFilter
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY
import warnings


class ResizeFilter(BaseFilter):
    """Filtro para redimensionar la imagen"""
    
    FILTER_NAME = "Resize"
    DESCRIPTION = "Redimensiona la imagen a un tamaño específico o por porcentaje"
    INPUTS = {"input_image": "image"}  # <-- MODIFICADO: ahora acepta input_image
    OUTPUTS = {
        "resized_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "mode": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Modo: 0=Por porcentaje, 1=Por dimensiones fijas"
        },
        "scale_percent": {
            "default": 100,
            "min": 10,
            "max": 200,
            "step": 5,
            "description": "Porcentaje de escala (solo modo 0)"
        },
        "width": {
            "default": 640,
            "min": 100,
            "max": 1920,
            "step": 10,
            "description": "Ancho en píxeles (solo modo 1)"
        },
        "height": {
            "default": 480,
            "min": 100,
            "max": 1080,
            "step": 10,
            "description": "Alto en píxeles (solo modo 1)"
        },
        "interpolation": {
            "default": 1,
            "min": 0,
            "max": 4,
            "step": 1,
            "description": "Interpolación: 0=NEAREST, 1=LINEAR, 2=AREA, 3=CUBIC, 4=LANCZOS4"
        }
    }
    
    INTERPOLATION_METHODS = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4
    ]
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        # <-- MODIFICADO: Ahora acepta input_image con fallback y advertencia
        input_img = inputs.get("input_image")
        if input_img is None:
            warnings.warn(f"[Resize] No se proporcionó 'input_image', usando imagen original.", stacklevel=2)
            input_img = original_image
        
        mode = self.params["mode"]
        interp = self.INTERPOLATION_METHODS[self.params["interpolation"]]
        
        if mode == 0:
            scale = self.params["scale_percent"] / 100.0
            new_width = int(input_img.shape[1] * scale)  # <-- Usar input_img
            new_height = int(input_img.shape[0] * scale)  # <-- Usar input_img
        else:
            new_width = self.params["width"]
            new_height = self.params["height"]
        
        resized = cv2.resize(input_img, (new_width, new_height), interpolation=interp)  # <-- Usar input_img
        
        return {
            "resized_image": resized,
            "sample_image": resized
        }
