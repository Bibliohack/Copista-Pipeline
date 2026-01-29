"""
Filtro: DenoiseNLMeans
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class DenoiseNLMeans(BaseFilter):
    """Reducción de ruido usando Non-Local Means"""
    
    FILTER_NAME = "DenoiseNLMeans"
    DESCRIPTION = "Aplica fastNlMeansDenoising para reducir ruido preservando bordes. Ideal para imágenes con ruido gaussiano."
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "denoised_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "h": {
            "default": 10,
            "min": 1,
            "max": 30,
            "step": 1,
            "description": "Fuerza del filtro. Mayor valor = más suavizado pero puede perder detalle."
        },
        "template_window_size": {
            "default": 7,
            "min": 3,
            "max": 21,
            "step": 2,
            "description": "Tamaño de la ventana de template (debe ser impar). Típico: 7."
        },
        "search_window_size": {
            "default": 21,
            "min": 7,
            "max": 51,
            "step": 2,
            "description": "Tamaño del área de búsqueda (debe ser impar). Típico: 21. Mayor = más lento pero mejor calidad."
        },
        "color_mode": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Modo: 0=Grayscale (más rápido), 1=Color (preserva colores)."
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        h = max(1, self.params["h"])
        template_size = self.params["template_window_size"] | 1  # Asegurar impar
        search_size = self.params["search_window_size"] | 1  # Asegurar impar
        color_mode = self.params["color_mode"]
        
        if color_mode == 0 or len(input_img.shape) == 2:
            # Modo grayscale
            if len(input_img.shape) == 3:
                gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = input_img
            
            result = cv2.fastNlMeansDenoising(gray, None, h, template_size, search_size)
            sample = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            # Modo color
            result = cv2.fastNlMeansDenoisingColored(input_img, None, h, h, template_size, search_size)
            sample = result
        
        return {
            "denoised_image": result,
            "sample_image": sample
        }
