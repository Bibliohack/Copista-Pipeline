"""
Filtro: GrayscaleFilter
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class GrayscaleFilter(BaseFilter):
    """Convierte la imagen a escala de grises"""
    
    FILTER_NAME = "Grayscale"
    DESCRIPTION = "Convierte la imagen a escala de grises con control de pesos RGB"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "grayscale_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "method": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Método: 0=OpenCV estándar, 1=Pesos personalizados"
        },
        "weight_r": {
            "default": 30,
            "min": 0,
            "max": 100,
            "step": 1,
            "description": "Peso del canal Rojo (0-100, solo método 1)"
        },
        "weight_g": {
            "default": 59,
            "min": 0,
            "max": 100,
            "step": 1,
            "description": "Peso del canal Verde (0-100, solo método 1)"
        },
        "weight_b": {
            "default": 11,
            "min": 0,
            "max": 100,
            "step": 1,
            "description": "Peso del canal Azul (0-100, solo método 1)"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        if len(input_img.shape) == 2:
            # Ya es grayscale
            gray = input_img
        elif self.params["method"] == 0:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            # Pesos personalizados
            total = self.params["weight_r"] + self.params["weight_g"] + self.params["weight_b"]
            if total == 0:
                total = 1
            wr = self.params["weight_r"] / total
            wg = self.params["weight_g"] / total
            wb = self.params["weight_b"] / total
            gray = (input_img[:,:,2] * wr + input_img[:,:,1] * wg + input_img[:,:,0] * wb).astype(np.uint8)
        
        # sample_image debe ser visualizable (3 canales para consistencia)
        sample = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(gray.shape) == 2 else gray
        
        return {
            "grayscale_image": gray,
            "sample_image": sample
        }
