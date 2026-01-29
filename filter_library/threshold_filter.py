"""
Filtro: ThresholdFilter
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class ThresholdFilter(BaseFilter):
    """Aplica umbralización a la imagen"""
    
    FILTER_NAME = "Threshold"
    DESCRIPTION = "Aplica umbralización binaria o adaptativa"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "threshold_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "method": {
            "default": 0,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": "Método: 0=Binario, 1=Otsu, 2=Adaptativo"
        },
        "threshold": {
            "default": 127,
            "min": 0,
            "max": 255,
            "step": 1,
            "description": "Valor de umbral (solo método binario)"
        },
        "max_value": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 1,
            "description": "Valor máximo para píxeles sobre umbral"
        },
        "adaptive_method": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Método adaptativo: 0=Mean, 1=Gaussian"
        },
        "block_size": {
            "default": 11,
            "min": 3,
            "max": 99,
            "step": 2,
            "description": "Tamaño de bloque para adaptativo (impar)"
        },
        "c_value": {
            "default": 2,
            "min": -20,
            "max": 20,
            "step": 1,
            "description": "Constante C para adaptativo"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        # Convertir a grayscale si es necesario
        if len(input_img.shape) == 3:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_img
        
        method = self.params["method"]
        max_val = self.params["max_value"]
        
        if method == 0:
            # Binario simple
            _, thresh = cv2.threshold(gray, self.params["threshold"], max_val, cv2.THRESH_BINARY)
        elif method == 1:
            # Otsu
            _, thresh = cv2.threshold(gray, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Adaptativo
            block_size = self.params["block_size"]
            if block_size % 2 == 0:
                block_size += 1
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if self.params["adaptive_method"] == 0 else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            thresh = cv2.adaptiveThreshold(gray, max_val, adaptive_method, cv2.THRESH_BINARY, block_size, self.params["c_value"])
        
        sample = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        return {
            "threshold_image": thresh,
            "sample_image": sample
        }
