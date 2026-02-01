"""
Filtro: ThresholdAdvanced
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class ThresholdAdvanced(BaseFilter):
    """Umbralización avanzada con múltiples métodos incluyendo OTSU y Adaptive"""
    
    FILTER_NAME = "ThresholdAdvanced"
    DESCRIPTION = "Umbralización con soporte para Binary, OTSU automático, y Adaptive (Mean/Gaussian)."
    
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
            "max": 7,
            "step": 1,
            "description": "Método: 0=BINARY, 1=BINARY_INV, 2=TRUNC, 3=TOZERO, 4=TOZERO_INV, 5=OTSU, 6=ADAPTIVE_MEAN, 7=ADAPTIVE_GAUSSIAN"
        },
        "threshold": {
            "default": 127,
            "min": 0,
            "max": 255,
            "step": 1,
            "description": "Valor de umbral (ignorado en OTSU y Adaptive)."
        },
        "max_value": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 1,
            "description": "Valor máximo para píxeles sobre umbral."
        },
        "block_size": {
            "default": 11,
            "min": 3,
            "max": 99,
            "step": 2,
            "description": "Tamaño de bloque para Adaptive (debe ser impar, solo métodos 6 y 7)."
        },
        "c_value": {
            "default": 2,
            "min": -20,
            "max": 20,
            "step": 1,
            "description": "Constante a restar del promedio en Adaptive (solo métodos 6 y 7)."
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
        thresh_val = self.params["threshold"]
        max_val = self.params["max_value"]
        block_size = self.params["block_size"] | 1  # Asegurar impar
        block_size = max(3, block_size)
        c_value = self.params["c_value"]
        
        if method <= 4:
            # Threshold estándar (BINARY, BINARY_INV, TRUNC, TOZERO, TOZERO_INV)
            _, result = cv2.threshold(gray, thresh_val, max_val, method)
        elif method == 5:
            # OTSU (calcula umbral automáticamente)
            _, result = cv2.threshold(gray, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 6:
            # Adaptive Mean
            result = cv2.adaptiveThreshold(
                gray, max_val, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, block_size, c_value
            )
        elif method == 7:
            # Adaptive Gaussian
            result = cv2.adaptiveThreshold(
                gray, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, c_value
            )
        else:
            _, result = cv2.threshold(gray, thresh_val, max_val, cv2.THRESH_BINARY)
        
        sample = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return {
            "threshold_image": result,
            "sample_image": sample
        }
