"""
Filtro: GaussianBlurFilter
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class GaussianBlurFilter(BaseFilter):
    """Aplica desenfoque gaussiano"""
    
    FILTER_NAME = "GaussianBlur"
    DESCRIPTION = "Aplica un filtro de desenfoque gaussiano"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "blurred_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "kernel_size": {
            "default": 5,
            "min": 1,
            "max": 31,
            "step": 2,
            "description": "Tamaño del kernel (debe ser impar)"
        },
        "sigma": {
            "default": 0,
            "min": 0,
            "max": 10,
            "step": 1,
            "description": "Sigma (0=auto calculado)"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        ksize = self.params["kernel_size"]
        if ksize % 2 == 0:
            ksize += 1
        
        sigma = self.params["sigma"]
        blurred = cv2.GaussianBlur(input_img, (ksize, ksize), sigma)
        
        return {
            "blurred_image": blurred,
            "sample_image": blurred
        }
