"""
Filtro: BrightnessContrastFilter
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class BrightnessContrastFilter(BaseFilter):
    """Ajusta brillo y contraste"""
    
    FILTER_NAME = "BrightnessContrast"
    DESCRIPTION = "Ajusta el brillo y contraste de la imagen"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "adjusted_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "brightness": {
            "default": 0,
            "min": -100,
            "max": 100,
            "step": 5,
            "description": "Ajuste de brillo (-100 a 100)"
        },
        "contrast": {
            "default": 100,
            "min": 0,
            "max": 200,
            "step": 5,
            "description": "Ajuste de contraste (0-200, 100=normal)"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        brightness = self.params["brightness"]
        contrast = self.params["contrast"] / 100.0
        
        # Aplicar contraste y brillo
        adjusted = cv2.convertScaleAbs(input_img, alpha=contrast, beta=brightness)
        
        return {
            "adjusted_image": adjusted,
            "sample_image": adjusted
        }
