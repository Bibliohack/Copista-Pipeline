"""
Filtro: ColorSpaceFilter
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class ColorSpaceFilter(BaseFilter):
    """Convierte entre espacios de color"""
    
    FILTER_NAME = "ColorSpace"
    DESCRIPTION = "Convierte la imagen a diferentes espacios de color y permite ver canales individuales"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "converted_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "color_space": {
            "default": 0,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": "Espacio: 0=BGR, 1=HSV, 2=LAB, 3=YCrCb"
        },
        "channel": {
            "default": 0,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": "Canal a mostrar: 0=Todos, 1=Canal1, 2=Canal2, 3=Canal3"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        # Asegurar que es BGR
        if len(input_img.shape) == 2:
            input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        
        cs = self.params["color_space"]
        
        if cs == 0:
            converted = input_img
        elif cs == 1:
            converted = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
        elif cs == 2:
            converted = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
        else:
            converted = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCrCb)
        
        channel = self.params["channel"]
        
        if channel == 0:
            sample = converted
        else:
            single_channel = converted[:, :, channel - 1]
            sample = cv2.cvtColor(single_channel, cv2.COLOR_GRAY2BGR)
        
        return {
            "converted_image": converted,
            "sample_image": sample
        }

# =============================================================================
# Filtros extraídos de page_border_pipeline.py
# =============================================================================
