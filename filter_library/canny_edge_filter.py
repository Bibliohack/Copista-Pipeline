"""
Filtro: CannyEdgeFilter
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class CannyEdgeFilter(BaseFilter):
    """Detector de bordes Canny"""
    
    FILTER_NAME = "CannyEdge"
    DESCRIPTION = "Detecta bordes usando el algoritmo de Canny"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "edge_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "threshold1": {
            "default": 50,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Umbral inferior para histéresis"
        },
        "threshold2": {
            "default": 150,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Umbral superior para histéresis"
        },
        "aperture_size": {
            "default": 3,
            "min": 3,
            "max": 7,
            "step": 2,
            "description": "Tamaño de apertura Sobel (3, 5 o 7)"
        },
        "l2_gradient": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Usar norma L2 para gradiente (0=No, 1=Sí)"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        # Convertir a grayscale si es necesario
        if len(input_img.shape) == 3:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_img
        
        aperture = self.params["aperture_size"]
        if aperture not in [3, 5, 7]:
            aperture = 3
        
        edges = cv2.Canny(
            gray,
            self.params["threshold1"],
            self.params["threshold2"],
            apertureSize=aperture,
            L2gradient=bool(self.params["l2_gradient"])
        )
        
        # sample_image en color para visualización
        sample = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return {
            "edge_image": edges,
            "sample_image": sample
        }
