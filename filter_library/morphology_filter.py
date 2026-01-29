"""
Filtro: MorphologyFilter
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class MorphologyFilter(BaseFilter):
    """Operaciones morfológicas"""
    
    FILTER_NAME = "Morphology"
    DESCRIPTION = "Aplica operaciones morfológicas (erosión, dilatación, apertura, cierre)"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "morphed_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "operation": {
            "default": 0,
            "min": 0,
            "max": 5,
            "step": 1,
            "description": "Operación: 0=Erosión, 1=Dilatación, 2=Apertura, 3=Cierre, 4=Gradiente, 5=TopHat"
        },
        "kernel_shape": {
            "default": 0,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": "Forma kernel: 0=Rectángulo, 1=Cruz, 2=Elipse"
        },
        "kernel_size": {
            "default": 5,
            "min": 1,
            "max": 21,
            "step": 2,
            "description": "Tamaño del kernel (impar)"
        },
        "iterations": {
            "default": 1,
            "min": 1,
            "max": 10,
            "step": 1,
            "description": "Número de iteraciones"
        }
    }
    
    OPERATIONS = [
        cv2.MORPH_ERODE,
        cv2.MORPH_DILATE,
        cv2.MORPH_OPEN,
        cv2.MORPH_CLOSE,
        cv2.MORPH_GRADIENT,
        cv2.MORPH_TOPHAT
    ]
    
    KERNEL_SHAPES = [
        cv2.MORPH_RECT,
        cv2.MORPH_CROSS,
        cv2.MORPH_ELLIPSE
    ]
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        ksize = self.params["kernel_size"]
        if ksize % 2 == 0:
            ksize += 1
        
        kernel = cv2.getStructuringElement(
            self.KERNEL_SHAPES[self.params["kernel_shape"]],
            (ksize, ksize)
        )
        
        op = self.OPERATIONS[self.params["operation"]]
        result = cv2.morphologyEx(input_img, op, kernel, iterations=self.params["iterations"])
        
        # Asegurar que sample_image sea visualizable
        if len(result.shape) == 2:
            sample = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            sample = result
        
        return {
            "morphed_image": result,
            "sample_image": sample
        }
