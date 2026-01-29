"""
Filtro: MorphologyAdvanced
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class MorphologyAdvanced(BaseFilter):
    """Operaciones morfológicas avanzadas con opción de inversión"""
    
    FILTER_NAME = "MorphologyAdvanced"
    DESCRIPTION = "Operaciones morfológicas (erosión, dilatación, apertura, cierre, gradiente, tophat, blackhat) con opción de invertir resultado."
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "morphed_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "operation": {
            "default": 2,
            "min": 0,
            "max": 6,
            "step": 1,
            "description": "Operación: 0=Erosión, 1=Dilatación, 2=Apertura, 3=Cierre, 4=Gradiente, 5=TopHat, 6=BlackHat"
        },
        "kernel_shape": {
            "default": 2,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": "Forma del kernel: 0=Rectángulo, 1=Cruz, 2=Elipse"
        },
        "kernel_size": {
            "default": 5,
            "min": 1,
            "max": 31,
            "step": 2,
            "description": "Tamaño del kernel (se fuerza a impar)."
        },
        "iterations": {
            "default": 1,
            "min": 1,
            "max": 10,
            "step": 1,
            "description": "Número de iteraciones."
        },
        "invert_output": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Invertir colores del resultado (0=No, 1=Sí)."
        }
    }
    
    OPERATIONS = [
        cv2.MORPH_ERODE,
        cv2.MORPH_DILATE,
        cv2.MORPH_OPEN,
        cv2.MORPH_CLOSE,
        cv2.MORPH_GRADIENT,
        cv2.MORPH_TOPHAT,
        cv2.MORPH_BLACKHAT
    ]
    
    KERNEL_SHAPES = [
        cv2.MORPH_RECT,
        cv2.MORPH_CROSS,
        cv2.MORPH_ELLIPSE
    ]
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        operation = self.params["operation"]
        kernel_shape = self.params["kernel_shape"]
        ksize = self.params["kernel_size"] | 1  # Asegurar impar
        ksize = max(1, ksize)
        iterations = max(1, self.params["iterations"])
        invert = self.params["invert_output"]
        
        # Crear kernel estructurante
        kernel = cv2.getStructuringElement(
            self.KERNEL_SHAPES[kernel_shape],
            (ksize, ksize)
        )
        
        # Aplicar operación morfológica
        op = self.OPERATIONS[operation]
        result = cv2.morphologyEx(input_img, op, kernel, iterations=iterations)
        
        # Invertir si está habilitado
        if invert == 1:
            result = cv2.bitwise_not(result)
        
        # Asegurar que sample_image sea visualizable
        if len(result.shape) == 2:
            sample = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            sample = result
        
        return {
            "morphed_image": result,
            "sample_image": sample
        }
