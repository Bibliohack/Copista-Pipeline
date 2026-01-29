"""
Filtro: ContourFilter
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class ContourFilter(BaseFilter):
    """Detecta y dibuja contornos"""
    
    FILTER_NAME = "Contours"
    DESCRIPTION = "Detecta contornos y genera datos de contornos"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "contours_data": "contours",
        "contour_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "mode": {
            "default": 1,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": "Modo: 0=EXTERNAL, 1=LIST, 2=CCOMP, 3=TREE"
        },
        "method": {
            "default": 1,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": "Método: 0=NONE, 1=SIMPLE, 2=TC89_L1, 3=TC89_KCOS"
        },
        "min_area": {
            "default": 100,
            "min": 0,
            "max": 10000,
            "step": 100,
            "description": "Área mínima de contorno"
        },
        "draw_thickness": {
            "default": 2,
            "min": 1,
            "max": 5,
            "step": 1,
            "description": "Grosor de línea para dibujar"
        },
        "color_r": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color R para contornos"
        },
        "color_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color G para contornos"
        },
        "color_b": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color B para contornos"
        }
    }
    
    MODES = [
        cv2.RETR_EXTERNAL,
        cv2.RETR_LIST,
        cv2.RETR_CCOMP,
        cv2.RETR_TREE
    ]
    
    METHODS = [
        cv2.CHAIN_APPROX_NONE,
        cv2.CHAIN_APPROX_SIMPLE,
        cv2.CHAIN_APPROX_TC89_L1,
        cv2.CHAIN_APPROX_TC89_KCOS
    ]
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        # Convertir a grayscale y binarizar si es necesario
        if len(input_img.shape) == 3:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_img
        
        # Si no es binaria, aplicar umbral
        if gray.max() > 1:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        else:
            binary = gray
        
        contours, hierarchy = cv2.findContours(
            binary,
            self.MODES[self.params["mode"]],
            self.METHODS[self.params["method"]]
        )
        
        # Filtrar por área mínima
        min_area = self.params["min_area"]
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        # Crear imagen de contornos usando input_img (no original_image)
        if len(input_img.shape) == 3:
            contour_img = input_img.copy()
        else:
            contour_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        
        color = (self.params["color_b"], self.params["color_g"], self.params["color_r"])
        cv2.drawContours(contour_img, filtered_contours, -1, color, self.params["draw_thickness"])
        
        # Preparar datos de contornos
        contours_data = []
        for c in filtered_contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            x, y, w, h = cv2.boundingRect(c)
            contours_data.append({
                "area": area,
                "perimeter": perimeter,
                "bounding_box": (x, y, w, h),
                "points": c.tolist()
            })
        
        return {
            "contours_data": contours_data,
            "contour_image": contour_img,
            "sample_image": contour_img
        }
