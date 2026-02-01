"""
Filtro: ContourFilter - MEJORADO
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class ContourFilter(BaseFilter):
    """Detecta y dibuja contornos"""
    
    FILTER_NAME = "Contours"
    DESCRIPTION = "Detecta contornos y genera datos de contornos. Incluye metadata con dimensiones de imagen."
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "contours_data": "contours",
        "contours_metadata": "metadata",
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
        
        h, w = input_img.shape[:2]
        
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
        
        # Preparar datos de contornos
        contours_data = []
        total_area = 0
        total_perimeter = 0
        
        for c in filtered_contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            x, y, w_box, h_box = cv2.boundingRect(c)
            
            total_area += area
            total_perimeter += perimeter
            
            contours_data.append({
                "area": area,
                "perimeter": perimeter,
                "bounding_box": (x, y, w_box, h_box),
                "points": c.tolist()
            })
        
        # Metadata con dimensiones e información de contornos
        image_area = w * h
        metadata = {
            "image_width": int(w),
            "image_height": int(h),
            "image_area": int(image_area),
            "total_contours": len(contours),
            "filtered_contours": len(filtered_contours),
            "rejected_contours": len(contours) - len(filtered_contours),
            "min_area_threshold": min_area,
            "total_contour_area": float(total_area),
            "total_contour_perimeter": float(total_perimeter),
            "coverage_percent": round((total_area / image_area) * 100, 2) if image_area > 0 else 0
        }
        
        result = {
            "contours_data": contours_data,
            "contours_metadata": metadata
        }
        
        # Solo generar visualización si no estamos en modo without_preview
        if not self.without_preview:
            # Crear imagen de contornos
            if len(input_img.shape) == 3:
                contour_img = input_img.copy()
            else:
                contour_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
            
            color = (self.params["color_b"], self.params["color_g"], self.params["color_r"])
            cv2.drawContours(contour_img, filtered_contours, -1, color, self.params["draw_thickness"])
            
            result["contour_image"] = contour_img
            result["sample_image"] = contour_img
        
        return result
