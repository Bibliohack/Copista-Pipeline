"""
Filtro: MinArcLength
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY
import warnings


class MinArcLength(BaseFilter):
    """Filtra contornos/bordes por longitud mínima de arco"""
    
    FILTER_NAME = "MinArcLength"
    DESCRIPTION = "Filtra una imagen de bordes, eliminando contornos cuya longitud de arco sea menor al mínimo especificado."
    
    INPUTS = {
        "edge_image": "image",
        "base_image": "image"  # <-- AÑADIDO: para visualización
    }
    
    OUTPUTS = {
        "filtered_edges": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "min_length": {
            "default": 100,
            "min": 1,
            "max": 1000,
            "step": 10,
            "description": "Longitud mínima de arco en píxeles. Contornos más cortos se eliminan."
        },
        "closed_contours": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Calcular longitud como contorno cerrado (0=No, 1=Sí)."
        },
        "line_thickness": {
            "default": 1,
            "min": 1,
            "max": 5,
            "step": 1,
            "description": "Grosor de línea al redibujar los contornos filtrados."
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        edge_img = inputs.get("edge_image")
        base_img = inputs.get("base_image", original_image)  # <-- MODIFICADO: fallback con advertencia
        
        if edge_img is None:
            warnings.warn(f"[MinArcLength] No se proporcionó 'edge_image', aplicando Canny a la imagen base.", stacklevel=2)
            # Si no hay edge_image, aplicar Canny a la imagen base
            if len(base_img.shape) == 3:
                gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = base_img
            edge_img = cv2.Canny(gray, 50, 150)
        
        # Asegurar que es grayscale
        if len(edge_img.shape) == 3:
            edge_img = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
        
        min_length = self.params["min_length"]
        closed = bool(self.params["closed_contours"])
        thickness = self.params["line_thickness"]
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear imagen resultado
        result = np.zeros_like(edge_img)
        
        # Filtrar y redibujar contornos que cumplan el mínimo
        for contour in contours:
            arc_length = cv2.arcLength(contour, closed=closed)
            if arc_length >= min_length:
                cv2.drawContours(result, [contour], -1, 255, thickness)
        
        # sample_image en BGR para visualización
        sample = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return {
            "filtered_edges": result,
            "sample_image": sample
        }
