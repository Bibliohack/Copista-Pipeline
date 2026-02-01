"""
Filtro: OverlayLinesFilter
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class OverlayLinesFilter(BaseFilter):
    """Filtro de visualización: dibuja líneas sobre la imagen original"""
    
    FILTER_NAME = "OverlayLines"
    DESCRIPTION = "Dibuja líneas detectadas sobre una imagen base (filtro de visualización)"
    INPUTS = {
        "base_image": "image",
        "lines_data": "lines"
    }
    OUTPUTS = {
        "overlay_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "color_preset": {
            "default": 0,
            "min": 0,
            "max": 5,
            "step": 1,
            "description": "Preset de color: 0=Verde, 1=Rojo, 2=Azul, 3=Amarillo, 4=Magenta, 5=Cian"
        },
        "line_thickness": {
            "default": 2,
            "min": 1,
            "max": 10,
            "step": 1,
            "description": "Grosor de las líneas"
        },
        "alpha": {
            "default": 100,
            "min": 0,
            "max": 100,
            "step": 10,
            "description": "Transparencia (0-100)"
        }
    }
    
    COLOR_PRESETS = [
        (0, 255, 0),    # Verde
        (0, 0, 255),    # Rojo
        (255, 0, 0),    # Azul
        (0, 255, 255),  # Amarillo
        (255, 0, 255),  # Magenta
        (255, 255, 0)   # Cian
    ]
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        base_img = inputs.get("base_image", original_image)
        lines_data = inputs.get("lines_data", [])
        
        if len(base_img.shape) == 2:
            overlay = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            overlay = base_img.copy()
        
        color = self.COLOR_PRESETS[self.params["color_preset"]]
        thickness = self.params["line_thickness"]
        
        # Crear capa para las líneas
        lines_layer = np.zeros_like(overlay)
        
        for line in lines_data:
            if "x1" in line:
                # Formato HoughLinesP
                cv2.line(lines_layer, 
                        (line["x1"], line["y1"]), 
                        (line["x2"], line["y2"]), 
                        color, thickness)
            elif "rho" in line:
                # Formato HoughLines standard
                rho = line["rho"]
                theta = line["theta"]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(lines_layer, (x1, y1), (x2, y2), color, thickness)
        
        # Aplicar transparencia
        alpha = self.params["alpha"] / 100.0
        result = cv2.addWeighted(overlay, 1.0, lines_layer, alpha, 0)
        
        return {
            "overlay_image": result,
            "sample_image": result
        }
