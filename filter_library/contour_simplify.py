"""
Filtro: ContourSimplify
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class ContourSimplify(BaseFilter):
    """Detecta y simplifica contornos usando approxPolyDP"""
    
    FILTER_NAME = "ContourSimplify"
    DESCRIPTION = "Detecta contornos, los simplifica con approxPolyDP y genera visualización con vértices. Útil para detección de formas geométricas."
    
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
            "default": 0,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": "Modo de recuperación: 0=EXTERNAL, 1=LIST, 2=CCOMP, 3=TREE"
        },
        "epsilon_factor": {
            "default": 20,
            "min": 1,
            "max": 100,
            "step": 1,
            "description": "Factor de simplificación: epsilon = perímetro / este_valor. Mayor = más simplificación."
        },
        "min_area": {
            "default": 1000,
            "min": 0,
            "max": 50000,
            "step": 100,
            "description": "Área mínima del contorno para mostrarlo."
        },
        "draw_vertices": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Dibujar vértices del polígono simplificado (0=No, 1=Sí)."
        },
        "vertex_radius": {
            "default": 5,
            "min": 1,
            "max": 15,
            "step": 1,
            "description": "Radio de los círculos de vértices."
        },
        "contour_thickness": {
            "default": 2,
            "min": 1,
            "max": 5,
            "step": 1,
            "description": "Grosor de línea del contorno."
        },
        "contour_color_r": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del contorno - componente Rojo."
        },
        "contour_color_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del contorno - componente Verde."
        },
        "contour_color_b": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del contorno - componente Azul."
        },
        "vertex_color_r": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de vértices - componente Rojo."
        },
        "vertex_color_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de vértices - componente Verde."
        },
        "vertex_color_b": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de vértices - componente Azul."
        }
    }
    
    MODES = [
        cv2.RETR_EXTERNAL,
        cv2.RETR_LIST,
        cv2.RETR_CCOMP,
        cv2.RETR_TREE
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
        
        mode = self.MODES[self.params["mode"]]
        epsilon_factor = max(1, self.params["epsilon_factor"])
        min_area = self.params["min_area"]
        draw_vertices = self.params["draw_vertices"]
        vertex_radius = self.params["vertex_radius"]
        thickness = self.params["contour_thickness"]
        
        contour_color = (
            self.params["contour_color_b"],
            self.params["contour_color_g"],
            self.params["contour_color_r"]
        )
        vertex_color = (
            self.params["vertex_color_b"],
            self.params["vertex_color_g"],
            self.params["vertex_color_r"]
        )
        
        # Encontrar contornos
        contours, hierarchy = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear imagen para dibujar usando input_img (no original_image)
        if len(input_img.shape) == 3:
            contour_img = input_img.copy()
        else:
            contour_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        
        # Preparar datos de contornos
        contours_data = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Simplificar contorno
                perimeter = cv2.arcLength(contour, True)
                epsilon = perimeter / epsilon_factor
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Dibujar contorno simplificado
                cv2.drawContours(contour_img, [approx], -1, contour_color, thickness)
                
                # Dibujar vértices si está habilitado
                if draw_vertices == 1:
                    for point in approx:
                        cv2.circle(contour_img, tuple(point[0]), vertex_radius, vertex_color, -1)
                
                # Guardar datos
                x, y, w, h = cv2.boundingRect(approx)
                contours_data.append({
                    "area": area,
                    "perimeter": perimeter,
                    "num_vertices": len(approx),
                    "bounding_box": (x, y, w, h),
                    "simplified_points": approx.tolist(),
                    "original_points": contour.tolist()
                })
        
        return {
            "contours_data": contours_data,
            "contour_image": contour_img,
            "sample_image": contour_img
        }
