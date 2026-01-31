"""
Filtros mejorados con metadata de dimensiones de imagen
========================================================

Todos los filtros que generan datos numéricos (líneas, contornos) ahora
incluyen un output adicional "*_metadata" con las dimensiones de la imagen
de referencia.

Esto permite:
- Escalar coordenadas entre diferentes resoluciones
- Validar que coordenadas estén dentro de límites
- Contextualizar métricas como áreas y perímetros
"""

# ==============================================================================
# 1. ClassifyLinesByAngle - MEJORADO
# ==============================================================================

"""
Filtro: ClassifyLinesByAngle
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class ClassifyLinesByAngle(BaseFilter):
    """Clasifica líneas en horizontales, verticales y otras según su ángulo"""
    
    FILTER_NAME = "ClassifyLinesByAngle"
    DESCRIPTION = "Clasifica líneas detectadas por Hough en horizontales, verticales y otras según tolerancia angular. Incluye metadata con dimensiones de imagen."
    
    INPUTS = {
        "base_image": "image",
        "lines_data": "lines"
    }
    
    OUTPUTS = {
        "horizontal_lines": "lines",
        "vertical_lines": "lines",
        "other_lines": "lines",
        "lines_metadata": "metadata",  # ✅ NUEVO
        "classified_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "angle_tolerance": {
            "default": 15,
            "min": 1,
            "max": 45,
            "step": 1,
            "description": "Tolerancia en grados para clasificar como horizontal (cerca de 0°/180°) o vertical (cerca de 90°)."
        },
        "horizontal_color_r": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas horizontales - Rojo."
        },
        "horizontal_color_g": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas horizontales - Verde."
        },
        "horizontal_color_b": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas horizontales - Azul."
        },
        "vertical_color_r": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas verticales - Rojo."
        },
        "vertical_color_g": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas verticales - Verde."
        },
        "vertical_color_b": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas verticales - Azul."
        },
        "line_thickness": {
            "default": 2,
            "min": 1,
            "max": 5,
            "step": 1,
            "description": "Grosor de las líneas en la visualización."
        }
    }
    
    def _get_line_angle(self, x1, y1, x2, y2):
        """Calcula el ángulo de una línea en grados (0-180)."""
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle < 0:
            angle += 180
        return angle
    
    def _is_horizontal(self, x1, y1, x2, y2, tolerance):
        """Determina si una línea es aproximadamente horizontal."""
        angle = self._get_line_angle(x1, y1, x2, y2)
        return angle < tolerance or angle > (180 - tolerance)
    
    def _is_vertical(self, x1, y1, x2, y2, tolerance):
        """Determina si una línea es aproximadamente vertical."""
        angle = self._get_line_angle(x1, y1, x2, y2)
        return abs(angle - 90) < tolerance
    
    def _convert_to_points_format(self, line, img_shape):
        """
        Convierte una línea al formato (x1, y1, x2, y2).
        Soporta formato HoughLinesP (ya tiene puntos) y HoughLines (rho, theta).
        """
        if "x1" in line:
            return (line["x1"], line["y1"], line["x2"], line["y2"])
        elif "rho" in line:
            rho = line["rho"]
            theta = line["theta"]
            h, w = img_shape[:2]
            
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            length = max(h, w) * 2
            x1 = int(x0 + length * (-b))
            y1 = int(y0 + length * (a))
            x2 = int(x0 - length * (-b))
            y2 = int(y0 - length * (a))
            
            return (x1, y1, x2, y2)
        else:
            return None
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        lines_data = inputs.get("lines_data", [])
        base_img = inputs.get("base_image", original_image)
        
        h, w = base_img.shape[:2]
        
        tolerance = self.params["angle_tolerance"]
        thickness = self.params["line_thickness"]
        
        h_color = (
            self.params["horizontal_color_b"],
            self.params["horizontal_color_g"],
            self.params["horizontal_color_r"]
        )
        v_color = (
            self.params["vertical_color_b"],
            self.params["vertical_color_g"],
            self.params["vertical_color_r"]
        )
        
        horizontal_lines = []
        vertical_lines = []
        other_lines = []
        
        if len(base_img.shape) == 2:
            vis_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = base_img.copy()
        
        for line in lines_data:
            points = self._convert_to_points_format(line, base_img.shape)
            if points is None:
                continue
            
            x1, y1, x2, y2 = points
            angle = self._get_line_angle(x1, y1, x2, y2)
            
            line_record = {
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "angle": float(angle)
            }
            
            if self._is_horizontal(x1, y1, x2, y2, tolerance):
                horizontal_lines.append(line_record)
                cv2.line(vis_img, (x1, y1), (x2, y2), h_color, thickness)
            elif self._is_vertical(x1, y1, x2, y2, tolerance):
                vertical_lines.append(line_record)
                cv2.line(vis_img, (x1, y1), (x2, y2), v_color, thickness)
            else:
                other_lines.append(line_record)
                cv2.line(vis_img, (x1, y1), (x2, y2), (128, 128, 128), 1)
        
        cv2.putText(vis_img, f"H:{len(horizontal_lines)} V:{len(vertical_lines)} Other:{len(other_lines)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # ✅ NUEVO: Metadata con dimensiones de imagen
        metadata = {
            "image_width": int(w),
            "image_height": int(h),
            "horizontal_count": len(horizontal_lines),
            "vertical_count": len(vertical_lines),
            "other_count": len(other_lines),
            "total_lines": len(lines_data),
            "angle_tolerance": tolerance
        }
        
        return {
            "horizontal_lines": horizontal_lines,
            "vertical_lines": vertical_lines,
            "other_lines": other_lines,
            "lines_metadata": metadata,  # ✅ NUEVO OUTPUT
            "classified_image": vis_img,
            "sample_image": vis_img
        }
