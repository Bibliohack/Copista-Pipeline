"""
Filtro: FilterLinesByOrientation
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class FilterLinesByOrientation(BaseFilter):
    """Filtra líneas por orientación, separando horizontales, verticales y descartando oblicuas"""
    
    FILTER_NAME = "FilterLinesByOrientation"
    DESCRIPTION = "Clasifica líneas detectadas por Hough en horizontales y verticales según tolerancia angular. Descarta líneas oblicuas. Incluye metadata con dimensiones de imagen."
    
    INPUTS = {
        "lines_data": "lines",
        "base_image": "image"
    }
    
    OUTPUTS = {
        "horizontal_lines": "lines",
        "vertical_lines": "lines",
        "lines_metadata": "metadata",
        "sample_image": "image"
    }
    
    PARAMS = {
        "h_v_tolerance": {
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
        "oblique_color_r": {
            "default": 128,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas oblicuas (descartadas) - Rojo."
        },
        "oblique_color_g": {
            "default": 128,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas oblicuas (descartadas) - Verde."
        },
        "oblique_color_b": {
            "default": 128,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas oblicuas (descartadas) - Azul."
        },
        "line_thickness": {
            "default": 2,
            "min": 1,
            "max": 10,
            "step": 1,
            "description": "Grosor de las líneas en la visualización."
        },
        "show_oblique": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar líneas oblicuas descartadas (0=No, 1=Sí)."
        },
        "show_info": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar información de conteo (0=No, 1=Sí)."
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
        
        tolerance = self.params["h_v_tolerance"]
        thickness = self.params["line_thickness"]
        show_oblique = bool(self.params["show_oblique"])
        show_info = bool(self.params["show_info"])
        
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
        oblique_color = (
            self.params["oblique_color_b"],
            self.params["oblique_color_g"],
            self.params["oblique_color_r"]
        )
        
        horizontal_lines = []
        vertical_lines = []
        oblique_lines = []
        
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
            elif self._is_vertical(x1, y1, x2, y2, tolerance):
                vertical_lines.append(line_record)
            else:
                oblique_lines.append(line_record)
        
        # Metadata con dimensiones de imagen
        metadata = {
            "image_width": int(w),
            "image_height": int(h),
            "horizontal_count": len(horizontal_lines),
            "vertical_count": len(vertical_lines),
            "oblique_count": len(oblique_lines),
            "total_lines": len(lines_data),
            "tolerance_degrees": tolerance,
            "filtered_lines": len(horizontal_lines) + len(vertical_lines),
            "discarded_lines": len(oblique_lines)
        }
        
        result = {
            "horizontal_lines": horizontal_lines,
            "vertical_lines": vertical_lines,
            "lines_metadata": metadata
        }
        
        # Solo generar visualización si no estamos en modo without_preview
        if not self.without_preview:
            if len(base_img.shape) == 2:
                vis_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
            else:
                vis_img = base_img.copy()
            
            # Dibujar líneas oblicuas primero (si está habilitado)
            if show_oblique:
                for line in oblique_lines:
                    cv2.line(vis_img, (line["x1"], line["y1"]), 
                            (line["x2"], line["y2"]), oblique_color, max(1, thickness - 1))
            
            # Dibujar líneas clasificadas (encima)
            for line in horizontal_lines:
                cv2.line(vis_img, (line["x1"], line["y1"]), 
                        (line["x2"], line["y2"]), h_color, thickness)
            
            for line in vertical_lines:
                cv2.line(vis_img, (line["x1"], line["y1"]), 
                        (line["x2"], line["y2"]), v_color, thickness)
            
            # Agregar información si está habilitado
            if show_info:
                info_lines = [
                    f"Tolerancia: {tolerance} grados",
                    f"Horizontales: {len(horizontal_lines)}",
                    f"Verticales: {len(vertical_lines)}",
                    f"Oblicuas: {len(oblique_lines)} (descartadas)"
                ]
                
                y_offset = 30
                for line_text in info_lines:
                    text_size = cv2.getTextSize(line_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Fondo semi-transparente
                    overlay = vis_img.copy()
                    cv2.rectangle(overlay, (8, y_offset - 20), 
                                (12 + text_size[0], y_offset + 5), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, vis_img, 0.4, 0, vis_img)
                    
                    # Texto
                    cv2.putText(vis_img, line_text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    y_offset += 30
                
                # Leyenda de colores
                legend_y = vis_img.shape[0] - 90
                
                # Horizontal
                cv2.line(vis_img, (10, legend_y), (30, legend_y), h_color, thickness)
                cv2.putText(vis_img, "Horizontales", (35, legend_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Vertical
                legend_y += 25
                cv2.line(vis_img, (10, legend_y), (30, legend_y), v_color, thickness)
                cv2.putText(vis_img, "Verticales", (35, legend_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Oblicuas
                if show_oblique:
                    legend_y += 25
                    cv2.line(vis_img, (10, legend_y), (30, legend_y), oblique_color, max(1, thickness - 1))
                    cv2.putText(vis_img, "Oblicuas (descartadas)", (35, legend_y + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            result["sample_image"] = vis_img
        
        return result
