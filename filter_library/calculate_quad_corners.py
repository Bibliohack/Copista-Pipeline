"""
Filtro: CalculateQuadCorners
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class CalculateQuadCorners(BaseFilter):
    """Calcula las 4 esquinas de un cuadrilátero a partir de líneas de borde"""
    
    FILTER_NAME = "CalculateQuadCorners"
    DESCRIPTION = "Calcula las 4 esquinas (top_left, top_right, bottom_left, bottom_right) intersectando las líneas de borde seleccionadas. Genera polígono de recorte."
    
    INPUTS = {
        "base_image": "image",  # <-- AÑADIDO: para visualización
        "selected_lines": "border_lines",
        "selection_metadata": "metadata"
    }
    
    OUTPUTS = {
        "corners": "quad_points",
        "corners_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "corner_radius": {
            "default": 10,
            "min": 3,
            "max": 30,
            "step": 1,
            "description": "Radio de los círculos que marcan las esquinas."
        },
        "corner_color_r": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de las esquinas - Rojo."
        },
        "corner_color_g": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de las esquinas - Verde."
        },
        "corner_color_b": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de las esquinas - Azul."
        },
        "polygon_color_r": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del polígono - Rojo."
        },
        "polygon_color_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del polígono - Verde."
        },
        "polygon_color_b": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del polígono - Azul."
        },
        "polygon_thickness": {
            "default": 3,
            "min": 1,
            "max": 10,
            "step": 1,
            "description": "Grosor del polígono."
        },
        "draw_labels": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Dibujar etiquetas de esquinas (0=No, 1=Sí)."
        }
    }
    
    def _line_intersection(self, line1, line2):
        """
        Calcula la intersección de dos líneas definidas por puntos.
        line1 y line2 son dicts con x1, y1, x2, y2
        Retorna (x, y) o None si son paralelas.
        """
        x1, y1 = line1["x1"], line1["y1"]
        x2, y2 = line1["x2"], line1["y2"]
        x3, y3 = line2["x1"], line2["y1"]
        x4, y4 = line2["x2"], line2["y2"]
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (int(round(x)), int(round(y)))
    
    def _get_line_y_at_x(self, line, x):
        """Calcula Y de una línea en posición X."""
        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
        if x2 == x1:
            return (y1 + y2) / 2
        slope = (y2 - y1) / (x2 - x1)
        return y1 + slope * (x - x1)
    
    def _get_line_x_at_y(self, line, y):
        """Calcula X de una línea en posición Y."""
        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
        if y2 == y1:
            return (x1 + x2) / 2
        slope = (x2 - x1) / (y2 - y1)
        return x1 + slope * (y - y1)
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        selected_lines = inputs.get("selected_lines", {})
        metadata = inputs.get("selection_metadata", {})
        base_img = inputs.get("base_image", original_image)  # <-- MODIFICADO: fallback con advertencia
        
        h, w = base_img.shape[:2]  # <-- Usar base_img, no original_image
        
        radius = self.params["corner_radius"]
        thickness = self.params["polygon_thickness"]
        draw_labels = self.params["draw_labels"]
        
        corner_color = (
            self.params["corner_color_b"],
            self.params["corner_color_g"],
            self.params["corner_color_r"]
        )
        polygon_color = (
            self.params["polygon_color_b"],
            self.params["polygon_color_g"],
            self.params["polygon_color_r"]
        )
        
        # Definir las 4 esquinas y sus líneas componentes
        corner_defs = [
            ("top_left", "top", "left"),
            ("top_right", "top", "right"),
            ("bottom_left", "bottom", "left"),
            ("bottom_right", "bottom", "right")
        ]
        
        corners = {}
        
        for corner_name, h_name, v_name in corner_defs:
            h_line = selected_lines.get(h_name)
            v_line = selected_lines.get(v_name)
            h_is_border = metadata.get(f"{h_name}_is_image_border", False)
            v_is_border = metadata.get(f"{v_name}_is_image_border", False)
            
            if h_line is not None and v_line is not None:
                # Ambas son líneas detectadas: calcular intersección
                intersection = self._line_intersection(h_line, v_line)
                if intersection is not None:
                    corners[corner_name] = {"x": intersection[0], "y": intersection[1], "type": "intersection"}
                else:
                    corners[corner_name] = None
            elif h_is_border and v_is_border:
                # Ambas son bordes de imagen: usar esquina de imagen
                if corner_name == "top_left":
                    corners[corner_name] = {"x": 0, "y": 0, "type": "image_corner"}
                elif corner_name == "top_right":
                    corners[corner_name] = {"x": w - 1, "y": 0, "type": "image_corner"}
                elif corner_name == "bottom_left":
                    corners[corner_name] = {"x": 0, "y": h - 1, "type": "image_corner"}
                elif corner_name == "bottom_right":
                    corners[corner_name] = {"x": w - 1, "y": h - 1, "type": "image_corner"}
            elif h_is_border and v_line is not None:
                # Horizontal es borde, vertical es línea
                y_border = 0 if h_name == "top" else h - 1
                x_at_border = self._get_line_x_at_y(v_line, y_border)
                corners[corner_name] = {"x": int(round(x_at_border)), "y": y_border, "type": "mixed_h_border"}
            elif v_is_border and h_line is not None:
                # Vertical es borde, horizontal es línea
                x_border = 0 if v_name == "left" else w - 1
                y_at_border = self._get_line_y_at_x(h_line, x_border)
                corners[corner_name] = {"x": x_border, "y": int(round(y_at_border)), "type": "mixed_v_border"}
            else:
                corners[corner_name] = None
        
        # Crear imagen de visualización usando base_img
        if len(base_img.shape) == 2:
            vis_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = base_img.copy()
        
        # Recolectar esquinas válidas en orden para el polígono
        corner_order = ["top_left", "top_right", "bottom_right", "bottom_left"]
        valid_corners = []
        
        for name in corner_order:
            if corners.get(name) is not None:
                valid_corners.append((corners[name]["x"], corners[name]["y"]))
        
        # Dibujar polígono si tenemos las 4 esquinas
        if len(valid_corners) == 4:
            polygon = np.array(valid_corners, dtype=np.int32)
            cv2.polylines(vis_img, [polygon], True, polygon_color, thickness)
            
            # Calcular área
            area = cv2.contourArea(polygon)
            area_percentage = (area / (w * h)) * 100
            
            # Agregar info al metadata
            corners["_polygon_area"] = float(area)
            corners["_polygon_area_percent"] = round(area_percentage, 2)
            corners["_valid"] = True
        else:
            corners["_valid"] = False
            corners["_corners_found"] = len(valid_corners)
        
        # Dibujar esquinas
        for name in corner_order:
            corner = corners.get(name)
            if corner is not None and "x" in corner:
                pt = (corner["x"], corner["y"])
                cv2.circle(vis_img, pt, radius, corner_color, -1)
                cv2.circle(vis_img, pt, radius + 3, corner_color, 2)
                
                if draw_labels == 1:
                    # Ajustar posición del label según la esquina
                    if "left" in name:
                        label_x = pt[0] + radius + 5
                    else:
                        label_x = pt[0] - radius - 80
                    if "top" in name:
                        label_y = pt[1] + radius + 20
                    else:
                        label_y = pt[1] - radius - 5
                    
                    cv2.putText(vis_img, name, (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, corner_color, 1)
        
        return {
            "corners": corners,
            "corners_image": vis_img,
            "sample_image": vis_img
        }


# =============================================================================
# FIN de Filtros basados en detectar_bordes_hough_probabilistico.py
# =============================================================================

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
