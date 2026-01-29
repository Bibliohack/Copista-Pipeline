"""
Filtro: SelectBorderLines
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class SelectBorderLines(BaseFilter):
    """Selecciona las líneas extremas que forman el borde de una página"""
    
    FILTER_NAME = "SelectBorderLines"
    DESCRIPTION = "Selecciona 4 líneas de borde (top, bottom, left, right) usando lógica de clustering y márgenes. Si no hay línea válida, usa el borde de imagen."
    
    INPUTS = {
        "base_image": "image",  # <-- AÑADIDO: para visualización
        "horizontal_lines": "lines",
        "vertical_lines": "lines"
    }
    
    OUTPUTS = {
        "selected_lines": "border_lines",
        "selection_metadata": "metadata",
        "selected_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "margin_top": {
            "default": 10,
            "min": 0,
            "max": 30,
            "step": 1,
            "description": "Margen superior (% del alto). Si línea extrema está más allá, usar borde imagen. 0=desactivado."
        },
        "margin_bottom": {
            "default": 10,
            "min": 0,
            "max": 30,
            "step": 1,
            "description": "Margen inferior (% del alto). Si línea extrema está más allá, usar borde imagen. 0=desactivado."
        },
        "margin_left": {
            "default": 10,
            "min": 0,
            "max": 30,
            "step": 1,
            "description": "Margen izquierdo (% del ancho). Si línea extrema está más allá, usar borde imagen. 0=desactivado."
        },
        "margin_right": {
            "default": 10,
            "min": 0,
            "max": 30,
            "step": 1,
            "description": "Margen derecho (% del ancho). Si línea extrema está más allá, usar borde imagen. 0=desactivado."
        },
        "cluster_top": {
            "default": 7,
            "min": 0,
            "max": 20,
            "step": 1,
            "description": "Cluster superior (% del alto). Agrupa líneas cercanas y selecciona la interior. 0=desactivado."
        },
        "cluster_bottom": {
            "default": 7,
            "min": 0,
            "max": 20,
            "step": 1,
            "description": "Cluster inferior (% del alto). Agrupa líneas cercanas y selecciona la interior. 0=desactivado."
        },
        "cluster_left": {
            "default": 7,
            "min": 0,
            "max": 20,
            "step": 1,
            "description": "Cluster izquierdo (% del ancho). Agrupa líneas cercanas y selecciona la interior. 0=desactivado."
        },
        "cluster_right": {
            "default": 7,
            "min": 0,
            "max": 20,
            "step": 1,
            "description": "Cluster derecho (% del ancho). Agrupa líneas cercanas y selecciona la interior. 0=desactivado."
        },
        "line_thickness": {
            "default": 2,
            "min": 1,
            "max": 5,
            "step": 1,
            "description": "Grosor de las líneas en la visualización."
        },
        "selected_color_r": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas seleccionadas - Rojo."
        },
        "selected_color_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas seleccionadas - Verde."
        },
        "selected_color_b": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas seleccionadas - Azul."
        },
        "border_color_r": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de bordes de imagen usados - Rojo."
        },
        "border_color_g": {
            "default": 165,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de bordes de imagen usados - Verde."
        },
        "border_color_b": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de bordes de imagen usados - Azul."
        }
    }
    
    def _get_line_y_at_x(self, x1, y1, x2, y2, x):
        """Calcula la coordenada Y de una línea en una posición X dada."""
        if x2 == x1:
            return (y1 + y2) / 2
        slope = (y2 - y1) / (x2 - x1)
        return y1 + slope * (x - x1)
    
    def _get_line_x_at_y(self, x1, y1, x2, y2, y):
        """Calcula la coordenada X de una línea en una posición Y dada."""
        if y2 == y1:
            return (x1 + x2) / 2
        slope = (x2 - x1) / (y2 - y1)
        return x1 + slope * (y - y1)
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        horizontal_lines = inputs.get("horizontal_lines", [])
        vertical_lines = inputs.get("vertical_lines", [])
        base_img = inputs.get("base_image", original_image)  # <-- MODIFICADO: fallback con advertencia
        
        h, w = base_img.shape[:2]  # <-- Usar base_img, no original_image
        center_x, center_y = w // 2, h // 2
        
        # Obtener parámetros
        margin_top_pct = self.params["margin_top"]
        margin_bottom_pct = self.params["margin_bottom"]
        margin_left_pct = self.params["margin_left"]
        margin_right_pct = self.params["margin_right"]
        
        cluster_top_pct = self.params["cluster_top"]
        cluster_bottom_pct = self.params["cluster_bottom"]
        cluster_left_pct = self.params["cluster_left"]
        cluster_right_pct = self.params["cluster_right"]
        
        thickness = self.params["line_thickness"]
        
        selected_color = (
            self.params["selected_color_b"],
            self.params["selected_color_g"],
            self.params["selected_color_r"]
        )
        border_color = (
            self.params["border_color_b"],
            self.params["border_color_g"],
            self.params["border_color_r"]
        )
        
        # Calcular márgenes y clusters en píxeles
        margin_top_px = int(h * margin_top_pct / 100) if margin_top_pct > 0 else 0
        margin_bottom_px = int(h * margin_bottom_pct / 100) if margin_bottom_pct > 0 else 0
        margin_left_px = int(w * margin_left_pct / 100) if margin_left_pct > 0 else 0
        margin_right_px = int(w * margin_right_pct / 100) if margin_right_pct > 0 else 0
        
        cluster_top_px = int(h * cluster_top_pct / 100) if cluster_top_pct > 0 else 0
        cluster_bottom_px = int(h * cluster_bottom_pct / 100) if cluster_bottom_pct > 0 else 0
        cluster_left_px = int(w * cluster_left_pct / 100) if cluster_left_pct > 0 else 0
        cluster_right_px = int(w * cluster_right_pct / 100) if cluster_right_pct > 0 else 0
        
        # Calcular posición característica de cada línea
        # Para horizontales: Y en el centro de la imagen
        # Para verticales: X en el centro de la imagen
        horizontal_with_y = []
        for line in horizontal_lines:
            x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
            y_at_center = self._get_line_y_at_x(x1, y1, x2, y2, center_x)
            horizontal_with_y.append({**line, "pos": y_at_center})
        
        vertical_with_x = []
        for line in vertical_lines:
            x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
            x_at_center = self._get_line_x_at_y(x1, y1, x2, y2, center_y)
            vertical_with_x.append({**line, "pos": x_at_center})
        
        # Estructuras de resultado
        selected = {"top": None, "bottom": None, "left": None, "right": None}
        metadata = {
            "image_size": {"width": w, "height": h},
            "margins_percent": {
                "top": margin_top_pct, "bottom": margin_bottom_pct,
                "left": margin_left_pct, "right": margin_right_pct
            },
            "margins_pixels": {
                "top": margin_top_px, "bottom": margin_bottom_px,
                "left": margin_left_px, "right": margin_right_px
            },
            "clusters_percent": {
                "top": cluster_top_pct, "bottom": cluster_bottom_pct,
                "left": cluster_left_pct, "right": cluster_right_pct
            },
            "clusters_pixels": {
                "top": cluster_top_px, "bottom": cluster_bottom_px,
                "left": cluster_left_px, "right": cluster_right_px
            },
            "top_is_image_border": False,
            "bottom_is_image_border": False,
            "left_is_image_border": False,
            "right_is_image_border": False
        }
        
        # --- Seleccionar TOP (horizontal con menor Y) ---
        if horizontal_with_y:
            sorted_by_y = sorted(horizontal_with_y, key=lambda x: x["pos"])
            extreme = sorted_by_y[0]
            
            if cluster_top_px > 0:
                cluster = [l for l in sorted_by_y if abs(l["pos"] - extreme["pos"]) <= cluster_top_px]
                candidate = sorted(cluster, key=lambda x: x["pos"], reverse=True)[0]
                metadata["top_cluster_size"] = len(cluster)
            else:
                candidate = extreme
                metadata["top_cluster_size"] = 1
            
            metadata["top_extreme_y"] = extreme["pos"]
            metadata["top_selected_y"] = candidate["pos"]
            
            if margin_top_pct > 0:
                if candidate["pos"] <= margin_top_px:
                    selected["top"] = candidate
                else:
                    metadata["top_is_image_border"] = True
                    metadata["top_candidate_rejected_y"] = candidate["pos"]
            else:
                selected["top"] = candidate
        else:
            metadata["top_is_image_border"] = True
        
        # --- Seleccionar BOTTOM (horizontal con mayor Y) ---
        if horizontal_with_y:
            sorted_by_y = sorted(horizontal_with_y, key=lambda x: x["pos"], reverse=True)
            extreme = sorted_by_y[0]
            
            if cluster_bottom_px > 0:
                cluster = [l for l in sorted_by_y if abs(l["pos"] - extreme["pos"]) <= cluster_bottom_px]
                candidate = sorted(cluster, key=lambda x: x["pos"])[0]
                metadata["bottom_cluster_size"] = len(cluster)
            else:
                candidate = extreme
                metadata["bottom_cluster_size"] = 1
            
            metadata["bottom_extreme_y"] = extreme["pos"]
            metadata["bottom_selected_y"] = candidate["pos"]
            
            if margin_bottom_pct > 0:
                if candidate["pos"] >= (h - margin_bottom_px):
                    selected["bottom"] = candidate
                else:
                    metadata["bottom_is_image_border"] = True
                    metadata["bottom_candidate_rejected_y"] = candidate["pos"]
            else:
                selected["bottom"] = candidate
        else:
            metadata["bottom_is_image_border"] = True
        
        # --- Seleccionar LEFT (vertical con menor X) ---
        if vertical_with_x:
            sorted_by_x = sorted(vertical_with_x, key=lambda x: x["pos"])
            extreme = sorted_by_x[0]
            
            if cluster_left_px > 0:
                cluster = [l for l in sorted_by_x if abs(l["pos"] - extreme["pos"]) <= cluster_left_px]
                candidate = sorted(cluster, key=lambda x: x["pos"], reverse=True)[0]
                metadata["left_cluster_size"] = len(cluster)
            else:
                candidate = extreme
                metadata["left_cluster_size"] = 1
            
            metadata["left_extreme_x"] = extreme["pos"]
            metadata["left_selected_x"] = candidate["pos"]
            
            if margin_left_pct > 0:
                if candidate["pos"] <= margin_left_px:
                    selected["left"] = candidate
                else:
                    metadata["left_is_image_border"] = True
                    metadata["left_candidate_rejected_x"] = candidate["pos"]
            else:
                selected["left"] = candidate
        else:
            metadata["left_is_image_border"] = True
        
        # --- Seleccionar RIGHT (vertical con mayor X) ---
        if vertical_with_x:
            sorted_by_x = sorted(vertical_with_x, key=lambda x: x["pos"], reverse=True)
            extreme = sorted_by_x[0]
            
            if cluster_right_px > 0:
                cluster = [l for l in sorted_by_x if abs(l["pos"] - extreme["pos"]) <= cluster_right_px]
                candidate = sorted(cluster, key=lambda x: x["pos"])[0]
                metadata["right_cluster_size"] = len(cluster)
            else:
                candidate = extreme
                metadata["right_cluster_size"] = 1
            
            metadata["right_extreme_x"] = extreme["pos"]
            metadata["right_selected_x"] = candidate["pos"]
            
            if margin_right_pct > 0:
                if candidate["pos"] >= (w - margin_right_px):
                    selected["right"] = candidate
                else:
                    metadata["right_is_image_border"] = True
                    metadata["right_candidate_rejected_x"] = candidate["pos"]
            else:
                selected["right"] = candidate
        else:
            metadata["right_is_image_border"] = True
        
        # Crear imagen de visualización usando base_img
        if len(base_img.shape) == 2:
            vis_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = base_img.copy()
        
        # Dibujar líneas seleccionadas o bordes de imagen
        for name, line in selected.items():
            is_border = metadata.get(f"{name}_is_image_border", False)
            
            if line is not None:
                x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
                cv2.line(vis_img, (x1, y1), (x2, y2), selected_color, thickness + 1)
            elif is_border:
                if name == "top":
                    cv2.line(vis_img, (0, 0), (w, 0), border_color, thickness)
                elif name == "bottom":
                    cv2.line(vis_img, (0, h-1), (w, h-1), border_color, thickness)
                elif name == "left":
                    cv2.line(vis_img, (0, 0), (0, h), border_color, thickness)
                elif name == "right":
                    cv2.line(vis_img, (w-1, 0), (w-1, h), border_color, thickness)
        
        # Convertir selected a formato serializable (sin 'pos')
        selected_clean = {}
        for name, line in selected.items():
            if line is not None:
                selected_clean[name] = {
                    "x1": line["x1"], "y1": line["y1"],
                    "x2": line["x2"], "y2": line["y2"],
                    "angle": line.get("angle", 0)
                }
            else:
                selected_clean[name] = None
        
        return {
            "selected_lines": selected_clean,
            "selection_metadata": metadata,
            "selected_image": vis_img,
            "sample_image": vis_img
        }
