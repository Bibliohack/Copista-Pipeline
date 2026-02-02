"""
Filtro: CalculateRectFromQuadCorners
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from .base_filter import BaseFilter, FILTER_REGISTRY


class CalculateRectFromQuadCorners(BaseFilter):
    """Calcula un rectángulo de crop a partir de las 4 esquinas de un cuadrilátero"""
    
    FILTER_NAME = "CalculateRectFromQuadCorners"
    DESCRIPTION = "Calcula coordenadas de un rectángulo para crop basado en las esquinas de un cuadrilátero. El rectángulo envuelve al polígono con opción de reducirlo hacia adentro (inset)."
    
    INPUTS = {
        "corners": "quad_points",
        "corners_metadata": "metadata",
        "base_image": "image"  # Para visualización y obtener dimensiones
    }
    
    OUTPUTS = {
        "crop_rect": "rect",
        "crop_metadata": "metadata",
        "sample_image": "image"
    }
    
    PARAMS = {
        "inset": {
            "default": 0,
            "min": 0,
            "max": 100,
            "step": 1,
            "description": "Reducción del rectángulo desde cada borde en píxeles. 0=Bounding completo, >0=Contraído hacia adentro."
        },
        "rect_color_r": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del rectángulo - Rojo."
        },
        "rect_color_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del rectángulo - Verde."
        },
        "rect_color_b": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del rectángulo - Azul."
        },
        "polygon_color_r": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del polígono original - Rojo."
        },
        "polygon_color_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del polígono original - Verde."
        },
        "polygon_color_b": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del polígono original - Azul."
        },
        "line_thickness": {
            "default": 3,
            "min": 1,
            "max": 10,
            "step": 1,
            "description": "Grosor de las líneas."
        },
        "show_info": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar información de dimensiones (0=No, 1=Sí)."
        }
    }
    
    def _get_bounding_rect(self, corners: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """
        Calcula el rectángulo que contiene completamente al polígono.
        
        Returns:
            Tuple (x1, y1, x2, y2)
        """
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        
        return (min(xs), min(ys), max(xs), max(ys))
    
    def _apply_inset(self, rect: Tuple[int, int, int, int], 
                     inset: int) -> Tuple[int, int, int, int]:
        """
        Aplica un inset (reducción) al rectángulo desde todos los bordes.
        
        Args:
            rect: (x1, y1, x2, y2)
            inset: Píxeles a reducir desde cada borde
            
        Returns:
            Rectángulo reducido (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = rect
        
        # Aplicar inset
        x1 += inset
        y1 += inset
        x2 -= inset
        y2 -= inset
        
        # Asegurar que no colapse (mínimo 10x10)
        if x2 - x1 < 10:
            cx = (x1 + x2) // 2
            x1 = cx - 5
            x2 = cx + 5
        
        if y2 - y1 < 10:
            cy = (y1 + y2) // 2
            y1 = cy - 5
            y2 = cy + 5
        
        return (x1, y1, x2, y2)
    
    def _clamp_to_image(self, rect: Tuple[int, int, int, int], 
                        width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Asegura que el rectángulo no exceda los límites de la imagen.
        
        Args:
            rect: (x1, y1, x2, y2)
            width: Ancho de imagen
            height: Alto de imagen
            
        Returns:
            Rectángulo clampeado (x1, y1, x2, y2)
        """
        x1 = max(0, min(width - 1, rect[0]))
        y1 = max(0, min(height - 1, rect[1]))
        x2 = max(0, min(width - 1, rect[2]))
        y2 = max(0, min(height - 1, rect[3]))
        
        # Asegurar que x1 < x2 y y1 < y2
        if x1 >= x2:
            x2 = x1 + 1
        if y1 >= y2:
            y2 = y1 + 1
        
        return (x1, y1, x2, y2)
    
    def _create_visualization(self, base_img: np.ndarray, 
                             corners_dict: Dict[str, Dict],
                             crop_rect: Tuple[int, int, int, int],
                             rect_color: Tuple[int, int, int],
                             polygon_color: Tuple[int, int, int],
                             thickness: int,
                             show_info: bool,
                             inset: int) -> np.ndarray:
        """Crea visualización con polígono original y rectángulo de crop"""
        
        if len(base_img.shape) == 2:
            vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            vis = base_img.copy()
        
        # Reconstruir polígono en el orden correcto para dibujarlo
        corner_order = ["top_left", "top_right", "bottom_right", "bottom_left"]
        polygon_points = []
        
        for name in corner_order:
            corner = corners_dict.get(name)
            if corner and "x" in corner and "y" in corner:
                polygon_points.append((corner["x"], corner["y"]))
        
        # Dibujar polígono original si tenemos los 4 puntos
        if len(polygon_points) == 4:
            polygon = np.array(polygon_points, dtype=np.int32)
            cv2.polylines(vis, [polygon], True, polygon_color, thickness)
            
            # Dibujar puntos del polígono
            for corner in polygon_points:
                cv2.circle(vis, corner, 5, polygon_color, -1)
        
        # Dibujar rectángulo de crop
        x1, y1, x2, y2 = crop_rect
        cv2.rectangle(vis, (x1, y1), (x2, y2), rect_color, thickness)
        
        # Información
        if show_info:
            rect_width = x2 - x1
            rect_height = y2 - y1
            rect_area = rect_width * rect_height
            
            if len(polygon_points) == 4:
                polygon = np.array(polygon_points, dtype=np.int32)
                polygon_area = cv2.contourArea(polygon)
                coverage = (rect_area / polygon_area * 100) if polygon_area > 0 else 0
            else:
                coverage = 0
            
            # Fondo para texto
            info_lines = [
                f"Inset: {inset} px",
                f"Rect: {rect_width}x{rect_height} px",
                f"Area: {rect_area} px2",
                f"Cobertura: {coverage:.1f}%"
            ]
            
            y_offset = 30
            for line in info_lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Fondo semi-transparente
                overlay = vis.copy()
                cv2.rectangle(overlay, (8, y_offset - 20), 
                            (12 + text_size[0], y_offset + 5), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
                
                # Texto
                cv2.putText(vis, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                y_offset += 30
            
            # Leyenda de colores
            legend_y = vis.shape[0] - 60
            
            # Polígono
            cv2.rectangle(vis, (10, legend_y), (30, legend_y + 15), polygon_color, -1)
            cv2.putText(vis, "Poligono original", (35, legend_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Rectángulo
            legend_y += 25
            cv2.rectangle(vis, (10, legend_y), (30, legend_y + 15), rect_color, -1)
            cv2.putText(vis, "Rectangulo crop", (35, legend_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        corners_data = inputs.get("corners", {})
        corners_metadata = inputs.get("corners_metadata", {})
        base_img = inputs.get("base_image", original_image)
        
        h, w = base_img.shape[:2]
        
        # Validar que tenemos 4 esquinas
        corner_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        valid_corners = []
        
        for name in corner_names:
            corner = corners_data.get(name)
            if corner is None or "x" not in corner or "y" not in corner:
                # Error: no tenemos las 4 esquinas
                error_msg = f"Missing or invalid corner: {name}"
                placeholder = np.ones((400, 600, 3), dtype=np.uint8) * 240
                cv2.putText(placeholder, "ERROR: Invalid corners data", (50, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(placeholder, error_msg, (50, 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                return {
                    "crop_rect": {},
                    "crop_metadata": {
                        "error": error_msg,
                        "valid": False
                    },
                    "sample_image": placeholder
                }
            
            valid_corners.append((corner["x"], corner["y"]))
        
        # Obtener parámetros
        inset = self.params["inset"]
        thickness = self.params["line_thickness"]
        show_info = bool(self.params["show_info"])
        
        rect_color = (
            self.params["rect_color_b"],
            self.params["rect_color_g"],
            self.params["rect_color_r"]
        )
        
        polygon_color = (
            self.params["polygon_color_b"],
            self.params["polygon_color_g"],
            self.params["polygon_color_r"]
        )
        
        # Calcular bounding rect
        crop_rect = self._get_bounding_rect(valid_corners)
        
        # Aplicar inset si es necesario
        if inset > 0:
            crop_rect = self._apply_inset(crop_rect, inset)
        
        # Clampear a límites de imagen
        crop_rect = self._clamp_to_image(crop_rect, w, h)
        
        x1, y1, x2, y2 = crop_rect
        
        # Calcular metadata
        rect_width = x2 - x1
        rect_height = y2 - y1
        rect_area = rect_width * rect_height
        
        polygon = np.array(valid_corners, dtype=np.int32)
        polygon_area = cv2.contourArea(polygon)
        
        crop_metadata = {
            "image_width": int(w),
            "image_height": int(h),
            "rect_width": int(rect_width),
            "rect_height": int(rect_height),
            "rect_area": int(rect_area),
            "polygon_area": float(polygon_area),
            "coverage_percent": round((rect_area / polygon_area * 100), 2) if polygon_area > 0 else 0,
            "inset": int(inset),
            "valid": True
        }
        
        # Preparar output del rectángulo
        crop_rect_output = {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "width": int(rect_width),
            "height": int(rect_height)
        }
        
        result = {
            "crop_rect": crop_rect_output,
            "crop_metadata": crop_metadata
        }
        
        # Solo generar visualización si no estamos en modo without_preview
        if not self.without_preview:
            vis = self._create_visualization(
                base_img, corners_data, crop_rect,
                rect_color, polygon_color, thickness, show_info, inset
            )
            result["sample_image"] = vis
        
        return result
