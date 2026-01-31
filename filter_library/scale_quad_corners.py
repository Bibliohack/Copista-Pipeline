"""
Filtro: ScaleQuadCorners
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from .base_filter import BaseFilter, FILTER_REGISTRY


class ScaleQuadCorners(BaseFilter):
    """Escala las coordenadas de un cuadrilátero a nuevas dimensiones de imagen"""
    
    FILTER_NAME = "ScaleQuadCorners"
    DESCRIPTION = "Escala las coordenadas de esquinas (quad_points) desde sus dimensiones originales a dimensiones de destino. Detecta automáticamente si recibe imagen o metadata como referencia de destino."
    
    INPUTS = {
        "corners": "quad_points",
        "corners_metadata": "metadata",
        "target_image": "image",  # Opcional: imagen de destino (también se usa como fondo de visualización)
        "target_metadata": "metadata",  # Opcional: metadata de destino
        "preview_image": "image"  # Opcional: imagen de fondo para visualización (si no hay target_image)
    }
    
    OUTPUTS = {
        "scaled_corners": "quad_points",
        "scaled_metadata": "metadata",
        "sample_image": "image"
    }
    
    PARAMS = {
        "scale_mode": {
            "default": 0,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": "Modo de escalado: 0=Proporcional (mantiene aspect ratio), 1=Stretch (estira a target), 2=Fit (ajusta manteniendo aspecto dentro de target)."
        },
        "round_coords": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Redondear coordenadas a enteros (0=No, 1=Sí)."
        },
        "clamp_to_bounds": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Forzar coordenadas dentro de límites de imagen (0=No, 1=Sí)."
        },
        "visualization_size": {
            "default": 800,
            "min": 400,
            "max": 1920,
            "step": 100,
            "description": "Ancho de la imagen de visualización."
        },
        "corner_radius": {
            "default": 8,
            "min": 3,
            "max": 20,
            "step": 1,
            "description": "Radio de los círculos de esquinas en visualización."
        },
        "show_info": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar información de escalado en visualización (0=No, 1=Sí)."
        }
    }
    
    def _get_target_dimensions(self, target_image: Optional[np.ndarray], 
                               target_metadata: Optional[Dict]) -> Optional[Tuple[int, int]]:
        """
        Obtiene las dimensiones de destino desde imagen o metadata.
        Prioriza target_image sobre target_metadata.
        
        Returns:
            Tuple (width, height) o None si no hay información
        """
        if target_image is not None:
            h, w = target_image.shape[:2]
            return (w, h)
        
        if target_metadata is not None:
            w = target_metadata.get("image_width")
            h = target_metadata.get("image_height")
            if w is not None and h is not None:
                return (int(w), int(h))
        
        return None
    
    def _scale_corner(self, corner: Dict, scale_x: float, scale_y: float, 
                     round_coords: bool, target_w: int, target_h: int, 
                     clamp: bool) -> Dict:
        """
        Escala una esquina individual.
        
        Args:
            corner: Dict con 'x', 'y', 'type'
            scale_x: Factor de escala en X
            scale_y: Factor de escala en Y
            round_coords: Si redondear a enteros
            target_w: Ancho de imagen destino (para clamp)
            target_h: Alto de imagen destino (para clamp)
            clamp: Si forzar dentro de límites
            
        Returns:
            Dict con coordenadas escaladas
        """
        if corner is None or "x" not in corner or "y" not in corner:
            return None
        
        x_scaled = corner["x"] * scale_x
        y_scaled = corner["y"] * scale_y
        
        if round_coords:
            x_scaled = round(x_scaled)
            y_scaled = round(y_scaled)
        
        if clamp:
            x_scaled = max(0, min(target_w - 1, x_scaled))
            y_scaled = max(0, min(target_h - 1, y_scaled))
        
        return {
            "x": int(x_scaled) if round_coords else x_scaled,
            "y": int(y_scaled) if round_coords else y_scaled,
            "type": corner.get("type", "unknown"),
            "original_x": corner["x"],
            "original_y": corner["y"]
        }
    
    def _calculate_scale_factors(self, source_w: int, source_h: int, 
                                 target_w: int, target_h: int, 
                                 mode: int) -> Tuple[float, float]:
        """
        Calcula los factores de escala según el modo.
        
        Args:
            source_w, source_h: Dimensiones originales
            target_w, target_h: Dimensiones destino
            mode: 0=Proporcional, 1=Stretch, 2=Fit
            
        Returns:
            Tuple (scale_x, scale_y)
        """
        if mode == 1:  # Stretch
            return (target_w / source_w, target_h / source_h)
        
        elif mode == 2:  # Fit (mantener aspecto, ajustar dentro de target)
            scale = min(target_w / source_w, target_h / source_h)
            return (scale, scale)
        
        else:  # Proporcional (default)
            scale_x = target_w / source_w
            scale_y = target_h / source_h
            return (scale_x, scale_y)
    
    def _create_visualization(self, scaled_corners: Dict, target_w: int, target_h: int,
                             source_w: int, source_h: int, scale_x: float, scale_y: float,
                             show_info: bool, vis_width: int, corner_radius: int,
                             background_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Crea una visualización de las esquinas escaladas.
        
        Args:
            background_image: Imagen de fondo opcional (target_image o preview_image)
        """
        # Calcular dimensiones de visualización manteniendo aspecto
        aspect = target_w / target_h
        vis_h = int(vis_width / aspect)
        
        # Crear imagen base
        if background_image is not None:
            # Usar imagen de fondo provista
            if len(background_image.shape) == 2:
                bg = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)
            else:
                bg = background_image.copy()
            
            # Redimensionar a tamaño de visualización
            vis_img = cv2.resize(bg, (vis_width, vis_h))
        else:
            # Crear fondo gris claro
            vis_img = np.ones((vis_h, vis_width, 3), dtype=np.uint8) * 240
        
        # Escala para visualización
        vis_scale_x = vis_width / target_w
        vis_scale_y = vis_h / target_h
        
        # Colores
        corner_color = (255, 0, 255)  # Magenta
        polygon_color = (0, 255, 255)  # Amarillo
        text_color = (255, 255, 255) if background_image is not None else (0, 0, 0)
        text_bg_color = (0, 0, 0) if background_image is not None else None
        
        # Recolectar esquinas válidas en orden para el polígono
        corner_order = ["top_left", "top_right", "bottom_right", "bottom_left"]
        valid_corners = []
        
        for name in corner_order:
            corner = scaled_corners.get(name)
            if corner is not None and "x" in corner:
                # Escalar a coordenadas de visualización
                vis_x = int(corner["x"] * vis_scale_x)
                vis_y = int(corner["y"] * vis_scale_y)
                valid_corners.append((vis_x, vis_y))
        
        # Dibujar polígono si tenemos las 4 esquinas
        if len(valid_corners) == 4:
            polygon = np.array(valid_corners, dtype=np.int32)
            cv2.polylines(vis_img, [polygon], True, polygon_color, 3)
        
        # Dibujar esquinas
        for name in corner_order:
            corner = scaled_corners.get(name)
            if corner is not None and "x" in corner:
                vis_x = int(corner["x"] * vis_scale_x)
                vis_y = int(corner["y"] * vis_scale_y)
                
                # Círculo de esquina
                cv2.circle(vis_img, (vis_x, vis_y), corner_radius, corner_color, -1)
                cv2.circle(vis_img, (vis_x, vis_y), corner_radius + 2, (0, 0, 0), 2)
                
                # Label con fondo si hay imagen de fondo
                label_offset = corner_radius + 5
                if "left" in name:
                    label_x = vis_x + label_offset
                else:
                    label_x = vis_x - 70
                if "top" in name:
                    label_y = vis_y + label_offset + 10
                else:
                    label_y = vis_y - label_offset
                
                # Fondo para el texto si hay imagen
                if text_bg_color is not None:
                    text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(vis_img, 
                                (label_x - 2, label_y - text_size[1] - 2),
                                (label_x + text_size[0] + 2, label_y + 2),
                                text_bg_color, -1)
                
                cv2.putText(vis_img, name, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Mostrar información de escalado
        if show_info:
            info_y = 20
            info_lines = [
                f"Source: {source_w}x{source_h}",
                f"Target: {target_w}x{target_h}",
                f"Scale: {scale_x:.3f}x, {scale_y:.3f}y",
                f"Corners: {len(valid_corners)}/4"
            ]
            
            for line in info_lines:
                # Fondo para el texto si hay imagen
                if text_bg_color is not None:
                    text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(vis_img, 
                                (8, info_y - text_size[1] - 2),
                                (12 + text_size[0], info_y + 2),
                                text_bg_color, -1)
                
                cv2.putText(vis_img, line, (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                info_y += 20
        
        return vis_img
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        corners = inputs.get("corners", {})
        corners_metadata = inputs.get("corners_metadata", {})
        target_image = inputs.get("target_image")
        target_metadata = inputs.get("target_metadata")
        preview_image = inputs.get("preview_image")
        
        # Validar que tenemos corners
        if not corners:
            # Sin corners, retornar vacío
            placeholder = np.ones((400, 600, 3), dtype=np.uint8) * 240
            cv2.putText(placeholder, "No corners data provided", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            return {
                "scaled_corners": {},
                "scaled_metadata": {
                    "error": "No corners provided",
                    "valid": False
                },
                "sample_image": placeholder
            }
        
        # Obtener dimensiones de origen desde corners_metadata
        source_w = corners_metadata.get("image_width")
        source_h = corners_metadata.get("image_height")
        
        if source_w is None or source_h is None:
            # Sin metadata de origen, retornar error
            placeholder = np.ones((400, 600, 3), dtype=np.uint8) * 240
            cv2.putText(placeholder, "Missing source metadata", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            return {
                "scaled_corners": corners,  # Retornar sin escalar
                "scaled_metadata": {
                    "error": "Missing source dimensions in corners_metadata",
                    "valid": False
                },
                "sample_image": placeholder
            }
        
        # Obtener dimensiones de destino
        target_dims = self._get_target_dimensions(target_image, target_metadata)
        
        if target_dims is None:
            # Sin dimensiones de destino, retornar sin escalar
            placeholder = np.ones((400, 600, 3), dtype=np.uint8) * 240
            cv2.putText(placeholder, "No target dimensions", (50, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(placeholder, "Provide target_image or target_metadata", (50, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            return {
                "scaled_corners": corners,  # Retornar sin escalar
                "scaled_metadata": {
                    "error": "No target dimensions provided",
                    "source_width": int(source_w),
                    "source_height": int(source_h),
                    "valid": False
                },
                "sample_image": placeholder
            }
        
        target_w, target_h = target_dims
        
        # Obtener parámetros
        scale_mode = self.params["scale_mode"]
        round_coords = bool(self.params["round_coords"])
        clamp = bool(self.params["clamp_to_bounds"])
        vis_width = self.params["visualization_size"]
        corner_radius = self.params["corner_radius"]
        show_info = bool(self.params["show_info"])
        
        # Calcular factores de escala
        scale_x, scale_y = self._calculate_scale_factors(
            source_w, source_h, target_w, target_h, scale_mode
        )
        
        # Escalar cada esquina
        scaled_corners = {}
        corner_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        
        for name in corner_names:
            corner = corners.get(name)
            if corner is not None:
                scaled_corners[name] = self._scale_corner(
                    corner, scale_x, scale_y, round_coords, 
                    target_w, target_h, clamp
                )
            else:
                scaled_corners[name] = None
        
        # Crear metadata
        valid_count = sum(1 for c in scaled_corners.values() if c is not None)
        
        scaled_metadata = {
            "source_width": int(source_w),
            "source_height": int(source_h),
            "target_width": int(target_w),
            "target_height": int(target_h),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "scale_mode": ["proportional", "stretch", "fit"][scale_mode],
            "corners_scaled": valid_count,
            "valid": valid_count == 4
        }
        
        # Si tenemos las 4 esquinas escaladas, calcular área del polígono
        if valid_count == 4:
            polygon_points = []
            for name in corner_names:
                corner = scaled_corners[name]
                if corner is not None:
                    polygon_points.append((corner["x"], corner["y"]))
            
            if len(polygon_points) == 4:
                polygon = np.array(polygon_points, dtype=np.int32)
                area = cv2.contourArea(polygon)
                area_percentage = (area / (target_w * target_h)) * 100
                
                scaled_metadata["polygon_area"] = float(area)
                scaled_metadata["polygon_area_percent"] = round(area_percentage, 2)
        
        # Determinar imagen de fondo para visualización
        # Prioridad: target_image > preview_image > None
        background_img = target_image if target_image is not None else preview_image
        
        # Crear visualización
        vis_img = self._create_visualization(
            scaled_corners, target_w, target_h, source_w, source_h,
            scale_x, scale_y, show_info, vis_width, corner_radius,
            background_img
        )
        
        return {
            "scaled_corners": scaled_corners,
            "scaled_metadata": scaled_metadata,
            "sample_image": vis_img
        }
