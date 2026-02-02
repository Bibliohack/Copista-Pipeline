"""
Filtro: RotateQuadCorners
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from .base_filter import BaseFilter, FILTER_REGISTRY


class RotateQuadCorners(BaseFilter):
    """Rota las coordenadas de un cuadrilátero alrededor del centro de la imagen"""
    
    FILTER_NAME = "RotateQuadCorners"
    DESCRIPTION = "Rota las coordenadas de las 4 esquinas de un cuadrilátero alrededor del centro de la imagen. Útil para aplicar corrección de rotación a las coordenadas detectadas."
    
    INPUTS = {
        "corners": "quad_points",
        "corners_metadata": "metadata",
        "rotation_angle": "float",  # Opcional: puede venir de CalculateRotationFromLines
        "base_image": "image"  # Para obtener dimensiones y visualización
    }
    
    OUTPUTS = {
        "rotated_corners": "quad_points",
        "rotated_metadata": "metadata",
        "sample_image": "image"
    }
    
    PARAMS = {
        "angle": {
            "default": 0,
            "min": -180,
            "max": 180,
            "step": 1,
            "description": "Ángulo de rotación en grados. Positivo=antihorario, Negativo=horario. Se ignora si viene rotation_angle de input."
        },
        "use_input_angle": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Usar ángulo del input rotation_angle (1) o del parámetro angle (0)."
        },
        "invert_angle": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Invertir el ángulo (1=Sí, 0=No). Útil para aplicar corrección de rotación a coordenadas."
        },
        "corner_radius": {
            "default": 8,
            "min": 3,
            "max": 20,
            "step": 1,
            "description": "Radio de los círculos de esquinas en visualización."
        },
        "original_color_r": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de esquinas originales - Rojo."
        },
        "original_color_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de esquinas originales - Verde."
        },
        "original_color_b": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de esquinas originales - Azul."
        },
        "rotated_color_r": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de esquinas rotadas - Rojo."
        },
        "rotated_color_g": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de esquinas rotadas - Verde."
        },
        "rotated_color_b": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de esquinas rotadas - Azul."
        },
        "line_thickness": {
            "default": 2,
            "min": 1,
            "max": 5,
            "step": 1,
            "description": "Grosor de las líneas de los polígonos."
        },
        "show_original": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar polígono original (0=No, 1=Sí)."
        },
        "show_info": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar información de rotación (0=No, 1=Sí)."
        }
    }
    
    def _rotate_point(self, point: Tuple[float, float], center: Tuple[float, float], 
                     angle_rad: float) -> Tuple[int, int]:
        """
        Rota un punto alrededor de un centro.
        
        Args:
            point: (x, y) del punto a rotar
            center: (cx, cy) centro de rotación
            angle_rad: Ángulo en radianes (positivo=antihorario)
            
        Returns:
            (x_rotated, y_rotated) como enteros
        """
        x, y = point
        cx, cy = center
        
        # Trasladar al origen
        x_translated = x - cx
        y_translated = y - cy
        
        # Aplicar rotación
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        x_rotated = x_translated * cos_a - y_translated * sin_a
        y_rotated = x_translated * sin_a + y_translated * cos_a
        
        # Trasladar de vuelta
        x_final = x_rotated + cx
        y_final = y_rotated + cy
        
        return (int(round(x_final)), int(round(y_final)))
    
    def _clamp_to_image(self, x: int, y: int, width: int, height: int) -> Tuple[int, int]:
        """Asegura que un punto esté dentro de los límites de la imagen"""
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        return (x, y)
    
    def _create_visualization(self, base_img: np.ndarray,
                            original_corners: Dict[str, Dict],
                            rotated_corners: Dict[str, Dict],
                            angle: float,
                            original_color: Tuple[int, int, int],
                            rotated_color: Tuple[int, int, int],
                            thickness: int,
                            corner_radius: int,
                            show_original: bool,
                            show_info: bool,
                            from_input: bool) -> np.ndarray:
        """Crea visualización con polígonos original y rotado"""
        
        if len(base_img.shape) == 2:
            vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            vis = base_img.copy()
        
        corner_order = ["top_left", "top_right", "bottom_right", "bottom_left"]
        
        # Dibujar polígono original si está habilitado
        if show_original:
            original_points = []
            for name in corner_order:
                corner = original_corners.get(name)
                if corner and "x" in corner and "y" in corner:
                    original_points.append((corner["x"], corner["y"]))
            
            if len(original_points) == 4:
                polygon = np.array(original_points, dtype=np.int32)
                cv2.polylines(vis, [polygon], True, original_color, thickness, cv2.LINE_AA)
                
                # Puntos originales
                for pt in original_points:
                    cv2.circle(vis, pt, corner_radius, original_color, -1)
                    cv2.circle(vis, pt, corner_radius + 2, original_color, 2)
        
        # Dibujar polígono rotado
        rotated_points = []
        for name in corner_order:
            corner = rotated_corners.get(name)
            if corner and "x" in corner and "y" in corner:
                rotated_points.append((corner["x"], corner["y"]))
        
        if len(rotated_points) == 4:
            polygon = np.array(rotated_points, dtype=np.int32)
            cv2.polylines(vis, [polygon], True, rotated_color, thickness + 1, cv2.LINE_AA)
            
            # Puntos rotados
            for pt in rotated_points:
                cv2.circle(vis, pt, corner_radius, rotated_color, -1)
                cv2.circle(vis, pt, corner_radius + 2, (0, 0, 0), 2)
        
        # Marcar centro de rotación
        h, w = vis.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
        cv2.circle(vis, (cx, cy), 8, (0, 255, 0), 2)
        cv2.putText(vis, "Centro", (cx + 12, cy + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Información
        if show_info:
            info_lines = [
                f"Rotacion: {angle:.2f} grados",
                f"Origen: {'Input' if from_input else 'Parametro'}",
                f"Esquinas rotadas: {len(rotated_points)}/4"
            ]
            
            y_offset = 30
            for line in info_lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Fondo semi-transparente
                overlay = vis.copy()
                cv2.rectangle(overlay, (8, y_offset - 20), 
                            (12 + text_size[0], y_offset + 5), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
                
                # Texto
                cv2.putText(vis, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                y_offset += 30
            
            # Leyenda
            legend_y = vis.shape[0] - 60
            
            if show_original:
                cv2.circle(vis, (15, legend_y), corner_radius, original_color, -1)
                cv2.putText(vis, "Original", (30, legend_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                legend_y += 25
            
            cv2.circle(vis, (15, legend_y), corner_radius, rotated_color, -1)
            cv2.putText(vis, "Rotado", (30, legend_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        corners_data = inputs.get("corners", {})
        corners_metadata = inputs.get("corners_metadata", {})
        input_angle = inputs.get("rotation_angle")
        base_img = inputs.get("base_image", original_image)
        
        h, w = base_img.shape[:2]
        center = (w / 2.0, h / 2.0)
        
        # Validar que tenemos corners
        corner_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        valid_input_corners = 0
        
        for name in corner_names:
            corner = corners_data.get(name)
            if corner is not None and "x" in corner and "y" in corner:
                valid_input_corners += 1
        
        if valid_input_corners == 0:
            # Error: no hay corners válidas
            error_msg = "No valid corners provided"
            placeholder = np.ones((400, 600, 3), dtype=np.uint8) * 240
            cv2.putText(placeholder, "ERROR: No corners data", (50, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(placeholder, error_msg, (50, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            return {
                "rotated_corners": {},
                "rotated_metadata": {
                    "error": error_msg,
                    "valid": False
                },
                "sample_image": placeholder
            }
        
        # Determinar ángulo a usar
        use_input = bool(self.params["use_input_angle"])
        
        if use_input and input_angle is not None:
            angle = float(input_angle)
            from_input = True
        else:
            angle = float(self.params["angle"])
            from_input = False
        
        # Invertir ángulo si está configurado
        if self.params["invert_angle"]:
            angle = -angle
        
        # Convertir a radianes
        angle_rad = np.radians(angle)
        
        # Rotar cada esquina
        rotated_corners = {}
        
        for name in corner_names:
            corner = corners_data.get(name)
            if corner is not None and "x" in corner and "y" in corner:
                # Rotar punto
                x_rot, y_rot = self._rotate_point(
                    (corner["x"], corner["y"]),
                    center,
                    angle_rad
                )
                
                # Clampear a límites de imagen
                x_rot, y_rot = self._clamp_to_image(x_rot, y_rot, w, h)
                
                rotated_corners[name] = {
                    "x": int(x_rot),
                    "y": int(y_rot),
                    "type": corner.get("type", "unknown"),
                    "original_x": corner["x"],
                    "original_y": corner["y"]
                }
            else:
                rotated_corners[name] = None
        
        # Contar esquinas válidas rotadas
        valid_rotated = sum(1 for c in rotated_corners.values() if c is not None)
        
        # Crear metadata
        rotated_metadata = {
            "image_width": int(w),
            "image_height": int(h),
            "rotation_angle": float(angle),
            "angle_source": "input" if from_input else "parameter",
            "inverted": bool(self.params["invert_angle"]),
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "corners_rotated": int(valid_rotated),
            "valid": valid_rotated == 4
        }
        
        # Calcular área del polígono rotado si tenemos las 4 esquinas
        if valid_rotated == 4:
            polygon_points = []
            for name in corner_names:
                corner = rotated_corners[name]
                if corner is not None:
                    polygon_points.append((corner["x"], corner["y"]))
            
            if len(polygon_points) == 4:
                polygon = np.array(polygon_points, dtype=np.int32)
                area = cv2.contourArea(polygon)
                area_percentage = (area / (w * h)) * 100
                
                rotated_metadata["polygon_area"] = float(area)
                rotated_metadata["polygon_area_percent"] = round(area_percentage, 2)
        
        result = {
            "rotated_corners": rotated_corners,
            "rotated_metadata": rotated_metadata
        }
        
        # Solo generar visualización si no estamos en modo without_preview
        if not self.without_preview:
            original_color = (
                self.params["original_color_b"],
                self.params["original_color_g"],
                self.params["original_color_r"]
            )
            
            rotated_color = (
                self.params["rotated_color_b"],
                self.params["rotated_color_g"],
                self.params["rotated_color_r"]
            )
            
            vis = self._create_visualization(
                base_img, corners_data, rotated_corners, angle,
                original_color, rotated_color,
                self.params["line_thickness"],
                self.params["corner_radius"],
                bool(self.params["show_original"]),
                bool(self.params["show_info"]),
                from_input
            )
            
            result["sample_image"] = vis
        
        return result
