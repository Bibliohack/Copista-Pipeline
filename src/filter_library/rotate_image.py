"""
Filtro: RotateImage
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class RotateImage(BaseFilter):
    """Rota la imagen por un ángulo especificado manteniendo las dimensiones originales"""
    
    FILTER_NAME = "RotateImage"
    DESCRIPTION = "Rota la imagen alrededor de su centro por un ángulo dado. Mantiene las dimensiones originales, rellenando con color de fondo las áreas vacías."
    
    INPUTS = {
        "input_image": "image",
        "rotation_angle": "float"  # Opcional: puede venir de otro filtro
    }
    
    OUTPUTS = {
        "rotated_image": "image",
        "rotation_metadata": "metadata",
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
        "interpolation": {
            "default": 1,
            "min": 0,
            "max": 4,
            "step": 1,
            "description": "Interpolación: 0=NEAREST (rápido), 1=LINEAR (default), 2=CUBIC (calidad), 3=AREA, 4=LANCZOS4 (máxima calidad)."
        },
        "background_r": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de fondo - Rojo."
        },
        "background_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de fondo - Verde."
        },
        "background_b": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de fondo - Azul."
        },
        "invert_angle": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Invertir el ángulo (1=Sí, 0=No). Útil para corregir rotación detectada."
        },
        "show_info": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar información de rotación en la imagen (0=No, 1=Sí)."
        },
        "show_center": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar cruz en el centro de rotación (0=No, 1=Sí)."
        }
    }
    
    INTERPOLATION_METHODS = {
        0: cv2.INTER_NEAREST,
        1: cv2.INTER_LINEAR,
        2: cv2.INTER_CUBIC,
        3: cv2.INTER_AREA,
        4: cv2.INTER_LANCZOS4
    }
    
    INTERPOLATION_NAMES = {
        0: "NEAREST",
        1: "LINEAR",
        2: "CUBIC",
        3: "AREA",
        4: "LANCZOS4"
    }
    
    def _rotate_image(self, image: np.ndarray, angle: float, 
                     background_color: Tuple[int, int, int],
                     interpolation: int) -> np.ndarray:
        """
        Rota la imagen manteniendo las dimensiones originales.
        
        Args:
            image: Imagen a rotar
            angle: Ángulo en grados (positivo=antihorario, negativo=horario)
            background_color: Color RGB para rellenar áreas vacías
            interpolation: Método de interpolación cv2.INTER_*
            
        Returns:
            Imagen rotada
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Crear matriz de rotación
        # Nota: OpenCV usa ángulos en sentido antihorario
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Aplicar rotación manteniendo dimensiones originales
        rotated = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (w, h),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=background_color
        )
        
        return rotated
    
    def _draw_center_cross(self, image: np.ndarray, size: int = 20, 
                          thickness: int = 2, color: Tuple[int, int, int] = (0, 255, 0)):
        """Dibuja una cruz en el centro de la imagen"""
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Línea horizontal
        cv2.line(image, (cx - size, cy), (cx + size, cy), color, thickness)
        # Línea vertical
        cv2.line(image, (cx, cy - size), (cx, cy + size), color, thickness)
        # Círculo central
        cv2.circle(image, (cx, cy), 3, color, -1)
    
    def _add_info_overlay(self, image: np.ndarray, angle: float, 
                         interpolation_name: str, from_input: bool):
        """Agrega información de rotación a la imagen"""
        info_lines = [
            f"Rotacion: {angle:.2f} grados",
            f"Interpolacion: {interpolation_name}",
            f"Origen: {'Input' if from_input else 'Parametro'}"
        ]
        
        y_offset = 30
        for line in info_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Fondo semi-transparente
            overlay = image.copy()
            cv2.rectangle(overlay, (8, y_offset - 20), 
                        (12 + text_size[0], y_offset + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # Texto
            cv2.putText(image, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset += 30
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        input_angle = inputs.get("rotation_angle")
        
        h, w = input_img.shape[:2]
        
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
        
        # Obtener parámetros
        interpolation_idx = self.params["interpolation"]
        interpolation = self.INTERPOLATION_METHODS.get(interpolation_idx, cv2.INTER_LINEAR)
        interpolation_name = self.INTERPOLATION_NAMES.get(interpolation_idx, "LINEAR")
        
        show_info = bool(self.params["show_info"])
        show_center = bool(self.params["show_center"])
        
        # Color de fondo
        background_color = (
            self.params["background_b"],
            self.params["background_g"],
            self.params["background_r"]
        )
        
        # Aplicar rotación
        rotated = self._rotate_image(input_img, angle, background_color, interpolation)
        
        # Crear metadata
        rotation_metadata = {
            "image_width": int(w),
            "image_height": int(h),
            "rotation_angle": float(angle),
            "angle_source": "input" if from_input else "parameter",
            "interpolation_method": interpolation_name,
            "background_color": {
                "r": self.params["background_r"],
                "g": self.params["background_g"],
                "b": self.params["background_b"]
            },
            "inverted": bool(self.params["invert_angle"])
        }
        
        result = {
            "rotated_image": rotated,
            "rotation_metadata": rotation_metadata
        }
        
        # Solo generar visualización si no estamos en modo without_preview
        if not self.without_preview:
            sample = rotated.copy()
            
            # Mostrar cruz en el centro si está habilitado
            if show_center:
                self._draw_center_cross(sample)
            
            # Agregar información si está habilitada
            if show_info:
                self._add_info_overlay(sample, angle, interpolation_name, from_input)
            
            result["sample_image"] = sample
        
        return result
