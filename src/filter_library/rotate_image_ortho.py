"""
Filtro: RotateImageOrtho
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class RotateImageOrtho(BaseFilter):
    """Rota la imagen en ángulos ortogonales (90°, -90°, 180°) sin pérdida de calidad"""
    
    FILTER_NAME = "RotateImageOrtho"
    DESCRIPTION = "Rota la imagen en múltiplos de 90 grados. Usa operaciones exactas sin interpolación, preservando calidad. Las dimensiones pueden cambiar según la rotación."
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "rotated_image": "image",
        "rotation_metadata": "metadata",
        "sample_image": "image"
    }
    
    PARAMS = {
        "rotation": {
            "default": 0,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": "Rotación: 0=0° (sin cambio), 1=90° antihorario, 2=180°, 3=270° antihorario (-90°)."
        },
        "show_info": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar información de rotación en la imagen (0=No, 1=Sí)."
        }
    }
    
    ROTATION_ANGLES = {
        0: 0,
        1: 90,
        2: 180,
        3: 270  # -90
    }
    
    ROTATION_NAMES = {
        0: "0° (sin cambio)",
        1: "90° antihorario",
        2: "180°",
        3: "270° antihorario (-90°)"
    }
    
    def _rotate_ortho(self, image: np.ndarray, rotation: int) -> np.ndarray:
        """
        Rota la imagen en ángulos ortogonales sin interpolación.
        
        Args:
            image: Imagen a rotar
            rotation: 0=0°, 1=90°, 2=180°, 3=270° (-90°)
            
        Returns:
            Imagen rotada
        """
        if rotation == 0:
            # Sin rotación
            return image.copy()
        elif rotation == 1:
            # 90° antihorario = cv2.ROTATE_90_COUNTERCLOCKWISE
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == 2:
            # 180° = cv2.ROTATE_180
            return cv2.rotate(image, cv2.ROTATE_180)
        elif rotation == 3:
            # 270° antihorario = 90° horario = cv2.ROTATE_90_CLOCKWISE
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            # Valor inválido, retornar sin cambios
            return image.copy()
    
    def _add_info_overlay(self, image: np.ndarray, rotation: int, 
                         original_dims: Tuple[int, int], new_dims: Tuple[int, int]):
        """Agrega información de rotación a la imagen"""
        angle = self.ROTATION_ANGLES[rotation]
        name = self.ROTATION_NAMES[rotation]
        orig_h, orig_w = original_dims
        new_h, new_w = new_dims
        
        dims_changed = (orig_w != new_w) or (orig_h != new_h)
        
        info_lines = [
            f"Rotacion: {name}",
            f"Original: {orig_w}x{orig_h}",
            f"Nuevo: {new_w}x{new_h}"
        ]
        
        if dims_changed:
            info_lines.append("Dimensiones cambiadas")
        
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
        
        orig_h, orig_w = input_img.shape[:2]
        
        # Obtener parámetros
        rotation = self.params["rotation"]
        show_info = bool(self.params["show_info"])
        
        # Validar rotation
        if rotation not in [0, 1, 2, 3]:
            rotation = 0
        
        # Aplicar rotación
        rotated = self._rotate_ortho(input_img, rotation)
        
        new_h, new_w = rotated.shape[:2]
        
        # Crear metadata
        rotation_metadata = {
            "original_width": int(orig_w),
            "original_height": int(orig_h),
            "rotated_width": int(new_w),
            "rotated_height": int(new_h),
            "rotation_code": int(rotation),
            "rotation_angle": int(self.ROTATION_ANGLES[rotation]),
            "rotation_name": self.ROTATION_NAMES[rotation],
            "dimensions_changed": (orig_w != new_w) or (orig_h != new_h)
        }
        
        result = {
            "rotated_image": rotated,
            "rotation_metadata": rotation_metadata
        }
        
        # Solo generar visualización si no estamos en modo without_preview
        if not self.without_preview:
            sample = rotated.copy()
            
            # Agregar información si está habilitada
            if show_info:
                self._add_info_overlay(sample, rotation, (orig_h, orig_w), (new_h, new_w))
            
            result["sample_image"] = sample
        
        return result
