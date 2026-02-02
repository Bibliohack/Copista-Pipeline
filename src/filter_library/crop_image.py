"""
Filtro: CropImage
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from .base_filter import BaseFilter, FILTER_REGISTRY


class CropImage(BaseFilter):
    """Recorta la imagen usando coordenadas de un rectángulo"""
    
    FILTER_NAME = "CropImage"
    DESCRIPTION = "Recorta la imagen según coordenadas de rectángulo. Puede usar datos de CalculateRectFromQuadCorners o coordenadas manuales."
    
    INPUTS = {
        "input_image": "image",
        "crop_rect": "rect",  # Opcional: de CalculateRectFromQuadCorners
        "crop_metadata": "metadata"  # Opcional: metadata del rectángulo
    }
    
    OUTPUTS = {
        "cropped_image": "image",
        "crop_metadata": "metadata",
        "sample_image": "image"
    }
    
    PARAMS = {
        "use_input_rect": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Usar rectángulo del input crop_rect (1) o coordenadas manuales (0)."
        },
        "x1": {
            "default": 0,
            "min": 0,
            "max": 5000,
            "step": 10,
            "description": "X inicial (esquina superior izquierda) - solo si use_input_rect=0."
        },
        "y1": {
            "default": 0,
            "min": 0,
            "max": 5000,
            "step": 10,
            "description": "Y inicial (esquina superior izquierda) - solo si use_input_rect=0."
        },
        "x2": {
            "default": 100,
            "min": 0,
            "max": 5000,
            "step": 10,
            "description": "X final (esquina inferior derecha) - solo si use_input_rect=0."
        },
        "y2": {
            "default": 100,
            "min": 0,
            "max": 5000,
            "step": 10,
            "description": "Y final (esquina inferior derecha) - solo si use_input_rect=0."
        },
        "clamp_to_image": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Ajustar coordenadas a límites de imagen (1=Sí, 0=No)."
        },
        "min_size": {
            "default": 10,
            "min": 1,
            "max": 100,
            "step": 1,
            "description": "Tamaño mínimo del crop (ancho y alto). Si es menor, se expande."
        },
        "show_rect_on_original": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar rectángulo sobre imagen original en preview (1=Sí, 0=No)."
        },
        "rect_color_r": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del rectángulo de preview - Rojo."
        },
        "rect_color_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del rectángulo de preview - Verde."
        },
        "rect_color_b": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del rectángulo de preview - Azul."
        },
        "rect_thickness": {
            "default": 3,
            "min": 1,
            "max": 10,
            "step": 1,
            "description": "Grosor del rectángulo de preview."
        }
    }
    
    def _validate_and_fix_rect(self, x1: int, y1: int, x2: int, y2: int,
                                img_width: int, img_height: int,
                                clamp: bool, min_size: int) -> Tuple[int, int, int, int]:
        """
        Valida y corrige las coordenadas del rectángulo.
        
        Args:
            x1, y1, x2, y2: Coordenadas del rectángulo
            img_width, img_height: Dimensiones de la imagen
            clamp: Si debe ajustar a límites de imagen
            min_size: Tamaño mínimo del rectángulo
            
        Returns:
            Tupla (x1, y1, x2, y2) validada y corregida
        """
        # Asegurar que x1 < x2 y y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Clampear a límites de imagen si está habilitado
        if clamp:
            x1 = max(0, min(img_width - 1, x1))
            y1 = max(0, min(img_height - 1, y1))
            x2 = max(0, min(img_width, x2))
            y2 = max(0, min(img_height, y2))
        
        # Asegurar tamaño mínimo
        width = x2 - x1
        height = y2 - y1
        
        if width < min_size:
            # Expandir desde el centro
            center_x = (x1 + x2) // 2
            x1 = max(0, center_x - min_size // 2)
            x2 = min(img_width, x1 + min_size)
            # Si aún es pequeño, ajustar x1
            if x2 - x1 < min_size:
                x1 = max(0, x2 - min_size)
        
        if height < min_size:
            # Expandir desde el centro
            center_y = (y1 + y2) // 2
            y1 = max(0, center_y - min_size // 2)
            y2 = min(img_height, y1 + min_size)
            # Si aún es pequeño, ajustar y1
            if y2 - y1 < min_size:
                y1 = max(0, y2 - min_size)
        
        return (x1, y1, x2, y2)
    
    def _create_preview(self, original_img: np.ndarray, 
                       cropped_img: np.ndarray,
                       x1: int, y1: int, x2: int, y2: int,
                       rect_color: Tuple[int, int, int],
                       thickness: int,
                       show_rect: bool) -> np.ndarray:
        """
        Crea preview mostrando el área recortada sobre la imagen original.
        """
        if not show_rect:
            # Solo retornar la imagen recortada
            return cropped_img
        
        # Crear visualización con rectángulo sobre original
        if len(original_img.shape) == 2:
            vis = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        else:
            vis = original_img.copy()
        
        # Dibujar rectángulo de crop
        cv2.rectangle(vis, (x1, y1), (x2, y2), rect_color, thickness)
        
        # Agregar información
        crop_w = x2 - x1
        crop_h = y2 - y1
        info_text = f"Crop: {crop_w}x{crop_h} px desde ({x1},{y1})"
        
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Fondo para texto
        overlay = vis.copy()
        cv2.rectangle(overlay, (8, 8), (12 + text_size[0], 38), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        
        cv2.putText(vis, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Combinar: imagen original arriba (reducida), crop abajo
        # Redimensionar original para que quepa arriba
        vis_h, vis_w = vis.shape[:2]
        crop_h_img, crop_w_img = cropped_img.shape[:2]
        
        # Calcular escala para que el original quepa en el ancho disponible
        max_preview_height = 300
        scale = min(1.0, max_preview_height / vis_h)
        preview_h = int(vis_h * scale)
        preview_w = int(vis_w * scale)
        
        vis_small = cv2.resize(vis, (preview_w, preview_h))
        
        # Si el crop es muy grande, también reducirlo
        max_crop_width = preview_w
        if crop_w_img > max_crop_width:
            crop_scale = max_crop_width / crop_w_img
            crop_preview_w = max_crop_width
            crop_preview_h = int(crop_h_img * crop_scale)
            crop_small = cv2.resize(cropped_img, (crop_preview_w, crop_preview_h))
        else:
            crop_small = cropped_img
        
        # Asegurar que ambas tengan el mismo ancho para vstack
        crop_h_small, crop_w_small = crop_small.shape[:2]
        
        if crop_w_small < preview_w:
            # Centrar el crop en un canvas del ancho del preview
            canvas = np.zeros((crop_h_small, preview_w, 3), dtype=np.uint8)
            canvas[:] = (128, 128, 128)  # Fondo gris
            offset_x = (preview_w - crop_w_small) // 2
            canvas[:, offset_x:offset_x+crop_w_small] = crop_small
            crop_small = canvas
        elif crop_w_small > preview_w:
            # Reducir el crop al ancho del preview
            scale = preview_w / crop_w_small
            crop_small = cv2.resize(crop_small, 
                                   (preview_w, int(crop_h_small * scale)))
        
        # Separador
        separator = np.ones((5, preview_w, 3), dtype=np.uint8) * 255
        
        # Combinar verticalmente
        combined = np.vstack([vis_small, separator, crop_small])
        
        # Agregar label "CROP" en el área del crop
        label_y = preview_h + 5 + 25
        cv2.putText(combined, "AREA RECORTADA:", (10, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(combined, "AREA RECORTADA:", (10, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return combined
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        input_rect = inputs.get("crop_rect")
        input_metadata = inputs.get("crop_metadata", {})
        
        img_h, img_w = input_img.shape[:2]
        
        # Obtener parámetros
        use_input = bool(self.params["use_input_rect"])
        clamp = bool(self.params["clamp_to_image"])
        min_size = self.params["min_size"]
        show_rect = bool(self.params["show_rect_on_original"])
        
        rect_color = (
            self.params["rect_color_b"],
            self.params["rect_color_g"],
            self.params["rect_color_r"]
        )
        thickness = self.params["rect_thickness"]
        
        # Determinar coordenadas del rectángulo
        from_input = False
        
        if use_input and input_rect is not None and isinstance(input_rect, dict):
            # Usar rectángulo del input
            x1 = input_rect.get("x1", 0)
            y1 = input_rect.get("y1", 0)
            x2 = input_rect.get("x2", img_w)
            y2 = input_rect.get("y2", img_h)
            from_input = True
        else:
            # Usar coordenadas manuales
            x1 = self.params["x1"]
            y1 = self.params["y1"]
            x2 = self.params["x2"]
            y2 = self.params["y2"]
        
        # Validar y corregir coordenadas
        x1, y1, x2, y2 = self._validate_and_fix_rect(
            x1, y1, x2, y2, img_w, img_h, clamp, min_size
        )
        
        # Realizar crop
        cropped = input_img[y1:y2, x1:x2].copy()
        
        crop_h, crop_w = cropped.shape[:2]
        
        # Crear metadata
        crop_metadata = {
            "original_width": int(img_w),
            "original_height": int(img_h),
            "crop_x1": int(x1),
            "crop_y1": int(y1),
            "crop_x2": int(x2),
            "crop_y2": int(y2),
            "crop_width": int(crop_w),
            "crop_height": int(crop_h),
            "crop_area": int(crop_w * crop_h),
            "source": "input" if from_input else "manual",
            "percentage_of_original": round((crop_w * crop_h) / (img_w * img_h) * 100, 2)
        }
        
        result = {
            "cropped_image": cropped,
            "crop_metadata": crop_metadata
        }
        
        # Solo generar visualización si no estamos en modo without_preview
        if not self.without_preview:
            preview = self._create_preview(
                input_img, cropped, x1, y1, x2, y2,
                rect_color, thickness, show_rect
            )
            result["sample_image"] = preview
        
        return result
