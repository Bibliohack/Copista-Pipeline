"""
Filtro: ScaleHOCR
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
import re
from xml.etree import ElementTree as ET
from .base_filter import BaseFilter, FILTER_REGISTRY


class ScaleHOCR(BaseFilter):
    """Escala las coordenadas de un archivo hOCR a nuevas dimensiones de imagen"""
    
    FILTER_NAME = "ScaleHOCR"
    DESCRIPTION = "Escala coordenadas bbox en hOCR desde dimensiones originales a dimensiones de destino. Útil cuando reduces resolución de imagen pero quieres mantener el OCR."
    
    INPUTS = {
        "hocr_data": "hocr",
        "hocr_metadata": "metadata",  # De TesseractOCR
        "target_image": "image",      # Opcional: imagen destino
        "target_metadata": "metadata" # Opcional: metadata destino
    }
    
    OUTPUTS = {
        "scaled_hocr": "hocr",
        "scaled_metadata": "metadata",
        "sample_image": "image"
    }
    
    PARAMS = {
        "scale_mode": {
            "default": 0,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": "Modo: 0=Proporcional, 1=Stretch, 2=Fit (mantiene aspecto)"
        },
        "round_coords": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Redondear coordenadas a enteros (0=No, 1=Sí)"
        },
        "update_ppageno": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Actualizar dimensiones en ppageno (0=No, 1=Sí)"
        },
        "visualization_overlay": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Overlay de info en visualización (0=No, 1=Sí)"
        }
    }
    
    def _get_target_dimensions(self, target_image: Optional[np.ndarray],
                               target_metadata: Optional[Dict]) -> Optional[Tuple[int, int]]:
        """
        Obtiene dimensiones de destino desde imagen o metadata.
        
        Returns:
            (width, height) o None
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
    
    def _calculate_scale_factors(self, source_w: int, source_h: int,
                                 target_w: int, target_h: int,
                                 mode: int) -> Tuple[float, float]:
        """Calcula factores de escala según modo"""
        if mode == 1:  # Stretch
            return (target_w / source_w, target_h / source_h)
        elif mode == 2:  # Fit
            scale = min(target_w / source_w, target_h / source_h)
            return (scale, scale)
        else:  # Proporcional (default)
            return (target_w / source_w, target_h / source_h)
    
    def _scale_bbox(self, bbox_str: str, scale_x: float, scale_y: float,
                   round_coords: bool) -> str:
        """
        Escala un bbox del formato "bbox x0 y0 x1 y1"
        
        Args:
            bbox_str: String como "bbox 100 200 300 400"
            scale_x, scale_y: Factores de escala
            round_coords: Si redondear a enteros
            
        Returns:
            String escalado "bbox x0' y0' x1' y1'"
        """
        match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', bbox_str)
        if not match:
            return bbox_str
        
        x0, y0, x1, y1 = map(int, match.groups())
        
        # Escalar
        x0_new = x0 * scale_x
        y0_new = y0 * scale_y
        x1_new = x1 * scale_x
        y1_new = y1 * scale_y
        
        if round_coords:
            x0_new = round(x0_new)
            y0_new = round(y0_new)
            x1_new = round(x1_new)
            y1_new = round(y1_new)
        
        # Reconstruir bbox
        new_bbox = f"bbox {int(x0_new)} {int(y0_new)} {int(x1_new)} {int(y1_new)}"
        
        # Reemplazar en el string original (mantener otros atributos como x_wconf)
        return re.sub(r'bbox\s+\d+\s+\d+\s+\d+\s+\d+', new_bbox, bbox_str)
    
    def _update_ppageno(self, ppageno_str: str, target_w: int, target_h: int) -> str:
        """
        Actualiza las dimensiones en ppageno
        
        Formato original: "ppageno 0; bbox 0 0 2480 3508; ppageno 0"
        O: "bbox 0 0 2480 3508"
        """
        # Buscar bbox en ppageno
        match = re.search(r'bbox\s+\d+\s+\d+\s+(\d+)\s+(\d+)', ppageno_str)
        if match:
            # Reemplazar con nuevas dimensiones
            new_str = re.sub(
                r'bbox\s+\d+\s+\d+\s+\d+\s+\d+',
                f'bbox 0 0 {target_w} {target_h}',
                ppageno_str
            )
            return new_str
        return ppageno_str
    
    def _scale_hocr(self, hocr: str, scale_x: float, scale_y: float,
                   target_w: int, target_h: int, round_coords: bool,
                   update_ppageno: bool) -> str:
        """
        Escala todas las coordenadas bbox en el hOCR
        
        Args:
            hocr: String hOCR completo
            scale_x, scale_y: Factores de escala
            target_w, target_h: Dimensiones objetivo
            round_coords: Si redondear coordenadas
            update_ppageno: Si actualizar dimensiones de página
            
        Returns:
            hOCR escalado
        """
        try:
            # Parsear como XML
            root = ET.fromstring(hocr)
            
            # Recorrer todos los elementos
            for elem in root.iter():
                if 'title' in elem.attrib:
                    title = elem.attrib['title']
                    
                    # Si tiene bbox, escalarlo
                    if 'bbox' in title:
                        # Caso especial: ppageno (dimensiones de página)
                        if 'class' in elem.attrib and 'ocr_page' in elem.attrib['class']:
                            if update_ppageno:
                                elem.attrib['title'] = self._update_ppageno(title, target_w, target_h)
                        else:
                            # Escalar bbox normal
                            elem.attrib['title'] = self._scale_bbox(title, scale_x, scale_y, round_coords)
            
            # Convertir de vuelta a string
            result = ET.tostring(root, encoding='unicode', method='html')
            return result
            
        except ET.ParseError as e:
            print(f"Error parseando hOCR para escalar: {e}")
            # Fallback: regex simple (menos preciso pero funciona)
            return self._scale_hocr_regex(hocr, scale_x, scale_y, round_coords)
    
    def _scale_hocr_regex(self, hocr: str, scale_x: float, scale_y: float,
                         round_coords: bool) -> str:
        """Fallback: escala usando regex (menos robusto pero funciona si XML falla)"""
        def replace_bbox(match):
            x0, y0, x1, y1 = map(int, match.groups())
            x0_new = int(round(x0 * scale_x)) if round_coords else x0 * scale_x
            y0_new = int(round(y0 * scale_y)) if round_coords else y0 * scale_y
            x1_new = int(round(x1 * scale_x)) if round_coords else x1 * scale_x
            y1_new = int(round(y1 * scale_y)) if round_coords else y1 * scale_y
            return f"bbox {x0_new} {y0_new} {x1_new} {y1_new}"
        
        return re.sub(
            r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',
            replace_bbox,
            hocr
        )
    
    def _create_visualization(self, target_image: Optional[np.ndarray],
                             source_w: int, source_h: int,
                             target_w: int, target_h: int,
                             scale_x: float, scale_y: float,
                             show_overlay: bool) -> np.ndarray:
        """Crea visualización del escalado"""
        # Crear imagen base
        if target_image is not None:
            if len(target_image.shape) == 2:
                vis = cv2.cvtColor(target_image, cv2.COLOR_GRAY2BGR)
            else:
                vis = target_image.copy()
        else:
            # Imagen placeholder
            vis = np.ones((min(target_h, 600), min(target_w, 800), 3), dtype=np.uint8) * 240
        
        if show_overlay:
            h, w = vis.shape[:2]
            
            # Overlay semi-transparente
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 0), (w, 100), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
            
            # Información
            cv2.putText(vis, "hOCR ESCALADO", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(vis, f"Origen: {source_w}x{source_h}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(vis, f"Destino: {target_w}x{target_h}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(vis, f"Escala: {scale_x:.3f}x, {scale_y:.3f}y", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        hocr_data = inputs.get("hocr_data")
        hocr_metadata = inputs.get("hocr_metadata", {})
        target_image = inputs.get("target_image")
        target_metadata = inputs.get("target_metadata")
        
        # Validar hOCR
        if not hocr_data or not hocr_data.strip():
            error_msg = "No hOCR data provided"
            return {
                "scaled_hocr": "",
                "scaled_metadata": {
                    "error": error_msg,
                    "valid": False
                },
                "sample_image": self._create_error_image(error_msg) if not self.without_preview else None
            }
        
        # Obtener dimensiones de origen
        source_w = hocr_metadata.get("image_width")
        source_h = hocr_metadata.get("image_height")
        
        if source_w is None or source_h is None:
            error_msg = "Missing source dimensions in hocr_metadata"
            return {
                "scaled_hocr": hocr_data,  # Retornar sin escalar
                "scaled_metadata": {
                    "error": error_msg,
                    "valid": False
                },
                "sample_image": self._create_error_image(error_msg) if not self.without_preview else None
            }
        
        # Obtener dimensiones de destino
        target_dims = self._get_target_dimensions(target_image, target_metadata)
        
        if target_dims is None:
            error_msg = "No target dimensions (provide target_image or target_metadata)"
            return {
                "scaled_hocr": hocr_data,  # Retornar sin escalar
                "scaled_metadata": {
                    "error": error_msg,
                    "source_width": int(source_w),
                    "source_height": int(source_h),
                    "valid": False
                },
                "sample_image": self._create_error_image(error_msg) if not self.without_preview else None
            }
        
        target_w, target_h = target_dims
        
        # Parámetros
        scale_mode = self.params["scale_mode"]
        round_coords = bool(self.params["round_coords"])
        update_ppageno = bool(self.params["update_ppageno"])
        show_overlay = bool(self.params["visualization_overlay"])
        
        # Calcular factores de escala
        scale_x, scale_y = self._calculate_scale_factors(
            source_w, source_h, target_w, target_h, scale_mode
        )
        
        # Escalar hOCR
        scaled_hocr = self._scale_hocr(
            hocr_data, scale_x, scale_y, target_w, target_h,
            round_coords, update_ppageno
        )
        
        # Metadata
        scaled_metadata = {
            "source_width": int(source_w),
            "source_height": int(source_h),
            "target_width": int(target_w),
            "target_height": int(target_h),
            "image_width": int(target_w),  # Para compatibilidad
            "image_height": int(target_h),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "scale_mode": ["proportional", "stretch", "fit"][scale_mode],
            "valid": True
        }
        
        result = {
            "scaled_hocr": scaled_hocr,
            "scaled_metadata": scaled_metadata
        }
        
        # Visualización
        if not self.without_preview:
            sample = self._create_visualization(
                target_image, source_w, source_h, target_w, target_h,
                scale_x, scale_y, show_overlay
            )
            result["sample_image"] = sample
        
        return result
    
    def _create_error_image(self, error_msg: str) -> np.ndarray:
        """Crea imagen de error"""
        vis = np.zeros((300, 600, 3), dtype=np.uint8)
        vis[:] = (40, 40, 40)
        
        cv2.putText(vis, "ERROR ESCALANDO hOCR", (50, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mensaje en líneas
        words = error_msg.split()
        lines = []
        current = []
        
        for word in words:
            current.append(word)
            if len(' '.join(current)) > 50:
                lines.append(' '.join(current[:-1]))
                current = [word]
        if current:
            lines.append(' '.join(current))
        
        y = 160
        for line in lines[:3]:
            cv2.putText(vis, line, (50, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y += 25
        
        return vis
