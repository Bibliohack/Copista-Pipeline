"""
Filtro: TesseractOCR
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import pytesseract
from PIL import Image
from .base_filter import BaseFilter, FILTER_REGISTRY


class TesseractOCR(BaseFilter):
    """Realiza OCR usando Tesseract y genera salida hOCR"""
    
    FILTER_NAME = "TesseractOCR"
    DESCRIPTION = "Extrae texto de la imagen usando Tesseract OCR y genera archivo hOCR con información de layout y coordenadas."
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "hocr_data": "hocr",
        "text_data": "text",
        "ocr_metadata": "metadata",
        "sample_image": "image"
    }
    
    PARAMS = {
        "language": {
            "default": 0,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": "Idioma: 0=Español, 1=Inglés, 2=Español+Inglés, 3=Auto"
        },
        "psm": {
            "default": 3,
            "min": 0,
            "max": 13,
            "step": 1,
            "description": "Page Segmentation Mode (PSM). 3=Auto, 6=Bloque uniforme, 11=Línea única"
        },
        "oem": {
            "default": 3,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": "OCR Engine Mode: 0=Legacy, 1=LSTM, 2=Legacy+LSTM, 3=Default"
        },
        "dpi": {
            "default": 300,
            "min": 72,
            "max": 600,
            "step": 50,
            "description": "DPI de la imagen para OCR (mayor=mejor calidad, más lento)"
        },
        "visualize_boxes": {
            "default": 1,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": "Visualizar: 0=Original, 1=Bounding boxes, 2=Palabras coloreadas"
        },
        "min_confidence": {
            "default": 0,
            "min": 0,
            "max": 100,
            "step": 5,
            "description": "Confianza mínima para mostrar palabra (0=todas)"
        }
    }
    
    LANGUAGE_CODES = {
        0: "spa",           # Español
        1: "eng",           # Inglés
        2: "spa+eng",       # Ambos
        3: ""               # Auto-detectar
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        h, w = input_img.shape[:2]
        
        # Parámetros
        lang_code = self.LANGUAGE_CODES[self.params["language"]]
        psm = self.params["psm"]
        oem = self.params["oem"]
        dpi = self.params["dpi"]
        vis_mode = self.params["visualize_boxes"]
        min_conf = self.params["min_confidence"]
        
        # Convertir a PIL Image para pytesseract
        if len(input_img.shape) == 2:
            pil_img = Image.fromarray(input_img)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
        
        # Configurar Tesseract
        custom_config = f'--oem {oem} --psm {psm}'
        if lang_code:
            custom_config = f'-l {lang_code} {custom_config}'
        
        # Obtener hOCR
        try:
            hocr_output = pytesseract.image_to_pdf_or_hocr(
                pil_img,
                extension='hocr',
                config=custom_config
            ).decode('utf-8')
        except pytesseract.TesseractError as e:
            hocr_output = f"<!-- Error en Tesseract: {e} -->"
        
        # Obtener texto plano
        try:
            text_output = pytesseract.image_to_string(
                pil_img,
                config=custom_config
            )
        except pytesseract.TesseractError as e:
            text_output = f"Error: {e}"
        
        # Obtener datos detallados para visualización y metadata
        try:
            data = pytesseract.image_to_data(
                pil_img,
                output_type=pytesseract.Output.DICT,
                config=custom_config
            )
        except pytesseract.TesseractError as e:
            data = {
                'text': [],
                'conf': [],
                'left': [],
                'top': [],
                'width': [],
                'height': []
            }
        
        # Filtrar por confianza
        n_boxes = len(data['text'])
        words_detected = 0
        total_confidence = 0
        
        filtered_boxes = []
        for i in range(n_boxes):
            conf = int(data['conf'][i]) if data['conf'][i] != -1 else 0
            text = data['text'][i].strip()
            
            if text and conf >= min_conf:
                words_detected += 1
                total_confidence += conf
                filtered_boxes.append({
                    'text': text,
                    'confidence': conf,
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i]
                })
        
        # Metadata
        avg_confidence = (total_confidence / words_detected) if words_detected > 0 else 0
        
        metadata = {
            "image_width": int(w),
            "image_height": int(h),
            "language": lang_code if lang_code else "auto",
            "psm": psm,
            "oem": oem,
            "dpi": dpi,
            "words_detected": words_detected,
            "total_words": len([t for t in data['text'] if t.strip()]),
            "average_confidence": round(avg_confidence, 2),
            "min_confidence_filter": min_conf
        }
        
        result = {
            "hocr_data": hocr_output,
            "text_data": text_output,
            "ocr_metadata": metadata
        }
        
        # Generar visualización si no estamos en modo without_preview
        if not self.without_preview:
            sample = self._create_visualization(
                input_img, filtered_boxes, vis_mode, min_conf
            )
            result["sample_image"] = sample
        
        return result
    
    def _create_visualization(self, img: np.ndarray, boxes: List[Dict],
                             mode: int, min_conf: int) -> np.ndarray:
        """Crea visualización del OCR"""
        if len(img.shape) == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()
        
        if mode == 0:
            # Original sin modificar
            return vis
        
        elif mode == 1:
            # Bounding boxes
            for box in boxes:
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                conf = box['confidence']
                
                # Color según confianza
                if conf >= 80:
                    color = (0, 255, 0)  # Verde
                elif conf >= 50:
                    color = (0, 255, 255)  # Amarillo
                else:
                    color = (0, 0, 255)  # Rojo
                
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                
                # Label con confianza
                label = f"{conf}%"
                cv2.putText(vis, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        elif mode == 2:
            # Palabras coloreadas por confianza
            overlay = vis.copy()
            
            for box in boxes:
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                conf = box['confidence']
                
                # Color según confianza
                if conf >= 80:
                    color = (0, 255, 0)  # Verde
                elif conf >= 50:
                    color = (0, 255, 255)  # Amarillo
                else:
                    color = (0, 0, 255)  # Rojo
                
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            
            # Blend
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        
        # Info general
        cv2.putText(vis, f"Palabras: {len(boxes)} (conf>={min_conf}%)",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis
