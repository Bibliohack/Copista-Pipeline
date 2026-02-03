"""
Filtro: HOCRtoPDF
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import tempfile
from datetime import datetime
import re
from xml.etree import ElementTree as ET

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from .base_filter import BaseFilter, FILTER_REGISTRY


class HOCRtoPDF(BaseFilter):
    """Genera PDF con capa OCR invisible desde hOCR + imagen usando reportlab"""
    
    FILTER_NAME = "HOCRtoPDF"
    DESCRIPTION = "Convierte imagen + hOCR a PDF searchable con capa de texto invisible. Requiere reportlab (pip install reportlab Pillow)."
    
    INPUTS = {
        "input_image": "image",
        "hocr_data": "hocr"
    }
    
    OUTPUTS = {
        "pdf_data": "pdf",
        "pdf_metadata": "metadata",
        "sample_image": "image"
    }
    
    PARAMS = {
        "dpi": {
            "default": 300,
            "min": 72,
            "max": 600,
            "step": 50,
            "description": "DPI del PDF resultante (debe coincidir con el usado en OCR)"
        },
        "image_format": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Formato de imagen: 0=PNG (sin pérdida), 1=JPEG (comprimido)"
        },
        "jpeg_quality": {
            "default": 85,
            "min": 50,
            "max": 100,
            "step": 5,
            "description": "Calidad JPEG (solo si image_format=1)"
        },
        "text_mode": {
            "default": 3,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": "Renderizado de texto: 0=Invisible, 1=Visible pequeño, 2=Visible normal, 3=Invisible mejorado"
        }
    }
    
    def _parse_hocr_bbox(self, title: str) -> Tuple[int, int, int, int]:
        """
        Extrae bounding box de hOCR title attribute
        
        Formato: "bbox 100 200 300 400"
        Returns: (x0, y0, x1, y1)
        """
        match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', title)
        if match:
            return tuple(map(int, match.groups()))
        return (0, 0, 0, 0)
    
    def _parse_hocr_words(self, hocr: str) -> List[Dict]:
        """
        Parsea hOCR y extrae palabras con sus coordenadas
        
        Returns:
            Lista de {text, x0, y0, x1, y1}
        """
        words = []
        
        try:
            # Parsear XML (hOCR es HTML válido como XML)
            root = ET.fromstring(hocr)
            
            # Buscar elementos ocrx_word
            for elem in root.iter():
                if 'class' in elem.attrib:
                    classes = elem.attrib['class'].split()
                    if 'ocrx_word' in classes:
                        text = elem.text or ''
                        text = text.strip()
                        
                        if text and 'title' in elem.attrib:
                            bbox = self._parse_hocr_bbox(elem.attrib['title'])
                            if bbox != (0, 0, 0, 0):
                                words.append({
                                    'text': text,
                                    'x0': bbox[0],
                                    'y0': bbox[1],
                                    'x1': bbox[2],
                                    'y1': bbox[3]
                                })
        except ET.ParseError as e:
            print(f"Error parseando hOCR: {e}")
        
        return words
    
    def _save_temp_image(self, image: np.ndarray, base_name: str) -> Path:
        """Guarda imagen temporal para el PDF"""
        temp_dir = Path(tempfile.gettempdir()) / "copista_pdf"
        temp_dir.mkdir(exist_ok=True)
        
        # Determinar formato
        if self.params["image_format"] == 0:
            img_ext = "png"
            encode_params = []
        else:
            img_ext = "jpg"
            quality = self.params["jpeg_quality"]
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        
        img_path = temp_dir / f"{base_name}.{img_ext}"
        cv2.imwrite(str(img_path), image, encode_params)
        
        return img_path
    
    def _generate_pdf_reportlab(self, img_path: Path, words: List[Dict],
                               output_path: Path, img_size: Tuple[int, int],
                               dpi: int) -> bool:
        """
        Genera PDF con reportlab
        
        Args:
            img_path: Path a la imagen
            words: Lista de palabras con coordenadas
            output_path: Path del PDF de salida
            img_size: (width, height) de la imagen original
            dpi: DPI para el PDF
        
        Returns:
            True si se generó exitosamente
        """
        try:
            img_width, img_height = img_size
            
            # Calcular tamaño de página en puntos (1 inch = 72 points)
            page_width = (img_width / dpi) * 72
            page_height = (img_height / dpi) * 72
            
            # Crear canvas
            c = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))
            
            # Dibujar imagen de fondo
            c.drawImage(str(img_path), 0, 0, width=page_width, height=page_height)
            
            # Modo de texto
            text_mode = self.params["text_mode"]
            
            if text_mode == 0:
                # Invisible - render mode 3
                c.setFillColorRGB(0, 0, 0, alpha=0)
            elif text_mode == 1:
                # Visible pequeño
                c.setFillColorRGB(0, 0, 1, alpha=0.3)
            elif text_mode == 2:
                # Visible normal
                c.setFillColorRGB(0, 0, 0)
            else:  # text_mode == 3
                # Invisible mejorado
                c.setFillColorRGB(1, 1, 1, alpha=0)  # Blanco transparente
            
            # Escala de coordenadas hOCR a puntos PDF
            scale_x = page_width / img_width
            scale_y = page_height / img_height
            
            # Dibujar cada palabra
            for word in words:
                text = word['text']
                x0, y0, x1, y1 = word['x0'], word['y0'], word['x1'], word['y1']
                
                # Convertir coordenadas
                # hOCR: origen arriba-izquierda, Y crece hacia abajo
                # PDF: origen abajo-izquierda, Y crece hacia arriba
                pdf_x = x0 * scale_x
                pdf_y = page_height - (y1 * scale_y)  # Invertir Y
                
                # Calcular tamaño de fuente aproximado
                word_width = (x1 - x0) * scale_x
                word_height = (y1 - y0) * scale_y
                
                # Estimar font size basado en altura del bbox
                font_size = word_height * 0.8
                font_size = max(1, min(72, font_size))  # Limitar entre 1 y 72
                
                # Ajustar ancho si es necesario (stretch horizontal)
                try:
                    c.setFont("Helvetica", font_size)
                    text_width = c.stringWidth(text, "Helvetica", font_size)
                    
                    if text_width > 0:
                        h_scale = (word_width / text_width) * 100
                        h_scale = max(50, min(200, h_scale))  # Limitar stretch
                    else:
                        h_scale = 100
                    
                    # Dibujar texto
                    text_obj = c.beginText(pdf_x, pdf_y)
                    text_obj.setFont("Helvetica", font_size)
                    text_obj.setHorizScale(h_scale)
                    text_obj.textLine(text)
                    c.drawText(text_obj)
                    
                except Exception as e:
                    # Si falla alguna palabra, continuar con las demás
                    continue
            
            # Guardar PDF
            c.save()
            return True
            
        except Exception as e:
            print(f"Error generando PDF con reportlab: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        hocr_data = inputs.get("hocr_data")
        
        h, w = input_img.shape[:2]
        
        # Validar reportlab
        if not REPORTLAB_AVAILABLE:
            error_msg = "reportlab no está instalado. Ejecuta: pip install reportlab Pillow"
            return {
                "pdf_data": None,
                "pdf_metadata": {
                    "error": error_msg,
                    "success": False,
                    "image_width": int(w),
                    "image_height": int(h)
                },
                "sample_image": self._create_error_image(error_msg) if not self.without_preview else None
            }
        
        # Validar hOCR
        if hocr_data is None or not hocr_data.strip():
            error_msg = "No se proporcionó hocr_data válido"
            return {
                "pdf_data": None,
                "pdf_metadata": {
                    "error": error_msg,
                    "success": False,
                    "image_width": int(w),
                    "image_height": int(h)
                },
                "sample_image": self._create_error_image(error_msg) if not self.without_preview else None
            }
        
        # Parsear palabras del hOCR
        words = self._parse_hocr_words(hocr_data)
        
        if not words:
            error_msg = "No se pudieron extraer palabras del hOCR"
            return {
                "pdf_data": None,
                "pdf_metadata": {
                    "error": error_msg,
                    "success": False,
                    "image_width": int(w),
                    "image_height": int(h),
                    "words_found": 0
                },
                "sample_image": self._create_error_image(error_msg) if not self.without_preview else None
            }
        
        # Generar nombre base único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"ocr_{timestamp}"
        
        # Guardar imagen temporal
        img_path = self._save_temp_image(input_img, base_name)
        
        # Generar PDF en temporal
        temp_pdf = Path(tempfile.gettempdir()) / "copista_pdf" / f"{base_name}.pdf"
        temp_pdf.parent.mkdir(parents=True, exist_ok=True)
        
        # Generar PDF
        dpi = self.params["dpi"]
        success = self._generate_pdf_reportlab(
            img_path, words, temp_pdf, (w, h), dpi
        )
        
        # Leer PDF como bytes
        pdf_bytes = None
        if success and temp_pdf.exists():
            with open(temp_pdf, 'rb') as f:
                pdf_bytes = f.read()
        
        # Metadata
        metadata = {
            "image_width": int(w),
            "image_height": int(h),
            "success": success,
            "dpi": dpi,
            "image_format": "PNG" if self.params["image_format"] == 0 else "JPEG",
            "words_processed": len(words)
        }
        
        if success and pdf_bytes:
            metadata["pdf_size_bytes"] = len(pdf_bytes)
            metadata["filename_suggestion"] = f"{base_name}.pdf"
        else:
            metadata["error"] = "Failed to generate PDF"
        
        # Limpiar archivos temporales
        try:
            img_path.unlink()
            if temp_pdf.exists():
                temp_pdf.unlink()
        except:
            pass
        
        result = {
            "pdf_data": pdf_bytes,
            "pdf_metadata": metadata
        }
        
        # Visualización
        if not self.without_preview:
            if success and pdf_bytes:
                sample = self._create_success_image(input_img, len(pdf_bytes), len(words))
            else:
                sample = self._create_error_image("Error generando PDF")
            result["sample_image"] = sample
        
        return result
    
    def _create_success_image(self, img: np.ndarray, pdf_size: int, 
                             word_count: int) -> np.ndarray:
        """Crea imagen de éxito con info del PDF"""
        if len(img.shape) == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()
        
        # Overlay semi-transparente
        h, w = vis.shape[:2]
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 100, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        
        # Texto
        size_kb = pdf_size / 1024
        cv2.putText(vis, "PDF GENERADO EXITOSAMENTE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Tamano: {size_kb:.1f} KB | Palabras: {word_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, "PDF en memoria (listo para guardar)", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return vis
    
    def _create_error_image(self, error_msg: str) -> np.ndarray:
        """Crea imagen de error"""
        vis = np.zeros((400, 700, 3), dtype=np.uint8)
        vis[:] = (40, 40, 40)
        
        cv2.putText(vis, "ERROR GENERANDO PDF", (50, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Dividir mensaje en líneas
        words = error_msg.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 60:
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        y = 200
        for line in lines[:4]:  # Max 4 líneas
            cv2.putText(vis, line, (50, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 30
        
        return vis
