"""
Filtro: DetectHistogramPeaks
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class DetectHistogramPeaks(BaseFilter):
    """Auto-detecta picos oscuro, claro y mínimo entre ellos del histograma"""
    
    FILTER_NAME = "DetectHistogramPeaks"
    DESCRIPTION = "Auto-detecta automáticamente los picos oscuro (fondo), claro (papel) y el mínimo entre ellos. Genera datos compatibles con NormalizeFromHistogram."
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "histogram_data": "histogram",
        "peaks_metadata": "metadata",
        "sample_image": "image"
    }
    
    PARAMS = {
        "dark_zone_max": {
            "default": 100,
            "min": 50,
            "max": 150,
            "step": 10,
            "description": "Límite superior de la zona oscura para buscar dark_peak."
        },
        "light_zone_min": {
            "default": 150,
            "min": 100,
            "max": 200,
            "step": 10,
            "description": "Límite inferior de la zona clara para buscar light_peak."
        },
        "visualization_mode": {
            "default": 0,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": "Visualización: 0=Histograma, 1=Píxeles coloreados, 2=Ambos (split)."
        },
        "histogram_height": {
            "default": 300,
            "min": 150,
            "max": 500,
            "step": 50,
            "description": "Altura de la imagen del histograma."
        },
        "pixel_highlight_mode": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Modo resaltado: 0=Solo valores exactos, 1=Rangos cercanos (±5)."
        },
        "show_counts": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar conteo de píxeles de cada pico (1=Sí, 0=No)."
        }
    }
    
    def _auto_detect_peaks(self, hist: np.ndarray, dark_zone_max: int, light_zone_min: int) -> Tuple[int, int, int]:
        """
        Auto-detecta los 3 puntos clave del histograma.
        
        Args:
            hist: Histograma de 256 valores
            dark_zone_max: Límite superior de zona oscura
            light_zone_min: Límite inferior de zona clara
        
        Returns:
            (dark_peak, min_between, light_peak)
        """
        # Buscar pico en zona oscura [0, dark_zone_max]
        dark_zone = hist[:dark_zone_max]
        dark_peak = int(np.argmax(dark_zone)) if dark_zone.max() > 0 else 30
        
        # Buscar pico en zona clara [light_zone_min, 255]
        light_zone = hist[light_zone_min:]
        light_peak = int(light_zone_min + np.argmax(light_zone)) if light_zone.max() > 0 else 220
        
        # Buscar mínimo entre picos
        if dark_peak < light_peak:
            between_zone = hist[dark_peak:light_peak+1]
            min_between = int(dark_peak + np.argmin(between_zone))
        else:
            min_between = (dark_peak + light_peak) // 2
        
        return (dark_peak, min_between, light_peak)
    
    def _create_histogram_visualization(self, hist: np.ndarray, 
                                       dark_peak: int, min_between: int, light_peak: int,
                                       hist_height: int, show_counts: bool) -> np.ndarray:
        """Crea visualización del histograma con marcadores de picos"""
        
        hist_width = 512  # 2 píxeles por bin
        margin = 50
        total_height = hist_height + 120
        total_width = hist_width + 2 * margin
        
        hist_img = np.zeros((total_height, total_width, 3), dtype=np.uint8)
        hist_img[:] = (40, 40, 40)  # Fondo gris oscuro
        
        # Normalizar para visualización
        hist_normalized = hist / hist.max() if hist.max() > 0 else hist
        
        # Dibujar barras del histograma
        for i in range(256):
            h = int(hist_normalized[i] * (hist_height - 20))
            x = margin + i * 2
            cv2.line(hist_img, (x, hist_height), (x, hist_height - h), (200, 200, 200), 1)
        
        # Dibujar marcador dark_peak (azul)
        x_dark = margin + dark_peak * 2
        cv2.line(hist_img, (x_dark, 10), (x_dark, hist_height), (255, 100, 0), 3)
        cv2.putText(hist_img, f"Dark:{dark_peak}", (x_dark - 30, hist_height + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)
        
        # Dibujar marcador mínimo (rojo/magenta)
        x_min = margin + min_between * 2
        cv2.line(hist_img, (x_min, 10), (x_min, hist_height), (128, 0, 255), 3)
        cv2.putText(hist_img, f"Min:{min_between}", (x_min - 25, hist_height + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 0, 255), 1)
        
        # Dibujar marcador light_peak (amarillo)
        x_light = margin + light_peak * 2
        cv2.line(hist_img, (x_light, 10), (x_light, hist_height), (0, 255, 255), 3)
        cv2.putText(hist_img, f"Light:{light_peak}", (x_light - 35, hist_height + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        # Dibujar eje X
        cv2.line(hist_img, (margin, hist_height), (margin + 512, hist_height), (150, 150, 150), 1)
        cv2.putText(hist_img, "0", (margin - 5, hist_height + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(hist_img, "255", (margin + 500, hist_height + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Título
        cv2.putText(hist_img, "HISTOGRAMA - PICOS AUTO-DETECTADOS", (margin, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Mostrar conteos si está habilitado
        if show_counts:
            y_counts = hist_height + 70
            cv2.putText(hist_img, f"Frecuencias:", (margin, y_counts),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            dark_count = int(hist[dark_peak])
            min_count = int(hist[min_between])
            light_count = int(hist[light_peak])
            
            cv2.putText(hist_img, f"Dark={dark_count}", (margin, y_counts + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 0), 1)
            cv2.putText(hist_img, f"Min={min_count}", (margin + 120, y_counts + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (128, 0, 255), 1)
            cv2.putText(hist_img, f"Light={light_count}", (margin + 230, y_counts + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        
        return hist_img
    
    def _create_pixel_visualization(self, gray: np.ndarray,
                                    dark_peak: int, min_between: int, light_peak: int,
                                    highlight_mode: int, show_counts: bool) -> np.ndarray:
        """
        Crea visualización coloreando píxeles según su clasificación.
        
        Args:
            gray: Imagen en escala de grises
            dark_peak, min_between, light_peak: Valores de los picos
            highlight_mode: 0=Solo exactos, 1=Rangos (±5)
        """
        h, w = gray.shape
        
        # Crear imagen RGB para colorear
        colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Crear máscaras según el modo
        if highlight_mode == 0:
            # Solo valores exactos
            mask_dark = (gray == dark_peak)
            mask_min = (gray == min_between)
            mask_light = (gray == light_peak)
        else:
            # Rangos cercanos (±5)
            tolerance = 5
            mask_dark = np.abs(gray.astype(int) - dark_peak) <= tolerance
            mask_min = np.abs(gray.astype(int) - min_between) <= tolerance
            mask_light = np.abs(gray.astype(int) - light_peak) <= tolerance
        
        # Colorear píxeles
        # Dark peak = Azul (255, 100, 0) en BGR
        colored[mask_dark] = (255, 100, 0)
        
        # Mínimo = Rojo/Magenta (128, 0, 255) en BGR
        colored[mask_min] = (128, 0, 255)
        
        # Light peak = Amarillo (0, 255, 255) en BGR
        colored[mask_light] = (0, 255, 255)
        
        # Agregar leyenda
        legend_height = 80
        legend = np.zeros((legend_height, w, 3), dtype=np.uint8)
        legend[:] = (40, 40, 40)
        
        y = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Título
        mode_text = "exactos" if highlight_mode == 0 else "rangos (±5)"
        cv2.putText(legend, f"PIXELES COLOREADOS ({mode_text}):", (10, y), font, 0.5, (255, 255, 255), 1)
        y += 25
        
        # Conteos
        if show_counts:
            dark_count = np.sum(mask_dark)
            min_count = np.sum(mask_min)
            light_count = np.sum(mask_light)
            
            # Cuadrados de color + texto
            # Dark
            cv2.rectangle(legend, (10, y-10), (30, y+5), (255, 100, 0), -1)
            cv2.putText(legend, f"Dark={dark_peak} ({dark_count} px)", (35, y+2),
                       font, 0.4, (255, 100, 0), 1)
            
            # Min
            cv2.rectangle(legend, (210, y-10), (230, y+5), (128, 0, 255), -1)
            cv2.putText(legend, f"Min={min_between} ({min_count} px)", (235, y+2),
                       font, 0.4, (128, 0, 255), 1)
            
            # Light
            cv2.rectangle(legend, (410, y-10), (430, y+5), (0, 255, 255), -1)
            cv2.putText(legend, f"Light={light_peak} ({light_count} px)", (435, y+2),
                       font, 0.4, (0, 255, 255), 1)
        else:
            # Solo cuadrados sin conteos
            cv2.rectangle(legend, (10, y-10), (30, y+5), (255, 100, 0), -1)
            cv2.putText(legend, f"Dark={dark_peak}", (35, y+2), font, 0.4, (255, 100, 0), 1)
            
            cv2.rectangle(legend, (150, y-10), (170, y+5), (128, 0, 255), -1)
            cv2.putText(legend, f"Min={min_between}", (175, y+2), font, 0.4, (128, 0, 255), 1)
            
            cv2.rectangle(legend, (290, y-10), (310, y+5), (0, 255, 255), -1)
            cv2.putText(legend, f"Light={light_peak}", (315, y+2), font, 0.4, (0, 255, 255), 1)
        
        # Combinar imagen coloreada + leyenda
        result = np.vstack([colored, legend])
        
        return result
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        # Convertir a grayscale si es necesario
        if len(input_img.shape) == 3:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_img.copy()
        
        h, w = gray.shape
        
        # Obtener parámetros
        dark_zone_max = self.params["dark_zone_max"]
        light_zone_min = self.params["light_zone_min"]
        vis_mode = self.params["visualization_mode"]
        hist_height = self.params["histogram_height"]
        highlight_mode = self.params["pixel_highlight_mode"]
        show_counts = bool(self.params["show_counts"])
        
        # Calcular histograma
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        # Auto-detectar picos
        dark_peak, min_between, light_peak = self._auto_detect_peaks(hist, dark_zone_max, light_zone_min)
        
        # Generar histogram_data compatible con NormalizeFromHistogram
        histogram_data = {
            "histogram": hist.tolist(),
            "dark_marker": int(dark_peak),
            "light_marker": int(light_peak)
        }
        
        # Crear metadata
        peaks_metadata = {
            "image_width": int(w),
            "image_height": int(h),
            "dark_peak": int(dark_peak),
            "min_between": int(min_between),
            "light_peak": int(light_peak),
            "dark_peak_frequency": int(hist[dark_peak]),
            "min_between_frequency": int(hist[min_between]),
            "light_peak_frequency": int(hist[light_peak]),
            "dark_zone_max": dark_zone_max,
            "light_zone_min": light_zone_min
        }
        
        result = {
            "histogram_data": histogram_data,
            "peaks_metadata": peaks_metadata
        }
        
        # Solo generar visualización si no estamos en modo without_preview
        if not self.without_preview:
            if vis_mode == 0:
                # Solo histograma
                sample = self._create_histogram_visualization(
                    hist, dark_peak, min_between, light_peak, hist_height, show_counts
                )
            elif vis_mode == 1:
                # Solo píxeles coloreados
                sample = self._create_pixel_visualization(
                    gray, dark_peak, min_between, light_peak, highlight_mode, show_counts
                )
            else:
                # Ambos (split vertical)
                hist_vis = self._create_histogram_visualization(
                    hist, dark_peak, min_between, light_peak, hist_height, show_counts
                )
                pixel_vis = self._create_pixel_visualization(
                    gray, dark_peak, min_between, light_peak, highlight_mode, show_counts
                )
                
                # Redimensionar histograma para que coincida en ancho con la imagen
                h_hist, w_hist = hist_vis.shape[:2]
                h_pix, w_pix = pixel_vis.shape[:2]
                
                if w_hist != w_pix:
                    hist_vis = cv2.resize(hist_vis, (w_pix, h_hist))
                
                # Combinar verticalmente
                sample = np.vstack([hist_vis, pixel_vis])
            
            result["sample_image"] = sample
        
        return result
