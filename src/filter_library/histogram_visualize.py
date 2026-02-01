"""
Filtro: HistogramVisualize
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class HistogramVisualize(BaseFilter):
    """Visualiza el histograma con marcadores de picos"""
    
    FILTER_NAME = "HistogramVisualize"
    DESCRIPTION = "Genera una visualización del histograma con marcadores para picos oscuros y claros. Útil para análisis previo a normalización."
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "histogram_data": "histogram",
        "sample_image": "image"
    }
    
    PARAMS = {
        "dark_marker": {
            "default": 30,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Posición del marcador de pico oscuro."
        },
        "light_marker": {
            "default": 220,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Posición del marcador de pico claro."
        },
        "show_auto_peaks": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar picos auto-detectados (0=No, 1=Sí)."
        },
        "histogram_height": {
            "default": 300,
            "min": 150,
            "max": 500,
            "step": 50,
            "description": "Altura de la imagen del histograma."
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        # Convertir a grayscale si es necesario
        if len(input_img.shape) == 3:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_img
        
        # Calcular histograma
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        dark_marker = self.params["dark_marker"]
        light_marker = self.params["light_marker"]
        show_auto = self.params["show_auto_peaks"]
        hist_height = self.params["histogram_height"]
        
        # Normalizar para visualización
        hist_normalized = hist / hist.max() if hist.max() > 0 else hist
        
        # Crear imagen del histograma
        hist_width = 512  # 2 píxeles por bin
        margin = 50
        total_height = hist_height + 100
        total_width = hist_width + 2 * margin
        
        hist_img = np.zeros((total_height, total_width, 3), dtype=np.uint8)
        hist_img[:] = (40, 40, 40)  # Fondo gris oscuro
        
        # Dibujar histograma
        for i in range(256):
            h = int(hist_normalized[i] * (hist_height - 20))
            x = margin + i * 2
            cv2.line(hist_img, (x, hist_height), (x, hist_height - h), (200, 200, 200), 1)
        
        # Dibujar marcador de dark_peak (azul)
        x_dark = margin + dark_marker * 2
        cv2.line(hist_img, (x_dark, 10), (x_dark, hist_height), (255, 100, 0), 2)
        cv2.putText(hist_img, f"Dark:{dark_marker}", (x_dark - 30, hist_height + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)
        
        # Dibujar marcador de light_peak (amarillo)
        x_light = margin + light_marker * 2
        cv2.line(hist_img, (x_light, 10), (x_light, hist_height), (0, 255, 255), 2)
        cv2.putText(hist_img, f"Light:{light_marker}", (x_light - 35, hist_height + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        # Auto-detectar y mostrar picos si está habilitado
        if show_auto == 1:
            dark_zone = hist[:100]
            light_zone = hist[150:]
            auto_dark = int(np.argmax(dark_zone)) if dark_zone.max() > 0 else 30
            auto_light = int(150 + np.argmax(light_zone)) if light_zone.max() > 0 else 220
            
            # Buscar mínimo entre picos
            if auto_dark < auto_light:
                between_zone = hist[auto_dark:auto_light+1]
                auto_min = int(auto_dark + np.argmin(between_zone))
            else:
                auto_min = (auto_dark + auto_light) // 2
            
            # Dibujar mínimo (magenta)
            x_min = margin + auto_min * 2
            cv2.line(hist_img, (x_min, 10), (x_min, hist_height), (255, 0, 255), 1)
            
            cv2.putText(hist_img, f"Auto: Dark={auto_dark}, Min={auto_min}, Light={auto_light}",
                        (margin, hist_height + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
        
        # Dibujar eje X
        cv2.line(hist_img, (margin, hist_height), (margin + 512, hist_height), (150, 150, 150), 1)
        cv2.putText(hist_img, "0", (margin - 5, hist_height + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(hist_img, "255", (margin + 500, hist_height + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Título
        cv2.putText(hist_img, "HISTOGRAMA", (margin, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return {
            "histogram_data": {"histogram": hist.tolist(), "dark_marker": dark_marker, "light_marker": light_marker},
            "sample_image": hist_img
        }


# =============================================================================
# FIN de Filtros extraídos de page_border_pipeline.py
# =============================================================================

# =============================================================================
# Filtros basados en detectar_bordes_hough_probabilistico.py
# =============================================================================
