"""
Filtro: HistogramFilter
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class HistogramFilter(BaseFilter):
    """Calcula y visualiza el histograma de la imagen"""
    
    FILTER_NAME = "Histogram"
    DESCRIPTION = "Calcula el histograma de la imagen y genera una visualización"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "histogram_data": "histogram",
        "sample_image": "image"
    }
    PARAMS = {
        "bins": {
            "default": 256,
            "min": 16,
            "max": 256,
            "step": 16,
            "description": "Número de bins del histograma"
        },
        "show_channels": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Mostrar canales separados (1) o combinado (0)"
        },
        "height": {
            "default": 300,
            "min": 100,
            "max": 500,
            "step": 50,
            "description": "Altura de la imagen del histograma"
        },
        "normalize": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Normalizar histograma (0=No, 1=Sí)"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        bins = self.params["bins"]
        hist_height = self.params["height"]
        hist_width = 512
        
        # Crear imagen para el histograma
        hist_img = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
        
        histogram_data = {}
        
        if len(input_img.shape) == 3 and self.params["show_channels"] == 1:
            # Histograma por canales (BGR)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR
            for i, color in enumerate(colors):
                hist = cv2.calcHist([input_img], [i], None, [bins], [0, 256])
                if self.params["normalize"]:
                    cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)
                histogram_data[f"channel_{i}"] = hist.flatten()
                
                bin_width = hist_width // bins
                for j in range(1, bins):
                    pt1 = (bin_width * (j-1), hist_height - int(hist[j-1]))
                    pt2 = (bin_width * j, hist_height - int(hist[j]))
                    cv2.line(hist_img, pt1, pt2, color, 2)
        else:
            # Histograma de intensidad (grayscale)
            if len(input_img.shape) == 3:
                gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = input_img
            
            hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
            if self.params["normalize"]:
                cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)
            histogram_data["intensity"] = hist.flatten()
            
            bin_width = hist_width // bins
            for j in range(1, bins):
                pt1 = (bin_width * (j-1), hist_height - int(hist[j-1]))
                pt2 = (bin_width * j, hist_height - int(hist[j]))
                cv2.line(hist_img, pt1, pt2, (255, 255, 255), 2)
        
        return {
            "histogram_data": histogram_data,
            "sample_image": hist_img
        }
