"""
Filtro: HoughLinesFilter - MEJORADO
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY
import warnings

class HoughLinesFilter(BaseFilter):
    """Detecta líneas usando la transformada de Hough"""
    
    FILTER_NAME = "HoughLines"
    DESCRIPTION = "Detecta líneas rectas usando la transformada de Hough. Incluye metadata con dimensiones de imagen."
    
    INPUTS = {
        "edge_image": "image",
        "base_image": "image"
    }
    
    OUTPUTS = {
        "lines_data": "lines",
        "lines_metadata": "metadata",  # ✅ NUEVO
        "sample_image": "image"
    }
    
    PARAMS = {
        "method": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Método: 0=Standard, 1=Probabilístico"
        },
        "rho": {
            "default": 1,
            "min": 1,
            "max": 10,
            "step": 1,
            "description": "Resolución de distancia en píxeles"
        },
        "theta_divisor": {
            "default": 180,
            "min": 30,
            "max": 360,
            "step": 10,
            "description": "Divisor para theta (pi/divisor)"
        },
        "threshold": {
            "default": 100,
            "min": 10,
            "max": 300,
            "step": 10,
            "description": "Umbral de acumulador"
        },
        "min_line_length": {
            "default": 50,
            "min": 10,
            "max": 200,
            "step": 10,
            "description": "Longitud mínima de línea (solo probabilístico)"
        },
        "max_line_gap": {
            "default": 10,
            "min": 1,
            "max": 50,
            "step": 5,
            "description": "Máximo gap entre segmentos (solo probabilístico)"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        edge_img = inputs.get("edge_image")
        base_img = inputs.get("base_image", original_image)
        
        if edge_img is None:
            warnings.warn(f"[HoughLines] No se proporcionó 'edge_image', aplicando Canny a la imagen base.", stacklevel=2)
            if len(base_img.shape) == 3:
                edge_img = cv2.Canny(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), 50, 150)
            else:
                edge_img = cv2.Canny(base_img, 50, 150)
        
        h, w = base_img.shape[:2]  # ✅ Obtener dimensiones
        
        rho = self.params["rho"]
        theta = np.pi / self.params["theta_divisor"]
        threshold = self.params["threshold"]
        
        if len(base_img.shape) == 3:
            sample = base_img.copy()
        else:
            sample = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        
        lines_data = []
        method_used = "standard" if self.params["method"] == 0 else "probabilistic"
        
        if self.params["method"] == 0:
            # Hough Standard
            lines = cv2.HoughLines(edge_img, rho, theta, threshold)
            if lines is not None:
                for line in lines:
                    rho_val, theta_val = line[0]
                    a = np.cos(theta_val)
                    b = np.sin(theta_val)
                    x0 = a * rho_val
                    y0 = b * rho_val
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(sample, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    lines_data.append({"rho": rho_val, "theta": theta_val})
        else:
            # Hough Probabilístico
            lines = cv2.HoughLinesP(
                edge_img, rho, theta, threshold,
                minLineLength=self.params["min_line_length"],
                maxLineGap=self.params["max_line_gap"]
            )
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(sample, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    lines_data.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        
        # ✅ NUEVO: Metadata con dimensiones e información del método
        metadata = {
            "image_width": int(w),
            "image_height": int(h),
            "total_lines": len(lines_data),
            "method": method_used,
            "rho": rho,
            "theta_degrees": float(np.degrees(theta)),
            "threshold": threshold
        }
        
        if self.params["method"] == 1:
            metadata["min_line_length"] = self.params["min_line_length"]
            metadata["max_line_gap"] = self.params["max_line_gap"]
        
        return {
            "lines_data": lines_data,
            "lines_metadata": metadata,  # ✅ NUEVO OUTPUT
            "sample_image": sample
        }
