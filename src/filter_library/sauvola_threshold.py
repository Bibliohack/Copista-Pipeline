"""
Filtro: SauvolaThreshold
"""

import cv2
import numpy as np
from typing import Dict, Any
from .base_filter import BaseFilter, FILTER_REGISTRY


class SauvolaThreshold(BaseFilter):
    """Binarización adaptativa local por el método de Sauvola (1999)"""

    FILTER_NAME = "SauvolaThreshold"
    DESCRIPTION = (
        "Binarización adaptativa local (Sauvola 1999). El umbral de cada píxel depende "
        "de la media Y la desviación estándar de su vecindad local. En zonas de fondo "
        "liso (std baja) el umbral sube evitando binarizar ruido; sobre texto (std alta) "
        "el umbral baja capturando bien los caracteres. Superior a los métodos adaptativos "
        "simples para documentos con fondo no uniforme o papel envejecido."
    )

    INPUTS = {
        "input_image": "image"
    }

    OUTPUTS = {
        "threshold_image": "image",
        "sample_image": "image"
    }

    PARAMS = {
        "window_size": {
            "default": 25,
            "min": 3,
            "max": 199,
            "step": 2,
            "description": "Tamaño de la ventana local (impar). Mayor = más contexto, mejor para texto grande"
        },
        "k": {
            "default": 0.2,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "description": "Sensibilidad a la varianza local. Menor k = umbral más bajo = más agresivo"
        },
        "r": {
            "default": 128,
            "min": 1,
            "max": 255,
            "step": 1,
            "description": "Rango dinámico de la desviación estándar (128 para imágenes de 8 bits)"
        },
        "invert": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "0=texto negro/fondo blanco, 1=texto blanco/fondo negro"
        },
        "show_comparison": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "0=solo resultado, 1=comparación lado a lado (original vs binarizado)"
        }
    }

    def _sauvola(self, gray: np.ndarray, window_size: int, k: float, r: float) -> np.ndarray:
        if window_size % 2 == 0:
            window_size += 1

        gray_f = gray.astype(np.float64)

        # Media local via box filter
        mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(window_size, window_size),
                             normalize=True, borderType=cv2.BORDER_REPLICATE)

        # Media de cuadrados para calcular varianza: Var = E[X²] - E[X]²
        mean_sq = cv2.boxFilter(gray_f * gray_f, ddepth=-1, ksize=(window_size, window_size),
                                normalize=True, borderType=cv2.BORDER_REPLICATE)

        variance = np.maximum(mean_sq - mean * mean, 0.0)
        std = np.sqrt(variance)

        # Umbral Sauvola: T = mean * (1 + k * (std/R - 1))
        threshold = mean * (1.0 + k * (std / r - 1.0))

        result = np.zeros_like(gray, dtype=np.uint8)
        result[gray_f >= threshold] = 255

        return result

    def _create_comparison(self, original: np.ndarray, result: np.ndarray) -> np.ndarray:
        orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR) if len(original.shape) == 2 else original
        res_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR) if len(result.shape) == 2 else result

        h, w = orig_bgr.shape[:2]
        sep = 20
        canvas = np.zeros((h, w * 2 + sep, 3), dtype=np.uint8)
        canvas[:, :w] = orig_bgr
        canvas[:, w + sep:] = res_bgr
        canvas[:, w:w + sep] = (100, 100, 100)
        cv2.putText(canvas, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "SAUVOLA", (w + sep + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return canvas

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)

        window_size = int(self.params["window_size"])
        k = float(self.params["k"])
        r = float(self.params["r"])
        invert = int(self.params["invert"])
        show_comparison = int(self.params["show_comparison"])

        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) if len(input_img.shape) == 3 else input_img.copy()

        binarized = self._sauvola(gray, window_size, k, r)

        if invert:
            binarized = cv2.bitwise_not(binarized)

        if self.without_preview:
            sample_image = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
        elif show_comparison:
            sample_image = self._create_comparison(gray, binarized)
        else:
            sample_image = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)

        return {
            "threshold_image": binarized,
            "sample_image": sample_image
        }
