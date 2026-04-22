"""
Filtro: RetinexFilter
"""

import cv2
import numpy as np
from typing import Dict, Any
from .base_filter import BaseFilter, FILTER_REGISTRY


class RetinexFilter(BaseFilter):
    """Corrección de iluminación mediante algoritmo Retinex (Land)"""

    FILTER_NAME = "RetinexFilter"
    DESCRIPTION = (
        "Corrección de iluminación mediante algoritmo Retinex (Land). "
        "SSR: log(I)-log(blur(I)). MSR: combinación de múltiples escalas para mayor robustez."
    )

    INPUTS = {
        "input_image": "image"
    }

    OUTPUTS = {
        "normalized_image": "image",
        "sample_image": "image"
    }

    PARAMS = {
        "method": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Método: 0=SSR (Single Scale, rápido), 1=MSR (Multi Scale, más robusto)"
        },
        "sigma": {
            "default": 100,
            "min": 10,
            "max": 300,
            "step": 10,
            "description": (
                "Sigma del blur gaussiano para SSR. "
                "En MSR se usan sigma, sigma*2 y sigma/2"
            )
        },
        "strength": {
            "default": 80,
            "min": 0,
            "max": 100,
            "step": 5,
            "description": "Intensidad de corrección: 0=imagen original, 100=Retinex completo"
        },
        "dynamic_range": {
            "default": 100,
            "min": 10,
            "max": 200,
            "step": 10,
            "description": (
                "Rango dinámico de la salida en % (100=rango completo 0-255). "
                "Controla el contraste final"
            )
        },
        "output_channel": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Canal a procesar: 0=luminancia LAB (recomendado), 1=escala de grises"
        },
        "show_comparison": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Salida del sample: 0=imagen normalizada (solo resultado), 1=comparación lado a lado (original vs resultado)"
        }
    }

    def _ssr(self, channel: np.ndarray, sigma: float) -> np.ndarray:
        """Single Scale Retinex"""
        img_float = channel.astype(np.float64) + 1.0  # evitar log(0)

        # Calcular ksize impar a partir de sigma
        ksize = int(sigma * 3) | 1  # |1 para asegurar impar
        ksize = max(3, ksize)

        blurred = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)
        blurred = np.maximum(blurred, 1.0)

        retinex = np.log(img_float) - np.log(blurred)
        return retinex

    def _msr(self, channel: np.ndarray, sigma: float) -> np.ndarray:
        """Multi Scale Retinex: promedio de 3 escalas"""
        scales = [sigma / 2, sigma, sigma * 2]
        result = np.zeros_like(channel, dtype=np.float64)
        for s in scales:
            result += self._ssr(channel, max(s, 5.0))
        return result / len(scales)

    def _normalize_retinex(self, retinex: np.ndarray, dynamic_range: int) -> np.ndarray:
        """Normaliza resultado retinex a [0, 255]"""
        # Recortar percentiles para robustez ante outliers
        p_low = np.percentile(retinex, 1)
        p_high = np.percentile(retinex, 99)
        retinex_clipped = np.clip(retinex, p_low, p_high)

        # Normalizar
        if p_high > p_low:
            normalized = (retinex_clipped - p_low) / (p_high - p_low)
        else:
            normalized = np.zeros_like(retinex)

        # Aplicar dynamic_range
        dr = dynamic_range / 100.0
        offset = (1.0 - dr) / 2.0
        normalized = normalized * dr + offset

        return (normalized * 255).astype(np.uint8)

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)

        method = self.params["method"]
        sigma = self.params["sigma"]
        strength = self.params["strength"]
        dynamic_range = self.params["dynamic_range"]
        output_channel = self.params["output_channel"]

        method_label = "SSR" if method == 0 else "MSR"

        if output_channel == 0 and len(input_img.shape) == 3:
            # LAB: procesar solo canal L
            lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            if method == 0:
                retinex = self._ssr(l, float(sigma))
            else:
                retinex = self._msr(l, float(sigma))

            corrected = self._normalize_retinex(retinex, dynamic_range)

            # Aplicar strength
            alpha = strength / 100.0
            l_final = np.clip(alpha * corrected.astype(np.float64) + (1 - alpha) * l.astype(np.float64),
                              0, 255).astype(np.uint8)

            lab_result = cv2.merge([l_final, a, b])
            result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)

            sample_orig = l
            sample_result = l_final
        else:
            # Gris
            if len(input_img.shape) == 3:
                gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = input_img.copy()

            if method == 0:
                retinex = self._ssr(gray, float(sigma))
            else:
                retinex = self._msr(gray, float(sigma))

            corrected = self._normalize_retinex(retinex, dynamic_range)

            alpha = strength / 100.0
            result = np.clip(alpha * corrected.astype(np.float64) + (1 - alpha) * gray.astype(np.float64),
                             0, 255).astype(np.uint8)

            sample_orig = gray
            sample_result = result

        show_comparison = int(self.params.get("show_comparison", 0))

        if not self.without_preview:
            if show_comparison:
                sample = self._create_comparison(sample_orig, sample_result, method_label)
            else:
                sample = cv2.cvtColor(sample_result, cv2.COLOR_GRAY2BGR)
        else:
            sample = cv2.cvtColor(sample_result, cv2.COLOR_GRAY2BGR)

        return {
            "normalized_image": result,
            "sample_image": sample
        }

    def _create_comparison(self, original: np.ndarray, processed: np.ndarray,
                           method_label: str) -> np.ndarray:
        """Crea visualización lado a lado (antes/después) del canal procesado"""
        h, w = original.shape[:2]

        canvas = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)

        orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        proc_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        canvas[:, :w] = orig_bgr
        canvas[:, w:w + 20] = (100, 100, 100)  # Separador gris 20px
        canvas[:, w + 20:] = proc_bgr

        cv2.putText(canvas, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"RETINEX ({method_label})", (w + 30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return canvas
