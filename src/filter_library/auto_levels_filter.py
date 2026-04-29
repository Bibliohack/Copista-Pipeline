"""
Filtro: AutoLevels
"""

import cv2
import numpy as np
from typing import Dict, Any
from .base_filter import BaseFilter, FILTER_REGISTRY


class AutoLevels(BaseFilter):
    """Estira el histograma al rango completo con ajuste de punto medio (gamma)"""

    FILTER_NAME = "AutoLevels"
    DESCRIPTION = (
        "Estira el histograma al rango completo: el valor mínimo pasa a 0, "
        "el máximo a 255. Recorte opcional por percentil para ignorar outliers. "
        "Ajuste de punto medio (gamma) para controlar los tonos medios."
    )

    INPUTS = {
        "input_image": "image"
    }

    OUTPUTS = {
        "normalized_image": "image",
        "sample_image": "image"
    }

    PARAMS = {
        "clip_low": {
            "default": 0.1,
            "min": 0.0,
            "max": 5.0,
            "step": 0.1,
            "description": "% de píxeles oscuros a recortar antes de estirar (0=ninguno, 1=1%)"
        },
        "clip_high": {
            "default": 0.1,
            "min": 0.0,
            "max": 5.0,
            "step": 0.1,
            "description": "% de píxeles claros a recortar antes de estirar (0=ninguno, 1=1%)"
        },
        "midpoint": {
            "default": 128,
            "min": 1,
            "max": 254,
            "step": 1,
            "description": (
                "Punto medio gamma: 128=neutro, >128=aclara tonos medios, <128=oscurece tonos medios. "
                "Equivale al slider central de Niveles en Photoshop."
            )
        },
        "output_channel": {
            "default": 0,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": (
                "Canal a normalizar: 0=luminancia (canal L en LAB, preserva color), "
                "1=escala de grises, 2=todos los canales BGR"
            )
        },
        "show_comparison": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "0=solo resultado, 1=comparación lado a lado (original vs resultado)"
        }
    }

    def _stretch_channel(self, channel: np.ndarray,
                         clip_low: float, clip_high: float,
                         midpoint: int) -> np.ndarray:
        """Estira un canal uint8 al rango completo con recorte percentil y gamma."""
        low_pct = max(0.0, min(49.9, clip_low))
        high_pct = max(0.0, min(49.9, clip_high))

        low_val = float(np.percentile(channel, low_pct))
        high_val = float(np.percentile(channel, 100.0 - high_pct))

        if high_val <= low_val:
            return channel.copy()

        c = channel.astype(np.float32)
        c = np.clip(c, low_val, high_val)
        c = (c - low_val) / (high_val - low_val)  # [0, 1]

        # Gamma desde punto medio: log(0.5) / log(midpoint/255)
        # midpoint=128 ≈ 0.502 → gamma ≈ 1.0 (neutro)
        # midpoint>128 → gamma<1 → aclara; midpoint<128 → gamma>1 → oscurece
        if midpoint != 128:
            mp = float(midpoint) / 255.0
            mp = max(1e-6, min(1.0 - 1e-6, mp))
            gamma = np.log(0.5) / np.log(mp)
            c = np.power(np.clip(c, 1e-7, 1.0), gamma)

        return np.clip(c * 255.0, 0, 255).astype(np.uint8)

    def _create_comparison(self, original: np.ndarray, result: np.ndarray) -> np.ndarray:
        if len(original.shape) == 2:
            orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            orig_bgr = original.copy()
        if len(result.shape) == 2:
            res_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            res_bgr = result.copy()

        h, w = orig_bgr.shape[:2]
        sep = 20
        canvas = np.zeros((h, w * 2 + sep, 3), dtype=np.uint8)
        canvas[:, :w] = orig_bgr
        canvas[:, w + sep:] = res_bgr
        canvas[:, w:w + sep] = (100, 100, 100)
        cv2.putText(canvas, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "AUTO LEVELS", (w + sep + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return canvas

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)

        clip_low = float(self.params["clip_low"])
        clip_high = float(self.params["clip_high"])
        midpoint = int(self.params["midpoint"])
        output_channel = int(self.params["output_channel"])
        show_comparison = int(self.params["show_comparison"])

        is_color = (len(input_img.shape) == 3 and input_img.shape[2] == 3)

        if output_channel == 0:
            if is_color:
                lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
                L, A, B = cv2.split(lab)
                L_out = self._stretch_channel(L, clip_low, clip_high, midpoint)
                lab_out = cv2.merge([L_out, A, B])
                normalized_image = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
            else:
                normalized_image = self._stretch_channel(input_img, clip_low, clip_high, midpoint)

        elif output_channel == 1:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) if is_color else input_img.copy()
            normalized_image = self._stretch_channel(gray, clip_low, clip_high, midpoint)

        else:
            if not is_color:
                input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
            channels = cv2.split(input_img)
            normalized_image = cv2.merge([
                self._stretch_channel(ch, clip_low, clip_high, midpoint)
                for ch in channels
            ])

        if self.without_preview:
            sample_image = normalized_image.copy() if len(normalized_image.shape) == 3 \
                else cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)
        elif show_comparison:
            sample_image = self._create_comparison(input_img, normalized_image)
        else:
            sample_image = normalized_image.copy() if len(normalized_image.shape) == 3 \
                else cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)

        return {
            "normalized_image": normalized_image,
            "sample_image": sample_image
        }
