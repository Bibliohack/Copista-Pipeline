"""
Filtro: CLAHEFilter
"""

import cv2
import numpy as np
from typing import Dict, Any
from .base_filter import BaseFilter, FILTER_REGISTRY


class CLAHEFilter(BaseFilter):
    """Ecualización adaptativa de histograma con límite de contraste (CLAHE)"""

    FILTER_NAME = "CLAHEFilter"
    DESCRIPTION = (
        "Ecualización adaptativa de histograma con límite de contraste (CLAHE). "
        "Normaliza el contraste local operando en el canal L del espacio LAB, "
        "preservando colores originales."
    )

    INPUTS = {
        "input_image": "image"
    }

    OUTPUTS = {
        "normalized_image": "image",
        "sample_image": "image"
    }

    PARAMS = {
        "clip_limit": {
            "default": 20,
            "min": 1,
            "max": 100,
            "step": 1,
            "description": (
                "Límite de recorte del contraste x10 (ej: 20 = clip_limit 2.0). "
                "Mayor=más contraste, puede generar ruido"
            )
        },
        "tile_size": {
            "default": 8,
            "min": 2,
            "max": 32,
            "step": 2,
            "description": "Tamaño de la rejilla de tiles (NxN). Mayor=más global, menor=más local"
        },
        "strength": {
            "default": 80,
            "min": 0,
            "max": 100,
            "step": 5,
            "description": "Intensidad de corrección: 0=imagen original, 100=CLAHE completo"
        },
        "output_channel": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": (
                "Canal a procesar: 0=luminancia LAB (recomendado para imágenes color), "
                "1=escala de grises"
            )
        },
        "show_comparison": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Salida del sample: 0=imagen normalizada (solo resultado), 1=comparación lado a lado (original vs resultado)"
        }
    }

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)

        clip = self.params["clip_limit"] / 10.0
        tile = self.params["tile_size"]
        strength = self.params["strength"]
        output_channel = self.params["output_channel"]

        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))

        if output_channel == 0 and len(input_img.shape) == 3:
            # LAB: procesar solo canal L
            lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = clahe.apply(l)
            # Aplicar strength
            alpha = strength / 100.0
            l_final = np.clip(alpha * l_clahe + (1 - alpha) * l, 0, 255).astype(np.uint8)
            lab_result = cv2.merge([l_final, a, b])
            result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)
            # Para sample: canal L original vs L procesado
            sample_orig = l
            sample_result = l_final
        else:
            # Gris
            if len(input_img.shape) == 3:
                gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = input_img.copy()
            gray_clahe = clahe.apply(gray)
            alpha = strength / 100.0
            result = np.clip(alpha * gray_clahe + (1 - alpha) * gray, 0, 255).astype(np.uint8)
            sample_orig = gray
            sample_result = result

        show_comparison = int(self.params.get("show_comparison", 0))

        if not self.without_preview:
            if show_comparison:
                sample = self._create_comparison(sample_orig, sample_result)
            else:
                sample = cv2.cvtColor(sample_result, cv2.COLOR_GRAY2BGR)
        else:
            sample = cv2.cvtColor(sample_result, cv2.COLOR_GRAY2BGR)

        return {
            "normalized_image": result,
            "sample_image": sample
        }

    def _create_comparison(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """Crea visualización lado a lado (antes/después) del canal procesado"""
        h, w = original.shape[:2]

        # Canvas BGR
        canvas = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)

        # Convertir a BGR para visualización
        orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        proc_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        canvas[:, :w] = orig_bgr
        canvas[:, w:w + 20] = (100, 100, 100)  # Separador gris 20px
        canvas[:, w + 20:] = proc_bgr

        cv2.putText(canvas, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "CLAHE", (w + 30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return canvas
