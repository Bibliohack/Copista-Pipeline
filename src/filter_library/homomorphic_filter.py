"""
Filtro: HomomorphicFilter
"""

import cv2
import numpy as np
from typing import Dict, Any
from .base_filter import BaseFilter, FILTER_REGISTRY


class HomomorphicFilter(BaseFilter):
    """Corrección de iluminación no uniforme mediante filtrado homomórfico"""

    FILTER_NAME = "HomomorphicFilter"
    DESCRIPTION = (
        "Corrección de iluminación no uniforme mediante filtrado homomórfico. "
        "Aplica log → FFT → filtro paso alto → IFFT → exp para separar iluminación de reflectancia."
    )

    INPUTS = {
        "input_image": "image"
    }

    OUTPUTS = {
        "normalized_image": "image",
        "sample_image": "image"
    }

    PARAMS = {
        "cutoff_frequency": {
            "default": 30,
            "min": 1,
            "max": 100,
            "step": 1,
            "description": "Frecuencia de corte como % del radio máximo. Menor=elimina más iluminación"
        },
        "gamma_low": {
            "default": 3,
            "min": 1,
            "max": 20,
            "step": 1,
            "description": "Ganancia para frecuencias bajas (iluminación) x10. Ej: 3 = 0.3. Menor=más supresión"
        },
        "gamma_high": {
            "default": 15,
            "min": 5,
            "max": 30,
            "step": 1,
            "description": "Ganancia para frecuencias altas (reflectancia) x10. Ej: 15 = 1.5. Mayor=más realce"
        },
        "strength": {
            "default": 80,
            "min": 0,
            "max": 100,
            "step": 5,
            "description": "Intensidad de corrección: 0=imagen original, 100=corrección completa"
        },
        "output_channel": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Canal a procesar: 0=luminancia (canal L en LAB, recomendado para color), 1=escala de grises"
        },
        "show_comparison": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Salida del sample: 0=imagen normalizada (solo resultado), 1=comparación lado a lado (original vs resultado)"
        }
    }

    def _homomorphic(self, channel: np.ndarray,
                     cutoff_frequency: int,
                     gamma_low: int,
                     gamma_high: int) -> np.ndarray:
        """
        Aplica el filtro homomórfico clásico a un canal uint8 [0,255].

        Pasos:
          1. Transformación logarítmica
          2. FFT
          3. Filtro Butterworth paso alto modificado con gammas
          4. IFFT
          5. Transformación exponencial inversa
          6. Normalización a [0, 255]
        """
        # 1. Log transform (evitar log(0))
        img_log = np.log1p(channel.astype(np.float64))

        # 2. FFT
        dft = np.fft.fft2(img_log)
        dft_shift = np.fft.fftshift(dft)

        # 3. Filtro Butterworth orden 2 con gammas
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        cutoff = cutoff_frequency / 100.0 * min(crow, ccol)

        # Matriz de distancias al centro
        y, x = np.mgrid[0:rows, 0:cols]
        D = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

        # Butterworth orden 2: suave, sin ringing
        H_hp = 1.0 / (1.0 + (cutoff / (D + 1e-6)) ** 2)  # paso alto [0,1]
        H_lp = 1.0 - H_hp                                  # paso bajo [0,1]

        # Combinar con gammas (normalizados /10)
        gl = gamma_low / 10.0
        gh = gamma_high / 10.0
        H = gl * H_lp + gh * H_hp

        # 4. Aplicar filtro
        filtered_shift = dft_shift * H

        # 5. IFFT
        filtered = np.fft.ifftshift(filtered_shift)
        img_filtered = np.real(np.fft.ifft2(filtered))

        # 6. Exp transform inversa
        result = np.expm1(img_filtered)

        # 7. Normalizar a [0, 255]
        result = np.clip(result, 0, None)
        min_val, max_val = result.min(), result.max()
        if max_val > min_val:
            result = (result - min_val) / (max_val - min_val) * 255.0
        else:
            result = np.zeros_like(channel, dtype=np.float64)

        return result.astype(np.uint8)

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)

        # Parámetros
        cutoff_frequency = int(self.params["cutoff_frequency"])
        gamma_low = int(self.params["gamma_low"])
        gamma_high = int(self.params["gamma_high"])
        strength = int(self.params["strength"])
        output_channel = int(self.params["output_channel"])
        show_comparison = int(self.params.get("show_comparison", 0))

        is_color = (len(input_img.shape) == 3 and input_img.shape[2] == 3)

        alpha = strength / 100.0

        if output_channel == 0:
            # --- Luminancia LAB ---
            if is_color:
                lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
                L, A, B_ch = cv2.split(lab)

                L_corrected = self._homomorphic(L, cutoff_frequency, gamma_low, gamma_high)

                # Mezcla con original según strength
                L_final = (
                    alpha * L_corrected.astype(np.float64)
                    + (1.0 - alpha) * L.astype(np.float64)
                ).astype(np.uint8)

                lab_out = cv2.merge([L_final, A, B_ch])
                normalized_image = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

                if not self.without_preview and show_comparison:
                    sample_image = self._create_comparison_gray(L, L_final)
                else:
                    sample_image = cv2.cvtColor(L_final, cv2.COLOR_GRAY2BGR)
            else:
                # Imagen ya en gris: tratar igual que modo 1
                gray = input_img.copy()
                corrected = self._homomorphic(gray, cutoff_frequency, gamma_low, gamma_high)
                normalized_image = (
                    alpha * corrected.astype(np.float64)
                    + (1.0 - alpha) * gray.astype(np.float64)
                ).astype(np.uint8)

                if not self.without_preview and show_comparison:
                    sample_image = self._create_comparison_gray(gray, normalized_image)
                else:
                    sample_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)

        else:
            # --- Escala de grises (output_channel == 1) ---
            if is_color:
                gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = input_img.copy()

            corrected = self._homomorphic(gray, cutoff_frequency, gamma_low, gamma_high)
            normalized_image = (
                alpha * corrected.astype(np.float64)
                + (1.0 - alpha) * gray.astype(np.float64)
            ).astype(np.uint8)

            if not self.without_preview and show_comparison:
                sample_image = self._create_comparison_gray(gray, normalized_image)
            else:
                sample_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)

        return {
            "normalized_image": normalized_image,
            "sample_image": sample_image
        }

    def _create_comparison_gray(self, original: np.ndarray,
                                result: np.ndarray) -> np.ndarray:
        """Crea comparación lado a lado (original vs homomórfico) en escala de grises."""
        h, w = original.shape[:2]

        sep = 20
        canvas = np.zeros((h, w * 2 + sep, 3), dtype=np.uint8)

        orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        res_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        canvas[:, :w] = orig_bgr
        canvas[:, w + sep:] = res_bgr
        canvas[:, w:w + sep] = (100, 100, 100)

        cv2.putText(canvas, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "HOMOMÓRFICO", (w + sep + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return canvas
