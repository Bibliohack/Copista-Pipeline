"""
Filtro: BackgroundNormalization
"""

import cv2
import numpy as np
from typing import Dict, Any
from .base_filter import BaseFilter, FILTER_REGISTRY


class BackgroundNormalization(BaseFilter):
    """Corrige iluminación no uniforme en imágenes de documentos fotografiados"""

    FILTER_NAME = "BackgroundNormalization"
    DESCRIPTION = (
        "Corrige iluminación no uniforme en imágenes de documentos fotografiados. "
        "Estima el fondo de iluminación (gaussiano, apertura morfológica o máximo local) "
        "y lo combina con la imagen original mediante diferentes modos de mezcla."
    )

    INPUTS = {
        "input_image": "image"
    }

    OUTPUTS = {
        "normalized_image": "image",
        "sample_image": "image"
    }

    PARAMS = {
        "blur_radius": {
            "default": 100,
            "min": 10,
            "max": 500,
            "step": 10,
            "description": "Radio del blur gaussiano para estimar el fondo de iluminación. Mayor = más suave"
        },
        "blend_mode": {
            "default": 1,
            "min": 0,
            "max": 10,
            "step": 1,
            "description": (
                "Modo de mezcla: 0=subtract, 1=divide, 2=retinex, 3=gamma_divide, "
                "4=overlay, 5=soft_light, 6=hard_light, 7=vivid_light, "
                "8=linear_light, 9=exclusion, 10=invert_soft_light (blur→invertir→soft_light)"
            )
        },
        "strength": {
            "default": 80,
            "min": 0,
            "max": 100,
            "step": 5,
            "description": "Intensidad de corrección (0=imagen original, 100=corrección completa)"
        },
        "gamma": {
            "default": 10,
            "min": 1,
            "max": 30,
            "step": 1,
            "description": (
                "Gamma para modo gamma_divide (10=gamma 1.0, valores menores=más oscuro, "
                "mayores=más claro)"
            )
        },
        "output_channel": {
            "default": 0,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": (
                "Canal a normalizar: 0=luminancia (canal L en LAB, recomendado), "
                "1=escala de grises, 2=todos los canales BGR"
            )
        },
        "background_method": {
            "default": 0,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": (
                "Método de estimación del fondo: "
                "0=Gaussiano (rápido, estándar), "
                "1=Cierre morfológico (dilata→erosiona, rellena texto oscuro con fondo claro, más preciso), "
                "2=Dilatación+Gaussian (envolvente máxima del fondo, más agresivo)"
            )
        },
        "morph_radius": {
            "default": 50,
            "min": 5,
            "max": 300,
            "step": 5,
            "description": "Radio del kernel morfológico (solo métodos 1 y 2). Debe ser mayor que el carácter más grande del texto"
        },
        "show_comparison": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Salida del sample: 0=imagen normalizada (solo resultado), 1=comparación lado a lado (original vs resultado)"
        }
    }

    # Nombres de los modos de mezcla para etiquetas
    _BLEND_NAMES = [
        "subtract", "divide", "retinex", "gamma_divide",
        "overlay", "soft_light", "hard_light", "vivid_light",
        "linear_light", "exclusion", "invert_soft_light"
    ]

    _BG_METHOD_NAMES = ["gauss", "morph_close", "max_local"]

    def _compute_background(self, channel: np.ndarray, blur_radius: int,
                            background_method: int, morph_radius: int = 50) -> np.ndarray:
        """
        Estima el fondo de iluminación.
        - 0: Gaussiano estándar (blur_radius controla el radio)
        - 1: Cierre morfológico (morph_radius) + Gaussian suavizado (blur_radius)
        - 2: Dilatación (morph_radius) + Gaussian suavizado (blur_radius)
        """
        # Kernel gaussiano (métodos 0 y suavizado final de 1/2)
        gksize = int(blur_radius) * 2 + 1
        gksize = max(3, gksize)
        sigma = blur_radius / 3.0

        if background_method == 0:
            background = cv2.GaussianBlur(channel, (gksize, gksize), sigma)

        elif background_method == 1:
            # Cierre morfológico: dilatación → erosión
            # La dilatación expande el fondo claro sobre el texto oscuro (rellena caracteres)
            # La erosión elimina artefactos brillantes introducidos por la dilatación
            mksize = int(morph_radius) * 2 + 1
            mksize = max(3, mksize)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mksize, mksize))
            background = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, kernel)
            # Gaussian para suavizar los bordes duros del kernel morfológico
            background = cv2.GaussianBlur(background, (gksize, gksize), sigma)

        else:
            # Dilatación pura + Gaussian
            mksize = int(morph_radius) * 2 + 1
            mksize = max(3, mksize)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mksize, mksize))
            background = cv2.dilate(channel, kernel)
            # Gaussian para suavizar los bordes duros del kernel morfológico
            background = cv2.GaussianBlur(background, (gksize, gksize), sigma)

        return background

    def _apply_blend(self, I: np.ndarray, B: np.ndarray,
                     blend_mode: int, gamma_param: float) -> np.ndarray:
        """
        Aplica el modo de mezcla indicado.
        I y B son float32 en [0, 1].
        Devuelve resultado float32 (puede estar fuera de [0,1] antes de clip).
        """
        # Evitar división por cero
        B_safe = np.clip(B, 1e-6, 1.0)

        if blend_mode == 0:
            # subtract
            result = I - B + 0.5

        elif blend_mode == 1:
            # divide
            mean_B = float(np.mean(B_safe))
            result = (I / B_safe) * mean_B

        elif blend_mode == 2:
            # retinex
            log_result = np.log(np.clip(I, 1e-6, 1.0)) - np.log(B_safe)
            # Normalizar al rango de I
            log_min = log_result.min()
            log_max = log_result.max()
            if log_max > log_min:
                result = (log_result - log_min) / (log_max - log_min)
                # Escalar al rango original de I
                I_min = float(I.min())
                I_max = float(I.max())
                result = result * (I_max - I_min) + I_min
            else:
                result = np.full_like(I, 0.5)

        elif blend_mode == 3:
            # gamma_divide
            g = gamma_param / 10.0
            result = (I / B_safe) ** g

        elif blend_mode == 4:
            # overlay: base = B, blend = I
            result = np.where(
                B < 0.5,
                2.0 * I * B,
                1.0 - 2.0 * (1.0 - I) * (1.0 - B)
            )

        elif blend_mode == 5:
            # soft_light (fórmula Pegtop simplificada)
            result = (1.0 - 2.0 * B) * I * I + 2.0 * B * I

        elif blend_mode == 6:
            # hard_light: I y B intercambiados respecto a overlay
            result = np.where(
                I < 0.5,
                2.0 * I * B,
                1.0 - 2.0 * (1.0 - I) * (1.0 - B)
            )

        elif blend_mode == 7:
            # vivid_light
            result = np.where(
                B < 0.5,
                1.0 - (1.0 - I) / (2.0 * B + 1e-6),
                I / (2.0 * (1.0 - B) + 1e-6)
            )

        elif blend_mode == 8:
            # linear_light
            result = I + 2.0 * B - 1.0

        elif blend_mode == 9:
            # exclusion
            result = I + B - 2.0 * I * B

        else:
            # invert_soft_light (modo 10): blur → invertir background → soft_light
            # El background invertido actúa como capa de corrección:
            # zonas oscuras (sombras) → B alto → B_inv bajo → soft_light aclara
            # zonas claras (sobreexpuestas) → B bajo → B_inv alto → soft_light oscurece
            B_inv = 1.0 - B
            result = (1.0 - 2.0 * B_inv) * I * I + 2.0 * B_inv * I

        return result.astype(np.float32)

    def _normalize_channel(self, channel: np.ndarray, blur_radius: int,
                            blend_mode: int, strength: int,
                            gamma_param: float, background_method: int = 0,
                            morph_radius: int = 50) -> np.ndarray:
        """
        Normaliza un canal en uint8 [0,255].
        Devuelve el canal corregido en uint8.
        """
        # Convertir a float [0,1]
        I = channel.astype(np.float32) / 255.0
        B = self._compute_background(channel, blur_radius, background_method, morph_radius).astype(np.float32) / 255.0

        corrected = self._apply_blend(I, B, blend_mode, gamma_param)

        # Aplicar strength
        alpha = strength / 100.0
        result = alpha * corrected + (1.0 - alpha) * I

        # Clip y convertir a uint8
        result = np.clip(result, 0.0, 1.0)
        return (result * 255.0).astype(np.uint8)

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)

        # Parámetros
        blur_radius = int(self.params["blur_radius"])
        blend_mode = int(self.params["blend_mode"])
        strength = int(self.params["strength"])
        gamma_param = float(self.params["gamma"])
        output_channel = int(self.params["output_channel"])
        background_method = int(self.params["background_method"])
        morph_radius = int(self.params["morph_radius"])
        show_comparison = int(self.params["show_comparison"])

        # Determinar si la entrada es color o gris
        is_color = (len(input_img.shape) == 3 and input_img.shape[2] == 3)

        blend_name = self._BLEND_NAMES[blend_mode] if 0 <= blend_mode < len(self._BLEND_NAMES) else str(blend_mode)
        bg_name = self._BG_METHOD_NAMES[background_method] if 0 <= background_method < len(self._BG_METHOD_NAMES) else str(background_method)
        label = f"{blend_name}/{bg_name}"

        if output_channel == 0:
            # --- Luminancia LAB ---
            if is_color:
                lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
                L, A, B_ch = cv2.split(lab)
                # Canal L está en [0,255] en OpenCV
                L_corrected = self._normalize_channel(
                    L, blur_radius, blend_mode, strength, gamma_param, background_method, morph_radius
                )
                lab_out = cv2.merge([L_corrected, A, B_ch])
                normalized_image = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

                if not self.without_preview:
                    # Comparación usando canal L en gris
                    sample_image = self._create_comparison_gray(
                        L, L_corrected, label
                    )
                else:
                    sample_image = None
            else:
                # Imagen ya es gris: tratar igual que modo 1
                gray = input_img.copy()
                normalized_image = self._normalize_channel(
                    gray, blur_radius, blend_mode, strength, gamma_param, background_method, morph_radius
                )
                if not self.without_preview:
                    sample_image = self._create_comparison_gray(
                        gray, normalized_image, blend_name
                    )
                else:
                    sample_image = None

        elif output_channel == 1:
            # --- Escala de grises ---
            if is_color:
                gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = input_img.copy()
            normalized_image = self._normalize_channel(
                gray, blur_radius, blend_mode, strength, gamma_param
            )
            if not self.without_preview:
                sample_image = self._create_comparison_gray(
                    gray, normalized_image, label
                )
            else:
                sample_image = None

        else:
            # --- Todos los canales BGR (output_channel == 2) ---
            if is_color:
                bgr = input_img.copy()
            else:
                bgr = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)

            channels = cv2.split(bgr)
            corrected_channels = [
                self._normalize_channel(ch, blur_radius, blend_mode, strength, gamma_param, background_method, morph_radius)
                for ch in channels
            ]
            normalized_image = cv2.merge(corrected_channels)

            if not self.without_preview:
                sample_image = self._create_comparison_bgr(
                    bgr, normalized_image, label
                )
            else:
                sample_image = None

        # Aplicar show_comparison: si 0, sample = imagen normalizada directamente
        if sample_image is not None and not show_comparison:
            if len(normalized_image.shape) == 2:
                sample_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)
            else:
                sample_image = normalized_image.copy()

        # Fallback: si sin_preview, devolver imagen normalizada como sample
        if sample_image is None:
            if len(normalized_image.shape) == 2:
                sample_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)
            else:
                sample_image = normalized_image.copy()

        return {
            "normalized_image": normalized_image,
            "sample_image": sample_image
        }

    def _create_comparison_gray(self, original: np.ndarray, result: np.ndarray,
                                blend_name: str) -> np.ndarray:
        """Crea comparación lado a lado en escala de grises."""
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
        cv2.putText(canvas, f"NORMALIZADO ({blend_name})", (w + sep + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return canvas

    def _create_comparison_bgr(self, original: np.ndarray, result: np.ndarray,
                               blend_name: str) -> np.ndarray:
        """Crea comparación lado a lado en BGR."""
        h, w = original.shape[:2]

        sep = 20
        canvas = np.zeros((h, w * 2 + sep, 3), dtype=np.uint8)

        canvas[:, :w] = original
        canvas[:, w + sep:] = result
        canvas[:, w:w + sep] = (100, 100, 100)

        cv2.putText(canvas, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"NORMALIZADO ({blend_name})", (w + sep + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return canvas
