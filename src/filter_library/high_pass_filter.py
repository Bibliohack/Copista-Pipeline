"""
Filtro: HighPassFilter
"""

import cv2
import numpy as np
from typing import Dict, Any
from .base_filter import BaseFilter, FILTER_REGISTRY


class HighPassFilter(BaseFilter):
    """Aplica filtro de paso alto para realzar bordes y detalles"""
    
    FILTER_NAME = "HighPass"
    DESCRIPTION = "Filtro de paso alto que realza bordes y detalles finos removiendo frecuencias bajas. Útil para mejorar nitidez antes de OCR o detección de bordes."
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "highpass_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "method": {
            "default": 0,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": "Método: 0=Gaussian, 1=Laplaciano, 2=Unsharp Mask, 3=FFT"
        },
        "kernel_size": {
            "default": 21,
            "min": 3,
            "max": 99,
            "step": 2,
            "description": "Tamaño del kernel (debe ser impar, para métodos 0 y 2)"
        },
        "sigma": {
            "default": 5,
            "min": 1,
            "max": 50,
            "step": 1,
            "description": "Sigma del desenfoque gaussiano (para métodos 0 y 2)"
        },
        "strength": {
            "default": 15,
            "min": 1,
            "max": 50,
            "step": 1,
            "description": "Fuerza del filtro (amplificación de detalles). 10=normal, 20=fuerte"
        },
        "add_original": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Agregar imagen original al resultado: 0=No (solo paso alto), 1=Sí (realce)"
        },
        "normalize_output": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Normalizar salida a rango 0-255: 0=No, 1=Sí"
        },
        "fft_cutoff": {
            "default": 30,
            "min": 1,
            "max": 100,
            "step": 5,
            "description": "Radio de corte en % para FFT (solo método 3). Menor=más paso alto"
        }
    }
    
    def _highpass_gaussian(self, gray: np.ndarray, ksize: int, sigma: float,
                          strength: float, add_original: bool) -> np.ndarray:
        """
        Filtro paso alto usando substracción gaussiana.
        
        highpass = original - gaussian_blur(original)
        Si add_original: resultado = original + strength * highpass
        """
        # Desenfoque gaussiano (frecuencias bajas)
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        
        # Paso alto = original - bajas frecuencias
        highpass = cv2.subtract(gray.astype(np.float32), blurred.astype(np.float32))
        
        # Amplificar
        highpass = highpass * (strength / 10.0)
        
        if add_original:
            # Combinar con original (realce)
            result = gray.astype(np.float32) + highpass
        else:
            # Solo paso alto
            result = highpass + 128  # Offset para visualización
        
        return result
    
    def _highpass_laplacian(self, gray: np.ndarray, strength: float,
                           add_original: bool) -> np.ndarray:
        """
        Filtro paso alto usando Laplaciano.
        
        El Laplaciano detecta cambios rápidos (bordes).
        """
        # Aplicar Laplaciano
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Amplificar
        laplacian = laplacian * (strength / 10.0)
        
        if add_original:
            # Combinar con original (realce de bordes)
            result = gray.astype(np.float32) + laplacian
        else:
            # Solo Laplaciano
            result = laplacian + 128  # Offset para visualización
        
        return result
    
    def _highpass_unsharp_mask(self, gray: np.ndarray, ksize: int, sigma: float,
                               strength: float) -> np.ndarray:
        """
        Unsharp Mask: método clásico de realce.
        
        resultado = original + amount * (original - blurred)
        """
        # Desenfoque
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        
        # Calcular máscara (diferencia)
        mask = cv2.subtract(gray.astype(np.float32), blurred.astype(np.float32))
        
        # Aplicar con fuerza
        amount = strength / 10.0
        result = gray.astype(np.float32) + amount * mask
        
        return result
    
    def _highpass_fft(self, gray: np.ndarray, cutoff_percent: int,
                     strength: float, add_original: bool) -> np.ndarray:
        """
        Filtro paso alto usando FFT (Transformada de Fourier).
        
        Bloquea frecuencias bajas en el dominio de frecuencia.
        """
        # FFT
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Crear máscara de paso alto (bloquear centro = frecuencias bajas)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Radio de corte (% del radio máximo)
        max_radius = min(crow, ccol)
        cutoff_radius = int(max_radius * cutoff_percent / 100.0)
        
        # Crear máscara (1 = pasar, 0 = bloquear)
        mask = np.ones((rows, cols, 2), np.float32)
        
        # Círculo en el centro = frecuencias bajas (bloquear)
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= cutoff_radius**2
        mask[mask_area] = 0
        
        # Aplicar máscara
        fshift = dft_shift * mask
        
        # IFFT
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        
        # Amplificar
        img_back = img_back * (strength / 10.0)
        
        if add_original:
            # Combinar con original
            result = gray.astype(np.float32) + img_back
        else:
            # Solo paso alto
            result = img_back
        
        return result
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        # Convertir a grayscale si es necesario
        if len(input_img.shape) == 3:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            gray = input_img.copy()
            is_color = False
        
        # Parámetros
        method = self.params["method"]
        ksize = self.params["kernel_size"] | 1  # Asegurar impar
        ksize = max(3, ksize)
        sigma = self.params["sigma"]
        strength = self.params["strength"]
        add_original = bool(self.params["add_original"])
        normalize = bool(self.params["normalize_output"])
        fft_cutoff = self.params["fft_cutoff"]
        
        # Aplicar filtro según método
        if method == 0:
            # Gaussian
            result = self._highpass_gaussian(gray, ksize, sigma, strength, add_original)
        elif method == 1:
            # Laplaciano
            result = self._highpass_laplacian(gray, strength, add_original)
        elif method == 2:
            # Unsharp Mask
            result = self._highpass_unsharp_mask(gray, ksize, sigma, strength)
        else:  # method == 3
            # FFT
            result = self._highpass_fft(gray, fft_cutoff, strength, add_original)
        
        # Normalizar si es necesario
        if normalize:
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            # Recortar valores extremos
            result = result.astype(np.float32)
            min_val = result.min()
            max_val = result.max()
            if max_val > min_val:
                result = ((result - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                result = np.zeros_like(gray)
        
        # Crear visualización side-by-side
        if not self.without_preview:
            sample = self._create_comparison(gray, result, method)
        else:
            sample = None
        
        return {
            "highpass_image": result,
            "sample_image": sample if sample is not None else cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        }
    
    def _create_comparison(self, original: np.ndarray, filtered: np.ndarray,
                          method: int) -> np.ndarray:
        """Crea visualización lado a lado (antes/después)"""
        # Asegurar mismo tamaño
        h, w = original.shape
        
        # Crear canvas
        canvas = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
        
        # Convertir a BGR
        orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        filt_bgr = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        
        # Colocar imágenes
        canvas[:, :w] = orig_bgr
        canvas[:, w+20:] = filt_bgr
        
        # Separador
        canvas[:, w:w+20] = (100, 100, 100)
        
        # Labels
        method_names = ["Gaussian", "Laplaciano", "Unsharp Mask", "FFT"]
        
        cv2.putText(canvas, "ORIGINAL", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(canvas, f"HIGH PASS ({method_names[method]})", (w + 30, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return canvas
