"""
Biblioteca de Filtros para el Sistema de Procesamiento de Imágenes
==================================================================

Cada filtro es una clase que hereda de BaseFilter y debe implementar:
- FILTER_NAME: nombre único del filtro
- DESCRIPTION: descripción del filtro
- INPUTS: dict con {nombre_input: tipo_dato} que necesita
- OUTPUTS: dict con {nombre_output: tipo_dato} que produce (siempre incluir 'sample_image')
- PARAMS: dict con {nombre_param: {default, min, max, step, description}}
- process(): método que ejecuta el filtro

Para agregar un nuevo filtro:
1. Crear una clase que herede de BaseFilter
2. Definir los atributos de clase requeridos
3. Implementar el método process()
4. El filtro se registra automáticamente
"""

import cv2
import numpy as np
import warnings  # <-- AÑADIDO
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple


# Registro global de filtros
FILTER_REGISTRY = {}


class BaseFilter(ABC):
    """Clase base para todos los filtros"""
    
    FILTER_NAME: str = "base"
    DESCRIPTION: str = "Filtro base"
    INPUTS: Dict[str, str] = {}  # {nombre: tipo}
    OUTPUTS: Dict[str, str] = {"sample_image": "image"}  # siempre debe incluir sample_image
    PARAMS: Dict[str, Dict] = {}  # {nombre: {default, min, max, step, description}}
    
    def __init_subclass__(cls, **kwargs):
        """Registra automáticamente cada subclase"""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'FILTER_NAME') and cls.FILTER_NAME != "base":
            FILTER_REGISTRY[cls.FILTER_NAME] = cls
    
    def __init__(self, params: Dict[str, Any] = None):
        """Inicializa el filtro con parámetros"""
        self.params = {}
        # Cargar defaults
        for name, config in self.PARAMS.items():
            self.params[name] = config.get('default', 0)
        # Sobrescribir con params proporcionados
        if params:
            for name, value in params.items():
                if name in self.PARAMS:
                    self.params[name] = value
    
    def set_param(self, name: str, value: Any):
        """Establece un parámetro"""
        if name in self.PARAMS:
            self.params[name] = value
    
    def get_param(self, name: str) -> Any:
        """Obtiene un parámetro"""
        return self.params.get(name)
    
    def get_params(self) -> Dict[str, Any]:
        """Obtiene todos los parámetros"""
        return self.params.copy()
    
    def get_help(self) -> str:
        """Retorna ayuda sobre los parámetros del filtro"""
        lines = [
            f"\n{'='*60}",
            f"FILTRO: {self.FILTER_NAME}",
            f"{'='*60}",
            f"Descripción: {self.DESCRIPTION}",
            f"\nEntradas requeridas:",
        ]
        if self.INPUTS:
            for name, dtype in self.INPUTS.items():
                lines.append(f"  - {name}: {dtype}")
        else:
            lines.append("  (ninguna - usa imagen original)")
        
        lines.append(f"\nSalidas producidas:")
        for name, dtype in self.OUTPUTS.items():
            lines.append(f"  - {name}: {dtype}")
        
        lines.append(f"\nParámetros:")
        if self.PARAMS:
            for name, config in self.PARAMS.items():
                lines.append(f"  [{name}]")
                lines.append(f"    Descripción: {config.get('description', 'Sin descripción')}")
                lines.append(f"    Default: {config.get('default', 'N/A')}")
                lines.append(f"    Rango: [{config.get('min', 'N/A')} - {config.get('max', 'N/A')}]")
                lines.append(f"    Step: {config.get('step', 1)}")
        else:
            lines.append("  (sin parámetros)")
        
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)
    
    @abstractmethod
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        """
        Procesa el filtro.
        
        Args:
            inputs: dict con los datos de entrada de otros filtros
            original_image: imagen original sin procesar
            
        Returns:
            dict con los outputs producidos (siempre debe incluir 'sample_image')
        """
        pass


# =============================================================================
# FILTROS DE EJEMPLO
# =============================================================================

class ResizeFilter(BaseFilter):
    """Filtro para redimensionar la imagen"""
    
    FILTER_NAME = "Resize"
    DESCRIPTION = "Redimensiona la imagen a un tamaño específico o por porcentaje"
    INPUTS = {"input_image": "image"}  # <-- MODIFICADO: ahora acepta input_image
    OUTPUTS = {
        "resized_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "mode": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Modo: 0=Por porcentaje, 1=Por dimensiones fijas"
        },
        "scale_percent": {
            "default": 100,
            "min": 10,
            "max": 200,
            "step": 5,
            "description": "Porcentaje de escala (solo modo 0)"
        },
        "width": {
            "default": 640,
            "min": 100,
            "max": 1920,
            "step": 10,
            "description": "Ancho en píxeles (solo modo 1)"
        },
        "height": {
            "default": 480,
            "min": 100,
            "max": 1080,
            "step": 10,
            "description": "Alto en píxeles (solo modo 1)"
        },
        "interpolation": {
            "default": 1,
            "min": 0,
            "max": 4,
            "step": 1,
            "description": "Interpolación: 0=NEAREST, 1=LINEAR, 2=AREA, 3=CUBIC, 4=LANCZOS4"
        }
    }
    
    INTERPOLATION_METHODS = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4
    ]
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        # <-- MODIFICADO: Ahora acepta input_image con fallback y advertencia
        input_img = inputs.get("input_image")
        if input_img is None:
            warnings.warn(f"[Resize] No se proporcionó 'input_image', usando imagen original.", stacklevel=2)
            input_img = original_image
        
        mode = self.params["mode"]
        interp = self.INTERPOLATION_METHODS[self.params["interpolation"]]
        
        if mode == 0:
            scale = self.params["scale_percent"] / 100.0
            new_width = int(input_img.shape[1] * scale)  # <-- Usar input_img
            new_height = int(input_img.shape[0] * scale)  # <-- Usar input_img
        else:
            new_width = self.params["width"]
            new_height = self.params["height"]
        
        resized = cv2.resize(input_img, (new_width, new_height), interpolation=interp)  # <-- Usar input_img
        
        return {
            "resized_image": resized,
            "sample_image": resized
        }


class GrayscaleFilter(BaseFilter):
    """Convierte la imagen a escala de grises"""
    
    FILTER_NAME = "Grayscale"
    DESCRIPTION = "Convierte la imagen a escala de grises con control de pesos RGB"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "grayscale_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "method": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Método: 0=OpenCV estándar, 1=Pesos personalizados"
        },
        "weight_r": {
            "default": 30,
            "min": 0,
            "max": 100,
            "step": 1,
            "description": "Peso del canal Rojo (0-100, solo método 1)"
        },
        "weight_g": {
            "default": 59,
            "min": 0,
            "max": 100,
            "step": 1,
            "description": "Peso del canal Verde (0-100, solo método 1)"
        },
        "weight_b": {
            "default": 11,
            "min": 0,
            "max": 100,
            "step": 1,
            "description": "Peso del canal Azul (0-100, solo método 1)"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        if len(input_img.shape) == 2:
            # Ya es grayscale
            gray = input_img
        elif self.params["method"] == 0:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            # Pesos personalizados
            total = self.params["weight_r"] + self.params["weight_g"] + self.params["weight_b"]
            if total == 0:
                total = 1
            wr = self.params["weight_r"] / total
            wg = self.params["weight_g"] / total
            wb = self.params["weight_b"] / total
            gray = (input_img[:,:,2] * wr + input_img[:,:,1] * wg + input_img[:,:,0] * wb).astype(np.uint8)
        
        # sample_image debe ser visualizable (3 canales para consistencia)
        sample = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(gray.shape) == 2 else gray
        
        return {
            "grayscale_image": gray,
            "sample_image": sample
        }


class GaussianBlurFilter(BaseFilter):
    """Aplica desenfoque gaussiano"""
    
    FILTER_NAME = "GaussianBlur"
    DESCRIPTION = "Aplica un filtro de desenfoque gaussiano"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "blurred_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "kernel_size": {
            "default": 5,
            "min": 1,
            "max": 31,
            "step": 2,
            "description": "Tamaño del kernel (debe ser impar)"
        },
        "sigma": {
            "default": 0,
            "min": 0,
            "max": 10,
            "step": 1,
            "description": "Sigma (0=auto calculado)"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        ksize = self.params["kernel_size"]
        if ksize % 2 == 0:
            ksize += 1
        
        sigma = self.params["sigma"]
        blurred = cv2.GaussianBlur(input_img, (ksize, ksize), sigma)
        
        return {
            "blurred_image": blurred,
            "sample_image": blurred
        }


class CannyEdgeFilter(BaseFilter):
    """Detector de bordes Canny"""
    
    FILTER_NAME = "CannyEdge"
    DESCRIPTION = "Detecta bordes usando el algoritmo de Canny"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "edge_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "threshold1": {
            "default": 50,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Umbral inferior para histéresis"
        },
        "threshold2": {
            "default": 150,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Umbral superior para histéresis"
        },
        "aperture_size": {
            "default": 3,
            "min": 3,
            "max": 7,
            "step": 2,
            "description": "Tamaño de apertura Sobel (3, 5 o 7)"
        },
        "l2_gradient": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Usar norma L2 para gradiente (0=No, 1=Sí)"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        # Convertir a grayscale si es necesario
        if len(input_img.shape) == 3:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_img
        
        aperture = self.params["aperture_size"]
        if aperture not in [3, 5, 7]:
            aperture = 3
        
        edges = cv2.Canny(
            gray,
            self.params["threshold1"],
            self.params["threshold2"],
            apertureSize=aperture,
            L2gradient=bool(self.params["l2_gradient"])
        )
        
        # sample_image en color para visualización
        sample = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return {
            "edge_image": edges,
            "sample_image": sample
        }


class ThresholdFilter(BaseFilter):
    """Aplica umbralización a la imagen"""
    
    FILTER_NAME = "Threshold"
    DESCRIPTION = "Aplica umbralización binaria o adaptativa"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "threshold_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "method": {
            "default": 0,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": "Método: 0=Binario, 1=Otsu, 2=Adaptativo"
        },
        "threshold": {
            "default": 127,
            "min": 0,
            "max": 255,
            "step": 1,
            "description": "Valor de umbral (solo método binario)"
        },
        "max_value": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 1,
            "description": "Valor máximo para píxeles sobre umbral"
        },
        "adaptive_method": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Método adaptativo: 0=Mean, 1=Gaussian"
        },
        "block_size": {
            "default": 11,
            "min": 3,
            "max": 99,
            "step": 2,
            "description": "Tamaño de bloque para adaptativo (impar)"
        },
        "c_value": {
            "default": 2,
            "min": -20,
            "max": 20,
            "step": 1,
            "description": "Constante C para adaptativo"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        # Convertir a grayscale si es necesario
        if len(input_img.shape) == 3:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_img
        
        method = self.params["method"]
        max_val = self.params["max_value"]
        
        if method == 0:
            # Binario simple
            _, thresh = cv2.threshold(gray, self.params["threshold"], max_val, cv2.THRESH_BINARY)
        elif method == 1:
            # Otsu
            _, thresh = cv2.threshold(gray, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Adaptativo
            block_size = self.params["block_size"]
            if block_size % 2 == 0:
                block_size += 1
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if self.params["adaptive_method"] == 0 else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            thresh = cv2.adaptiveThreshold(gray, max_val, adaptive_method, cv2.THRESH_BINARY, block_size, self.params["c_value"])
        
        sample = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        return {
            "threshold_image": thresh,
            "sample_image": sample
        }


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


class HoughLinesFilter(BaseFilter):
    """Detecta líneas usando la transformada de Hough"""
    
    FILTER_NAME = "HoughLines"
    DESCRIPTION = "Detecta líneas rectas usando la transformada de Hough"
    INPUTS = {
        "edge_image": "image",
        "base_image": "image"  # <-- AÑADIDO: para visualización
    }
    OUTPUTS = {
        "lines_data": "lines",
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
        base_img = inputs.get("base_image", original_image)  # <-- MODIFICADO: fallback con advertencia
        
        if edge_img is None:
            warnings.warn(f"[HoughLines] No se proporcionó 'edge_image', aplicando Canny a la imagen base.", stacklevel=2)
            # Si no hay edge_image, usar la imagen base
            if len(base_img.shape) == 3:
                edge_img = cv2.Canny(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), 50, 150)
            else:
                edge_img = cv2.Canny(base_img, 50, 150)
        
        rho = self.params["rho"]
        theta = np.pi / self.params["theta_divisor"]
        threshold = self.params["threshold"]
        
        # Crear imagen para visualización usando base_img
        if len(base_img.shape) == 3:
            sample = base_img.copy()
        else:
            sample = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        
        lines_data = []
        
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
        
        return {
            "lines_data": lines_data,
            "sample_image": sample
        }


class MorphologyFilter(BaseFilter):
    """Operaciones morfológicas"""
    
    FILTER_NAME = "Morphology"
    DESCRIPTION = "Aplica operaciones morfológicas (erosión, dilatación, apertura, cierre)"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "morphed_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "operation": {
            "default": 0,
            "min": 0,
            "max": 5,
            "step": 1,
            "description": "Operación: 0=Erosión, 1=Dilatación, 2=Apertura, 3=Cierre, 4=Gradiente, 5=TopHat"
        },
        "kernel_shape": {
            "default": 0,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": "Forma kernel: 0=Rectángulo, 1=Cruz, 2=Elipse"
        },
        "kernel_size": {
            "default": 5,
            "min": 1,
            "max": 21,
            "step": 2,
            "description": "Tamaño del kernel (impar)"
        },
        "iterations": {
            "default": 1,
            "min": 1,
            "max": 10,
            "step": 1,
            "description": "Número de iteraciones"
        }
    }
    
    OPERATIONS = [
        cv2.MORPH_ERODE,
        cv2.MORPH_DILATE,
        cv2.MORPH_OPEN,
        cv2.MORPH_CLOSE,
        cv2.MORPH_GRADIENT,
        cv2.MORPH_TOPHAT
    ]
    
    KERNEL_SHAPES = [
        cv2.MORPH_RECT,
        cv2.MORPH_CROSS,
        cv2.MORPH_ELLIPSE
    ]
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        ksize = self.params["kernel_size"]
        if ksize % 2 == 0:
            ksize += 1
        
        kernel = cv2.getStructuringElement(
            self.KERNEL_SHAPES[self.params["kernel_shape"]],
            (ksize, ksize)
        )
        
        op = self.OPERATIONS[self.params["operation"]]
        result = cv2.morphologyEx(input_img, op, kernel, iterations=self.params["iterations"])
        
        # Asegurar que sample_image sea visualizable
        if len(result.shape) == 2:
            sample = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            sample = result
        
        return {
            "morphed_image": result,
            "sample_image": sample
        }


class ContourFilter(BaseFilter):
    """Detecta y dibuja contornos"""
    
    FILTER_NAME = "Contours"
    DESCRIPTION = "Detecta contornos y genera datos de contornos"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "contours_data": "contours",
        "contour_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "mode": {
            "default": 1,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": "Modo: 0=EXTERNAL, 1=LIST, 2=CCOMP, 3=TREE"
        },
        "method": {
            "default": 1,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": "Método: 0=NONE, 1=SIMPLE, 2=TC89_L1, 3=TC89_KCOS"
        },
        "min_area": {
            "default": 100,
            "min": 0,
            "max": 10000,
            "step": 100,
            "description": "Área mínima de contorno"
        },
        "draw_thickness": {
            "default": 2,
            "min": 1,
            "max": 5,
            "step": 1,
            "description": "Grosor de línea para dibujar"
        },
        "color_r": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color R para contornos"
        },
        "color_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color G para contornos"
        },
        "color_b": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color B para contornos"
        }
    }
    
    MODES = [
        cv2.RETR_EXTERNAL,
        cv2.RETR_LIST,
        cv2.RETR_CCOMP,
        cv2.RETR_TREE
    ]
    
    METHODS = [
        cv2.CHAIN_APPROX_NONE,
        cv2.CHAIN_APPROX_SIMPLE,
        cv2.CHAIN_APPROX_TC89_L1,
        cv2.CHAIN_APPROX_TC89_KCOS
    ]
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        # Convertir a grayscale y binarizar si es necesario
        if len(input_img.shape) == 3:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_img
        
        # Si no es binaria, aplicar umbral
        if gray.max() > 1:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        else:
            binary = gray
        
        contours, hierarchy = cv2.findContours(
            binary,
            self.MODES[self.params["mode"]],
            self.METHODS[self.params["method"]]
        )
        
        # Filtrar por área mínima
        min_area = self.params["min_area"]
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        # Crear imagen de contornos usando input_img (no original_image)
        if len(input_img.shape) == 3:
            contour_img = input_img.copy()
        else:
            contour_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        
        color = (self.params["color_b"], self.params["color_g"], self.params["color_r"])
        cv2.drawContours(contour_img, filtered_contours, -1, color, self.params["draw_thickness"])
        
        # Preparar datos de contornos
        contours_data = []
        for c in filtered_contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            x, y, w, h = cv2.boundingRect(c)
            contours_data.append({
                "area": area,
                "perimeter": perimeter,
                "bounding_box": (x, y, w, h),
                "points": c.tolist()
            })
        
        return {
            "contours_data": contours_data,
            "contour_image": contour_img,
            "sample_image": contour_img
        }


class OverlayLinesFilter(BaseFilter):
    """Filtro de visualización: dibuja líneas sobre la imagen original"""
    
    FILTER_NAME = "OverlayLines"
    DESCRIPTION = "Dibuja líneas detectadas sobre una imagen base (filtro de visualización)"
    INPUTS = {
        "base_image": "image",
        "lines_data": "lines"
    }
    OUTPUTS = {
        "overlay_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "color_preset": {
            "default": 0,
            "min": 0,
            "max": 5,
            "step": 1,
            "description": "Preset de color: 0=Verde, 1=Rojo, 2=Azul, 3=Amarillo, 4=Magenta, 5=Cian"
        },
        "line_thickness": {
            "default": 2,
            "min": 1,
            "max": 10,
            "step": 1,
            "description": "Grosor de las líneas"
        },
        "alpha": {
            "default": 100,
            "min": 0,
            "max": 100,
            "step": 10,
            "description": "Transparencia (0-100)"
        }
    }
    
    COLOR_PRESETS = [
        (0, 255, 0),    # Verde
        (0, 0, 255),    # Rojo
        (255, 0, 0),    # Azul
        (0, 255, 255),  # Amarillo
        (255, 0, 255),  # Magenta
        (255, 255, 0)   # Cian
    ]
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        base_img = inputs.get("base_image", original_image)
        lines_data = inputs.get("lines_data", [])
        
        if len(base_img.shape) == 2:
            overlay = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            overlay = base_img.copy()
        
        color = self.COLOR_PRESETS[self.params["color_preset"]]
        thickness = self.params["line_thickness"]
        
        # Crear capa para las líneas
        lines_layer = np.zeros_like(overlay)
        
        for line in lines_data:
            if "x1" in line:
                # Formato HoughLinesP
                cv2.line(lines_layer, 
                        (line["x1"], line["y1"]), 
                        (line["x2"], line["y2"]), 
                        color, thickness)
            elif "rho" in line:
                # Formato HoughLines standard
                rho = line["rho"]
                theta = line["theta"]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(lines_layer, (x1, y1), (x2, y2), color, thickness)
        
        # Aplicar transparencia
        alpha = self.params["alpha"] / 100.0
        result = cv2.addWeighted(overlay, 1.0, lines_layer, alpha, 0)
        
        return {
            "overlay_image": result,
            "sample_image": result
        }


class BrightnessContrastFilter(BaseFilter):
    """Ajusta brillo y contraste"""
    
    FILTER_NAME = "BrightnessContrast"
    DESCRIPTION = "Ajusta el brillo y contraste de la imagen"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "adjusted_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "brightness": {
            "default": 0,
            "min": -100,
            "max": 100,
            "step": 5,
            "description": "Ajuste de brillo (-100 a 100)"
        },
        "contrast": {
            "default": 100,
            "min": 0,
            "max": 200,
            "step": 5,
            "description": "Ajuste de contraste (0-200, 100=normal)"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        brightness = self.params["brightness"]
        contrast = self.params["contrast"] / 100.0
        
        # Aplicar contraste y brillo
        adjusted = cv2.convertScaleAbs(input_img, alpha=contrast, beta=brightness)
        
        return {
            "adjusted_image": adjusted,
            "sample_image": adjusted
        }


class ColorSpaceFilter(BaseFilter):
    """Convierte entre espacios de color"""
    
    FILTER_NAME = "ColorSpace"
    DESCRIPTION = "Convierte la imagen a diferentes espacios de color y permite ver canales individuales"
    INPUTS = {
        "input_image": "image"
    }
    OUTPUTS = {
        "converted_image": "image",
        "sample_image": "image"
    }
    PARAMS = {
        "color_space": {
            "default": 0,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": "Espacio: 0=BGR, 1=HSV, 2=LAB, 3=YCrCb"
        },
        "channel": {
            "default": 0,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": "Canal a mostrar: 0=Todos, 1=Canal1, 2=Canal2, 3=Canal3"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        # Asegurar que es BGR
        if len(input_img.shape) == 2:
            input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        
        cs = self.params["color_space"]
        
        if cs == 0:
            converted = input_img
        elif cs == 1:
            converted = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
        elif cs == 2:
            converted = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
        else:
            converted = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCrCb)
        
        channel = self.params["channel"]
        
        if channel == 0:
            sample = converted
        else:
            single_channel = converted[:, :, channel - 1]
            sample = cv2.cvtColor(single_channel, cv2.COLOR_GRAY2BGR)
        
        return {
            "converted_image": converted,
            "sample_image": sample
        }

# =============================================================================
# Filtros extraídos de page_border_pipeline.py
# =============================================================================


class NormalizePeaks(BaseFilter):
    """Normaliza la imagen mapeando rangos de intensidad basados en picos"""
    
    FILTER_NAME = "NormalizePeaks"
    DESCRIPTION = "Normaliza la imagen mapeando [dark_peak, light_peak] a [dark_target, light_target]. Útil para ajustar contraste basado en histograma."
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "normalized_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "dark_peak": {
            "default": 30,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Valor de gris del pico oscuro (fondo). Píxeles <= a este valor se mapean al target oscuro."
        },
        "light_peak": {
            "default": 220,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Valor de gris del pico claro (papel). Píxeles >= a este valor se mapean al target claro."
        },
        "dark_target": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Valor objetivo para píxeles oscuros (típicamente 0 = negro)."
        },
        "light_target": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Valor objetivo para píxeles claros (típicamente 255 = blanco)."
        },
        "auto_detect": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Auto-detectar picos del histograma (0=No, 1=Sí). Si es 1, ignora dark_peak y light_peak."
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        # Convertir a grayscale si es necesario
        if len(input_img.shape) == 3:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_img.copy()
        
        # Auto-detectar picos si está habilitado
        if self.params["auto_detect"] == 1:
            dark_peak, light_peak = self._auto_detect_peaks(gray)
        else:
            dark_peak = self.params["dark_peak"]
            light_peak = self.params["light_peak"]
        
        dark_target = self.params["dark_target"]
        light_target = self.params["light_target"]
        
        # Asegurar que los valores son válidos
        dark_peak = max(0, min(254, dark_peak))
        light_peak = max(dark_peak + 1, min(255, light_peak))
        
        # Normalizar: mapear [dark_peak, light_peak] -> [dark_target, light_target]
        img = gray.astype(np.float32)
        
        # Clipping y escalado
        img = np.clip(img, dark_peak, light_peak)
        img = (img - dark_peak) / (light_peak - dark_peak)  # Normalizar a [0, 1]
        img = img * (light_target - dark_target) + dark_target  # Escalar a targets
        
        result = np.clip(img, 0, 255).astype(np.uint8)
        
        # sample_image en BGR para visualización
        sample = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return {
            "normalized_image": result,
            "sample_image": sample
        }
    
    def _auto_detect_peaks(self, gray: np.ndarray) -> tuple:
        """Auto-detecta picos oscuro y claro del histograma"""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        # Buscar pico en zona oscura (0-100) y zona clara (150-255)
        dark_zone = hist[:100]
        light_zone = hist[150:]
        
        dark_peak = int(np.argmax(dark_zone)) if dark_zone.max() > 0 else 30
        light_peak = int(150 + np.argmax(light_zone)) if light_zone.max() > 0 else 220
        
        return dark_peak, light_peak


class MinArcLength(BaseFilter):
    """Filtra contornos/bordes por longitud mínima de arco"""
    
    FILTER_NAME = "MinArcLength"
    DESCRIPTION = "Filtra una imagen de bordes, eliminando contornos cuya longitud de arco sea menor al mínimo especificado."
    
    INPUTS = {
        "edge_image": "image",
        "base_image": "image"  # <-- AÑADIDO: para visualización
    }
    
    OUTPUTS = {
        "filtered_edges": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "min_length": {
            "default": 100,
            "min": 1,
            "max": 1000,
            "step": 10,
            "description": "Longitud mínima de arco en píxeles. Contornos más cortos se eliminan."
        },
        "closed_contours": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Calcular longitud como contorno cerrado (0=No, 1=Sí)."
        },
        "line_thickness": {
            "default": 1,
            "min": 1,
            "max": 5,
            "step": 1,
            "description": "Grosor de línea al redibujar los contornos filtrados."
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        edge_img = inputs.get("edge_image")
        base_img = inputs.get("base_image", original_image)  # <-- MODIFICADO: fallback con advertencia
        
        if edge_img is None:
            warnings.warn(f"[MinArcLength] No se proporcionó 'edge_image', aplicando Canny a la imagen base.", stacklevel=2)
            # Si no hay edge_image, aplicar Canny a la imagen base
            if len(base_img.shape) == 3:
                gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = base_img
            edge_img = cv2.Canny(gray, 50, 150)
        
        # Asegurar que es grayscale
        if len(edge_img.shape) == 3:
            edge_img = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
        
        min_length = self.params["min_length"]
        closed = bool(self.params["closed_contours"])
        thickness = self.params["line_thickness"]
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear imagen resultado
        result = np.zeros_like(edge_img)
        
        # Filtrar y redibujar contornos que cumplan el mínimo
        for contour in contours:
            arc_length = cv2.arcLength(contour, closed=closed)
            if arc_length >= min_length:
                cv2.drawContours(result, [contour], -1, 255, thickness)
        
        # sample_image en BGR para visualización
        sample = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return {
            "filtered_edges": result,
            "sample_image": sample
        }


class DenoiseNLMeans(BaseFilter):
    """Reducción de ruido usando Non-Local Means"""
    
    FILTER_NAME = "DenoiseNLMeans"
    DESCRIPTION = "Aplica fastNlMeansDenoising para reducir ruido preservando bordes. Ideal para imágenes con ruido gaussiano."
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "denoised_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "h": {
            "default": 10,
            "min": 1,
            "max": 30,
            "step": 1,
            "description": "Fuerza del filtro. Mayor valor = más suavizado pero puede perder detalle."
        },
        "template_window_size": {
            "default": 7,
            "min": 3,
            "max": 21,
            "step": 2,
            "description": "Tamaño de la ventana de template (debe ser impar). Típico: 7."
        },
        "search_window_size": {
            "default": 21,
            "min": 7,
            "max": 51,
            "step": 2,
            "description": "Tamaño del área de búsqueda (debe ser impar). Típico: 21. Mayor = más lento pero mejor calidad."
        },
        "color_mode": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Modo: 0=Grayscale (más rápido), 1=Color (preserva colores)."
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        h = max(1, self.params["h"])
        template_size = self.params["template_window_size"] | 1  # Asegurar impar
        search_size = self.params["search_window_size"] | 1  # Asegurar impar
        color_mode = self.params["color_mode"]
        
        if color_mode == 0 or len(input_img.shape) == 2:
            # Modo grayscale
            if len(input_img.shape) == 3:
                gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = input_img
            
            result = cv2.fastNlMeansDenoising(gray, None, h, template_size, search_size)
            sample = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            # Modo color
            result = cv2.fastNlMeansDenoisingColored(input_img, None, h, h, template_size, search_size)
            sample = result
        
        return {
            "denoised_image": result,
            "sample_image": sample
        }


class ThresholdAdvanced(BaseFilter):
    """Umbralización avanzada con múltiples métodos incluyendo OTSU y Adaptive"""
    
    FILTER_NAME = "ThresholdAdvanced"
    DESCRIPTION = "Umbralización con soporte para Binary, OTSU automático, y Adaptive (Mean/Gaussian)."
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "threshold_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "method": {
            "default": 0,
            "min": 0,
            "max": 7,
            "step": 1,
            "description": "Método: 0=BINARY, 1=BINARY_INV, 2=TRUNC, 3=TOZERO, 4=TOZERO_INV, 5=OTSU, 6=ADAPTIVE_MEAN, 7=ADAPTIVE_GAUSSIAN"
        },
        "threshold": {
            "default": 127,
            "min": 0,
            "max": 255,
            "step": 1,
            "description": "Valor de umbral (ignorado en OTSU y Adaptive)."
        },
        "max_value": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 1,
            "description": "Valor máximo para píxeles sobre umbral."
        },
        "block_size": {
            "default": 11,
            "min": 3,
            "max": 99,
            "step": 2,
            "description": "Tamaño de bloque para Adaptive (debe ser impar, solo métodos 6 y 7)."
        },
        "c_value": {
            "default": 2,
            "min": -20,
            "max": 20,
            "step": 1,
            "description": "Constante a restar del promedio en Adaptive (solo métodos 6 y 7)."
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        # Convertir a grayscale si es necesario
        if len(input_img.shape) == 3:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_img
        
        method = self.params["method"]
        thresh_val = self.params["threshold"]
        max_val = self.params["max_value"]
        block_size = self.params["block_size"] | 1  # Asegurar impar
        block_size = max(3, block_size)
        c_value = self.params["c_value"]
        
        if method <= 4:
            # Threshold estándar (BINARY, BINARY_INV, TRUNC, TOZERO, TOZERO_INV)
            _, result = cv2.threshold(gray, thresh_val, max_val, method)
        elif method == 5:
            # OTSU (calcula umbral automáticamente)
            _, result = cv2.threshold(gray, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 6:
            # Adaptive Mean
            result = cv2.adaptiveThreshold(
                gray, max_val, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, block_size, c_value
            )
        elif method == 7:
            # Adaptive Gaussian
            result = cv2.adaptiveThreshold(
                gray, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, c_value
            )
        else:
            _, result = cv2.threshold(gray, thresh_val, max_val, cv2.THRESH_BINARY)
        
        sample = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return {
            "threshold_image": result,
            "sample_image": sample
        }


class MorphologyAdvanced(BaseFilter):
    """Operaciones morfológicas avanzadas con opción de inversión"""
    
    FILTER_NAME = "MorphologyAdvanced"
    DESCRIPTION = "Operaciones morfológicas (erosión, dilatación, apertura, cierre, gradiente, tophat, blackhat) con opción de invertir resultado."
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "morphed_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "operation": {
            "default": 2,
            "min": 0,
            "max": 6,
            "step": 1,
            "description": "Operación: 0=Erosión, 1=Dilatación, 2=Apertura, 3=Cierre, 4=Gradiente, 5=TopHat, 6=BlackHat"
        },
        "kernel_shape": {
            "default": 2,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": "Forma del kernel: 0=Rectángulo, 1=Cruz, 2=Elipse"
        },
        "kernel_size": {
            "default": 5,
            "min": 1,
            "max": 31,
            "step": 2,
            "description": "Tamaño del kernel (se fuerza a impar)."
        },
        "iterations": {
            "default": 1,
            "min": 1,
            "max": 10,
            "step": 1,
            "description": "Número de iteraciones."
        },
        "invert_output": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Invertir colores del resultado (0=No, 1=Sí)."
        }
    }
    
    OPERATIONS = [
        cv2.MORPH_ERODE,
        cv2.MORPH_DILATE,
        cv2.MORPH_OPEN,
        cv2.MORPH_CLOSE,
        cv2.MORPH_GRADIENT,
        cv2.MORPH_TOPHAT,
        cv2.MORPH_BLACKHAT
    ]
    
    KERNEL_SHAPES = [
        cv2.MORPH_RECT,
        cv2.MORPH_CROSS,
        cv2.MORPH_ELLIPSE
    ]
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        operation = self.params["operation"]
        kernel_shape = self.params["kernel_shape"]
        ksize = self.params["kernel_size"] | 1  # Asegurar impar
        ksize = max(1, ksize)
        iterations = max(1, self.params["iterations"])
        invert = self.params["invert_output"]
        
        # Crear kernel estructurante
        kernel = cv2.getStructuringElement(
            self.KERNEL_SHAPES[kernel_shape],
            (ksize, ksize)
        )
        
        # Aplicar operación morfológica
        op = self.OPERATIONS[operation]
        result = cv2.morphologyEx(input_img, op, kernel, iterations=iterations)
        
        # Invertir si está habilitado
        if invert == 1:
            result = cv2.bitwise_not(result)
        
        # Asegurar que sample_image sea visualizable
        if len(result.shape) == 2:
            sample = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            sample = result
        
        return {
            "morphed_image": result,
            "sample_image": sample
        }


class ContourSimplify(BaseFilter):
    """Detecta y simplifica contornos usando approxPolyDP"""
    
    FILTER_NAME = "ContourSimplify"
    DESCRIPTION = "Detecta contornos, los simplifica con approxPolyDP y genera visualización con vértices. Útil para detección de formas geométricas."
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "contours_data": "contours",
        "contour_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "mode": {
            "default": 0,
            "min": 0,
            "max": 3,
            "step": 1,
            "description": "Modo de recuperación: 0=EXTERNAL, 1=LIST, 2=CCOMP, 3=TREE"
        },
        "epsilon_factor": {
            "default": 20,
            "min": 1,
            "max": 100,
            "step": 1,
            "description": "Factor de simplificación: epsilon = perímetro / este_valor. Mayor = más simplificación."
        },
        "min_area": {
            "default": 1000,
            "min": 0,
            "max": 50000,
            "step": 100,
            "description": "Área mínima del contorno para mostrarlo."
        },
        "draw_vertices": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Dibujar vértices del polígono simplificado (0=No, 1=Sí)."
        },
        "vertex_radius": {
            "default": 5,
            "min": 1,
            "max": 15,
            "step": 1,
            "description": "Radio de los círculos de vértices."
        },
        "contour_thickness": {
            "default": 2,
            "min": 1,
            "max": 5,
            "step": 1,
            "description": "Grosor de línea del contorno."
        },
        "contour_color_r": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del contorno - componente Rojo."
        },
        "contour_color_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del contorno - componente Verde."
        },
        "contour_color_b": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del contorno - componente Azul."
        },
        "vertex_color_r": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de vértices - componente Rojo."
        },
        "vertex_color_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de vértices - componente Verde."
        },
        "vertex_color_b": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de vértices - componente Azul."
        }
    }
    
    MODES = [
        cv2.RETR_EXTERNAL,
        cv2.RETR_LIST,
        cv2.RETR_CCOMP,
        cv2.RETR_TREE
    ]
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        input_img = inputs.get("input_image", original_image)
        
        # Convertir a grayscale y binarizar si es necesario
        if len(input_img.shape) == 3:
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_img
        
        # Si no es binaria, aplicar umbral
        if gray.max() > 1:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        else:
            binary = gray
        
        mode = self.MODES[self.params["mode"]]
        epsilon_factor = max(1, self.params["epsilon_factor"])
        min_area = self.params["min_area"]
        draw_vertices = self.params["draw_vertices"]
        vertex_radius = self.params["vertex_radius"]
        thickness = self.params["contour_thickness"]
        
        contour_color = (
            self.params["contour_color_b"],
            self.params["contour_color_g"],
            self.params["contour_color_r"]
        )
        vertex_color = (
            self.params["vertex_color_b"],
            self.params["vertex_color_g"],
            self.params["vertex_color_r"]
        )
        
        # Encontrar contornos
        contours, hierarchy = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear imagen para dibujar usando input_img (no original_image)
        if len(input_img.shape) == 3:
            contour_img = input_img.copy()
        else:
            contour_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        
        # Preparar datos de contornos
        contours_data = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Simplificar contorno
                perimeter = cv2.arcLength(contour, True)
                epsilon = perimeter / epsilon_factor
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Dibujar contorno simplificado
                cv2.drawContours(contour_img, [approx], -1, contour_color, thickness)
                
                # Dibujar vértices si está habilitado
                if draw_vertices == 1:
                    for point in approx:
                        cv2.circle(contour_img, tuple(point[0]), vertex_radius, vertex_color, -1)
                
                # Guardar datos
                x, y, w, h = cv2.boundingRect(approx)
                contours_data.append({
                    "area": area,
                    "perimeter": perimeter,
                    "num_vertices": len(approx),
                    "bounding_box": (x, y, w, h),
                    "simplified_points": approx.tolist(),
                    "original_points": contour.tolist()
                })
        
        return {
            "contours_data": contours_data,
            "contour_image": contour_img,
            "sample_image": contour_img
        }


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


class ClassifyLinesByAngle(BaseFilter):
    """Clasifica líneas en horizontales, verticales y otras según su ángulo"""
    
    FILTER_NAME = "ClassifyLinesByAngle"
    DESCRIPTION = "Clasifica líneas detectadas por Hough en horizontales, verticales y otras según tolerancia angular. Soporta formato HoughLines y HoughLinesP."
    
    INPUTS = {
        "base_image": "image",  # <-- AÑADIDO: para visualización
        "lines_data": "lines"
    }
    
    OUTPUTS = {
        "horizontal_lines": "lines",
        "vertical_lines": "lines",
        "other_lines": "lines",
        "classified_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "angle_tolerance": {
            "default": 15,
            "min": 1,
            "max": 45,
            "step": 1,
            "description": "Tolerancia en grados para clasificar como horizontal (cerca de 0°/180°) o vertical (cerca de 90°)."
        },
        "horizontal_color_r": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas horizontales - Rojo."
        },
        "horizontal_color_g": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas horizontales - Verde."
        },
        "horizontal_color_b": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas horizontales - Azul."
        },
        "vertical_color_r": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas verticales - Rojo."
        },
        "vertical_color_g": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas verticales - Verde."
        },
        "vertical_color_b": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas verticales - Azul."
        },
        "line_thickness": {
            "default": 2,
            "min": 1,
            "max": 5,
            "step": 1,
            "description": "Grosor de las líneas en la visualización."
        }
    }
    
    def _get_line_angle(self, x1, y1, x2, y2):
        """Calcula el ángulo de una línea en grados (0-180)."""
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle < 0:
            angle += 180
        return angle
    
    def _is_horizontal(self, x1, y1, x2, y2, tolerance):
        """Determina si una línea es aproximadamente horizontal."""
        angle = self._get_line_angle(x1, y1, x2, y2)
        return angle < tolerance or angle > (180 - tolerance)
    
    def _is_vertical(self, x1, y1, x2, y2, tolerance):
        """Determina si una línea es aproximadamente vertical."""
        angle = self._get_line_angle(x1, y1, x2, y2)
        return abs(angle - 90) < tolerance
    
    def _convert_to_points_format(self, line, img_shape):
        """
        Convierte una línea al formato (x1, y1, x2, y2).
        Soporta formato HoughLinesP (ya tiene puntos) y HoughLines (rho, theta).
        """
        if "x1" in line:
            # Formato HoughLinesP - ya tiene puntos
            return (line["x1"], line["y1"], line["x2"], line["y2"])
        elif "rho" in line:
            # Formato HoughLines standard - convertir de polar a puntos
            rho = line["rho"]
            theta = line["theta"]
            h, w = img_shape[:2]
            
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Extender la línea
            length = max(h, w) * 2
            x1 = int(x0 + length * (-b))
            y1 = int(y0 + length * (a))
            x2 = int(x0 - length * (-b))
            y2 = int(y0 - length * (a))
            
            return (x1, y1, x2, y2)
        else:
            # Formato desconocido
            return None
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        lines_data = inputs.get("lines_data", [])
        base_img = inputs.get("base_image", original_image)  # <-- MODIFICADO: fallback con advertencia
        
        tolerance = self.params["angle_tolerance"]
        thickness = self.params["line_thickness"]
        
        h_color = (
            self.params["horizontal_color_b"],
            self.params["horizontal_color_g"],
            self.params["horizontal_color_r"]
        )
        v_color = (
            self.params["vertical_color_b"],
            self.params["vertical_color_g"],
            self.params["vertical_color_r"]
        )
        
        horizontal_lines = []
        vertical_lines = []
        other_lines = []
        
        # Crear imagen para visualización usando base_img
        if len(base_img.shape) == 2:
            vis_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = base_img.copy()
        
        for line in lines_data:
            points = self._convert_to_points_format(line, base_img.shape)  # <-- Usar base_img.shape
            if points is None:
                continue
            
            x1, y1, x2, y2 = points
            angle = self._get_line_angle(x1, y1, x2, y2)
            
            line_record = {
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "angle": float(angle)
            }
            
            if self._is_horizontal(x1, y1, x2, y2, tolerance):
                horizontal_lines.append(line_record)
                cv2.line(vis_img, (x1, y1), (x2, y2), h_color, thickness)
            elif self._is_vertical(x1, y1, x2, y2, tolerance):
                vertical_lines.append(line_record)
                cv2.line(vis_img, (x1, y1), (x2, y2), v_color, thickness)
            else:
                other_lines.append(line_record)
                # Otras líneas en gris tenue
                cv2.line(vis_img, (x1, y1), (x2, y2), (128, 128, 128), 1)
        
        # Agregar texto informativo
        cv2.putText(vis_img, f"H:{len(horizontal_lines)} V:{len(vertical_lines)} Other:{len(other_lines)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return {
            "horizontal_lines": horizontal_lines,
            "vertical_lines": vertical_lines,
            "other_lines": other_lines,
            "classified_image": vis_img,
            "sample_image": vis_img
        }


class SelectBorderLines(BaseFilter):
    """Selecciona las líneas extremas que forman el borde de una página"""
    
    FILTER_NAME = "SelectBorderLines"
    DESCRIPTION = "Selecciona 4 líneas de borde (top, bottom, left, right) usando lógica de clustering y márgenes. Si no hay línea válida, usa el borde de imagen."
    
    INPUTS = {
        "base_image": "image",  # <-- AÑADIDO: para visualización
        "horizontal_lines": "lines",
        "vertical_lines": "lines"
    }
    
    OUTPUTS = {
        "selected_lines": "border_lines",
        "selection_metadata": "metadata",
        "selected_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "margin_top": {
            "default": 10,
            "min": 0,
            "max": 30,
            "step": 1,
            "description": "Margen superior (% del alto). Si línea extrema está más allá, usar borde imagen. 0=desactivado."
        },
        "margin_bottom": {
            "default": 10,
            "min": 0,
            "max": 30,
            "step": 1,
            "description": "Margen inferior (% del alto). Si línea extrema está más allá, usar borde imagen. 0=desactivado."
        },
        "margin_left": {
            "default": 10,
            "min": 0,
            "max": 30,
            "step": 1,
            "description": "Margen izquierdo (% del ancho). Si línea extrema está más allá, usar borde imagen. 0=desactivado."
        },
        "margin_right": {
            "default": 10,
            "min": 0,
            "max": 30,
            "step": 1,
            "description": "Margen derecho (% del ancho). Si línea extrema está más allá, usar borde imagen. 0=desactivado."
        },
        "cluster_top": {
            "default": 7,
            "min": 0,
            "max": 20,
            "step": 1,
            "description": "Cluster superior (% del alto). Agrupa líneas cercanas y selecciona la interior. 0=desactivado."
        },
        "cluster_bottom": {
            "default": 7,
            "min": 0,
            "max": 20,
            "step": 1,
            "description": "Cluster inferior (% del alto). Agrupa líneas cercanas y selecciona la interior. 0=desactivado."
        },
        "cluster_left": {
            "default": 7,
            "min": 0,
            "max": 20,
            "step": 1,
            "description": "Cluster izquierdo (% del ancho). Agrupa líneas cercanas y selecciona la interior. 0=desactivado."
        },
        "cluster_right": {
            "default": 7,
            "min": 0,
            "max": 20,
            "step": 1,
            "description": "Cluster derecho (% del ancho). Agrupa líneas cercanas y selecciona la interior. 0=desactivado."
        },
        "line_thickness": {
            "default": 2,
            "min": 1,
            "max": 5,
            "step": 1,
            "description": "Grosor de las líneas en la visualización."
        },
        "selected_color_r": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas seleccionadas - Rojo."
        },
        "selected_color_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas seleccionadas - Verde."
        },
        "selected_color_b": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de líneas seleccionadas - Azul."
        },
        "border_color_r": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de bordes de imagen usados - Rojo."
        },
        "border_color_g": {
            "default": 165,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de bordes de imagen usados - Verde."
        },
        "border_color_b": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de bordes de imagen usados - Azul."
        }
    }
    
    def _get_line_y_at_x(self, x1, y1, x2, y2, x):
        """Calcula la coordenada Y de una línea en una posición X dada."""
        if x2 == x1:
            return (y1 + y2) / 2
        slope = (y2 - y1) / (x2 - x1)
        return y1 + slope * (x - x1)
    
    def _get_line_x_at_y(self, x1, y1, x2, y2, y):
        """Calcula la coordenada X de una línea en una posición Y dada."""
        if y2 == y1:
            return (x1 + x2) / 2
        slope = (x2 - x1) / (y2 - y1)
        return x1 + slope * (y - y1)
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        horizontal_lines = inputs.get("horizontal_lines", [])
        vertical_lines = inputs.get("vertical_lines", [])
        base_img = inputs.get("base_image", original_image)  # <-- MODIFICADO: fallback con advertencia
        
        h, w = base_img.shape[:2]  # <-- Usar base_img, no original_image
        center_x, center_y = w // 2, h // 2
        
        # Obtener parámetros
        margin_top_pct = self.params["margin_top"]
        margin_bottom_pct = self.params["margin_bottom"]
        margin_left_pct = self.params["margin_left"]
        margin_right_pct = self.params["margin_right"]
        
        cluster_top_pct = self.params["cluster_top"]
        cluster_bottom_pct = self.params["cluster_bottom"]
        cluster_left_pct = self.params["cluster_left"]
        cluster_right_pct = self.params["cluster_right"]
        
        thickness = self.params["line_thickness"]
        
        selected_color = (
            self.params["selected_color_b"],
            self.params["selected_color_g"],
            self.params["selected_color_r"]
        )
        border_color = (
            self.params["border_color_b"],
            self.params["border_color_g"],
            self.params["border_color_r"]
        )
        
        # Calcular márgenes y clusters en píxeles
        margin_top_px = int(h * margin_top_pct / 100) if margin_top_pct > 0 else 0
        margin_bottom_px = int(h * margin_bottom_pct / 100) if margin_bottom_pct > 0 else 0
        margin_left_px = int(w * margin_left_pct / 100) if margin_left_pct > 0 else 0
        margin_right_px = int(w * margin_right_pct / 100) if margin_right_pct > 0 else 0
        
        cluster_top_px = int(h * cluster_top_pct / 100) if cluster_top_pct > 0 else 0
        cluster_bottom_px = int(h * cluster_bottom_pct / 100) if cluster_bottom_pct > 0 else 0
        cluster_left_px = int(w * cluster_left_pct / 100) if cluster_left_pct > 0 else 0
        cluster_right_px = int(w * cluster_right_pct / 100) if cluster_right_pct > 0 else 0
        
        # Calcular posición característica de cada línea
        # Para horizontales: Y en el centro de la imagen
        # Para verticales: X en el centro de la imagen
        horizontal_with_y = []
        for line in horizontal_lines:
            x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
            y_at_center = self._get_line_y_at_x(x1, y1, x2, y2, center_x)
            horizontal_with_y.append({**line, "pos": y_at_center})
        
        vertical_with_x = []
        for line in vertical_lines:
            x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
            x_at_center = self._get_line_x_at_y(x1, y1, x2, y2, center_y)
            vertical_with_x.append({**line, "pos": x_at_center})
        
        # Estructuras de resultado
        selected = {"top": None, "bottom": None, "left": None, "right": None}
        metadata = {
            "image_size": {"width": w, "height": h},
            "margins_percent": {
                "top": margin_top_pct, "bottom": margin_bottom_pct,
                "left": margin_left_pct, "right": margin_right_pct
            },
            "margins_pixels": {
                "top": margin_top_px, "bottom": margin_bottom_px,
                "left": margin_left_px, "right": margin_right_px
            },
            "clusters_percent": {
                "top": cluster_top_pct, "bottom": cluster_bottom_pct,
                "left": cluster_left_pct, "right": cluster_right_pct
            },
            "clusters_pixels": {
                "top": cluster_top_px, "bottom": cluster_bottom_px,
                "left": cluster_left_px, "right": cluster_right_px
            },
            "top_is_image_border": False,
            "bottom_is_image_border": False,
            "left_is_image_border": False,
            "right_is_image_border": False
        }
        
        # --- Seleccionar TOP (horizontal con menor Y) ---
        if horizontal_with_y:
            sorted_by_y = sorted(horizontal_with_y, key=lambda x: x["pos"])
            extreme = sorted_by_y[0]
            
            if cluster_top_px > 0:
                cluster = [l for l in sorted_by_y if abs(l["pos"] - extreme["pos"]) <= cluster_top_px]
                candidate = sorted(cluster, key=lambda x: x["pos"], reverse=True)[0]
                metadata["top_cluster_size"] = len(cluster)
            else:
                candidate = extreme
                metadata["top_cluster_size"] = 1
            
            metadata["top_extreme_y"] = extreme["pos"]
            metadata["top_selected_y"] = candidate["pos"]
            
            if margin_top_pct > 0:
                if candidate["pos"] <= margin_top_px:
                    selected["top"] = candidate
                else:
                    metadata["top_is_image_border"] = True
                    metadata["top_candidate_rejected_y"] = candidate["pos"]
            else:
                selected["top"] = candidate
        else:
            metadata["top_is_image_border"] = True
        
        # --- Seleccionar BOTTOM (horizontal con mayor Y) ---
        if horizontal_with_y:
            sorted_by_y = sorted(horizontal_with_y, key=lambda x: x["pos"], reverse=True)
            extreme = sorted_by_y[0]
            
            if cluster_bottom_px > 0:
                cluster = [l for l in sorted_by_y if abs(l["pos"] - extreme["pos"]) <= cluster_bottom_px]
                candidate = sorted(cluster, key=lambda x: x["pos"])[0]
                metadata["bottom_cluster_size"] = len(cluster)
            else:
                candidate = extreme
                metadata["bottom_cluster_size"] = 1
            
            metadata["bottom_extreme_y"] = extreme["pos"]
            metadata["bottom_selected_y"] = candidate["pos"]
            
            if margin_bottom_pct > 0:
                if candidate["pos"] >= (h - margin_bottom_px):
                    selected["bottom"] = candidate
                else:
                    metadata["bottom_is_image_border"] = True
                    metadata["bottom_candidate_rejected_y"] = candidate["pos"]
            else:
                selected["bottom"] = candidate
        else:
            metadata["bottom_is_image_border"] = True
        
        # --- Seleccionar LEFT (vertical con menor X) ---
        if vertical_with_x:
            sorted_by_x = sorted(vertical_with_x, key=lambda x: x["pos"])
            extreme = sorted_by_x[0]
            
            if cluster_left_px > 0:
                cluster = [l for l in sorted_by_x if abs(l["pos"] - extreme["pos"]) <= cluster_left_px]
                candidate = sorted(cluster, key=lambda x: x["pos"], reverse=True)[0]
                metadata["left_cluster_size"] = len(cluster)
            else:
                candidate = extreme
                metadata["left_cluster_size"] = 1
            
            metadata["left_extreme_x"] = extreme["pos"]
            metadata["left_selected_x"] = candidate["pos"]
            
            if margin_left_pct > 0:
                if candidate["pos"] <= margin_left_px:
                    selected["left"] = candidate
                else:
                    metadata["left_is_image_border"] = True
                    metadata["left_candidate_rejected_x"] = candidate["pos"]
            else:
                selected["left"] = candidate
        else:
            metadata["left_is_image_border"] = True
        
        # --- Seleccionar RIGHT (vertical con mayor X) ---
        if vertical_with_x:
            sorted_by_x = sorted(vertical_with_x, key=lambda x: x["pos"], reverse=True)
            extreme = sorted_by_x[0]
            
            if cluster_right_px > 0:
                cluster = [l for l in sorted_by_x if abs(l["pos"] - extreme["pos"]) <= cluster_right_px]
                candidate = sorted(cluster, key=lambda x: x["pos"])[0]
                metadata["right_cluster_size"] = len(cluster)
            else:
                candidate = extreme
                metadata["right_cluster_size"] = 1
            
            metadata["right_extreme_x"] = extreme["pos"]
            metadata["right_selected_x"] = candidate["pos"]
            
            if margin_right_pct > 0:
                if candidate["pos"] >= (w - margin_right_px):
                    selected["right"] = candidate
                else:
                    metadata["right_is_image_border"] = True
                    metadata["right_candidate_rejected_x"] = candidate["pos"]
            else:
                selected["right"] = candidate
        else:
            metadata["right_is_image_border"] = True
        
        # Crear imagen de visualización usando base_img
        if len(base_img.shape) == 2:
            vis_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = base_img.copy()
        
        # Dibujar líneas seleccionadas o bordes de imagen
        for name, line in selected.items():
            is_border = metadata.get(f"{name}_is_image_border", False)
            
            if line is not None:
                x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
                cv2.line(vis_img, (x1, y1), (x2, y2), selected_color, thickness + 1)
            elif is_border:
                if name == "top":
                    cv2.line(vis_img, (0, 0), (w, 0), border_color, thickness)
                elif name == "bottom":
                    cv2.line(vis_img, (0, h-1), (w, h-1), border_color, thickness)
                elif name == "left":
                    cv2.line(vis_img, (0, 0), (0, h), border_color, thickness)
                elif name == "right":
                    cv2.line(vis_img, (w-1, 0), (w-1, h), border_color, thickness)
        
        # Convertir selected a formato serializable (sin 'pos')
        selected_clean = {}
        for name, line in selected.items():
            if line is not None:
                selected_clean[name] = {
                    "x1": line["x1"], "y1": line["y1"],
                    "x2": line["x2"], "y2": line["y2"],
                    "angle": line.get("angle", 0)
                }
            else:
                selected_clean[name] = None
        
        return {
            "selected_lines": selected_clean,
            "selection_metadata": metadata,
            "selected_image": vis_img,
            "sample_image": vis_img
        }


class CalculateQuadCorners(BaseFilter):
    """Calcula las 4 esquinas de un cuadrilátero a partir de líneas de borde"""
    
    FILTER_NAME = "CalculateQuadCorners"
    DESCRIPTION = "Calcula las 4 esquinas (top_left, top_right, bottom_left, bottom_right) intersectando las líneas de borde seleccionadas. Genera polígono de recorte."
    
    INPUTS = {
        "base_image": "image",  # <-- AÑADIDO: para visualización
        "selected_lines": "border_lines",
        "selection_metadata": "metadata"
    }
    
    OUTPUTS = {
        "corners": "quad_points",
        "corners_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "corner_radius": {
            "default": 10,
            "min": 3,
            "max": 30,
            "step": 1,
            "description": "Radio de los círculos que marcan las esquinas."
        },
        "corner_color_r": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de las esquinas - Rojo."
        },
        "corner_color_g": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de las esquinas - Verde."
        },
        "corner_color_b": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de las esquinas - Azul."
        },
        "polygon_color_r": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del polígono - Rojo."
        },
        "polygon_color_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del polígono - Verde."
        },
        "polygon_color_b": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color del polígono - Azul."
        },
        "polygon_thickness": {
            "default": 3,
            "min": 1,
            "max": 10,
            "step": 1,
            "description": "Grosor del polígono."
        },
        "draw_labels": {
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Dibujar etiquetas de esquinas (0=No, 1=Sí)."
        }
    }
    
    def _line_intersection(self, line1, line2):
        """
        Calcula la intersección de dos líneas definidas por puntos.
        line1 y line2 son dicts con x1, y1, x2, y2
        Retorna (x, y) o None si son paralelas.
        """
        x1, y1 = line1["x1"], line1["y1"]
        x2, y2 = line1["x2"], line1["y2"]
        x3, y3 = line2["x1"], line2["y1"]
        x4, y4 = line2["x2"], line2["y2"]
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (int(round(x)), int(round(y)))
    
    def _get_line_y_at_x(self, line, x):
        """Calcula Y de una línea en posición X."""
        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
        if x2 == x1:
            return (y1 + y2) / 2
        slope = (y2 - y1) / (x2 - x1)
        return y1 + slope * (x - x1)
    
    def _get_line_x_at_y(self, line, y):
        """Calcula X de una línea en posición Y."""
        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
        if y2 == y1:
            return (x1 + x2) / 2
        slope = (x2 - x1) / (y2 - y1)
        return x1 + slope * (y - y1)
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        selected_lines = inputs.get("selected_lines", {})
        metadata = inputs.get("selection_metadata", {})
        base_img = inputs.get("base_image", original_image)  # <-- MODIFICADO: fallback con advertencia
        
        h, w = base_img.shape[:2]  # <-- Usar base_img, no original_image
        
        radius = self.params["corner_radius"]
        thickness = self.params["polygon_thickness"]
        draw_labels = self.params["draw_labels"]
        
        corner_color = (
            self.params["corner_color_b"],
            self.params["corner_color_g"],
            self.params["corner_color_r"]
        )
        polygon_color = (
            self.params["polygon_color_b"],
            self.params["polygon_color_g"],
            self.params["polygon_color_r"]
        )
        
        # Definir las 4 esquinas y sus líneas componentes
        corner_defs = [
            ("top_left", "top", "left"),
            ("top_right", "top", "right"),
            ("bottom_left", "bottom", "left"),
            ("bottom_right", "bottom", "right")
        ]
        
        corners = {}
        
        for corner_name, h_name, v_name in corner_defs:
            h_line = selected_lines.get(h_name)
            v_line = selected_lines.get(v_name)
            h_is_border = metadata.get(f"{h_name}_is_image_border", False)
            v_is_border = metadata.get(f"{v_name}_is_image_border", False)
            
            if h_line is not None and v_line is not None:
                # Ambas son líneas detectadas: calcular intersección
                intersection = self._line_intersection(h_line, v_line)
                if intersection is not None:
                    corners[corner_name] = {"x": intersection[0], "y": intersection[1], "type": "intersection"}
                else:
                    corners[corner_name] = None
            elif h_is_border and v_is_border:
                # Ambas son bordes de imagen: usar esquina de imagen
                if corner_name == "top_left":
                    corners[corner_name] = {"x": 0, "y": 0, "type": "image_corner"}
                elif corner_name == "top_right":
                    corners[corner_name] = {"x": w - 1, "y": 0, "type": "image_corner"}
                elif corner_name == "bottom_left":
                    corners[corner_name] = {"x": 0, "y": h - 1, "type": "image_corner"}
                elif corner_name == "bottom_right":
                    corners[corner_name] = {"x": w - 1, "y": h - 1, "type": "image_corner"}
            elif h_is_border and v_line is not None:
                # Horizontal es borde, vertical es línea
                y_border = 0 if h_name == "top" else h - 1
                x_at_border = self._get_line_x_at_y(v_line, y_border)
                corners[corner_name] = {"x": int(round(x_at_border)), "y": y_border, "type": "mixed_h_border"}
            elif v_is_border and h_line is not None:
                # Vertical es borde, horizontal es línea
                x_border = 0 if v_name == "left" else w - 1
                y_at_border = self._get_line_y_at_x(h_line, x_border)
                corners[corner_name] = {"x": x_border, "y": int(round(y_at_border)), "type": "mixed_v_border"}
            else:
                corners[corner_name] = None
        
        # Crear imagen de visualización usando base_img
        if len(base_img.shape) == 2:
            vis_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = base_img.copy()
        
        # Recolectar esquinas válidas en orden para el polígono
        corner_order = ["top_left", "top_right", "bottom_right", "bottom_left"]
        valid_corners = []
        
        for name in corner_order:
            if corners.get(name) is not None:
                valid_corners.append((corners[name]["x"], corners[name]["y"]))
        
        # Dibujar polígono si tenemos las 4 esquinas
        if len(valid_corners) == 4:
            polygon = np.array(valid_corners, dtype=np.int32)
            cv2.polylines(vis_img, [polygon], True, polygon_color, thickness)
            
            # Calcular área
            area = cv2.contourArea(polygon)
            area_percentage = (area / (w * h)) * 100
            
            # Agregar info al metadata
            corners["_polygon_area"] = float(area)
            corners["_polygon_area_percent"] = round(area_percentage, 2)
            corners["_valid"] = True
        else:
            corners["_valid"] = False
            corners["_corners_found"] = len(valid_corners)
        
        # Dibujar esquinas
        for name in corner_order:
            corner = corners.get(name)
            if corner is not None and "x" in corner:
                pt = (corner["x"], corner["y"])
                cv2.circle(vis_img, pt, radius, corner_color, -1)
                cv2.circle(vis_img, pt, radius + 3, corner_color, 2)
                
                if draw_labels == 1:
                    # Ajustar posición del label según la esquina
                    if "left" in name:
                        label_x = pt[0] + radius + 5
                    else:
                        label_x = pt[0] - radius - 80
                    if "top" in name:
                        label_y = pt[1] + radius + 20
                    else:
                        label_y = pt[1] - radius - 5
                    
                    cv2.putText(vis_img, name, (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, corner_color, 1)
        
        return {
            "corners": corners,
            "corners_image": vis_img,
            "sample_image": vis_img
        }


# =============================================================================
# FIN de Filtros basados en detectar_bordes_hough_probabilistico.py
# =============================================================================

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_filter(name: str) -> type:
    """Obtiene una clase de filtro por nombre"""
    return FILTER_REGISTRY.get(name)


def list_filters() -> List[str]:
    """Lista todos los filtros disponibles"""
    return list(FILTER_REGISTRY.keys())


def get_filter_info(name: str) -> Dict:
    """Obtiene información sobre un filtro"""
    filter_class = FILTER_REGISTRY.get(name)
    if filter_class:
        return {
            "name": filter_class.FILTER_NAME,
            "description": filter_class.DESCRIPTION,
            "inputs": filter_class.INPUTS,
            "outputs": filter_class.OUTPUTS,
            "params": filter_class.PARAMS
        }
    return None


if __name__ == "__main__":
    # Test: listar filtros disponibles
    print("Filtros disponibles:")
    print("-" * 40)
    for name in list_filters():
        info = get_filter_info(name)
        print(f"  {name}: {info['description']}")
    print("-" * 40)
    
    # Mostrar ayuda de un filtro
    f = GrayscaleFilter()
    print(f.get_help())
