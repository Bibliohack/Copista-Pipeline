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
    INPUTS = {}  # No requiere inputs de otros filtros, usa la imagen original
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
        mode = self.params["mode"]
        interp = self.INTERPOLATION_METHODS[self.params["interpolation"]]
        
        if mode == 0:
            scale = self.params["scale_percent"] / 100.0
            new_width = int(original_image.shape[1] * scale)
            new_height = int(original_image.shape[0] * scale)
        else:
            new_width = self.params["width"]
            new_height = self.params["height"]
        
        resized = cv2.resize(original_image, (new_width, new_height), interpolation=interp)
        
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
        "edge_image": "image"
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
        if edge_img is None:
            # Si no hay edge_image, usar la imagen original
            if len(original_image.shape) == 3:
                edge_img = cv2.Canny(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY), 50, 150)
            else:
                edge_img = cv2.Canny(original_image, 50, 150)
        
        rho = self.params["rho"]
        theta = np.pi / self.params["theta_divisor"]
        threshold = self.params["threshold"]
        
        # Crear imagen para visualización
        if len(original_image.shape) == 3:
            sample = original_image.copy()
        else:
            sample = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
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
        
        # Crear imagen de contornos
        if len(original_image.shape) == 3:
            contour_img = original_image.copy()
        else:
            contour_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
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
