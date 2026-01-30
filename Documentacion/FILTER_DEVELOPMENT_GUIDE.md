# Guía Técnica: Creación de Filtros para filter_library

## Resumen

Este documento explica cómo crear nuevos filtros para el sistema de procesamiento de imágenes. Cada filtro es una clase Python que hereda de `BaseFilter`, se ubica en su propio archivo y se registra automáticamente en el sistema.

---

## Arquitectura del Sistema

### Flujo de datos

```
Imagen Original
      ↓
  [Filtro 0] → outputs: {output_name: data, sample_image: img}
      ↓
  [Filtro 1] → recibe inputs de filtros anteriores → produce outputs
      ↓
  [Filtro N] → ...
      ↓
  Resultado Final
```

### Estructura de archivos

```
proyecto/
├── filter_library/
│   ├── __init__.py              # Exporta todos los filtros
│   ├── base_filter.py           # Clase base y FILTER_REGISTRY
│   ├── resize_filter.py         # Un filtro por archivo
│   ├── grayscale_filter.py
│   ├── gaussian_blur_filter.py
│   ├── canny_edge_filter.py
│   ├── ...
│   └── mi_nuevo_filtro.py       # Tu nuevo filtro
├── pipeline.json                # Define qué filtros usar y cómo conectarlos
├── params.json                  # Almacena los valores de parámetros configurados
└── param_configurator.py        # GUI para ajustar parámetros y visualizar resultados
```

### Convención de nombres

| Clase | Archivo |
|-------|---------|
| `MiFiltro` | `mi_filtro.py` |
| `GaussianBlurFilter` | `gaussian_blur_filter.py` |
| `CannyEdgeFilter` | `canny_edge_filter.py` |
| `ThresholdAdvanced` | `threshold_advanced.py` |

Se usa **snake_case** para archivos, **PascalCase** para clases.

---

## Crear un Nuevo Filtro

### Paso 1: Crear el archivo

Crear `filter_library/mi_filtro.py`:

```python
"""
Filtro: MiFiltro
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class MiFiltro(BaseFilter):
    """Descripción breve del filtro"""
    
    # === ATRIBUTOS DE CLASE OBLIGATORIOS ===
    
    FILTER_NAME = "MiFiltro"  # Identificador único, usado en pipeline.json
    DESCRIPTION = "Descripción detallada de lo que hace el filtro"
    
    INPUTS = {
        # Entradas que este filtro necesita de filtros anteriores
        # Formato: "nombre_input": "tipo_dato"
        "input_image": "image"
    }
    
    OUTPUTS = {
        # Salidas que este filtro produce
        # Formato: "nombre_output": "tipo_dato"
        # OBLIGATORIO: siempre debe incluir "sample_image": "image"
        "mi_resultado": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        # Parámetros ajustables por el usuario
        # Cada parámetro es un dict con: default, min, max, step, description
        "mi_parametro": {
            "default": 50,
            "min": 0,
            "max": 100,
            "step": 5,
            "description": "Descripción del parámetro para el usuario"
        }
    }
    
    # === MÉTODO OBLIGATORIO ===
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        """
        Procesa el filtro.
        
        Args:
            inputs: Dict con datos de filtros anteriores (según INPUTS definidos)
            original_image: Imagen original sin procesar (siempre disponible)
        
        Returns:
            Dict con todos los outputs definidos en OUTPUTS
        """
        # Obtener imagen de entrada
        img = inputs.get("input_image", original_image)
        
        # Obtener parámetros
        param_valor = self.params["mi_parametro"]
        
        # Procesar...
        resultado = self._hacer_algo(img, param_valor)
        
        # Retornar TODOS los outputs declarados en OUTPUTS
        return {
            "mi_resultado": resultado,
            "sample_image": resultado  # Para visualización
        }
    
    # === MÉTODOS AUXILIARES (opcionales) ===
    
    def _hacer_algo(self, img, valor):
        """Método privado auxiliar"""
        # ...
        return img
```

### Paso 2: Registrar en \_\_init\_\_.py

Editar `filter_library/__init__.py` y agregar:

```python
# En la sección de imports
from .mi_filtro import MiFiltro

# En la lista __all__
__all__ = [
    # ... otros filtros ...
    "MiFiltro",
]
```

### Paso 3: Usar en pipeline.json

```json
{
    "3": {
        "filter_name": "MiFiltro",
        "inputs": {
            "input_image": "2.otra_salida"
        }
    }
}
```

¡Listo! El filtro se registra automáticamente al importar el módulo.

---

## Atributos de Clase en Detalle

### FILTER_NAME (str, obligatorio)

Identificador único del filtro. Se usa en `pipeline.json` para referenciar el filtro.

```python
FILTER_NAME = "GaussianBlur"
```

**Reglas:**
- Debe ser único en toda la biblioteca
- Usar PascalCase
- Sin espacios ni caracteres especiales

---

### DESCRIPTION (str, obligatorio)

Descripción legible del filtro. Se muestra en la ayuda (tecla `h`).

```python
DESCRIPTION = "Aplica un filtro de desenfoque gaussiano a la imagen"
```

---

### INPUTS (Dict[str, str], obligatorio)

Define qué datos necesita el filtro de filtros anteriores.

```python
INPUTS = {
    "input_image": "image",      # Una imagen
    "edge_image": "image",       # Otra imagen (ej: bordes)
    "lines_data": "lines"        # Datos estructurados
}
```

**Formato:** `"nombre_que_usarás_en_process": "tipo_de_dato"`

**Tipos de datos comunes:**
| Tipo | Descripción | Ejemplo de uso |
|------|-------------|----------------|
| `"image"` | numpy.ndarray (imagen BGR o grayscale) | Resultado de cualquier filtro de imagen |
| `"lines"` | Lista de líneas detectadas | Salida de HoughLines |
| `"contours"` | Lista de contornos | Salida de detección de contornos |
| `"histogram"` | Datos de histograma | Salida de cálculo de histograma |
| `"border_lines"` | Líneas de borde seleccionadas | Salida de SelectBorderLines |
| `"quad_points"` | 4 puntos de un cuadrilátero | Salida de CalculateQuadCorners |
| `"metadata"` | Diccionario con metadatos | Información adicional del procesamiento |

**Si el filtro no necesita inputs de otros filtros:**
```python
INPUTS = {}  # Usará original_image directamente
```

---

### OUTPUTS (Dict[str, str], obligatorio)

Define qué datos produce el filtro.

```python
OUTPUTS = {
    "blurred_image": "image",    # Output principal
    "sample_image": "image"      # OBLIGATORIO para visualización
}
```

**IMPORTANTE:** 
- `"sample_image": "image"` es **OBLIGATORIO** en todos los filtros
- `sample_image` es lo que se muestra en el visualizador GUI
- Puede ser igual al output principal o una representación visual de datos

**Ejemplo para filtro que produce datos (no imagen):**
```python
OUTPUTS = {
    "lines_data": "lines",       # Datos estructurados
    "sample_image": "image"      # Visualización de las líneas
}
```

---

### PARAMS (Dict[str, Dict], obligatorio)

Define parámetros ajustables por el usuario mediante trackbars.

```python
PARAMS = {
    "threshold": {
        "default": 127,      # Valor inicial
        "min": 0,            # Valor mínimo del trackbar
        "max": 255,          # Valor máximo del trackbar
        "step": 1,           # Incremento al ajustar
        "description": "Umbral de binarización (0-255)"
    },
    "kernel_size": {
        "default": 5,
        "min": 1,
        "max": 31,
        "step": 2,           # Solo valores impares
        "description": "Tamaño del kernel (debe ser impar)"
    }
}
```

**Si el filtro no tiene parámetros:**
```python
PARAMS = {}
```

**Limitaciones de los trackbars de OpenCV:**
- Solo valores enteros
- El step se aplica al incrementar/decrementar

**Workaround para valores decimales:**
```python
# Definir como entero 0-100, luego dividir en process()
"opacity": {
    "default": 100,
    "min": 0,
    "max": 100,
    "step": 5,
    "description": "Opacidad en porcentaje (0-100)"
}

# En process():
opacity = self.params["opacity"] / 100.0  # Convertir a 0.0-1.0
```

---

## Método process() en Detalle

### Firma

```python
def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
```

### Parámetros de entrada

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `inputs` | `Dict[str, Any]` | Datos de filtros anteriores según lo definido en `INPUTS` |
| `original_image` | `np.ndarray` | Imagen original cargada, siempre disponible |

### Acceso a inputs

```python
def process(self, inputs, original_image):
    # Obtener input con fallback a original_image
    img = inputs.get("input_image", original_image)
    
    # Obtener input obligatorio (lanzará KeyError si no existe)
    edge_img = inputs["edge_image"]
    
    # Verificar si un input existe
    if "optional_data" in inputs:
        data = inputs["optional_data"]
```

### Acceso a parámetros

```python
def process(self, inputs, original_image):
    # Los parámetros están en self.params
    threshold = self.params["threshold"]
    kernel_size = self.params["kernel_size"]
    
    # Validación/corrección de parámetros
    if kernel_size % 2 == 0:
        kernel_size += 1  # Asegurar impar
```

### Valor de retorno

**DEBE** retornar un diccionario con **TODOS** los outputs definidos en `OUTPUTS`:

```python
def process(self, inputs, original_image):
    # ... procesamiento ...
    
    return {
        "output_1": resultado_1,
        "output_2": resultado_2,
        "sample_image": imagen_para_visualizar  # OBLIGATORIO
    }
```

---

## Consideraciones sobre sample_image

### Caso 1: El output principal es una imagen

```python
OUTPUTS = {
    "blurred_image": "image",
    "sample_image": "image"
}

def process(self, inputs, original_image):
    img = inputs.get("input_image", original_image)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    return {
        "blurred_image": blurred,
        "sample_image": blurred  # Mismo que output principal
    }
```

### Caso 2: El output principal son datos (no imagen)

```python
OUTPUTS = {
    "histogram_data": "histogram",
    "sample_image": "image"
}

def process(self, inputs, original_image):
    img = inputs.get("input_image", original_image)
    
    # Calcular histograma (datos)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    # Crear visualización del histograma (imagen)
    hist_img = np.zeros((300, 256, 3), dtype=np.uint8)
    # ... dibujar histograma en hist_img ...
    
    return {
        "histogram_data": hist,      # Datos para otros filtros
        "sample_image": hist_img     # Visualización para el usuario
    }
```

### Caso 3: Filtro de visualización (combina datos de múltiples fuentes)

```python
OUTPUTS = {
    "overlay_image": "image",
    "sample_image": "image"
}

def process(self, inputs, original_image):
    base_img = inputs.get("base_image", original_image)
    lines = inputs.get("lines_data", [])
    
    # Dibujar líneas sobre la imagen
    result = base_img.copy()
    for line in lines:
        cv2.line(result, ...)
    
    return {
        "overlay_image": result,
        "sample_image": result
    }
```

---

## Consideraciones sobre Dimensiones de Imagen

### Problema con original_image

Si el pipeline incluye un filtro `Resize`, las dimensiones de `original_image` no coincidirán con las de las imágenes procesadas. 

**Incorrecto:**
```python
def process(self, inputs, original_image):
    h, w = original_image.shape[:2]  # ¡Dimensiones incorrectas!
    # ...
```

**Correcto - usar imagen de referencia del pipeline:**
```python
INPUTS = {
    "base_image": "image",  # Imagen de referencia para dimensiones
    "lines_data": "lines"
}

def process(self, inputs, original_image):
    base_img = inputs.get("base_image", original_image)
    h, w = base_img.shape[:2]  # Dimensiones correctas
    # ...
```

---

## Imports Necesarios

Cada archivo de filtro debe incluir sus propios imports:

```python
"""
Filtro: MiFiltro
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY

# Imports adicionales según necesidad:
import warnings  # Si usas warnings.warn()
import math      # Si usas funciones matemáticas
```

**Nota:** Los imports no se heredan de otros archivos. Cada módulo debe importar explícitamente lo que usa.

---

## Compatibilidad con el Sistema de Cache

### Restricción importante

**Solo los filtros que producen exclusivamente imágenes pueden estar antes de un checkpoint.**

El sistema de cache guarda el `sample_image` de cada filtro. Si un filtro produce datos estructurados (como `lines_data`), esos datos no se pueden recuperar del cache.

### Filtros compatibles con checkpoint (todos los outputs son "image"):

```python
OUTPUTS = {
    "grayscale_image": "image",
    "sample_image": "image"
}
```

### Filtros NO compatibles con checkpoint (tienen outputs no-imagen):

```python
OUTPUTS = {
    "lines_data": "lines",      # ← Esto impide checkpoint
    "sample_image": "image"
}
```

---

## Ejemplos Completos

### Ejemplo 1: Filtro simple de imagen

Archivo: `filter_library/invert_filter.py`

```python
"""
Filtro: InvertFilter
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class InvertFilter(BaseFilter):
    """Invierte los colores de la imagen"""
    
    FILTER_NAME = "Invert"
    DESCRIPTION = "Invierte los colores de la imagen (negativo)"
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "inverted_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {}  # Sin parámetros
    
    def process(self, inputs, original_image):
        img = inputs.get("input_image", original_image)
        inverted = cv2.bitwise_not(img)
        
        return {
            "inverted_image": inverted,
            "sample_image": inverted
        }
```

### Ejemplo 2: Filtro con múltiples parámetros

Archivo: `filter_library/bilateral_filter.py`

```python
"""
Filtro: BilateralFilter
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class BilateralFilter(BaseFilter):
    """Filtro bilateral para suavizado preservando bordes"""
    
    FILTER_NAME = "Bilateral"
    DESCRIPTION = "Suaviza la imagen preservando los bordes"
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "filtered_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "d": {
            "default": 9,
            "min": 1,
            "max": 25,
            "step": 2,
            "description": "Diámetro del vecindario de píxeles"
        },
        "sigma_color": {
            "default": 75,
            "min": 10,
            "max": 200,
            "step": 5,
            "description": "Sigma en el espacio de color"
        },
        "sigma_space": {
            "default": 75,
            "min": 10,
            "max": 200,
            "step": 5,
            "description": "Sigma en el espacio de coordenadas"
        }
    }
    
    def process(self, inputs, original_image):
        img = inputs.get("input_image", original_image)
        
        d = self.params["d"]
        sigma_color = self.params["sigma_color"]
        sigma_space = self.params["sigma_space"]
        
        filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        
        return {
            "filtered_image": filtered,
            "sample_image": filtered
        }
```

### Ejemplo 3: Filtro que produce datos estructurados

Archivo: `filter_library/circle_detector.py`

```python
"""
Filtro: CircleDetector
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class CircleDetector(BaseFilter):
    """Detecta círculos usando la transformada de Hough"""
    
    FILTER_NAME = "CircleDetector"
    DESCRIPTION = "Detecta círculos en la imagen usando HoughCircles"
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "circles_data": "circles",   # Datos estructurados
        "sample_image": "image"      # Visualización
    }
    
    PARAMS = {
        "dp": {
            "default": 1,
            "min": 1,
            "max": 3,
            "step": 1,
            "description": "Ratio de resolución del acumulador"
        },
        "min_dist": {
            "default": 50,
            "min": 10,
            "max": 200,
            "step": 10,
            "description": "Distancia mínima entre centros de círculos"
        },
        "param1": {
            "default": 50,
            "min": 10,
            "max": 200,
            "step": 10,
            "description": "Umbral superior para Canny"
        },
        "param2": {
            "default": 30,
            "min": 10,
            "max": 100,
            "step": 5,
            "description": "Umbral del acumulador"
        },
        "min_radius": {
            "default": 10,
            "min": 0,
            "max": 100,
            "step": 5,
            "description": "Radio mínimo de círculos"
        },
        "max_radius": {
            "default": 100,
            "min": 10,
            "max": 300,
            "step": 10,
            "description": "Radio máximo de círculos"
        }
    }
    
    def process(self, inputs, original_image):
        img = inputs.get("input_image", original_image)
        
        # Convertir a grayscale si es necesario
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Detectar círculos
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.params["dp"],
            minDist=self.params["min_dist"],
            param1=self.params["param1"],
            param2=self.params["param2"],
            minRadius=self.params["min_radius"],
            maxRadius=self.params["max_radius"]
        )
        
        # Preparar datos estructurados
        circles_data = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                circles_data.append({
                    "x": int(circle[0]),
                    "y": int(circle[1]),
                    "radius": int(circle[2])
                })
        
        # Crear visualización
        if len(img.shape) == 2:
            sample = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            sample = img.copy()
        
        for c in circles_data:
            cv2.circle(sample, (c["x"], c["y"]), c["radius"], (0, 255, 0), 2)
            cv2.circle(sample, (c["x"], c["y"]), 2, (0, 0, 255), 3)
        
        return {
            "circles_data": circles_data,
            "sample_image": sample
        }
```

### Ejemplo 4: Filtro que combina múltiples inputs

Archivo: `filter_library/blend_images.py`

```python
"""
Filtro: BlendImages
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class BlendImages(BaseFilter):
    """Mezcla dos imágenes con transparencia ajustable"""
    
    FILTER_NAME = "BlendImages"
    DESCRIPTION = "Combina dos imágenes usando alpha blending"
    
    INPUTS = {
        "image_a": "image",
        "image_b": "image"
    }
    
    OUTPUTS = {
        "blended_image": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "alpha": {
            "default": 50,
            "min": 0,
            "max": 100,
            "step": 5,
            "description": "Peso de imagen A (0-100, imagen B = 100 - alpha)"
        }
    }
    
    def process(self, inputs, original_image):
        img_a = inputs.get("image_a", original_image)
        img_b = inputs.get("image_b", original_image)
        
        # Asegurar mismo tamaño
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        
        alpha = self.params["alpha"] / 100.0
        beta = 1.0 - alpha
        
        blended = cv2.addWeighted(img_a, alpha, img_b, beta, 0)
        
        return {
            "blended_image": blended,
            "sample_image": blended
        }
```

---

## Uso en pipeline.json

Una vez creado el filtro, se usa en `pipeline.json` así:

```json
{
    "filters": {
        "0": {
            "filter_name": "Resize",
            "inputs": {}
        },
        "1": {
            "filter_name": "MiFiltro",
            "inputs": {
                "input_image": "0.resized_image"
            }
        },
        "2": {
            "filter_name": "OtroFiltro",
            "inputs": {
                "input_image": "1.mi_resultado"
            }
        }
    }
}
```

**Formato de referencia:** `"número_filtro.nombre_output"`

---

## Checklist para Nuevo Filtro

- [ ] Crear archivo `filter_library/mi_filtro.py`
- [ ] Incluir imports necesarios (cv2, numpy, typing, base_filter)
- [ ] La clase hereda de `BaseFilter`
- [ ] `FILTER_NAME` es único y en PascalCase
- [ ] `DESCRIPTION` describe claramente la función
- [ ] `INPUTS` lista todas las entradas necesarias
- [ ] `OUTPUTS` incluye `"sample_image": "image"`
- [ ] `PARAMS` tiene `default`, `min`, `max`, `step`, `description` para cada parámetro
- [ ] `process()` retorna **todos** los outputs definidos
- [ ] `process()` maneja el caso donde `inputs` puede estar vacío
- [ ] Agregar import en `filter_library/__init__.py`
- [ ] Agregar a la lista `__all__` en `__init__.py`
- [ ] Si produce solo imágenes, es compatible con checkpoint
- [ ] Los valores de parámetros se validan/corrigen si es necesario (ej: kernel impar)

---

## Registro Automático

**No es necesario registrar manualmente el filtro.** La clase `BaseFilter` usa `__init_subclass__` para registrar automáticamente cada subclase:

```python
class BaseFilter(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'FILTER_NAME') and cls.FILTER_NAME != "base":
            FILTER_REGISTRY[cls.FILTER_NAME] = cls
```

El filtro se registra automáticamente cuando se importa. Por eso es necesario agregar el import en `__init__.py`.
