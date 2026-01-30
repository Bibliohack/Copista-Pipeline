# Referencia Rápida: Creación de Filtros

## Estructura de Archivos

```
filter_library/
├── __init__.py          # Importa todos los filtros
├── base_filter.py       # Clase base
└── mi_filtro.py         # Un archivo por filtro
```

## Plantilla de Filtro

Crear archivo `filter_library/mi_filtro.py`:

```python
"""
Filtro: MiFiltro
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class MiFiltro(BaseFilter):
    FILTER_NAME = "MiFiltro"                    # Identificador único (PascalCase)
    DESCRIPTION = "Qué hace el filtro"
    
    INPUTS = {
        "input_image": "image"                  # Entradas de otros filtros. {} si usa original
    }
    
    OUTPUTS = {
        "resultado": "image",
        "sample_image": "image"                 # OBLIGATORIO: imagen para visualización
    }
    
    PARAMS = {
        "mi_param": {
            "default": 50,
            "min": 0,
            "max": 100,
            "step": 5,
            "description": "Descripción del parámetro"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        img = inputs.get("input_image", original_image)
        valor = self.params["mi_param"]
        
        # ... procesamiento con OpenCV ...
        
        return {
            "resultado": imagen_procesada,
            "sample_image": imagen_procesada    # Lo que se muestra en el visualizador
        }
```

## Registrar en \_\_init\_\_.py

```python
from .mi_filtro import MiFiltro

__all__ = [
    # ...
    "MiFiltro",
]
```

## Reglas

1. **Cada filtro en su propio archivo** → `nombre_del_filtro.py` (snake_case)

2. **Imports obligatorios en cada archivo:**
   ```python
   import cv2
   import numpy as np
   from typing import Dict, Any, List, Tuple
   from .base_filter import BaseFilter, FILTER_REGISTRY
   ```

3. **OUTPUTS siempre debe incluir `"sample_image": "image"`** → Es lo que muestra el visualizador

4. **Tipos de datos para INPUTS/OUTPUTS:**
   - `"image"` → numpy.ndarray (BGR o grayscale)
   - `"lines"` → lista de líneas detectadas
   - `"contours"` → lista de contornos
   - `"border_lines"` → líneas de borde seleccionadas
   - `"quad_points"` → 4 puntos de un cuadrilátero
   - `"metadata"` → diccionario con metadatos

5. **Si el filtro produce datos (no imagen)**, sample_image debe ser una representación visual:
   ```python
   OUTPUTS = {
       "lines_data": "lines",      # Datos
       "sample_image": "image"     # Visualización de los datos
   }
   ```

6. **Restricción de cache:** Solo filtros con todos los outputs tipo `"image"` pueden estar antes de un checkpoint

7. **No usar `original_image.shape` para dimensiones** si hay Resize en el pipeline:
   ```python
   # Incorrecto
   h, w = original_image.shape[:2]
   
   # Correcto - usar imagen del pipeline
   base_img = inputs.get("base_image", original_image)
   h, w = base_img.shape[:2]
   ```

## Uso en pipeline.json

```json
{
    "3": {
        "filter_name": "MiFiltro",
        "inputs": {
            "input_image": "2.grayscale_image"
        }
    }
}
```

Formato de referencia: `"número_filtro.nombre_output"`

## Checklist Rápido

- [ ] Archivo creado en `filter_library/`
- [ ] Imports incluidos
- [ ] Hereda de `BaseFilter`
- [ ] `FILTER_NAME` único
- [ ] `OUTPUTS` incluye `sample_image`
- [ ] `process()` retorna todos los outputs
- [ ] Import agregado en `__init__.py`
- [ ] Agregado a `__all__`
