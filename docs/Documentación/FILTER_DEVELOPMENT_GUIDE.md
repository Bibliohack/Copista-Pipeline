# Guía Técnica: Creación de Filtros para filter_library.py

## Resumen

Este documento explica cómo crear nuevos filtros para el sistema de procesamiento de imágenes. Cada filtro es una clase Python que hereda de `BaseFilter` y se registra automáticamente en el sistema.

---

## Arquitectura del Sistema

### Flujo de datos

```
Imagen Original
      ↓
  [Filtro resize] → outputs: {resized_image: img, sample_image: img}
      ↓
  [Filtro blur] → recibe inputs de filtros anteriores → produce outputs
      ↓
  [Filtro canny] → ...
      ↓
  Resultado Final
```

### Archivos involucrados

| Archivo | Función |
|---------|---------|
| `filter_library/` | Contiene todas las clases de filtros |
| `pipeline.json` | Define qué filtros usar y cómo conectarlos |
| `params.json` | Almacena los valores de parámetros configurados |
| `param_configurator.py` | GUI para ajustar parámetros y visualizar resultados |

---

## Estructura de un Filtro

### Plantilla básica (filtro de imagen)

```python
class MiFiltro(BaseFilter):
    """Descripción breve del filtro"""
    
    # === ATRIBUTOS DE CLASE OBLIGATORIOS ===
    
    FILTER_NAME = "MiFiltro"  # Identificador único, usado en pipeline.json
    DESCRIPTION = "Descripción detallada de lo que hace el filtro"
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "mi_resultado": "image",
        "sample_image": "image"  # OBLIGATORIO para visualización
    }
    
    PARAMS = {
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
        img = inputs.get("input_image", original_image)
        param_valor = self.params["mi_parametro"]
        
        # Procesar...
        resultado = self._hacer_algo(img, param_valor)
        
        return {
            "mi_resultado": resultado,
            "sample_image": resultado
        }
```

### Plantilla para filtros que producen datos con coordenadas

**IMPORTANTE:** Si tu filtro produce datos con coordenadas absolutas (líneas, contornos, puntos), 
**DEBE** incluir un output de metadata con las dimensiones de la imagen de referencia.

```python
class DetectarLineas(BaseFilter):
    """Detecta líneas en la imagen"""
    
    FILTER_NAME = "DetectarLineas"
    DESCRIPTION = "Detecta líneas rectas en la imagen"
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "lines_data": "lines",          # Datos con coordenadas
        "lines_metadata": "metadata",   # ✅ OBLIGATORIO: metadata con dimensiones
        "sample_image": "image"         # Visualización
    }
    
    PARAMS = {...}
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        img = inputs.get("input_image", original_image)
        h, w = img.shape[:2]  # ✅ Obtener dimensiones
        
        # Detectar líneas...
        lines_data = [
            {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
            {"x1": 15, "y1": 25, "x2": 105, "y2": 205},
            # ...
        ]
        
        # ✅ IMPORTANTE: Crear metadata con dimensiones
        # Mínimo requerido: image_width, image_height
        metadata = {
            "image_width": int(w),
            "image_height": int(h),
            # Opcional: otros datos relevantes
            "total_lines": len(lines_data),
            "detection_threshold": self.params.get("threshold", 0)
        }
        
        # Crear visualización...
        vis = self._draw_lines(img, lines_data)
        
        return {
            "lines_data": lines_data,
            "lines_metadata": metadata,  # ✅ Retornar metadata
            "sample_image": vis
        }
```

### ¿Por qué metadata es obligatoria para datos con coordenadas?

**Problema sin metadata:**
```python
# Usuario detecta líneas en imagen pequeña (640x480)
lines = detectar_lineas(imagen_pequeña)  # [{"x1": 100, "y1": 50, ...}]

# Usuario quiere aplicarlas a imagen grande (1920x1080)
# ❌ NO SABE qué resolución tenían las coordenadas originales
# ❌ NO PUEDE escalar correctamente
```

**Solución con metadata:**
```python
# Filtro retorna datos + metadata
lines_data = [{"x1": 100, "y1": 50, ...}]
metadata = {"image_width": 640, "image_height": 480}

# Usuario puede escalar correctamente
scale_x = 1920 / metadata["image_width"]   # 3.0
scale_y = 1080 / metadata["image_height"]  # 2.25
scaled_x = 100 * scale_x  # 300
```

### Convenciones de nombres para metadata

| Tipo de Datos | Nombre del Output | Claves Mínimas |
|---------------|-------------------|----------------|
| Líneas | `lines_metadata` | `image_width`, `image_height` |
| Contornos | `contours_metadata` | `image_width`, `image_height`, `image_area` |
| Puntos/Esquinas | `corners_metadata` o `points_metadata` | `image_width`, `image_height` |

**Reglas:**
- Nombres en snake_case
- Sin prefijo `_`: usar `"image_width"` NO `"_image_width"`
- Siempre incluir al menos `image_width` e `image_height`

---

## Uso en pipeline.json

Una vez creado el filtro, se usa en `pipeline.json` así:

```json
{
    "filters": {
        "resize": {
            "filter_name": "Resize",
            "inputs": {}
        },
        "mi_filtro": {
            "filter_name": "MiFiltro",
            "inputs": {
                "input_image": "resize.resized_image"
            }
        },
        "otro_filtro": {
            "filter_name": "OtroFiltro",
            "inputs": {
                "input_image": "mi_filtro.mi_resultado"
            }
        }
    }
}
```

**Formato de referencia:** `"filter_id.nombre_output"`

### Características del sistema de IDs:

- **IDs semánticos**: Usa nombres descriptivos como `"resize"`, `"blur"`, `"canny"`
- **Orden implícito**: El orden en el JSON define el orden de ejecución
- **Inserción fácil**: Agregar un filtro entre otros es trivial

### Ejemplo: Insertar un filtro

```json
{
    "filters": {
        "resize": {...},
        "grayscale": {...},
        "denoise": {  // ← NUEVO - Solo lo insertas aquí
            "filter_name": "DenoiseNLMeans",
            "inputs": {"input_image": "grayscale.grayscale_image"}
        },
        "blur": {
            "inputs": {"input_image": "denoise.denoised_image"}  // ← Solo cambias esto
        }
    }
}
```

El filtro `denoise` se ejecutará entre `grayscale` y `blur` automáticamente.

---

## Optimización para Batch Processing

Filtros que producen datos (no imágenes) y cuya visualización es costosa
pueden verificar `self.without_preview` para omitir la generación de sample_image.

### Cuándo implementar esta optimización:

**SÍ implementar:**
- Filtros que dibujan visualizaciones complejas (líneas, contornos, gráficos)
- Cuando generar sample_image tiene costo significativo
- Ejemplos: HoughLines, ClassifyLinesByAngle, HistogramVisualize

**NO implementar:**
- Filtros donde sample_image == output principal
- Cuando generar sample_image no tiene costo extra
- Ejemplos: CannyEdge, GaussianBlur, Threshold

---

## Checklist para Nuevo Filtro

- [ ] La clase hereda de `BaseFilter`
- [ ] `FILTER_NAME` es único y en PascalCase
- [ ] `DESCRIPTION` describe claramente la función
- [ ] `INPUTS` lista todas las entradas necesarias (puede ser `{}` si es filtro de corpus)
- [ ] `OUTPUTS` incluye `"sample_image": "image"`
- [ ] **Si produce datos con coordenadas, incluye output `*_metadata` con `image_width` e `image_height`**
- [ ] `PARAMS` tiene `default`, `min`, `max`, `step`, `description` para parámetros numéricos
- [ ] **Parámetros de texto (rutas, patrones): solo `default` y `description`, sin `min`/`max`/`step`**
- [ ] `process()` retorna **todos** los outputs definidos
- [ ] `process()` maneja el caso donde `inputs` puede estar vacío
- [ ] Si produce solo imágenes, es compatible con checkpoint
- [ ] Los valores de parámetros se validan/corrigen si es necesario (ej: kernel impar)
- [ ] **Las claves de metadata usan snake_case sin prefijo `_`**
- [ ] Importación añadida en `filter_library/__init__.py` y nombre en `__all__`
- [ ] Si comparte código con otro filtro, el código compartido va en `_modulo_interno.py`

---

## Patrones especiales

### Filtros que leen de carpeta externa (sin inputs del pipeline)

Algunos filtros operan a nivel de **corpus**, no de imagen individual. No reciben inputs del pipeline; toda su información la obtienen de una carpeta configurada como parámetro.

**Ejemplo:** `FindPeakProportion` lee JSONs de `source_folder` y calcula la proporción más frecuente del corpus.

```python
class MiFiltroDeCorpus(BaseFilter):
    """Lee archivos externos; no consume inputs del pipeline."""

    FILTER_NAME = "MiFiltroDeCorpus"
    DESCRIPTION = "Analiza una carpeta de archivos para obtener estadísticas de corpus."

    INPUTS = {}  # Sin inputs del pipeline

    OUTPUTS = {
        "corpus_data": "metadata",
        "sample_image": "image"
    }

    PARAMS = {
        "source_folder": {
            "default": "",
            "description": "Carpeta con los archivos a analizar. Obligatorio."
            # Sin min/max/step: es un parámetro de texto (ruta)
        },
        "file_pattern": {
            "default": "*.json",
            "description": "Patrón glob para filtrar archivos."
            # Sin min/max/step: es un parámetro de texto
        },
        "umbral": {
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "description": "Umbral numérico (sí tiene min/max/step)."
        }
    }

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        source_folder = self.params.get("source_folder", "")
        # original_image puede ignorarse; el filtro lee de disco
        # ...
```

**Reglas para parámetros de texto (rutas, patrones glob, nombres):**
- Omitir las claves `min`, `max` y `step`.
- Solo incluir `default` y `description`.
- El configurador gráfico mostrará un campo de texto en lugar de slider.

**Cuándo usar este patrón:**
- Estadísticas de corpus (proporciones, histogramas globales).
- Filtros que requieren un archivo de calibración externo.
- Se recomienda complementar con un script independiente en `scripts/`. Ver `FILTROS_PROPORCION_CORPUS.md` y `BATCH_PRE_POST_PROCESS.md`.

---

### Módulos internos de utilidad (no filtros)

Cuando varios filtros comparten código de geometría, matemáticas u otras utilidades, ese código va en un **módulo interno** dentro de `filter_library/`. Los módulos internos se distinguen por el prefijo `_` en el nombre del archivo.

**Características:**
- El archivo empieza con `_`: `_mi_modulo.py`.
- **No** define ninguna subclase de `BaseFilter`.
- **No** se registra en `FILTER_REGISTRY`.
- **No** se importa en `__init__.py`.
- Los filtros que lo necesitan lo importan directamente con import relativo:

```python
# En refine_polygon_by_area.py
from ._polygon_geometry import (
    add_positions, limit_lines, exhaustive_search,
    candidate_to_border_lines,
)
```

**Ejemplo existente:** `_polygon_geometry.py` contiene funciones de geometría de polígonos compartidas por `RefinePolygonByArea` y `RefinePolygonByCanny`.

**Cuándo crear un módulo interno:**
- El mismo bloque de código (>20 líneas) sería copiado en dos o más filtros.
- La lógica es puramente utilitaria y no tiene sentido exponerla como un filtro independiente.
- Funciones matemáticas, parsers de formato, helpers de visualización compartidos.

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

Simplemente define la clase en `filter_library/` y estará disponible.

**Para un filtro normal**, además del archivo `.py`, hay que añadir la importación en `filter_library/__init__.py`:

```python
# En filter_library/__init__.py
from .mi_filtro import MiFiltro
```

Y añadir el nombre a `__all__`:

```python
__all__ = [
    # ... filtros existentes ...
    "MiFiltro",
]
```

Los módulos internos (`_*.py`) **no** se añaden a `__init__.py`.
