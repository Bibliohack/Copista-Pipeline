# Sistema de Filtros de Imagen Modular

Sistema en Python para aplicar una sucesión configurable de filtros sobre imágenes.

## Uso Rápido

```bash
# Instalar dependencias
pip install opencv-python numpy

# Crear los archivos de configuración desde los ejemplos
cp samples/pipeline.json pipeline.json
cp samples/checkpoints.json checkpoints.json
cp samples/params.json params.json

# Ejecutar el configurador GUI
python param_configurator.py [ruta_a_carpeta_imagenes]

# Si no se indica ruta, usa el directorio actual
python param_configurator.py

# Limpiar todo el cache al iniciar
python param_configurator.py --clear-cache
```

## Estructura de Archivos

```
image_filter_system/
├── pipeline.json           # Define qué filtros aplicar y sus conexiones
├── params.json             # Parámetros guardados de los filtros
├── checkpoints.json         # Configuración del checkpoint de cache
├── filter_library/         # Biblioteca de filtros disponibles
│   ├── __init__.py
│   ├── base_filter.py
│   ├── resize_filter.py
│   └── ...
├── param_configurator.py   # GUI para configurar parámetros
├── sync_pipeline_params.py # Sincronizador pipeline ↔ params
└── carpeta_imagenes/
    └── .cache/             # Cache de filtros (generado automáticamente)
        └── {filtro_id}/
            └── {imagen}.png
```

## Controles del Configurador GUI

| Tecla | Acción |
|-------|--------|
| `a` / `d` | Imagen anterior / siguiente |
| `ESPACIO` | Avanzar al siguiente filtro (visualización) |
| `BACKSPACE` | Retroceder al filtro anterior (visualización) |
| `PgDown` | Avanzar al siguiente filtro (edición) |
| `PgUp` | Retroceder al filtro anterior (edición) |
| `↑` / `↓` | Navegar entre parámetros |
| `←` / `→` | Decrementar / incrementar valor del parámetro |
| `c` | Marcar/desmarcar filtro actual como checkpoint |
| `s` | Guardar parámetros en params.json |
| `r` | Recargar parámetros desde params.json |
| `h` | Mostrar ayuda del filtro actual |
| `q` / `ESC` | Salir |

## Sistema de Cache (Checkpoints)

El sistema permite marcar filtros como "checkpoints" para acelerar el procesamiento.

### Funcionamiento

1. **Marcar como checkpoint**: Presionando `c` en el filtro deseado (ej: filtro `blur`) se agrega/quita de la lista de checkpoints
2. **Generación de cache**: Al navegar imágenes, se guarda automáticamente el resultado del checkpoint en `.cache/{filtro_id}/{imagen}.png`
3. **Uso del cache**: Si estás visualizando un filtro posterior al último checkpoint (ej: filtro `canny`), los filtros anteriores no se ejecutan - se carga directamente desde cache

### Modificación de parámetros previos al último checkpoint

Si modificas parámetros de un filtro anterior o igual al último checkpoint:
- El cache se **ignora temporalmente** (verás los cambios en tiempo real)
- El cache **no se borra** hasta que guardes con `s`
- Al guardar, se muestra una advertencia y se borra el cache

Ver mas detalle en [Documentacion/FUNCIONAMIENTO_DE_CACHE_Y_CHECKPOINTS.md](Documentacion/FUNCIONAMIENTO_DE_CACHE_Y_CHECKPOINTS.md)

## Archivos de Configuración

### pipeline.json - Configuración del Pipeline

Define la cadena de filtros a aplicar. Cada filtro tiene:
- **ID único** (clave del dict): Identifica el filtro semánticamente
- `filter_name`: Nombre del filtro (debe existir en la biblioteca)
- `inputs`: Diccionario que mapea las entradas requeridas a salidas de filtros anteriores

**Formato de referencias:** `"filter_id.nombre_output"`

**Referencia especial:** Use `"original.image"` para referenciar la imagen original sin procesar.

**El orden de los filtros en el JSON determina el orden de ejecución.**

```json
{
    "filters": {
        "resize": {
            "filter_name": "Resize",
            "description": "Redimensionar imagen inicial",
            "inputs": {}
        },
        "grayscale": {
            "filter_name": "Grayscale",
            "description": "Convertir a escala de grises",
            "inputs": {
                "input_image": "resize.resized_image"
            }
        },
        "canny": {
            "filter_name": "CannyEdge",
            "description": "Detectar bordes",
            "inputs": {
                "input_image": "grayscale.grayscale_image"
            }
        }
    }
}
```

#### Características del sistema de IDs:

- **Legibilidad**: `"blur.blurred_image"` es más claro que un número
- **Inserción fácil**: Agregar filtros entre otros es trivial
- **Orden implícito**: El orden visual en el JSON define el orden de ejecución

#### Insertar filtros entre otros:

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

No necesitas renumerar nada.

### params.json - Parámetros Guardados

Se genera/actualiza automáticamente al presionar 's'.
Si no existe, los filtros usan sus valores por defecto.

Los parámetros se guardan por ID del filtro:

```json
{
    "version": "1.0",
    "filter_params": {
        "resize": {
            "filter_name": "Resize",
            "params": {
                "scale_percent": 50,
                "interpolation": 1
            }
        },
        "blur": {
            "filter_name": "GaussianBlur",
            "params": {
                "kernel_size": 5
            }
        }
    }
}
```

Si modificas `pipeline.json`, ejecuta `sync_pipeline_params.py` para mantener sincronización.

### checkpoints.json - Configuración del Checkpoint

```json
{
    "checkpoints": [
        "resize",
        "denoise"
    ],
    "last_modified": "2025-01-31T10:30:00"
}
```

## Agregar Nuevos Filtros

Para agregar un filtro a la biblioteca:

```python
# En filter_library/mi_filtro.py
class MiNuevoFiltro(BaseFilter):
    FILTER_NAME = "MiNuevoFiltro"
    DESCRIPTION = "Descripción del filtro"
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "mi_output": "image",
        "sample_image": "image"  # OBLIGATORIO para visualización
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
    
    def process(self, inputs, original_image):
        img = inputs.get("input_image", original_image)
        # ... procesamiento ...
        return {
            "mi_output": resultado,
            "sample_image": imagen_para_visualizar
        }
```

El filtro se registra automáticamente al definir la clase.

Luego agregar a `filter_library/__init__.py`:

```python
from .mi_filtro import MiNuevoFiltro

__all__ = [
    # ... otros filtros
    "MiNuevoFiltro",
]
```

Ver documentación completa en:
- [FILTER_REFERENCE.md](docs/FILTER_REFERENCE.md) - Referencia rápida
- [FILTER_DEVELOPMENT_GUIDE.md](docs/FILTER_DEVELOPMENT_GUIDE.md) - Guía detallada

## Filtros Disponibles

| Filtro | Descripción | Inputs | Outputs |
|--------|-------------|--------|---------|
| `Resize` | Redimensiona la imagen | - | resized_image |
| `BrightnessContrast` | Ajusta brillo y contraste | input_image | adjusted_image |
| `Grayscale` | Convierte a escala de grises | input_image | grayscale_image |
| `GaussianBlur` | Aplica desenfoque gaussiano | input_image | blurred_image |
| `CannyEdge` | Detecta bordes con Canny | input_image | edge_image |
| `Threshold` | Umbralización binaria/adaptativa | input_image | threshold_image |
| `Histogram` | Calcula y visualiza histograma | input_image | histogram_data |
| `HoughLines` | Detecta líneas con Hough | edge_image, base_image | lines_data |
| `Morphology` | Operaciones morfológicas | input_image | morphed_image |
| `Contours` | Detecta contornos | input_image | contours_data, contour_image |
| `ColorSpace` | Convierte espacios de color | input_image | converted_image |
| `OverlayLines` | Visualiza líneas sobre imagen | base_image, lines_data | overlay_image |
| `ClassifyLinesByAngle` | Clasifica líneas H/V | lines_data, base_image | horizontal_lines, vertical_lines |
| `SelectBorderLines` | Selecciona líneas de borde | horizontal_lines, vertical_lines | selected_lines |
| `CalculateQuadCorners` | Calcula esquinas del quad | selected_lines | corners |
| `DetectPageSkew` | Detecta inclinación de página | lines_data, base_image | skew_angle |

(la lista no es exhaustiva!)

## Conceptos Clave

### sample_image
Todo filtro DEBE producir un output llamado `sample_image`. Es lo que se muestra en el visualizador.

### Filtros de Visualización
Algunos filtros no procesan realmente la imagen, solo combinan datos para visualización (ej: `OverlayLines`).

### Flujo de Datos
Los filtros pueden tomar datos de cualquier filtro anterior, no solo del inmediato.

### Reutilización de Filtros
El mismo filtro puede usarse múltiples veces en el pipeline con diferentes parámetros:

```json
{
    "resize_down": {
        "filter_name": "Resize",
        "inputs": {},
        "params": {"scale_percent": 50}
    },
    "grayscale": {
        "filter_name": "Grayscale",
        "inputs": {"input_image": "resize_down.resized_image"}
    },
    "resize_up": {
        "filter_name": "Resize",
        "inputs": {"input_image": "grayscale.grayscale_image"},
        "params": {"scale_percent": 200}
    }
}
```

## Scripts Disponibles

| Script | Propósito |
|--------|-----------|
| `param_configurator.py` | Configurador GUI principal |
| `sync_pipeline_params.py` | Sincronizador de pipeline.json ↔ params.json |

## Comandos Útiles

```bash
# Ejecutar configurador
python param_configurator.py [carpeta_imagenes]

# Validar sincronización sin GUI
python sync_pipeline_params.py --validate-only

# Limpiar parámetros huérfanos automáticamente
python sync_pipeline_params.py --auto-clean

# Resolver problemas interactivamente
python sync_pipeline_params.py

# Limpiar todo el cache
python param_configurator.py --clear-cache
```

## Requisitos

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy

**Nota**: El sistema usa `OrderedDict` para garantizar orden de filtros en Python < 3.7, pero se recomienda Python 3.7+.

## Documentación Adicional

- **[FILTER_REFERENCE.md](Documentacion/FILTER_REFERENCE.md)** - Referencia rápida para crear filtros
- **[FILTER_DEVELOPMENT_GUIDE.md](Documentacion/FILTER_DEVELOPMENT_GUIDE.md)** - Guía técnica detallada
