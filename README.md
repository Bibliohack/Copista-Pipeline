# Sistema de Filtros de Imagen Modular

Sistema en Python para aplicar una sucesión configurable de filtros sobre imágenes.

## Uso Rápido

```bash
# Instalar dependencias
pip install opencv-python numpy python3-tqdm

# Ejecutar el configurador GUI
python src/param_configurator.py [ruta_a_carpeta_imagenes] --pipeline ./examples/Basic_Sample

# Ejecutar el procesamiento en lote
python src/batch_processor.py ./examples/Basic_Sample

# Si no se indica ruta, usa el directorio actual
python src/param_configurator.py

# Limpiar todo el cache al iniciar
python src/param_configurator.py --clear-cache
```

## Estructura de Archivos

```
Copista-Pipeline/
├── src/
│   ├── param_configurator.py   # GUI para configurar parámetros
│   ├── batch_processor.py      # Procesamiento en lote
│   ├── sync_pipeline_params.py # Sincronizador pipeline ↔ params
│   ├── core/                   # Clases compartidas
│   │   ├── __init__.py
│   │   └── pipeline_classes.py
│   └── filter_library/         # Biblioteca de filtros disponibles
│       ├── __init__.py
│       ├── base_filter.py
│       ├── resize_filter.py
│       └── ...
├── examples/                   # Ejemplos de configuracion/proyectos
|   ├── Basic_Sample/        
|   |   ├── pipeline.json       # Define qué filtros aplicar y sus conexiones
|   |   ├── params.json         # Parámetros guardados de los filtros
|   |   ├── checkpoints.json    # Configuración de los checkpoints de cache
|   |   └── batch_config.json   # Configuración para procesamiento en lote
|   └── .../
└── __data/                     # (el repo aun no tiene ejemplos de datos imagenes!)
    └── raw/
        └── .cache/             # Cache de filtros (generado automáticamente)
            └── {filtro_id}/
                └── {imagen}.png
```

## Configurador de parámetros (con interfaz gráfica) - param_configurator.py

El objetivo de este script es permitir al usuario hacer pruebas en vivo sobre los filtros 
seleccionados en pipeline.json a través de una GUI y establecer los parametros óptimos

### Controles

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

### Sistema de Cache (Checkpoints)

El sistema permite marcar filtros como "checkpoints" para acelerar el procesamiento.

### Funcionamiento

1. **Marcar como checkpoint**: Presionando `c` en el filtro deseado (ej: filtro `blur`) se agrega/quita de la lista de checkpoints
2. **Generación de cache**: Al navegar imágenes, se guarda automáticamente el resultado del checkpoint en `.cache/{filtro_id}/{imagen}.png`
3. **Uso del cache**: Si estás visualizando un filtro posterior al último checkpoint (ej: filtro `canny`), los filtros anteriores no se ejecutan - se carga directamente desde cache

Usar checkpoints puede mejorar significativamente el rendimiento en filtros que generan visualizaciones complejas.

### Modificación de parámetros previos al último checkpoint

Si modificas parámetros de un filtro anterior o igual al último checkpoint:
- El cache se **ignora temporalmente** (verás los cambios en tiempo real)
- El cache **no se borra** hasta que guardes con `s`
- Al guardar, se muestra una advertencia y se borra el cache

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
Ver mas detalle en [docs/Documentación/FUNCIONAMIENTO_DE_CACHE_Y_CHECKPOINTS.md](docs/Documentación/FUNCIONAMIENTO_DE_CACHE_Y_CHECKPOINTS.md)

---

## Procesamiento en Lote (sin interfaz gráfica) - batch_processor.py

Este script procesa imágenes en lote sin interfaz gráfica. Aplica los mismos filtros y parámetros definidos en `param_configurator.py` 
a cada imagen desde una carpeta de origen y genera como salida imágenes o datos en múltiples destinos. El archivo de configuración 
`batch_config.json` permite seleccionar que filtros generan una salida, hacia que destino y con variantes (prefijo/sufijo) opcionales 
en el nombre.

### Características

1. **Sin GUI**: Solo terminal con progress bar
2. **Sin cache**: Procesa todo desde cero
3. **`without_preview=True`**: Optimizado, sin generar visualizaciones innecesarias
4. **Multi-target**: Guarda múltiples outputs del pipeline
5. **Tipos de output soportados**:
   - `image` → PNG/JPG
   - `lines`, `contours`, `metadata`, etc. → JSON
   - `float` → TXT o JSON

**C. Optimización de procesamiento**:
- Determinar el **último filtro necesario** entre todos los targets
- Procesar pipeline **una sola vez** hasta ese filtro
- Extraer múltiples outputs de esa ejecución única

### Ejemplo de configuración - batch_config.json

```json
{
  "version": "1.0",
  "description": "Configuración de ejemplo para batch processing",
  "source_folder": "__data/raw",
  "log_file": "__data/logs/batch_processing.log",
  "targets": [
    {
      "filter_id": "resize",
      "output_name": "resized_image",
      "destination": {
        "folder": "__data/processed/resized",
        "prefix": "",
        "suffix": "_resized",
        "extension": "png"
      }
    },
    {
      "filter_id": "canny_border",
      "output_name": "edge_image",
      "destination": {
        "folder": "__data/processed/edges",
        "prefix": "edge_",
        "suffix": "",
        "extension": "png"
      }
    }
  ]
}
```

#### Validaciones

**B. Validación de targets en batch_config.json**:
- ✅ `filter_id` existe en pipeline.json
- ✅ `output_name` existe en `OUTPUTS` del filtro
- ✅ Carpetas de destino se pueden crear
- ✅ Extensiones coherentes con tipo de dato

---

## Filtros: Archivos de Configuración

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
| `batch_processor.py` | Procesamiento en lote |
| `sync_pipeline_params.py` | Sincronizador de pipeline.json ↔ params.json |

## Comandos Útiles

```bash
# Ejecutar configurador
python src/param_configurator.py [carpeta_imagenes] --pipeline [pipeline_json_files]

# Validar sincronización sin GUI
python src/sync_pipeline_params.py --validate-only

# Limpiar parámetros huérfanos automáticamente
python src/sync_pipeline_params.py --auto-clean

# Resolver problemas interactivamente
python src/sync_pipeline_params.py

# Limpiar todo el cache
python src/param_configurator.py --clear-cache
```

## Requisitos

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy
- tqdm

**Nota**: El sistema usa `OrderedDict` para garantizar orden de filtros en Python < 3.7, pero se recomienda Python 3.7+.

## Documentación Adicional

- **[FILTER_REFERENCE.md](docs/Documentación/FILTER_REFERENCE.md)** - Referencia rápida para crear filtros
- **[FILTER_DEVELOPMENT_GUIDE.md](docs/Documentación/FILTER_DEVELOPMENT_GUIDE.md)** - Guía técnica detallada
