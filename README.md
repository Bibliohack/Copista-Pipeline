# Sistema de Filtros de Imagen Modular

Sistema en Python para aplicar una sucesión configurable de filtros sobre imágenes.

https://github.com/user-attachments/assets/e84fdcec-30dc-49f0-81f8-e44c3913897a

## Uso Rápido

```bash
# Instalar dependencias
pip install opencv-python numpy

# Crear los archivos de configuración desde los ejemplos
cp samples/pipeline.json pipeline.json
cp samples/checkpoint.json checkpoint.json
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
proyecto/
├── pipeline.json           # (A) Define qué filtros aplicar y sus conexiones
├── params.json             # (B) Parámetros guardados de los filtros
├── checkpoint.json         # Configuración del checkpoint de cache
├── filter_library/         # (C) Biblioteca de filtros (módulo)
│   ├── __init__.py         #     Exporta todos los filtros
│   ├── base_filter.py      #     Clase base y FILTER_REGISTRY
│   ├── resize_filter.py    #     Un filtro por archivo
│   ├── grayscale_filter.py
│   ├── ...
│   └── [mi_filtro.py]      #     Nuevos filtros van aquí
├── param_configurator.py   # (D) GUI para configurar parámetros
├── sync_pipeline_params.py # (E) Sincronizador pipeline ↔ params
└── carpeta_imagenes/
    └── .cache/             # Cache de filtros (generado automáticamente)
        └── {filtro}/
            └── {imagen}.png
```

## ⭐ Sistema de Sincronización Pipeline ↔ Parámetros

### ¿Qué problema resuelve?

Cuando modificas `pipeline.json` (agregas, eliminas o reordenas filtros), los parámetros en `params.json` pueden quedar desalineados porque se guardan por índice numérico.

**El sincronizador detecta y corrige automáticamente estos problemas.**

### Validación Automática

`param_configurator.py` ahora valida automáticamente la sincronización al iniciar:

```bash
$ python param_configurator.py

🔍 Validando sincronización pipeline ↔ params...

# CASO 1: Solo filtros nuevos (continúa normalmente)
⚠️  AVISOS DETECTADOS (no bloqueantes)
2 filtro(s) nuevo(s) detectado(s):
  • Índice 3: CannyEdge (usará defaults)
  • Índice 4: Morphology (usará defaults)
✅ Puedes continuar. param_configurator.py funcionará normalmente.

# CASO 2: Errores críticos (bloquea ejecución)
❌ VALIDACIÓN FALLIDA - ERRORES CRÍTICOS
Se encontraron 1 errores bloqueantes:
  • Índice 2: Cambio de filtro
    Era: GaussianBlur
    Ahora: CannyEdge

⚠️  Debes ejecutar: python sync_pipeline_params.py
```

### Uso del Sincronizador

```bash
# Modo interactivo: te pregunta qué hacer con cada problema
python sync_pipeline_params.py

# Modo automático: limpia huérfanos automáticamente
python sync_pipeline_params.py --auto-clean

# Solo validar sin hacer cambios
python sync_pipeline_params.py --validate-only
```

### Tipos de Problemas Detectados

| Tipo | Criticidad | Descripción |
|------|------------|-------------|
| **MISMATCH** | ❌ Bloqueante | Mismo índice, diferente filtro (ej: índice 2 era Blur, ahora es Canny) |
| **ORPHAN** | ⚠️ Advertencia | Parámetros sin filtro correspondiente (pueden limpiarse) |
| **MOVED** | ⚠️ Advertencia | Filtro movido de posición (puede corregirse) |
| **NEW** | ℹ️ Info (OK) | Filtros nuevos sin parámetros (usarán defaults) |

**Solo MISMATCH bloquea param_configurator.py**

### Flujo de Trabajo Recomendado

#### Agregar filtros nuevos (simple)
```bash
# Edita pipeline.json, agrega filtros al final
python param_configurator.py
# ✅ Detecta filtros nuevos, avisa, y continúa normalmente
```

#### Modificar pipeline existente (requiere sincronización)
```bash
# 1. Edita pipeline.json (elimina, mueve, cambia filtros)
# 2. Sincroniza
python sync_pipeline_params.py
# 3. Continúa normalmente
python param_configurator.py
```

### Ejemplo de Sesión Interactiva

```bash
$ python sync_pipeline_params.py

============================================================
❌ ERRORES BLOQUEANTES: 1
⚠️  ADVERTENCIAS: 1
ℹ️  INFORMACIÓN: 1
============================================================

❌ CAMBIOS DE FILTRO - BLOQUEANTE (1):
  Índice 2:
    Pipeline: CannyEdge
    Params:   GaussianBlur

⚠️  PARÁMETROS HUÉRFANOS - ADVERTENCIA (1):
  Índice 5: Threshold
    (este filtro ya no está en pipeline.json)

ℹ️  FILTROS NUEVOS - OK (1):
  Índice 3: Morphology

¿Deseas corregir estos problemas? (s/n): s

[1/3] CAMBIO DE FILTRO en índice 2
  Pipeline: CannyEdge
  Params:   GaussianBlur

Opciones:
  1) Usar parámetros guardados para el NUEVO filtro (si son compatibles)
  2) Descartar parámetros antiguos (usar defaults del nuevo filtro)
  3) Cancelar (mantener como está)

Elige opción (1/2/3): 2
  ✓ Parámetros antiguos descartados

[2/3] PARÁMETROS HUÉRFANOS en índice 5
  Filtro: Threshold
  (ya no existe en pipeline.json)

Opciones:
  1) Eliminar estos parámetros
  2) Mantener (por si lo vuelves a usar)

Elige opción (1/2): 1
  ✓ Parámetros eliminados

[3/3] FILTRO NUEVO en índice 3
  Filtro: Morphology
  ✅ Esto está OK. Usará valores por defecto.

✅ Cambios guardados en params.json
```

Para más detalles, consulta la [Documentación Completa de Sincronización](docs/SINCRONIZACION.md)

## Controles del Configurador GUI

| Tecla | Acción |
|-------|--------|
| `a` / `d` | Imagen anterior / siguiente |
| `ESPACIO` | Avanzar al siguiente filtro (visualización) |
| `BACKSPACE` | Retroceder al filtro anterior (visualización) |
| `↑` / `↓` | Navegar entre parámetros |
| `←` / `→` | Decrementar / incrementar valor del parámetro |
| `PgUp` / `PgDown` | Cambiar filtro a editar (sin cambiar vista) |
| `c` | Marcar/desmarcar filtro actual como checkpoint |
| `s` | Guardar parámetros en params.json |
| `r` | Recargar parámetros desde params.json |
| `h` | Mostrar ayuda del filtro actual |
| `q` / `ESC` | Salir |

## Sistema de Cache (Checkpoints)

El sistema permite marcar un filtro como "checkpoint" para acelerar el procesamiento.

### Funcionamiento

1. **Marcar checkpoint**: Presiona `c` en el filtro deseado (ej: filtro 2)
2. **Generación de cache**: Al navegar imágenes, se guarda automáticamente el resultado del checkpoint en `.cache/{filtro}/{imagen}.png`
3. **Uso del cache**: Si estás visualizando un filtro posterior al checkpoint (ej: filtro 5), los filtros 0, 1, 2 no se ejecutan - se carga directamente desde cache

### Solo un checkpoint activo

Solo puede haber un checkpoint a la vez. Si marcas otro filtro como checkpoint, el anterior se elimina junto con su cache.

### Modificación de parámetros pre-checkpoint

Si modificas parámetros de un filtro anterior o igual al checkpoint:
- El cache se **ignora temporalmente** (verás los cambios en tiempo real)
- El cache **no se borra** hasta que guardes con `s`
- Al guardar, se muestra una advertencia y se borra el cache

### Ejemplo de uso

```
Pipeline: [0:Resize] → [1:Grayscale] → [2:Blur] → [3:Canny] → [4:Hough] → [5:Overlay]
                                         ↑
                                    CHECKPOINT

- Visualizando filtro 5, navegando imágenes con 'a'/'d':
  → Filtros 0,1,2 NO se ejecutan (se usa cache del filtro 2)
  → Solo se ejecutan filtros 3,4,5

- Retrocediendo a filtro 1 con BACKSPACE:
  → Se ejecutan filtros 0,1 (el cache no aplica)

- Modificando parámetros del filtro 1:
  → ignore_cache = True
  → Se ejecuta todo el pipeline
  → Al guardar con 's': advertencia + borrado de cache
```

## Archivos de Configuración

### (A) pipeline.json - Configuración del Pipeline

Define la cadena de filtros a aplicar. Cada filtro tiene:
- `filter_name`: Nombre del filtro (debe existir en la biblioteca)
- `inputs`: Diccionario que mapea las entradas requeridas a salidas de filtros anteriores

Formato de referencias: `"numero_filtro.nombre_output"`

```json
{
    "filters": {
        "0": {
            "filter_name": "Resize",
            "inputs": {}
        },
        "1": {
            "filter_name": "Grayscale",
            "inputs": {
                "input_image": "0.resized_image"
            }
        },
        "2": {
            "filter_name": "CannyEdge",
            "inputs": {
                "input_image": "1.grayscale_image"
            }
        }
    }
}
```

### (B) params.json - Parámetros Guardados

Se genera/actualiza automáticamente al presionar 's'.
Si no existe, los filtros usan sus valores por defecto.

**Importante**: Los parámetros se guardan por índice numérico. Si modificas `pipeline.json`, ejecuta `sync_pipeline_params.py` para mantener la sincronización.

### checkpoint.json - Configuración del Checkpoint

```json
{
    "checkpoint_filter": "2",
    "last_modified": "2024-01-15T10:30:00"
}
```

## Agregar Nuevos Filtros

La biblioteca de filtros está organizada en módulos individuales dentro de `filter_library/`.

### Paso 1: Crear el archivo del filtro

Crear `filter_library/mi_nuevo_filtro.py`:

```python
"""
Filtro: MiNuevoFiltro
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class MiNuevoFiltro(BaseFilter):
    FILTER_NAME = "MiNuevoFiltro"
    DESCRIPTION = "Descripción del filtro"
    INPUTS = {
        "input_image": "image"  # Entradas requeridas
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

### Paso 2: Registrar en \_\_init\_\_.py

Editar `filter_library/__init__.py`:

```python
# Agregar el import
from .mi_nuevo_filtro import MiNuevoFiltro

# Agregar a __all__
__all__ = [
    # ... otros filtros ...
    "MiNuevoFiltro",
]
```

El filtro se registra automáticamente al importar el módulo.

Ver documentación completa en:
- [FILTER_REFERENCE.md](FILTER_REFERENCE.md) - Referencia rápida
- [FILTER_DEVELOPMENT_GUIDE.md](FILTER_DEVELOPMENT_GUIDE.md) - Guía detallada

## Filtros Disponibles

### Filtros Básicos

| Filtro | Descripción | Inputs | Outputs |
|--------|-------------|--------|---------|
| `Resize` | Redimensiona la imagen | input_image | resized_image |
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

### Filtros Avanzados

| Filtro | Descripción | Inputs | Outputs |
|--------|-------------|--------|---------|
| `NormalizePeaks` | Normaliza imagen por picos de histograma | input_image | normalized_image |
| `MinArcLength` | Filtra contornos por longitud mínima de arco | edge_image, base_image | filtered_edges |
| `DenoiseNLMeans` | Reducción de ruido Non-Local Means | input_image | denoised_image |
| `ThresholdAdvanced` | Umbralización con OTSU y adaptativa | input_image | threshold_image |
| `MorphologyAdvanced` | Morfología con TopHat, BlackHat e inversión | input_image | morphed_image |
| `ContourSimplify` | Simplifica contornos con approxPolyDP | input_image | contours_data |
| `HistogramVisualize` | Visualiza histograma con marcadores | input_image | histogram_data |

### Filtros de Detección de Bordes de Página

| Filtro | Descripción | Inputs | Outputs |
|--------|-------------|--------|---------|
| `ClassifyLinesByAngle` | Clasifica líneas en horizontales/verticales | lines_data, base_image | horizontal_lines, vertical_lines |
| `SelectBorderLines` | Selecciona líneas extremas de borde | horizontal_lines, vertical_lines, base_image | selected_lines, selection_metadata |
| `CalculateQuadCorners` | Calcula 4 esquinas del polígono | selected_lines, selection_metadata, base_image | corners |

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
    "0": {"filter_name": "Resize", "params": {"scale": 50}},
    "1": {"filter_name": "Grayscale", ...},
    "2": {"filter_name": "Resize", "params": {"scale": 200}}
}
```

### Dimensiones de Imagen
Si el pipeline incluye un `Resize`, los filtros que necesitan dimensiones deben usar una imagen de referencia del pipeline, no `original_image`:

```python
# Incorrecto
h, w = original_image.shape[:2]

# Correcto
base_img = inputs.get("base_image", original_image)
h, w = base_img.shape[:2]
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

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy

## Documentación Adicional

- **[FILTER_REFERENCE.md](FILTER_REFERENCE.md)** - Referencia rápida para crear filtros
- **[FILTER_DEVELOPMENT_GUIDE.md](FILTER_DEVELOPMENT_GUIDE.md)** - Guía técnica detallada para desarrolladores
