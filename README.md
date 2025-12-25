# Sistema de Filtros de Imagen Modular

Sistema en Python para aplicar una sucesión configurable de filtros sobre imágenes.

<video width="640" height="360" controls>
  <source src="samples/screencapt.mp4" type="video/mp4">
  Tu navegador no soporta el elemento de video.
</video>



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
image_filter_system/
├── pipeline.json           # (A) Define qué filtros aplicar y sus conexiones
├── params.json             # (B) Parámetros guardados de los filtros
├── filter_library.py       # (C) Biblioteca de filtros disponibles
├── param_configurator.py   # (D) GUI para configurar parámetros
├── checkpoint.json         # Configuración del checkpoint de cache
└── carpeta_imagenes/
    └── .cache/             # Cache de filtros (generado automáticamente)
        └── {filtro}/
            └── {imagen}.png
```


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

### checkpoint.json - Configuración del Checkpoint

```json
{
    "checkpoint_filter": "2",
    "last_modified": "2024-01-15T10:30:00"
}
```

## Agregar Nuevos Filtros

Para agregar un filtro a la biblioteca (`filter_library.py`):

```python
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

El filtro se registra automáticamente al definir la clase.

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
| `HoughLines` | Detecta líneas con Hough | edge_image | lines_data |
| `Morphology` | Operaciones morfológicas | input_image | morphed_image |
| `Contours` | Detecta contornos | input_image | contours_data, contour_image |
| `ColorSpace` | Convierte espacios de color | input_image | converted_image |
| `OverlayLines` | Visualiza líneas sobre imagen | base_image, lines_data | overlay_image |

## Conceptos Clave

### sample_image
Todo filtro DEBE producir un output llamado `sample_image`. Es lo que se muestra en el visualizador.

### Filtros de Visualización
Algunos filtros no procesan realmente la imagen, solo combinan datos para visualización (ej: `OverlayLines`).

### Flujo de Datos
Los filtros pueden tomar datos de cualquier filtro anterior, no solo del inmediato.

## Requisitos

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
