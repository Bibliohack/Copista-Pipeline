# Filtros: Proporción de Corpus

Estos dos filtros implementan la **estrategia de proporción invariante** para mejorar y validar la detección de bordes de página. La idea central es que todas las páginas del corpus tienen la misma proporción de papel (independientemente del zoom o la perspectiva), y ese valor puede extraerse estadísticamente del corpus completo para usarse como restricción en el pipeline de recorte.

Ver contexto completo en `MEJORA_DETECCION_BORDES.md`.

---

## CalculatePolygonProportion

**Archivo**: `src/filter_library/calculate_polygon_proportion.py`

### Qué hace

Recibe el polígono detectado de una imagen (4 esquinas en orden horario) y calcula su proporción:

```
proporcion = avg(lado_izquierdo, lado_derecho) / lado_superior
```

Donde cada lado es la distancia euclidiana entre dos esquinas del cuadrilátero:

```
TL ──── top_side ──── TR
│                      │
left_side          right_side
│                      │
BL ─── bottom_side ─── BR
```

Esta proporción es **invariante al zoom**: si la misma página se fotografía desde más cerca o más lejos, la proporción no cambia. Es característica del tamaño físico del papel del corpus.

### Inputs / Outputs

| Nombre            | Tipo       | Descripción |
|-------------------|------------|-------------|
| `gt_data`         | `metadata` | Dict con clave `polygon` (salida de `PolygonToGTFormat`). |
| `proportion_data` | `metadata` | Proporción calculada + longitudes de cada lado. |
| `sample_image`    | `image`    | Visualización del polígono con las medidas anotadas. |

### Formato de `proportion_data`

```json
{
  "proportion":   0.741646,
  "top_side":     4908.0,
  "right_side":   3640.0,
  "bottom_side":  4908.0,
  "left_side":    3640.0,
  "valid":        true,
  "image_width":  6000,
  "image_height": 4000
}
```

Si el polígono está incompleto o es inválido, `proportion` es `null` y `valid` es `false`.

### Parámetros

| Parámetro          | Default | Descripción |
|--------------------|---------|-------------|
| `visualization_size` | `900` | Ancho de la imagen de muestra. |
| `show_side_lengths`  | `1`   | Mostrar longitud de cada lado en la visualización (`0`/`1`). |

### Uso en pipeline

```json
"calcular_proporcion": {
    "filter_name": "CalculatePolygonProportion",
    "description": "Calcular proporción del polígono detectado",
    "inputs": {
        "gt_data": "polygon_to_gt.gt_data"
    }
}
```

---

## FindPeakProportion

**Archivo**: `src/filter_library/find_peak_proportion.py`

### Qué hace

Lee una **carpeta completa** de archivos JSON (polígonos o proporciones), recopila todas las proporciones del corpus, construye un histograma con suavizado gaussiano y determina el **pico** (valor más frecuente).

Este filtro opera a nivel de corpus, no de imagen individual. La imagen que recibe del pipeline es ignorada; toda su información la extrae de los archivos JSON de la carpeta configurada.

### Nota arquitectónica

Este filtro **no encaja en el flujo normal por imagen**. Las formas correctas de usarlo son:

1. **Como script independiente**: usar `scripts/find_peak_proportion.py` (recomendado).
2. **Como postprocess en `batch_config.json`**: se ejecuta automáticamente al terminar el batch. Ver `BATCH_PRE_POST_PROCESS.md`.
3. **Como filtro de "disparo único"**: incluirlo en un pipeline que se procesa sobre una sola imagen ficticia.

### Inputs / Outputs

| Nombre         | Tipo       | Descripción |
|----------------|------------|-------------|
| *(ninguno)*    | —          | No consume inputs del pipeline. |
| `peak_data`    | `metadata` | Proporción pico + estadísticas del corpus. |
| `sample_image` | `image`    | Histograma visual con el pico marcado. |

### Formato de `peak_data`

```json
{
  "peak_proportion":   0.737114,
  "peak_count":        9,
  "mean":              0.739948,
  "std":               0.026055,
  "min":               0.694649,
  "max":               0.817326,
  "n_bins":            13,
  "valid_proportions": 50,
  "skipped":           0,
  "errors":            0,
  "source_folder":     "...",
  "valid":             true
}
```

### Parámetros

| Parámetro          | Default    | Descripción |
|--------------------|------------|-------------|
| `source_folder`    | `""`       | Carpeta con archivos JSON a analizar. **Obligatorio.** |
| `file_pattern`     | `*.json`   | Patrón glob para filtrar archivos. |
| `bin_size`         | `0.01`     | Tamaño de bin del histograma. |
| `smoothing_sigma`  | `1.5`      | Sigma del suavizado gaussiano (`0` = sin suavizado). |
| `histogram_height` | `350`      | Altura de la imagen de histograma. |
| `histogram_width`  | `900`      | Ancho de la imagen de histograma. |

### Formatos de JSON aceptados como entrada

El filtro detecta automáticamente el formato de cada archivo:

| Clave presente | Acción |
|----------------|--------|
| `proportion`   | Usa el valor directamente. |
| `polygon`      | Calcula la proporción `avg(izq, der) / superior`. |
| Ninguna        | El archivo se cuenta como saltado. |

Esto permite alimentar el filtro tanto con salida directa de `PolygonToGTFormat` (`.det.json`, `.gt.json`) como con salida de `CalculatePolygonProportion` guardada como JSON.

---

## Flujo de trabajo: pipeline de calibración

```
Corpus de imágenes
      │
      ▼
[Pipeline Heraldo_Claude]
      │  (batch_processor)
      ▼
 .det.json por imagen
      │
      ▼ (postprocess)
[find_peak_proportion.py]
      │
      ▼
 peak_proportion.json
      │
      ▼
[Pipeline de recorte]   ← consume peak_proportion como restricción
```

### Relación con las estrategias de mejora

| Estrategia              | Filtro/Script                   | Rol |
|-------------------------|---------------------------------|-----|
| Detección Hough (base)  | `HoughLines`, `SelectBorderLines` | Detecta líneas |
| Proporción invariante   | `CalculatePolygonProportion`    | Mide la proporción por imagen |
|                         | `FindPeakProportion` / script   | Encuentra la proporción del corpus |
| Projection Profile      | *(por implementar)*             | Complementa Hough |
