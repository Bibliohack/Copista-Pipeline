# Filtros: Refinamiento de Polígono por Proporción

Estos filtros buscan la combinación óptima de 4 líneas Hough que produce un polígono
con la proporción de papel más cercana al objetivo del corpus. Son alternativas o
complementos a `SelectBorderLines`: producen el mismo formato de salida (`border_lines`)
y son compatibles directamente con `CalculateQuadCorners`.

## Motivación

`SelectBorderLines` elige las 4 líneas más extremas del pool Hough. Cuando hay papeles,
objetos o páginas adyacentes que sobresalen en los márgenes, esas líneas extremas pueden
corresponder a elementos externos, no al borde real del papel. El resultado es un polígono
incorrecto. Los filtros de refinamiento usan la **proporción conocida del papel** (obtenida
vía calibración del corpus) como restricción para encontrar las 4 líneas correctas dentro
del conjunto completo de líneas Hough, no solo las más extremas.

Ver contexto completo en `MEJORA_DETECCION_BORDES.md` y `FILTROS_PROPORCION_CORPUS.md`.

---

## Módulo de utilidades compartidas: `_polygon_geometry.py`

Ambos filtros comparten código de geometría a través de este módulo interno (no es un filtro,
no se registra en `FILTER_REGISTRY`). Funciones principales:

| Función | Descripción |
|---------|-------------|
| `line_intersection(l1, l2)` | Intersección de dos líneas (x1,y1,x2,y2). None si paralelas. |
| `polygon_from_lines(top, bottom, left, right)` | Calcula los 4 vértices [TL,TR,BR,BL]. |
| `polygon_proportion(poly)` | `avg(izq,der) / superior`. |
| `polygon_area(poly)` | Área por fórmula de Gauss (shoelace). |
| `add_positions(lines, w, h, is_h)` | Añade posición característica a cada línea. |
| `limit_lines(lines, max_n)` | Muestrea uniformemente preservando extremos. |
| `exhaustive_search(...)` | Prueba todas las combinaciones y devuelve top-K. |
| `candidate_to_border_lines(cand)` | Convierte candidato al formato de `SelectBorderLines`. |

---

## Algoritmo común: búsqueda exhaustiva

Ambos filtros comparten el núcleo de búsqueda:

1. Para cada par `(h_top, h_bottom)` con `h_top.pos < h_bottom.pos`:
   Para cada par `(v_left, v_right)` con `v_left.pos < v_right.pos`:
   - Calcular los 4 vértices (intersecciones de las 4 líneas)
   - Calcular proporción y área del cuadrilátero
   - Calcular score compuesto

2. Devolver los top-K candidatos ordenados por score descendente.

### Score compuesto

```
Con target_area > 0:
  score = w_prop × (1 - |prop - P_obj| / P_obj)
        + w_area × (1 - |area - A_obj| / A_obj)

Sin target_area (= 0):
  score = w_prop × (1 - |prop - P_obj| / P_obj)
        + w_area × (area / img_area)        ← maximizar tamaño
```

### Límite de líneas

Para evitar cuelgues con pools grandes, `max_lines` (default 20) limita el número de líneas
por orientación antes de la búsqueda. Las líneas se muestrean uniformemente preservando
siempre las más extremas (primera y última por posición).

Complejidad máxima: `max_lines² × max_lines² = 20⁴ ≈ 160.000 combinaciones`.
En la práctica Hough devuelve <20 líneas por orientación, así que el límite rara vez actúa.

---

## RefinePolygonByArea

**Archivo**: `src/filter_library/refine_polygon_by_area.py`

### Cuándo usarlo

Cuando se conoce la **proporción del papel** (siempre necesaria) y, opcionalmente, el
**área esperada** del polígono en la resolución de trabajo. El área es constante en
corpus de zoom fijo (todas las fotos tomadas desde la misma distancia).

### Inputs

| Nombre             | Tipo         | Descripción |
|--------------------|--------------|-------------|
| `horizontal_lines` | `lines`      | Todas las líneas horizontales de `ClassifyLinesByAngle`. |
| `vertical_lines`   | `lines`      | Todas las líneas verticales de `ClassifyLinesByAngle`. |
| `lines_metadata`   | `metadata`   | Dimensiones de imagen (de `ClassifyLinesByAngle`). |
| `proportion_data`  | `metadata`   | Opcional. Dict con `peak_proportion` o `proportion` (de `FindPeakProportion` o `CalculatePolygonProportion`). |

### Outputs

| Nombre               | Tipo           | Descripción |
|----------------------|----------------|-------------|
| `selected_lines`     | `border_lines` | Las 4 líneas óptimas (compatible con `CalculateQuadCorners`). |
| `selection_metadata` | `metadata`     | Score, proporción obtenida, estadísticas de búsqueda. |
| `sample_image`       | `image`        | Visualización con polígono seleccionado y candidatos alternativos. |

### Parámetros

| Parámetro            | Default | Descripción |
|----------------------|---------|-------------|
| `target_proportion`  | `0.0`   | Proporción objetivo. Si es 0, se lee de `proportion_data`. |
| `target_area`        | `0.0`   | Área objetivo en px² (resolución de trabajo). 0 = maximizar tamaño. |
| `max_lines`          | `20`    | Límite de seguridad por orientación. |
| `proportion_weight`  | `0.6`   | Peso del término de proporción en el score. |
| `area_weight`        | `0.4`   | Peso del término de área. |
| `top_k`              | `3`     | Candidatos a conservar internamente. |
| `visualization_size` | `900`   | Ancho de la imagen de muestra. |

### Uso en pipeline (con área conocida)

```json
"refine_borders": {
    "filter_name": "RefinePolygonByArea",
    "description": "Refinar bordes por proporción + área del corpus",
    "inputs": {
        "horizontal_lines": "angle_classify.horizontal_lines",
        "vertical_lines":   "angle_classify.vertical_lines",
        "lines_metadata":   "angle_classify.lines_metadata",
        "proportion_data":  "calibracion.peak_data"
    }
}
```

Con `target_area` y `target_proportion` como parámetros directos:

```json
"refine_borders": {
    "filter_name": "RefinePolygonByArea",
    "inputs": {
        "horizontal_lines": "angle_classify.horizontal_lines",
        "vertical_lines":   "angle_classify.vertical_lines",
        "lines_metadata":   "angle_classify.lines_metadata"
    }
}
```
Y en `params.json`:
```json
"refine_borders": {
    "target_proportion": 0.737,
    "target_area": 14500000
}
```

---

## RefinePolygonByCanny

**Archivo**: `src/filter_library/refine_polygon_by_canny.py`

### Cuándo usarlo

Cuando **no se conoce el área** exacta pero se dispone de la imagen de bordes (Canny).
El filtro primero evalúa qué líneas del pool tienen respaldo real en los bordes de la imagen,
descarta las que no lo tienen, y luego busca la proporción óptima entre las supervivientes.

También con área opcional (Estrategia B+3 de la planificación).

### Inputs

| Nombre             | Tipo         | Descripción |
|--------------------|--------------|-------------|
| `horizontal_lines` | `lines`      | Todas las líneas horizontales de `ClassifyLinesByAngle`. |
| `vertical_lines`   | `lines`      | Todas las líneas verticales de `ClassifyLinesByAngle`. |
| `lines_metadata`   | `metadata`   | Dimensiones de imagen. |
| `canny_image`      | `image`      | Imagen de bordes (salida de `CannyEdge`). |
| `proportion_data`  | `metadata`   | Opcional. Dict con proporción objetivo (misma que `RefinePolygonByArea`). |

### Outputs

Idénticos a `RefinePolygonByArea` (compatible con `CalculateQuadCorners`).

### Parámetros adicionales respecto a RefinePolygonByArea

| Parámetro           | Default | Descripción |
|---------------------|---------|-------------|
| `min_canny_support` | `0.25`  | Fracción mínima de puntos de la línea con borde Canny detectado (0-1). Líneas por debajo se descartan. |
| `canny_band_px`     | `5`     | Radio en píxeles de la banda alrededor de cada punto muestreado. |
| `canny_sample_step` | `8`     | Distancia entre puntos de muestreo a lo largo de la línea (px). |

### Cálculo del soporte Canny

Para cada línea se muestrean puntos a lo largo de ella (cada `canny_sample_step` píxeles).
En cada punto se busca si hay algún píxel activo en la imagen Canny dentro de una banda de
`canny_band_px` píxeles. El soporte es la fracción de puntos con borde encontrado:

```
soporte = puntos_con_borde / total_puntos_válidos
```

Las líneas con `soporte < min_canny_support` se descartan antes de la búsqueda exhaustiva.
Esto reduce el pool de forma inteligente: solo participan líneas que coinciden con bordes reales.

### Visualización

El `sample_image` usa código de colores:
- **Rojo tenue**: líneas descartadas por bajo soporte Canny
- **Azul/verde tenue**: líneas supervivientes no seleccionadas
- **Amarillo/naranja prominente**: las 4 líneas del mejor candidato
- **Verde**: polígono resultante
- **Gris**: polígonos de candidatos alternativos

---

## Posición en el pipeline

Ambos filtros se insertan **entre** `ClassifyLinesByAngle` y `CalculateQuadCorners`,
reemplazando a `SelectBorderLines`:

```
HoughLines
    ↓
ClassifyLinesByAngle  →  horizontal_lines, vertical_lines, lines_metadata
    ↓
RefinePolygonByArea       ← recibe horizontal/vertical COMPLETOS
    o                     ← (no las 4 de SelectBorderLines)
RefinePolygonByCanny      ← también recibe canny_image
    ↓  selected_lines, selection_metadata
CalculateQuadCorners      ← misma conexión que antes
    ↓
ScaleQuadCorners
    ↓
PolygonToGTFormat
```

---

## Tabla comparativa de estrategias de selección de bordes

| Filtro                  | Necesita             | Fortaleza | Limitación |
|-------------------------|----------------------|-----------|------------|
| `SelectBorderLines`     | Solo líneas          | Simple, rápido | Confundido por objetos externos |
| `RefinePolygonByArea`   | Proporción + área    | Muy preciso con corpus uniforme | Requiere calibración de área |
| `RefinePolygonByCanny`  | Proporción + Canny   | Robusto sin datos de zoom | Falla si el borde real no tiene borde Canny |
| Combinación de los tres | Todo                 | Máxima robustez | Mayor complejidad de configuración |
