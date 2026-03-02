# Proyecto: Mejora de Detección de Bordes de Página

## Contexto

El corpus Heraldo consiste en fotografías de páginas de periódico histórico. Cada imagen muestra la página **rotada 90°** respecto a su orientación natural. El pipeline principal (`Heraldo_Derecha`, `Heraldo_Izquierda`) detecta los bordes físicos del papel para recortarlo correctamente.

### El problema

La detección de bordes falla en varios escenarios frecuentes en el corpus:

- **Papeles que sobresalen** de los márgenes (hojas intercaladas, protectores)
- **Páginas adyacentes que aparecen** en los bordes de la imagen
- Estas interferencias confunden la detección de líneas de Hough incluso en los bordes que en teoría tienen buen contraste figura/fondo (laterales y superior)

El borde inferior (lomo/pliegue donde se unen dos páginas) también es problemático, aunque es un problema secundario comparado con los anteriores.

### Objetivo

Desarrollar y evaluar estrategias que mejoren la robustez de la detección del polígono de borde de página, midiendo el progreso de forma objetiva mediante la métrica **IoU (Intersection over Union)** entre el polígono detectado automáticamente y anotaciones manuales de referencia (ground truth).

---

## Estructura de directorios

```
__data/
  ground_truth/
    Derecha/
      001.gt.json         ← anotaciones manuales (extensión .gt.json)
      004.gt.json
      ...

  experiments/
    001_baseline/
      pipeline.json       ← snapshot del pipeline.json usado en este experimento
      params.json         ← snapshot del params.json usado en este experimento
      metrics.json        ← resultados IoU generados por iou_metrics.py
      results/
        Derecha/
          001.det.json    ← polígono detectado por el pipeline (extensión .det.json)
          004.det.json
          ...
    002_nombre_experimento/
      ...

  heraldo_raw_100/
    Derecha/              ← 100 imágenes del corpus (subconjunto de evaluación)
    Izquierda/
```

**Convención de extensiones:**

| Extensión | Significado |
|-----------|-------------|
| `.gt.json` | Anotación manual de ground truth — **nunca sobreescribir** |
| `.det.json` | Detección automática del pipeline en un experimento |

---

## Herramientas

### 1. `src/ground_truth_annotator.py` — Anotación manual

GUI (tkinter) para marcar manualmente las 4 esquinas del borde físico del papel en cada imagen. Genera los archivos `.gt.json`.

```bash
python3 src/ground_truth_annotator.py <carpeta_imagenes> --output <carpeta_gt>

# Ejemplo para el corpus de evaluación:
python3 src/ground_truth_annotator.py __data/heraldo_raw_100/Derecha/ \
        --output __data/ground_truth/Derecha/
```

**Controles:**

| Acción | Control |
|--------|---------|
| Colocar punto (orden: TL → TR → BR → BL) | Click izquierdo |
| Mover punto colocado | Arrastrar izquierdo |
| Zoom centrado en cursor | Rueda del ratón |
| Paneo | Arrastrar botón derecho |
| Reset zoom | `R` |
| Imagen anterior / siguiente | `←` / `→` |
| Guardar | `S` o botón Guardar |

**Formato de salida** (`.gt.json`):
```json
{
  "image_file": "001.jpg",
  "image_width": 6000,
  "image_height": 4000,
  "polygon": [
    {"x": 120, "y": 95},
    {"x": 5880, "y": 88},
    {"x": 5910, "y": 3960},
    {"x": 108, "y": 3971}
  ]
}
```

- Coordenadas absolutas en píxeles de la imagen original
- 4 puntos en sentido horario desde esquina superior izquierda
- Se incluyen las dimensiones de la imagen para normalización posterior

---

### 2. `examples/Heraldo_Claude/` — Pipeline de detección

Pipeline de trabajo derivado de `Heraldo_Derecha`, simplificado para evaluación:

- **Sin corrección de rotación** (se eliminaron 9 filtros): el objetivo actual es sólo mejorar la detección del polígono
- **Termina en `PolygonToGTFormat`**: convierte las esquinas detectadas a coordenadas originales y las empaqueta en formato compatible con `.gt.json`

Cadena de filtros:

```
original.image
  → main_resize             (escala a resolución de trabajo, ej. 4000px)
  → resize_for_detect_borders (reduce para detección)
  → contrast / histogram_autopeaks / normalize_from_histogram_peaks
  → blur → denoise
  → canny_border → hough_border
  → angle_classify → set_borders → quad_corners
  → scale_to_original       (escala esquinas a resolución main_resize)
  → polygon_to_gt           (re-escala a original, empaqueta como .det.json)
```

**Ejecución:**
```bash
cd /ruta/proyecto
python3 src/batch_processor.py --pipeline examples/Heraldo_Claude/
```

El destino de los resultados se configura en `examples/Heraldo_Claude/batch_config.json`. Actualizar la carpeta del experimento antes de cada corrida.

---

### 3. `src/iou_metrics.py` — Evaluación IoU

Compara `.det.json` (pipeline) contra `.gt.json` (anotación manual) y genera un informe de métricas.

#### Evaluar un experimento

```bash
python3 src/iou_metrics.py evaluate __data/experiments/001_baseline/

# Con carpeta GT explícita:
python3 src/iou_metrics.py evaluate __data/experiments/002_nombre/ \
        --gt __data/ground_truth/
```

Genera `__data/experiments/001_baseline/metrics.json` y muestra por consola:

```
====================================================
  001_baseline
====================================================
  Imágenes evaluadas   : 50
  Detecciones fallidas : 3
  IoU medio            : 0.8812
  IoU mediana          : 0.9034
  IoU std              : 0.0941
  IoU mín / máx        : 0.4210 / 0.9741
====================================================
  Por imagen:
    [✓] Derecha/001  IoU=0.9284
    [✗] Derecha/007  IoU=0.0000   ← detección incompleta
    ...
```

El símbolo `[✗]` indica que el pipeline no encontró las 4 esquinas y usó puntos de fallback (bordes de imagen), lo que típicamente produce IoU bajo.

#### Comparar todos los experimentos

```bash
python3 src/iou_metrics.py compare
```

Muestra una tabla ordenada por IoU medio descendente:

```
═══════════════════════════════════════════════════════════════════════════
  COMPARATIVA DE EXPERIMENTOS — IoU de polígono detectado vs ground truth
═══════════════════════════════════════════════════════════════════════════
Experimento              N  Fallidos    Media  Mediana     Std     Mín     Máx
───────────────────────────────────────────────────────────────────────────
003_projection_profile  50         1   0.9210   0.9380  0.0610  0.7100  0.9830
002_mejor_canny         50         2   0.8950   0.9100  0.0720  0.5900  0.9741
001_baseline            50         3   0.8812   0.9034  0.0941  0.4210  0.9741
```

---

## Flujo de trabajo para un experimento

```
1. Modificar examples/Heraldo_Claude/pipeline.json y/o params.json
         ↓
2. Actualizar batch_config.json:
   - "folder": "__data/experiments/NNN_nombre/results/Derecha/"
         ↓
3. Correr el pipeline:
   python3 src/batch_processor.py --pipeline examples/Heraldo_Claude/
         ↓
4. Guardar snapshot del pipeline usado:
   cp examples/Heraldo_Claude/pipeline.json __data/experiments/NNN_nombre/
   cp examples/Heraldo_Claude/params.json   __data/experiments/NNN_nombre/
         ↓
5. Evaluar:
   python3 src/iou_metrics.py evaluate __data/experiments/NNN_nombre/
         ↓
6. Ver comparativa:
   python3 src/iou_metrics.py compare
```

---

## Estrategias de mejora planificadas

### Estrategia 1 (implementada): Baseline Hough
El pipeline actual usa detección de bordes Canny seguida de transformada de Hough. Es el punto de partida para medir mejoras.

**Referencia:** `experiments/001_baseline/`

### Estrategia 2: Projection Profile (paper Shamqoli)
Aplicar histogramas de proyección sobre la imagen de bordes (Prewitt):
- **Picos** en los extremos del histograma → borde físico del papel
- **Fase 2 del paper**: refinamiento con cuartiles (LQ/UQ) para detectar texto de página vecina y recalcular ese borde
- Particularmente útil cuando la línea física no es clara y el texto de la página adyacente contamina el borde

Paper de referencia: `docs/Info_relacionada/Shamqoli-...-BorderDetection...pdf`

### Estrategia 3: Proporción invariante del corpus
Todas las páginas del Heraldo son la misma hoja física → misma proporción ancho/alto:
- Acumular detecciones del corpus → histograma de proporciones → pico = proporción real
- Cuando una detección diverge del pico → recalcular el borde usando la proporción correcta
- Especialmente útil para inferir el borde inferior (lomo) cuando falla

### Las tres estrategias son complementarias

| Técnica | Fortaleza |
|---------|-----------|
| Hough lines (actual) | Detecta líneas físicas rectas con precisión cuando el contraste es bueno |
| Projection profile | Robusto cuando la línea no es visible; maneja texto de página vecina |
| Proporción invariante | Valida y corrige cuando alguno de los dos falla |

---

## Formato de los archivos JSON

### `.gt.json` — Ground truth manual

```json
{
  "image_file": "001.jpg",
  "image_width": 6000,
  "image_height": 4000,
  "polygon": [
    {"x": 120, "y": 95},
    {"x": 5880, "y": 88},
    {"x": 5910, "y": 3960},
    {"x": 108,  "y": 3971}
  ]
}
```

### `.det.json` — Detección automática del pipeline

```json
{
  "image_file": "",
  "image_width": 6000,
  "image_height": 4000,
  "polygon": [
    {"x": 408,  "y": 336,  "type": "intersection"},
    {"x": 5316, "y": 336,  "type": "intersection"},
    {"x": 5316, "y": 3976, "type": "intersection"},
    {"x": 408,  "y": 3976, "type": "intersection"}
  ],
  "all_corners_found": true
}
```

El campo `type` en `.det.json` indica la confiabilidad de cada esquina:

| Valor | Significado |
|-------|-------------|
| `intersection` | Intersección real de dos líneas de Hough — más confiable |
| `mixed_h_border` | Una línea de Hough + borde horizontal de imagen |
| `mixed_v_border` | Una línea de Hough + borde vertical de imagen |
| `image_corner` | Esquina de imagen pura — el pipeline no encontró nada |

### `metrics.json` — Resultados de evaluación

```json
{
  "experiment": "001_baseline",
  "evaluated_at": "2026-03-02T12:00:00",
  "gt_folder": "__data/ground_truth",
  "summary": {
    "evaluated": 50,
    "failed_detection": 3,
    "mean_iou": 0.8812,
    "median_iou": 0.9034,
    "std_iou": 0.0941,
    "min_iou": 0.4210,
    "max_iou": 0.9741
  },
  "results": [
    {"image": "001", "subset": "Derecha", "iou": 0.9284, "all_corners_found": true},
    ...
  ]
}
```
