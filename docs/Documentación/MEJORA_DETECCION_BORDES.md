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
    001_derecha.gt.json   ← anotaciones manuales (extensión .gt.json)
    001_izquierda.gt.json
    004_derecha.gt.json
    ...

  experiments/
    001_baseline/
      pipeline.json       ← snapshot del pipeline.json usado en este experimento
      params.json         ← snapshot del params.json usado en este experimento
      metrics.json        ← resultados IoU generados por iou_metrics.py
      results/
        001_derecha.det.json    ← polígono detectado por el pipeline (extensión .det.json)
        001_izquierda.det.json
        004_derecha.det.json
        ...
    002_nombre_experimento/
      ...

  heraldo_raw_100/
    001_derecha.jpg       ← 100 imágenes del corpus (subconjunto de evaluación, estructura plana)
    001_izquierda.jpg
    004_derecha.jpg
    ...
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
python3 src/ground_truth_annotator.py __data/heraldo_raw_100/ \
        --output __data/ground_truth/
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
  Imágenes evaluadas   : 100
  Detecciones fallidas : 0
  IoU medio            : 0.9575
  IoU mediana          : 0.9695
  IoU std              : 0.0381
  IoU mín / máx        : 0.8047 / 0.9935
====================================================
  Por imagen:
    [✓] 001_derecha  IoU=0.9284
    [✗] 022_izquierda  IoU=0.0000   ← detección incompleta
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
   - "folder": "__data/experiments/NNN_nombre/results/"
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

## Arquitectura de agentes IA

El proceso de experimentación se organiza en cuatro agentes especializados, cada uno con un contexto acotado y responsabilidades claras.

### Agente Orquestador
Decide la estrategia de alto nivel. Interpreta resultados cualitativamente ("el IoU bajo está concentrado en imágenes con papeles que sobresalen, no es un problema de parámetros sino de lógica"), decide qué familia de técnicas probar, cuándo abandonar una línea y cuándo combinar estrategias. Crea nuevos filtros cuando la estrategia lo requiere. Es el único que se comunica con el humano para validar decisiones importantes.

### Agente Experimentador
Recibe del Orquestador una consigna acotada ("explorá el espacio de parámetros de `hough_border` y `set_borders`") y ejecuta el loop autónomamente:

```
modificar params.json → correr batch_processor → evaluar iou_metrics → decidir próximo paso
```

Conoce los rangos válidos de cada parámetro y una heurística de búsqueda. Devuelve al Orquestador una tabla de resultados con observaciones ("el parámetro `cluster_left` tiene el mayor impacto en IoU").

### Agente Analista
Recibe los `.det.json` y `.gt.json` de los casos con IoU bajo y clasifica los fallos: ¿es una esquina concreta que siempre falla? ¿un tipo de imagen específico? ¿el borde inferior en particular? Produce un diagnóstico accionable ("el 80% de los fallos son en `bottom_left`, casi siempre de tipo `image_corner`") que el Orquestador usa para decidir qué atacar. Sin este agente los números de IoU son opacos; con él se convierten en dirección concreta.

### Agente Implementador
Conoce en profundidad el código del sistema: arquitectura de filtros (BaseFilter, FILTER_REGISTRY, convenciones de metadata), filtros existentes como referencia de patrones, pipeline.json, batch_processor y herramientas de evaluación. Recibe del Orquestador una especificación técnica ("implementar projection profile según sección 3.2 del paper Shamqoli") y devuelve el filtro integrado y listo para experimentar. Sus fuentes de conocimiento son `FILTER_DEVELOPMENT_GUIDE.md`, `FILTER_REFERENCE.md` y `METADATA_CONVENTION.md`.

### Flujo típico

```
Orquestador
  │
  ├─► Analista ──────► diagnóstico de fallos
  │                         │
  │         ◄───────────────┘
  │
  ├─► Implementador ──► nuevo filtro o modificación al sistema
  │
  └─► Experimentador ─► tabla de resultados por experimento
            │
            ◄─── reporta al Orquestador para siguiente decisión
```

> **Nota:** el Agente Implementador necesita bastante contexto de código para operar bien. En la práctica puede dividirse en un implementador liviano (recibe spec + docs técnicos, escribe el código) validado por el Orquestador (que verifica coherencia con el proyecto).

---

## Estrategias de mejora

### Estrategia 1 (implementada): Baseline Hough
El pipeline actual usa detección de bordes Canny seguida de transformada de Hough. Es el punto de partida para medir mejoras.

**Referencia:** `experiments/001_baseline/`

### Estrategia 2 (implementada): Projection Profile (paper Shamqoli)

Detecta los bordes de la página mediante histogramas de proyección sobre la imagen de bordes Canny.

**Evolución (ver bitácora `bitacora/2026-03-13_experimentos_ppb.md`):**

**v1 — argmax (experimento 005, IoU=0.5375):**
- `top/bottom = argmax(H)` en las zonas superior/inferior del histograma
- Falla porque el pico más alto puede ser una línea de texto interna, no el borde físico

**v2 — onset + pico local (experimento 006, IoU=0.6961):**
- `_find_onset_peak`: escanea desde cada extremo buscando la *primera* señal significativa (`H[i] >= max(H) × onset_threshold`)
- Busca el pico máximo en el vecindario `[onset, onset + onset_neighborhood)`
- El borde físico siempre aparece antes que el contenido impreso → onset encuentra la señal correcta

**v3 — PPB + MinArcLength (experimento 007, IoU=0.9451, mínimo=0.8508):**
- Aplica `MinArcLength(min_length=401px)` sobre el Canny **antes** de calcular los perfiles de proyección
- Elimina el texto impreso del mapa de bordes (caracteres individuales tienen arcos cortos; los bordes físicos tienen arcos largos)
- Resultado: los histogramas quedan limpios y el onset converge directamente al borde físico del papel
- **Cero fallos catastróficos** (mínimo IoU = 0.8508 vs 0.8047 del baseline Hough)

**Parámetros del 007:**

| Parámetro | Valor | Descripción |
|---|---|---|
| `min_length` (MinArcLength) | 401 px | Umbral de longitud mínima de arco Canny |
| `onset_threshold` | 0.07 | Fracción del máximo del perfil para definir señal |
| `onset_neighborhood` | 20 | Vecindario de búsqueda del pico desde el onset |
| `search_zone_h` | 0.25 | Fracción de la imagen donde buscar bordes H |
| `search_zone_v` | 0.30 | Fracción de la imagen donde buscar bordes V |
| `use_phase2` | 0 | Fase 2 (cuartiles Shamqoli) desactivada |

Paper de referencia: `docs/Info_relacionada/Shamqoli-...-BorderDetection...pdf`

### Estrategia 3 (implementada): Proporción invariante del corpus

Todas las páginas del Heraldo son la misma hoja física → misma proporción `avg(izq, der) / superior`.

**Calibración del corpus** (implementada):
- `CalculatePolygonProportion`: mide la proporción de la detección de cada imagen.
- `FindPeakProportion` / `scripts/find_peak_proportion.py`: acumula todas las proporciones del corpus y encuentra el pico mediante histograma con suavizado gaussiano.

**Uso como restricción de detección** (implementado):
- `RefinePolygonByArea`: búsqueda exhaustiva sobre el pool completo de líneas Hough buscando la combinación con proporción + área más cercana al objetivo. Reemplaza a `SelectBorderLines`.
- `RefinePolygonByCanny`: igual que `RefinePolygonByArea` pero con pre-filtrado por soporte en imagen Canny para descartar líneas sin respaldo real antes de la búsqueda.

Ver `FILTROS_PROPORCION_CORPUS.md` y `FILTROS_REFINAMIENTO_POLIGONO.md` para documentación detallada.

### Las tres estrategias son complementarias

| Técnica | Fortaleza | Estado |
|---------|-----------|--------|
| Hough lines (001) | Detecta líneas físicas rectas con precisión cuando el contraste es bueno | Implementada — IoU=0.9575, mínimo=0.8047 |
| Proporción invariante (002-004) | Valida y corrige cuando SelectBorderLines elige líneas de elementos externos | Implementada — IoU menor al baseline por mayor complejidad de búsqueda |
| Projection profile (007) | Robusto cuando la línea no es visible; mínimo IoU garantizado | Implementada — IoU=0.9451, mínimo=0.8508, cero fallos |

---

## Estado actual de implementación

### Filtros implementados

| Filtro | Función en la estrategia |
|--------|--------------------------|
| `HoughLines` | Detección base de líneas (Estrategia 1) |
| `ClassifyLinesByAngle` | Separar líneas por orientación |
| `SelectBorderLines` | Selección simple de las 4 líneas más extremas (Estrategia 1) |
| `CalculateQuadCorners` | Calcular vértices del polígono |
| `CalculatePolygonProportion` | Medir la proporción de una detección individual (Estrategia 3 — calibración) |
| `FindPeakProportion` | Encontrar la proporción más frecuente del corpus (Estrategia 3 — calibración) |
| `RefinePolygonByArea` | Búsqueda por proporción + área (Estrategia 3 — restricción) |
| `RefinePolygonByCanny` | Búsqueda con soporte Canny (Estrategia 3 — restricción) |
| `PolygonToGTFormat` | Convertir resultado a formato de evaluación |
| `MinArcLength` | Filtra bordes Canny por longitud mínima de arco — clave en experimento 007 |
| `ProjectionProfileBorder` | Histograma de proyección con onset+pico local (Estrategia 2) |
| `FilterLinesByPPBZone` | Filtra líneas Hough a la zona de búsqueda del PPB |

### Herramientas de soporte implementadas

| Herramienta | Ubicación | Función |
|-------------|-----------|---------|
| `find_peak_proportion.py` | `scripts/` | Script autónomo de calibración de corpus |
| `ground_truth_annotator.py` | `src/` | GUI para anotación manual de ground truth |
| `iou_metrics.py` | `src/` | Evaluación IoU de experimentos |
| `debug_projection_profile.py` | `scripts/` | Histogramas PPB + Canny de una imagen para diagnóstico |
| `_polygon_geometry.py` | `src/filter_library/` | Módulo interno de geometría compartido |

### Tabla de experimentos

| Exp | Descripción | IoU medio | Std | Mín | Fallidos |
|---|---|---|---|---|---|
| 001_baseline | SelectBorderLines (Hough) | **0.9575** | 0.0381 | 0.8047 | 0 |
| 002_refine_polygon | RefinePolygonByCanny sin área | 0.8243 | 0.2910 | — | 1 |
| 003_refine_area_fraction | + target_area_fraction | 0.8239 | 0.2526 | — | 1 |
| 004_refine_zones | + restricciones zona | 0.7803 | 0.3470 | — | 16 |
| 005_projection_profile | PPB argmax | 0.5375 | 0.1395 | — | 0 |
| 006_ppb_onset | PPB onset+pico local | 0.6961 | 0.0469 | 0.600 | 0 |
| **007_min_arc_canny** | **PPB + MinArcLength(401px) pre-PPB** | **0.9451** | **0.0329** | **0.8508** | **0** |

**Conclusión actual:** el 001 (Hough) sigue siendo el más preciso (IoU=0.9575) pero con fallos catastróficos ocasionales. El 007 (PPB + MinArcLength) es el más robusto: IoU casi igual (0.9451), mínimo garantizado de 0.8508, cero fallos. En producción se usa el 001; el 007 es candidato para pipeline híbrido de validación cruzada (próximo experimento 008).

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
  "evaluated_at": "2026-03-13T12:00:00",
  "gt_folder": "__data/ground_truth",
  "summary": {
    "evaluated": 100,
    "failed_detection": 0,
    "mean_iou": 0.9575,
    "median_iou": 0.9695,
    "std_iou": 0.0381,
    "min_iou": 0.8047,
    "max_iou": 0.9935
  },
  "results": [
    {"image": "001_derecha", "iou": 0.9284, "all_corners_found": true},
    ...
  ]
}
```
