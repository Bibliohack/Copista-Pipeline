# Propuestas de Filtros: Análisis Bibliográfico

**Documento preparado por:** copista-analista
**Fecha:** Marzo 2026
**Fuentes analizadas:**
- `BIBLIOGRAFIA_DETECCION_BORDES.md` (20 papers, 2 catálogos de técnicas, 6 secciones)
- `diferenciar_lineas_con_segmentos_densos.md` (3 técnicas de post-filtrado Hough)
- `programacion_basada_en_nodos.md` (referencias de UI, no directamente relevante al problema de bordes)

---

## Resumen ejecutivo

El análisis bibliográfico identifica cuatro oportunidades de mejora sin requerimientos de datos
de entrenamiento, aplicables directamente al corpus Heraldo con los filtros actuales como
base. La más urgente es el **Projection Profile** (Shamqoli 2013), que ya estaba identificada
como Estrategia 2 pendiente en `MEJORA_DETECCION_BORDES.md`: ataca directamente el problema
de páginas adyacentes visibles sin necesidad de GPU ni anotaciones. Le sigue en prioridad el
**filtrado por densidad de segmentos Canny** sobre líneas Hough individuales, que resolvería
casos donde `RefinePolygonByCanny` descarta líneas válidas porque la imagen Canny es ruidosa
cerca del borde. Como tercera vía, el **criterio de contraste interior/exterior** (Skoryukina
2020) complementa al sistema de proporción existente para descartar cuadriláteros que engloban
fondo oscuro. Finalmente, el **LSD (Line Segment Detector)** como alternativa a Hough mejoraría
la calidad del pool de líneas de entrada en imágenes con ruido. Las técnicas de aprendizaje
profundo (U-Net, regresión de esquinas, HED) son viables a mediano plazo pero requieren
infraestructura de GPU o anotación intensiva.

---

## Propuestas priorizadas

---

### P1: ProjectionProfileBorder

**Origen:** Shamqoli & Khosravi (2013), "Border detection of document images scanned from
large books"; también Stamatopoulos & Gatos (2007), "Automatic Borders Detection of Camera
Document Images"; Shafait et al. (2007/2010), "Page Frame Detection for Marginal Noise Removal"

**Problema que resuelve:** El pipeline actual (Hough + `SelectBorderLines` /
`RefinePolygonByArea`) falla cuando una página adyacente ocupa una franja significativa del
borde de la imagen, porque el pool de líneas Hough no tiene ninguna línea que coincida con el
borde real del papel: todas las líneas detectadas corresponden a estructuras internas del texto.
El Projection Profile opera sobre acumulaciones de píxeles por fila/columna, lo que le permite
detectar la *transición* entre la zona de texto de la página adyacente y la zona vacía entre
páginas, independientemente de si esa transición produce un borde Canny claro.

**Descripción técnica:** Para cada fila de la imagen (o columna, según la orientación), se
calcula la suma de píxeles de borde (imagen Prewitt o Canny binaria). El perfil resultante
tiene dos tipos de picos: picos altos distribuidos corresponden a filas con texto (muchos
bordes de letras); picos concentrados en zonas muy localizadas corresponden a bordes físicos
del papel. Detectar los "valles" (mínimos locales) en el perfil separa las regiones de
contenido de las regiones de margen. En la Fase 2 del paper Shamqoli, si una de las cuatro
fronteras detectadas cae dentro de una zona de texto denso (diagnóstico por cuartiles del
perfil), se interpreta como texto de página adyacente y se recalcula el borde buscando el
siguiente valle hacia el interior. Esto es directamente el problema de las páginas adyacentes
del corpus Heraldo.

**Inputs propuestos:**

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| `edge_image` | `image` | Imagen de bordes (salida de `CannyEdge` o imagen Prewitt). |
| `proportion_data` | `metadata` | Opcional. Proporción objetivo del corpus (de `FindPeakProportion`), usada para resolver ambigüedad cuando el perfil tiene múltiples valles. |

**Outputs propuestos:**

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| `border_lines` | `border_lines` | Las 4 líneas (top, bottom, left, right) en el mismo formato que `SelectBorderLines`, compatibles con `CalculateQuadCorners`. |
| `projection_metadata` | `metadata` | Perfiles de proyección horizontal y vertical, posiciones de bordes detectados, indicadores de "borde en zona de texto" por cada lado. |
| `sample_image` | `image` | Visualización de los perfiles como gráficas superpuestas a la imagen. |

**Posición en el pipeline:** Alternativa a `SelectBorderLines` y a los filtros de refinamiento.
Puede recibir la imagen Canny que ya produce el pipeline actual (`canny_border`) sin añadir
preprocesamiento nuevo.

```
HoughLines → ClassifyLinesByAngle
                   ↓
CannyEdge  →  ProjectionProfileBorder   ← nuevo filtro (usa Canny directamente)
                   ↓ border_lines
             CalculateQuadCorners
```

O bien como validador/complemento después de `RefinePolygonByArea`, usando `projection_metadata`
para diagnosticar qué bordes del polígono caen en zonas de texto y refinar solo esos.

**Complejidad:** Baja. La operación central es `np.sum(edge_image, axis=1)` (o `axis=0`) sobre
la imagen Canny ya disponible, seguida de detección de picos y valles con `scipy.signal.find_peaks`
o lógica similar. La Fase 2 (detección de texto de página adyacente por cuartiles) añade una
docena de líneas de NumPy. No requiere GPU, red neuronal ni datos de entrenamiento. El paper
de referencia (`Shamqoli-...-BorderDetection...pdf`) está disponible en `docs/Info_relacionada/`.

**Prioridad:** Alta. Es la única de las propuestas clásicas que ataca directamente el caso de
*páginas adyacentes* (el problema más frecuente según `MEJORA_DETECCION_BORDES.md`), sin
depender de que Hough haya detectado una línea en la posición correcta. El propio documento
`MEJORA_DETECCION_BORDES.md` la lista como Estrategia 2 pendiente. Además, la bibliografía
muestra que este enfoque fue evaluado específicamente en imágenes de libros digitalizados con
páginas dobles, que es el caso análogo al corpus Heraldo.

---

### P2: FilterLinesBySegmentDensity

**Origen:** `diferenciar_lineas_con_segmentos_densos.md` (análisis interno, dic/2025), que
describe tres variantes: HoughLinesP con post-filtrado por segmentos continuos, análisis de
densidad local con ventana deslizante, y preprocesamiento morfológico antes de Hough.

**Problema que resuelve:** `RefinePolygonByCanny` usa el soporte Canny como criterio de
filtrado de líneas, pero el soporte se calcula como fracción de puntos con borde en una
banda alrededor de cada línea, sin distinguir si esos bordes son *contiguos* (borde físico
del papel) o *dispersos* (bordes de letras individuales de texto). Una línea de Hough que
pase por una fila de texto denso puede tener soporte Canny comparable al de la línea del
borde del papel, resultando en falsos positivos que confunden al filtro de refinamiento.
El criterio de densidad de segmentos continuos discrimina estructuralmente entre ambos casos.

**Descripción técnica:** Para cada línea Hough candidata, se muestrean los píxeles de la
imagen Canny a lo largo de la línea (ya implementado en `RefinePolygonByCanny`). En lugar
de calcular solo la fracción de puntos activos (soporte), se analiza la *distribución* de
esos puntos: se identifican "runs" de píxeles activos consecutivos (segmentos continuos).
Una línea de borde físico tiene un único segmento largo (o muy pocos segmentos largos); una
línea de texto tiene muchos segmentos cortos separados. El criterio de descarte es: la línea
se considera válida solo si tiene al menos un segmento continuo cuya longitud supera un
umbral mínimo (por ejemplo, 50 píxeles), o si la suma de los segmentos que superan una
longitud mínima equivale a al menos el 60% de la longitud total de la línea.

**Inputs propuestos:**

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| `horizontal_lines` | `lines` | Líneas horizontales de `ClassifyLinesByAngle`. |
| `vertical_lines` | `lines` | Líneas verticales de `ClassifyLinesByAngle`. |
| `lines_metadata` | `metadata` | Dimensiones de imagen (de `ClassifyLinesByAngle`). |
| `canny_image` | `image` | Imagen de bordes (salida de `CannyEdge`). |

**Outputs propuestos:**

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| `horizontal_lines` | `lines` | Líneas horizontales filtradas (solo las con segmentos densos). |
| `vertical_lines` | `lines` | Líneas verticales filtradas. |
| `lines_metadata` | `metadata` | Dimensiones de imagen (pasada sin cambios, para compatibilidad). |
| `filter_metadata` | `metadata` | Número de líneas descartadas por orientación, estadísticas de densidad. |

**Posición en el pipeline:** Entre `ClassifyLinesByAngle` y `RefinePolygonByArea` /
`RefinePolygonByCanny`, como pre-filtro de calidad del pool de líneas:

```
ClassifyLinesByAngle
        ↓
FilterLinesBySegmentDensity   ← nuevo filtro (reduce el pool antes de la búsqueda exhaustiva)
        ↓
RefinePolygonByArea  o  RefinePolygonByCanny
        ↓
CalculateQuadCorners
```

**Complejidad:** Baja. El código base ya existe en `RefinePolygonByCanny` (el muestreo de
píxeles a lo largo de líneas). Solo se añade el análisis de runs. El documento
`diferenciar_lineas_con_segmentos_densos.md` proporciona pseudocódigo detallado de la función
`has_dense_segments()`. Toda la lógica es NumPy puro, sin dependencias adicionales.

**Prioridad:** Media. No resuelve un caso nuevo por sí solo, pero aumenta la calidad del
pool de entrada a los filtros de refinamiento existentes, reduciendo la probabilidad de que
una línea de texto "compita" con la línea del borde real en la búsqueda exhaustiva. Es
especialmente útil cuando el corpus tiene imágenes con texto muy denso cerca del margen.

---

### P3: ScorePolygonByContrast

**Origen:** Skoryukina et al. (2020), "Approach for Document Detection by Contours and
Contrasts", arXiv:2008.02615. Evaluado en el dataset MIDV-500; reporta reducción del 40%
en errores de ordenamiento de hipótesis.

**Problema que resuelve:** Cuando la búsqueda exhaustiva de `RefinePolygonByArea` o
`RefinePolygonByCanny` tiene varios candidatos con scores de proporción similares, el
desempate actual usa solo el área. En casos donde un objeto externo (papel que sobresale)
tiene proporciones similares a las del papel principal, el score de proporción + área no
alcanza para discriminar. El contraste entre el interior del cuadrilátero (el papel del
periódico, relativamente claro) y el exterior (fondo fotográfico, generalmente más oscuro)
es un criterio adicional robusto que no requiere parámetros del corpus.

**Descripción técnica:** Para cada cuadrilátero candidato que ya haya pasado el filtrado
por proporción, se calcula el contraste como la diferencia de luminosidad media entre el
área interior del polígono y el área exterior:

```
contraste = |luminosidad_media_interior - luminosidad_media_exterior|
```

El área interior se obtiene con `cv2.fillPoly()` (máscara interior) y el área exterior es
la imagen completa menos esa máscara. El cuadrilátero con mayor contraste, entre los
candidatos con scores de proporción similares, se selecciona como solución. Como variante
más robusta, el contraste puede calcularse solo en una banda estrecha alrededor del borde
del polígono (no toda el área interior/exterior), que es más rápido y menos sensible a
contenido interior variable.

**Inputs propuestos:**

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| `original_image` | `image` | Imagen en escala de grises o color (para calcular luminosidad). Se recomienda usar la imagen normalizada o la imagen de trabajo, no la original a máxima resolución. |
| `selection_metadata` | `metadata` | Salida de `RefinePolygonByArea` o `RefinePolygonByCanny`, que contiene la lista de candidatos top-K con sus coordenadas y scores. |

**Outputs propuestos:**

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| `selected_lines` | `border_lines` | Las 4 líneas del candidato con mejor contraste (compatible con `CalculateQuadCorners`). |
| `contrast_metadata` | `metadata` | Contraste calculado para cada candidato, candidato seleccionado. |
| `sample_image` | `image` | Visualización con candidatos coloreados por contraste. |

**Posición en el pipeline:** Post-proceso de los filtros de refinamiento existentes. Solo
actúa cuando los top-K candidatos tienen scores similares (diferencia < umbral configurable):

```
RefinePolygonByArea o RefinePolygonByCanny
        ↓ selection_metadata (top-K candidatos)
ScorePolygonByContrast
        ↓ selected_lines (el mejor por contraste)
CalculateQuadCorners
```

**Complejidad:** Baja-Media. La operación de máscara con `cv2.fillPoly()` y cálculo de
media sobre la máscara es simple. La complejidad crece si se implementa la variante de
banda estrecha (requiere dilatar y restar máscaras). Para documentos de alta resolución,
se puede submuestrear la imagen antes del cálculo de contraste sin pérdida significativa
de discriminación. El principal riesgo es que en el corpus Heraldo el fondo puede ser
variable (fondos claros, superficies blancas), reduciendo la utilidad del criterio en
algunos casos.

**Prioridad:** Media. Complementa los filtros existentes sin reemplazarlos; es un desempate
de bajo costo que puede añadirse como parámetro opcional. No resuelve el caso donde el
cuadrilátero correcto no está en el top-K de `RefinePolygonByArea`, sino que mejora la
selección cuando sí está. Útil particularmente para el caso de papeles que sobresalen con
proporciones similares a la página principal.

---

### P4: LSDLineDetector

**Origen:** Paper 5 (Kaiserslautern, 2018), "A Robust Page Frame Detection Method for
Complex Historical Document Images", que usa el detector LSD (Line Segment Detector)
como componente central; catálogo de técnicas clásicas de `BIBLIOGRAFIA_DETECCION_BORDES.md`,
sección "Detector de Segmentos de Línea (LSD)".

**Problema que resuelve:** La transformada de Hough estándar (`HoughLines`) acumula votos
en el espacio de parámetros (rho, theta) y puede producir "líneas fantasma" donde múltiples
bordes débiles de texto se superponen en el espacio de Hough, generando líneas que no
corresponden a ningún segmento continuo real. Esto contamina el pool de entrada a
`RefinePolygonByArea` y `RefinePolygonByCanny`. LSD opera directamente sobre gradientes
de la imagen y produce únicamente segmentos de línea que tienen soporte continuo en la
imagen original, sin parámetros de umbral delicados.

**Descripción técnica:** LSD (Grompone von Gioi et al., 2010, disponible como
`cv2.createLineSegmentDetector()` en OpenCV) analiza los gradientes locales de la imagen,
agrupa píxeles con gradiente similar en "line-support regions" y ajusta un segmento de línea
a cada región. El resultado es una lista de segmentos (x1,y1,x2,y2) con longitud, precisión
y score. Se propone usarlo como alternativa o complemento a `HoughLines`:

1. Aplicar `cv2.createLineSegmentDetector().detect(gray_image)`.
2. Filtrar por longitud mínima (descartar segmentos cortos de letras).
3. Clasificar por ángulo (como hace `ClassifyLinesByAngle`).
4. De cada segmento, extraer la línea infinita en formato (rho, theta) para compatibilidad
   con `CalculateQuadCorners`, o bien extender los segmentos a la anchura/altura de la imagen.

El paper de Kaiserslautern combina LSD con operaciones morfológicas previas (para reforzar
bordes del papel) y un algoritmo de matching geométrico posterior, lo que produce un sistema
robusto para documentos históricos del siglo XVI-XIX con calidad de imagen variable.

**Inputs propuestos:**

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| `gray_image` | `image` | Imagen en escala de grises (preprocesada con blur/denoise). |
| `image_metadata` | `metadata` | Dimensiones de imagen, para extensión de segmentos a bordes de imagen. |

**Outputs propuestos:**

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| `horizontal_lines` | `lines` | Segmentos horizontales detectados, en el mismo formato que `ClassifyLinesByAngle`. |
| `vertical_lines` | `lines` | Segmentos verticales detectados. |
| `lines_metadata` | `metadata` | Dimensiones de imagen, número de segmentos detectados por orientación. |
| `sample_image` | `image` | Visualización de los segmentos detectados por orientación. |

**Posición en el pipeline:** Reemplaza la combinación `HoughLines + ClassifyLinesByAngle`
con un único filtro que produce la misma interfaz de salida:

```
gray_image → LSDLineDetector → horizontal_lines, vertical_lines, lines_metadata
                                        ↓
                          RefinePolygonByArea o RefinePolygonByCanny
```

O bien, como variante conservadora, los segmentos LSD se fusionan con las líneas Hough
existentes para enriquecer el pool antes de la búsqueda exhaustiva.

**Complejidad:** Media. `cv2.createLineSegmentDetector()` está disponible en OpenCV sin
instalación adicional. La complejidad está en la conversión de segmentos LSD al formato
de líneas esperado por los filtros downstream: los segmentos LSD tienen endpoints
(x1,y1,x2,y2) mientras que los filtros actuales trabajan con (rho,theta) de Hough. Se
requiere implementar la conversión y asegurar que `CalculateQuadCorners` o la capa de
compatibilidad maneje ambos formatos. Alternativamente, LSD puede usarse solo como
pre-filtro: sus segmentos definen las zonas de búsqueda para Hough, reduciendo el ruido.

**Prioridad:** Baja-Media. LSD mejora la calidad del pool de líneas pero el sistema de
refinamiento existente (especialmente `RefinePolygonByCanny`) ya mitiga parcialmente el
problema de líneas fantasma de Hough. La prioridad sería mayor si los experimentos con
`RefinePolygonByCanny` muestran que el pool Hough sigue conteniendo muchas líneas sin
soporte real incluso después del filtrado por soporte.

---

### P5: MorphologicalPreprocessForEdges

**Origen:** Paper 5 (Kaiserslautern, 2018); documento `diferenciar_lineas_con_segmentos_densos.md`,
sección "Morfología antes de Hough"; catálogo de técnicas clásicas de `BIBLIOGRAFIA_DETECCION_BORDES.md`,
sección "Morfología Matemática".

**Problema que resuelve:** La imagen Canny actual contiene bordes de letras, manchas de
degradación y ruido fotográfico mezclados con el borde físico del papel. Cuando el borde
del papel tiene bajo contraste (degradación, iluminación no uniforme cerca del lomo), las
operaciones morfológicas pueden reforzar los segmentos largos (borde del papel) y suprimir
los segmentos cortos (letras, ruido), mejorando la calidad de la imagen de bordes antes de
Hough o LSD.

**Descripción técnica:** Sobre la imagen Canny (binaria), aplicar:

1. **Cierre morfológico** con kernel estrecho y largo (ej. 5×1 para horizontal, 1×5 para
   vertical): une los pequeños gaps en bordes físicos del papel que Canny puede segmentar
   en varios segmentos cortos.
2. **Apertura morfológica** con kernel más largo (ej. 15×1 y 1×15): elimina los componentes
   cortos que sobrevivieron al cierre (bordes de letras aisladas, manchas pequeñas).
3. Combinar las dos imágenes resultantes (horizontal + vertical) con `cv2.bitwise_or()`.

El resultado es una imagen de bordes donde solo sobreviven los segmentos físicamente largos
y continuos, que corresponden con mayor probabilidad a los bordes del papel que a estructuras
de texto.

**Inputs propuestos:**

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| `canny_image` | `image` | Imagen Canny binaria (salida de `CannyEdge`). |

**Outputs propuestos:**

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| `morpho_edge_image` | `image` | Imagen de bordes morfológicamente procesada. |

**Posición en el pipeline:** Entre `CannyEdge` y `HoughLines` (o `LSDLineDetector`):

```
CannyEdge
    ↓
MorphologicalPreprocessForEdges   ← nuevo filtro (pre-filtro sobre Canny)
    ↓ morpho_edge_image
HoughLines   (usa la imagen morfológica en lugar del Canny directo)
```

El filtro existente `MorphologyFilter` podría cubrir parcialmente esta necesidad si se
configura con los kernels correctos. Sin embargo, dado que la lógica implica dos operaciones
en dos orientaciones distintas y luego una combinación, probablemente requiere un filtro
específico o una extensión del existente.

**Complejidad:** Baja. `cv2.morphologyEx()` con `MORPH_CLOSE` y `MORPH_OPEN` sobre imagen
binaria. El diseño de los kernels (tamaños, orientaciones) requiere experimentación en el
corpus, pero la implementación en sí es trivial. Antes de crear un filtro nuevo, verificar
si `MorphologyFilter` o `MorphologyAdvanced` ya en el pipeline puede encadenarse para lograr
el mismo efecto ajustando parámetros.

**Prioridad:** Baja. Mejora incremental sobre el Canny existente. Su impacto depende de
cuánto del ruido actual en la imagen Canny proviene de bordes cortos (texto) vs. bordes
largos confundibles con el papel. Útil como experimento antes de adoptar LSD o técnicas
más costosas.

---

## Técnicas descartadas

### HED (Holistically-Nested Edge Detection)
**Por qué se descarta (para implementación inmediata):** Requiere cargar un modelo Caffe
pre-entrenado en BSDS500 (imágenes naturales, no documentos históricos), lo que implica
gestionar archivos de modelo externos y dependencia de `cv2.dnn`. El modelo produciría
mapas de bordes de todas las estructuras de la imagen, no específicamente el borde del papel,
y requeriría el mismo post-procesamiento Hough del pipeline actual. La mejora sobre Canny
en documentos históricos con fine-tuning es demostrable, pero sin fine-tuning el beneficio
es incierto. Se recomienda evaluar cuando los experimentos con técnicas clásicas hayan
alcanzado su límite.

### FCN / U-Net / dhSegment (segmentación semántica con deep learning)
**Por qué se descartan:** Requieren anotación de máscaras de segmentación para el corpus
Heraldo (cientos de imágenes con mask de píxel a pixel), entrenamiento en GPU, y
mantenimiento de pesos del modelo. El retorno es alto en términos de robustez, pero el
costo de entrada es desproporcionado respecto a las mejoras clásicas aún no exploradas.
Candidatas para fase mediano plazo si los métodos clásicos no superan un IoU objetivo.

### Regresión directa de esquinas (DenseNet/MobileNet)
**Por qué se descarta:** Mismo problema que FCN: requiere anotación de las 4 esquinas en
cientos de imágenes (aunque más simple que las máscaras pixel a pixel), infraestructura de
entrenamiento GPU, y la regresión puede ser imprecisa en esquinas ocluidas por el lomo.
El pipeline existente ya localiza esquinas con buena precisión cuando el pool de líneas
Hough es correcto; el problema principal está en la selección de líneas, no en el cálculo
de esquinas.

### GrabCut
**Por qué se descarta:** Requiere un rectángulo inicial de inicialización que a su vez
requiere una detección previa. No añade robustez si la inicialización inicial ya falla.
Útil como refinamiento sub-pixel en casos donde el borde ya está aproximadamente localizado,
pero ese refinamiento no es la limitación actual.

### Active Contours / Snakes
**Por qué se descartan:** Requieren inicialización próxima al borde real y son sensibles
al ruido en la imagen de gradientes. Si `SelectBorderLines` falla, el contorno inicial
estaría lejos del borde real, impidiendo la convergencia. La restricción de convexidad
necesaria para un cuadrilátero no es nativa del modelo de snake estándar.

### Watershed
**Por qué se descarta:** Requiere marcadores adecuados (foreground, background) que son
difíciles de calcular automáticamente en imágenes donde el fondo puede incluir páginas
adyacentes del mismo color que el papel principal. Sin marcadores correctos, la
segmentación produce resultados impredecibles.

### Transformers para rectificación (DocTr, DocScanner)
**Por qué se descartan:** Orientados a dewarping (corrección de curvatura), no a detección
de bordes en sentido estricto. El corpus Heraldo no presenta páginas curvadas sino planas
con objetos externos en los márgenes. El lomo/pliegue es un problema secundario que no
justifica la complejidad de un modelo de rectificación completo.

### DewarpNet / DocUNet
**Por qué se descartan:** Igual que los transformers de rectificación. Además, los modelos
pre-entrenados están entrenados con imágenes de documentos modernos fotografiados con
smartphone, no con fondos de fotografía de archivo como el corpus Heraldo. El transfer
learning requeriría re-entrenamiento con el corpus propio.

### Programación basada en nodos (Dear PyGui, Ryven, NodeGraphQt)
**Por qué se descarta:** El documento `programacion_basada_en_nodos.md` cubre alternativas
de UI para construir el editor visual del pipeline, no técnicas de detección de bordes.
No es aplicable al problema técnico analizado en este informe.

---

## Conexiones con la implementación existente

### P1 (ProjectionProfileBorder) con la implementación actual

- Reutiliza la imagen `canny_image` ya producida por el pipeline (sin preprocesamiento nuevo).
- Produce `border_lines` en el mismo formato que `SelectBorderLines`, por lo que
  `CalculateQuadCorners` no necesita modificación.
- Puede recibir `proportion_data` de `FindPeakProportion` como restricción de desambiguación
  cuando hay múltiples valles plausibles en el perfil, creando sinergia con la Estrategia 3.
- El paper de referencia (Shamqoli) está disponible en `docs/Info_relacionada/`.

### P2 (FilterLinesBySegmentDensity) con la implementación actual

- La lógica de muestreo de píxeles a lo largo de líneas ya está implementada en
  `RefinePolygonByCanny` (módulo `_polygon_geometry.py`). P2 reutiliza ese código base,
  añadiendo únicamente el análisis de "runs" continuas.
- Los outputs de P2 (`horizontal_lines`, `vertical_lines`, `lines_metadata`) son
  exactamente los inputs que `RefinePolygonByArea` y `RefinePolygonByCanny` ya esperan:
  el filtro se inserta en el pipeline sin modificar los filtros downstream.

### P3 (ScorePolygonByContrast) con la implementación actual

- Consume `selection_metadata` de los filtros de refinamiento existentes, que ya exponen
  la lista de candidatos top-K con sus geometrías. Si `RefinePolygonByArea` actualmente
  no serializa las geometrías completas de los candidatos en su metadata, sería necesaria
  una modificación menor para exponerlas.
- El candidato final de P3 sigue siendo compatible con `CalculateQuadCorners` porque
  produce `border_lines` en el mismo formato.

### P4 (LSDLineDetector) con la implementación actual

- LSD produce segmentos (x1,y1,x2,y2); los filtros actuales trabajan con (rho,theta) de
  Hough. Requiere una capa de conversión o modificar `_polygon_geometry.py` para aceptar
  ambos formatos como posición característica de la línea.
- Si se implementa como enriquecimiento del pool Hough (fusionar ambos), la compatibilidad
  es inmediata porque `RefinePolygonByArea` ya acepta cualquier conjunto de líneas con
  posición característica calculada.

### P5 (MorphologicalPreprocessForEdges) con la implementación actual

- Antes de implementar P5 como filtro nuevo, verificar si encadenar dos instancias del
  filtro `MorphologyFilter` existente (una con MORPH_CLOSE, otra con MORPH_OPEN, en dos
  orientaciones) produce el resultado equivalente. Si los parámetros del filtro existente
  permiten kernels asimétricos (ej. 15×1), la propuesta puede implementarse sin código nuevo,
  solo con configuración.

---

## Orden de experimentación sugerido

| Orden | Propuesta | Justificación |
|-------|-----------|---------------|
| 1 | P1: ProjectionProfileBorder | Ataca el caso de páginas adyacentes (problema más frecuente); ya identificada como Estrategia 2 pendiente |
| 2 | P2: FilterLinesBySegmentDensity | Bajo costo, reutiliza código existente; mejora la entrada a filtros ya implementados |
| 3 | P3: ScorePolygonByContrast | Complemento de desempate; solo vale si los top-K de refinamiento ya son buenos |
| 4 | P5: MorphologicalPreprocess | Verificar primero con filtros morfológicos existentes; implementar solo si el experimento lo justifica |
| 5 | P4: LSDLineDetector | Mayor costo de integración; considerar si P1 + P2 no alcanzan el IoU objetivo |

---

*Documento generado en Marzo 2026 para el proyecto Copista-Pipeline.*
*Solo propuestas fundamentadas en los documentos fuente listados al inicio.*
