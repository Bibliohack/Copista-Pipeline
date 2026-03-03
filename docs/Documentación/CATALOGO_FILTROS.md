# Catálogo de Filtros

Lista completa de los 42 filtros registrados en `FILTER_REGISTRY`, organizados por categoría funcional.

Para instrucciones sobre cómo crear nuevos filtros, ver `FILTER_DEVELOPMENT_GUIDE.md`.
Para detalles sobre filtros específicos, ver los archivos `.py` correspondientes en `src/filter_library/`.

---

## Procesamiento de imagen

Filtros que transforman directamente el contenido visual de la imagen.

| FILTER_NAME | Archivo | Descripción |
|-------------|---------|-------------|
| `Resize` | `resize_filter.py` | Redimensiona la imagen por porcentaje, dimensiones fijas o ancho proporcional. Tres modos: `percent`, `fixed`, `proportional`. |
| `Grayscale` | `grayscale_filter.py` | Convierte la imagen a escala de grises. Dos métodos: OpenCV estándar o pesos RGB personalizables. |
| `GaussianBlur` | `gaussian_blur_filter.py` | Aplica desenfoque gaussiano. El kernel_size se fuerza a impar automáticamente. |
| `CannyEdge` | `canny_edge_filter.py` | Detecta bordes con el algoritmo de Canny. Parámetros: threshold1, threshold2, aperture_size, l2_gradient. |
| `Threshold` | `threshold_filter.py` | Umbralización de imagen. Tres métodos: Binary, Otsu, Adaptive. |
| `ThresholdAdvanced` | `threshold_advanced.py` | Umbralización extendida. 8 métodos (0–7), incluyendo ADAPTIVE_MEAN y ADAPTIVE_GAUSSIAN. |
| `Morphology` | `morphology_filter.py` | Operaciones morfológicas básicas. 6 operaciones (erosión, dilatación, apertura, cierre, gradiente, tophat), 3 formas de kernel. |
| `MorphologyAdvanced` | `morphology_advanced.py` | Como `Morphology` pero añade BlackHat (operación 7) e `invert_output`. |
| `BrightnessContrast` | `brightness_contrast_filter.py` | Ajusta brillo y contraste (alpha × img + beta). |
| `ColorSpace` | `color_space_filter.py` | Convierte entre espacios de color (BGR, HSV, LAB, YCrCb) con extracción opcional de canal individual. |
| `HighPass` | `high_pass_filter.py` | Filtro de paso alto para realzar bordes. 4 métodos: Gaussian subtraction, Laplacian, Unsharp Mask, FFT. |
| `DenoiseNLMeans` | `denoise_nl_means.py` | Reducción de ruido con `fastNlMeansDenoising` preservando bordes. Compatible con grayscale y color. |

**Inputs típicos:** `input_image`
**Outputs típicos:** output de imagen + `sample_image`

---

## Detección y análisis de datos

Filtros que extraen información estadística o detectan elementos estructurales de la imagen.

| FILTER_NAME | Archivo | Descripción |
|-------------|---------|-------------|
| `Histogram` | `histogram_filter.py` | Calcula el histograma de la imagen. Produce `histogram_data` (por canal o intensidad) + visualización. |
| `HistogramVisualize` | `histogram_visualize.py` | Histograma con marcadores de picos oscuro/claro y auto-detección. Genera `histogram_data` con `dark_marker` y `light_marker`. |
| `DetectHistogramPeaks` | `detect_histogram_peaks.py` | Auto-detecta picos en el histograma: `dark_peak`, `min_between`, `light_peak`. Produce `histogram_data` compatible con `NormalizeFromHistogram`. |
| `NormalizePeaks` | `normalize_peaks.py` | Normaliza mapeando `[dark_peak, light_peak]` → `[dark_target, light_target]`. Modo `auto_detect` disponible. |
| `NormalizeFromHistogram` | `normalize_from_histogram.py` | Normaliza con mapeado de 3 puntos usando `dark_marker`/`light_marker` de `HistogramVisualize`. Modos: piecewise o lineal. |
| `HoughLines` | `hough_lines_filter.py` | Detecta líneas con la transformada de Hough. Implementa `without_preview`. Dos variantes: Standard y Probabilistic. Produce `lines_data` + `lines_metadata` con dimensiones. |
| `Contours` | `contour_filter.py` | Detecta contornos con metadata completa incluyendo `coverage_percent`. Implementa `without_preview`. |
| `ContourSimplify` | `contour_simplify.py` | Simplifica contornos con `approxPolyDP`. Guarda tanto `simplified_points` como `original_points`. |
| `MinArcLength` | `min_arc_length.py` | Filtra imagen de bordes eliminando contornos más cortos que `min_length`. |
| `OverlayLines` | `overlay_lines_filter.py` | Dibuja líneas detectadas sobre una imagen base. Compatible con HoughLinesP (`x1,y1,x2,y2`) y HoughLines (`rho,theta`). |

---

## Clasificación y análisis de líneas

Filtros especializados en procesar el pool de líneas Hough para extraer geometría de la página.

| FILTER_NAME | Archivo | Descripción |
|-------------|---------|-------------|
| `ClassifyLinesByAngle` | `classify_lines_by_angle.py` | Clasifica líneas Hough en `horizontal_lines`, `vertical_lines` y `other_lines`. Implementa `without_preview`. Compatible con HoughLinesP y HoughLines. Produce `lines_metadata` con dimensiones. |
| `FilterLinesByOrientation` | `filter_lines_by_orientation.py` | Como `ClassifyLinesByAngle` pero descarta las oblicuas (sin `other_lines`). Implementa `without_preview`. |
| `DetectPageSkew` | `detect_page_skew.py` | Detecta inclinación de página mediante clustering de líneas. Implementa `without_preview`. Valida perpendicularidad. |
| `CalculateRotationFromLines` | `calculate_rotation_from_lines.py` | Calcula ángulo de rotación de la página por clusters. Implementa `without_preview`. Tres modos: solo H, solo V, o ambos. |

---

## Transformaciones geométricas

Filtros que transforman la geometría de la imagen o de las coordenadas detectadas.

| FILTER_NAME | Archivo | Descripción |
|-------------|---------|-------------|
| `RotateImage` | `rotate_image.py` | Rota la imagen por ángulo dado. Implementa `without_preview`. Puede leer `rotation_angle` del pipeline o usar parámetro manual. Opción `invert_angle`. |
| `RotateImageOrtho` | `rotate_image_ortho.py` | Rota en múltiplos de 90° sin interpolación (cv2.rotate). Implementa `without_preview`. Opciones: 0°, 90° CCW, 180°, 270° CCW. |
| `RotateQuadCorners` | `rotate_quad_corners.py` | Rota coordenadas de esquinas alrededor del centro de imagen. Implementa `without_preview`. |
| `ScaleQuadCorners` | `scale_quad_corners.py` | Escala coordenadas de esquinas a nuevas dimensiones. Tres modos: proportional, stretch, fit. |
| `CalculateRectFromQuadCorners` | `calc_rect_from_quad_corners.py` | Calcula rectángulo de crop como bounding box de las 4 esquinas con inset opcional. Implementa `without_preview`. |
| `CropImage` | `crop_image.py` | Recorta la imagen usando `crop_rect` del pipeline o coordenadas manuales. Implementa `without_preview`. |

---

## Polígono y bordes de página

Filtros del flujo principal de detección del polígono de borde físico del papel.

| FILTER_NAME | Archivo | Descripción |
|-------------|---------|-------------|
| `SelectBorderLines` | `select_border_lines.py` | Selecciona las 4 líneas de borde (top/bottom/left/right) del pool clasificado usando las más extremas con clustering y márgenes. Fallback a borde de imagen. Implementa `without_preview`. Salida: `selected_lines` (formato `border_lines`). |
| `CalculateQuadCorners` | `calculate_quad_corners.py` | Calcula las 4 esquinas intersectando las líneas de borde. Implementa `without_preview`. Maneja tipos: `intersection`, `image_corner`, `mixed_h_border`, `mixed_v_border`. |
| `RefinePolygonByArea` | `refine_polygon_by_area.py` | Búsqueda exhaustiva sobre todas las líneas Hough para encontrar la combinación con proporción + área más cercana al objetivo del corpus. Reemplaza `SelectBorderLines`. Implementa `without_preview`. Ver `FILTROS_REFINAMIENTO_POLIGONO.md`. |
| `RefinePolygonByCanny` | `refine_polygon_by_canny.py` | Como `RefinePolygonByArea` pero con pre-filtrado por soporte Canny: descarta líneas que no coinciden con bordes reales antes de la búsqueda exhaustiva. Implementa `without_preview`. Ver `FILTROS_REFINAMIENTO_POLIGONO.md`. |
| `PolygonToGTFormat` | `polygon_to_gt_format.py` | Convierte las esquinas detectadas (coordenadas de resolución de trabajo) a formato `.gt.json` / `.det.json` con coordenadas de la imagen original. |

**Posición en el pipeline:**
```
HoughLines → ClassifyLinesByAngle → SelectBorderLines (o RefinePolygon*) → CalculateQuadCorners → ScaleQuadCorners → PolygonToGTFormat
```

---

## Proporción de corpus

Filtros para calibrar y usar la proporción física invariante de las páginas del corpus.

| FILTER_NAME | Archivo | Descripción |
|-------------|---------|-------------|
| `CalculatePolygonProportion` | `calculate_polygon_proportion.py` | Calcula la proporción `avg(izq, der) / superior` del polígono detectado en una imagen. Produce `proportion_data`. Ver `FILTROS_PROPORCION_CORPUS.md`. |
| `FindPeakProportion` | `find_peak_proportion.py` | Lee una **carpeta de JSONs** (no recibe inputs del pipeline), acumula proporciones del corpus, y determina la proporción más frecuente mediante histograma con suavizado gaussiano. Produce `peak_data`. Ver `FILTROS_PROPORCION_CORPUS.md`. |

**Nota:** `FindPeakProportion` no consume inputs del pipeline (`INPUTS = {}`). Se usa como script independiente o en postprocess de batch. Ver `FILTROS_PROPORCION_CORPUS.md` para la arquitectura completa.

---

## OCR y PDF

Filtros para reconocimiento óptico de caracteres y generación de PDF con capa de texto.

| FILTER_NAME | Archivo | Descripción |
|-------------|---------|-------------|
| `TesseractOCR` | `tesseract_ocr.py` | OCR con pytesseract. Genera salida hOCR con coordenadas de palabras. Implementa `without_preview`. Idiomas: `spa`, `eng`, `spa+eng`, `auto`. |
| `HOCRtoPDF` | `hocr_to_pdf.py` | Genera PDF con capa de texto invisible desde imagen + hOCR. Implementa `without_preview`. Requiere reportlab. |
| `ScaleHOCR` | `scale_hocr.py` | Escala las coordenadas bbox de un hOCR a nuevas dimensiones de imagen. Usa parser ET con fallback a regex. |

---

## Resumen por categoría

| Categoría | Cantidad |
|-----------|----------|
| Procesamiento de imagen | 12 |
| Detección y análisis de datos | 10 |
| Clasificación y análisis de líneas | 4 |
| Transformaciones geométricas | 6 |
| Polígono y bordes de página | 5 |
| Proporción de corpus | 2 |
| OCR y PDF | 3 |
| **Total** | **42** |

---

## Módulos internos (no filtros)

Módulos en `src/filter_library/` que comienzan con `_` y no se registran en `FILTER_REGISTRY`.

| Módulo | Descripción |
|--------|-------------|
| `_polygon_geometry.py` | Utilidades geométricas compartidas por `RefinePolygonByArea` y `RefinePolygonByCanny`. Funciones: `line_intersection`, `polygon_from_lines`, `polygon_proportion`, `polygon_area`, `add_positions`, `limit_lines`, `exhaustive_search`, `candidate_to_border_lines`, `extract_target_proportion`, `make_refinement_sample`. Ver `FILTROS_REFINAMIENTO_POLIGONO.md`. |
