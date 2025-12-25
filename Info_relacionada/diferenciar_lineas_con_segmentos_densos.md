[Claude.ai dic/2025]

Cómo diferenciar entre **líneas con segmentos densos localmente** (bordes de página/imagen) vs **líneas con puntos distribuidos uniformemente pero espaciados** (líneas de texto).

## Soluciones posibles

### 1. **Transformada de Hough Probabilística con post-filtrado por segmentos**

La idea es usar `HoughLinesP` y luego analizar la **estructura interna** de cada línea detectada:

```python
import cv2
import numpy as np

def detect_edge_lines(edges, min_segment_length=50, min_density_ratio=0.6):
    """
    Detecta líneas que tienen segmentos densos (bordes) 
    vs líneas dispersas (texto)
    """
    # Hough probabilístico con parámetros permisivos
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=100,
        maxLineGap=30  # Permitimos gaps grandes inicialmente
    )
    
    if lines is None:
        return []
    
    valid_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Extraer los píxeles a lo largo de la línea
        num_points = max(abs(x2-x1), abs(y2-y1))
        if num_points == 0:
            continue
            
        xs = np.linspace(x1, x2, num_points).astype(int)
        ys = np.linspace(y1, y2, num_points).astype(int)
        
        # Obtener valores de borde a lo largo de la línea
        line_values = edges[ys, xs]
        
        # Analizar segmentos continuos
        if has_dense_segments(line_values, min_segment_length, min_density_ratio):
            valid_lines.append(line)
    
    return valid_lines


def has_dense_segments(line_values, min_segment_length, min_density_ratio):
    """
    Verifica si la línea tiene al menos un segmento 
    con alta densidad de puntos continuos
    """
    binary = (line_values > 0).astype(int)
    
    # Encontrar runs de 1s (segmentos continuos)
    segments = []
    current_run = 0
    
    for val in binary:
        if val == 1:
            current_run += 1
        else:
            if current_run > 0:
                segments.append(current_run)
            current_run = 0
    if current_run > 0:
        segments.append(current_run)
    
    # Criterio: existe al menos un segmento largo
    if segments:
        max_segment = max(segments)
        if max_segment >= min_segment_length:
            return True
    
    # Alternativa: la suma de segmentos "buenos" es significativa
    good_segments = [s for s in segments if s >= min_segment_length // 3]
    if sum(good_segments) / len(line_values) >= min_density_ratio:
        return True
    
    return False
```

### 2. **Análisis de "Local Density" sobre líneas Hough estándar**

Usar Hough estándar y luego proyectar píxeles sobre cada línea candidata, analizando la distribución:

```python
def analyze_line_density(edges, rho, theta, window_size=20, density_threshold=0.7):
    """
    Para una línea (rho, theta), analiza si tiene 
    ventanas con alta densidad local
    """
    h, w = edges.shape
    
    # Generar puntos sobre la línea
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    points = []
    for t in range(-max(h, w), max(h, w)):
        x = int(rho * cos_t - t * sin_t)
        y = int(rho * sin_t + t * cos_t)
        if 0 <= x < w and 0 <= y < h:
            points.append((x, y, edges[y, x]))
    
    if len(points) < window_size:
        return False
    
    # Sliding window para encontrar segmentos densos
    values = np.array([p[2] for p in points])
    
    max_density = 0
    for i in range(len(values) - window_size):
        window = values[i:i+window_size]
        density = np.sum(window > 0) / window_size
        max_density = max(max_density, density)
    
    return max_density >= density_threshold
```

### 3. **Morfología antes de Hough** (más simple pero efectivo)

Antes de aplicar Hough, usar operaciones morfológicas que favorezcan líneas continuas:

```python
def preprocess_for_page_edges(edges):
    """
    Pre-procesa para favorecer bordes continuos sobre texto
    """
    # Dilatar ligeramente para cerrar pequeños gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # horizontal
    closed_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # vertical
    closed_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    
    # Combinar
    closed = cv2.bitwise_or(closed_h, closed_v)
    
    # Ahora eliminar elementos pequeños/dispersos
    # usando apertura con kernel más grande
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    opened_h = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    opened_v = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    return cv2.bitwise_or(opened_h, opened_v)
```

## Recomendación

Combinaría las aproximaciones:

1. **Morfología primero** para cerrar gaps pequeños en bordes reales
2. **Hough probabilístico** con `maxLineGap` moderado
3. **Post-filtrado por segmentos densos** para eliminar falsos positivos de texto

¿Querés que desarrolle alguna de estas soluciones más a fondo o que arme un ejemplo completo con imágenes de prueba?
