# Normalización de Iluminación No Uniforme

**Fecha de inicio:** Abril 2026
**Contexto:** Las capturas fotográficas del Heraldo presentan gradientes de iluminación causados por la fuente de luz, la curvatura del papel y el entorno. Al aplicar contraste, esta no-uniformidad se amplifica, degradando la calidad de imagen para OCR y visualización.

---

## Problema

Las imágenes capturadas muestran variaciones suaves de iluminación (gradientes de esquina a esquina, zonas más oscuras en los bordes, reflexiones locales). Al normalizar el contraste globalmente, estas variaciones se hacen más notorias en lugar de corregirse.

**Modelo físico:**
`I(x,y) = R(x,y) × L(x,y)`
Donde `R` = reflectancia del papel/texto (lo que queremos) y `L` = iluminación no uniforme (lo que queremos eliminar). El objetivo es recuperar `R`.

**Requisito de salida:** la imagen resultante debe mantener la tonalidad de color del fondo (tono cálido del papel). Por eso se trabaja preferentemente en canal L del espacio LAB (`output_channel=0`), que solo modifica luminancia y deja intactos los canales de color.

---

## Técnicas implementadas

### 1. BackgroundNormalization (`background_normalization.py`) ✓ MEJOR RESULTADO

Estima el fondo de iluminación y lo combina con la imagen original mediante distintos modos de mezcla.

**Parámetros:**

| Parámetro | Rango | Descripción |
|---|---|---|
| `blur_radius` | 10–500 | Radio del Gaussian final (suaviza el fondo estimado) |
| `blend_mode` | 0–9 | Modo de combinación imagen/fondo |
| `strength` | 0–100 | Intensidad de corrección (0=original, 100=completo) |
| `gamma` | 1–30 | Gamma para modo `gamma_divide` (÷10) |
| `output_channel` | 0–2 | 0=canal L (LAB), 1=gris, 2=BGR |
| `background_method` | 0–2 | Método de estimación del fondo (ver abajo) |
| `morph_radius` | 5–300 | Radio del kernel morfológico (solo métodos 1 y 2) |
| `show_comparison` | 0–1 | 0=solo resultado, 1=comparación lado a lado |

**Métodos de estimación del fondo (`background_method`):**

| Método | Algoritmo | Cuándo usar |
|---|---|---|
| 0 | Gaussiano puro | Rápido. El texto contamina levemente el estimado |
| 1 | **Cierre morfológico** (dilata→erosiona) + Gaussian | Más preciso: dilata rellena el texto oscuro con fondo claro, erosiona elimina artefactos. Equivalente al algoritmo *rolling ball* |
| 2 | Dilatación pura + Gaussian | Más agresivo que el cierre, sin erosión final |

> **Nota importante:** la operación correcta para fondo claro / texto oscuro es el **cierre** (dilatación→erosión), no la apertura (erosión→dilatación). La dilatación expande el fondo claro sobre el texto, luego la erosión limpia los artefactos. `morph_radius` debe ser mayor que el carácter más grande del texto.

**10 modos de mezcla (`blend_mode`):**

| Modo | Fórmula | Característica |
|---|---|---|
| 0 subtract | `I - B + 0.5` | Corrección aditiva |
| 1 divide | `I / B × mean(B)` | Corrección multiplicativa (modelo físico correcto) |
| 2 retinex | `exp(log(I) - log(B))` | Estable en zonas oscuras |
| 3 gamma_divide | `(I/B)^γ` | Divide con control de intensidad |
| 4 overlay | Photoshop overlay | Preserva textura |
| 5 soft_light | Fórmula Pegtop | Versión suave del overlay |
| 6 hard_light | Overlay con I/B invertidos | Más contraste |
| 7 vivid_light | burn/dodge según B | Máximo contraste local |
| 8 linear_light | `I + 2B - 1` | Balance lineal |
| 9 exclusion | `I + B - 2IB` | Efecto suave |

---

### 2. CLAHEFilter (`clahe_filter.py`)
Ecualización adaptativa de histograma con límite de contraste. Opera en bloques locales del canal L.

**Resultado en comparación:** inferior a BackgroundNormalization para este corpus. Puede complementarlo en una etapa posterior.

**Parámetros clave:** `clip_limit` (1–100, ×0.1), `tile_size` (2–32), `strength` (0–100), `show_comparison` (0–1)

---

### 3. RetinexFilter (`retinex_filter.py`)
Algoritmo Retinex de Land. SSR (Single Scale) y MSR (Multi Scale, promedio de 3 escalas).

**Resultado en comparación:** inferior a BackgroundNormalization para este corpus.

**Parámetros clave:** `method` (0=SSR, 1=MSR), `sigma` (10–300), `dynamic_range` (10–200), `strength` (0–100), `show_comparison` (0–1)

---

### 4. HomomorphicFilter (`homomorphic_filter.py`)
Filtrado homomórfico en dominio frecuencial. Filtro Butterworth orden 2.

`log(I) → FFT → H_Butterworth(γ_low, γ_high) → IFFT → exp`

**Resultado en comparación:** inferior a BackgroundNormalization para este corpus. Puede ser útil cuando el gradiente es muy pronunciado y hay zonas muy oscuras.

**Parámetros clave:** `cutoff_frequency` (1–100%), `gamma_low` (×0.1), `gamma_high` (×0.1), `strength` (0–100), `show_comparison` (0–1)

---

## Experimentos

### 001 — Comparación de técnicas (`__data/normalizacion/001_comparacion_tecnicas/`)
Pipeline de comparación: los 4 filtros toman el mismo input como alternativas independientes.

**Conclusión:** `BackgroundNormalization` da el mejor resultado visual. Mantiene la tonalidad cálida del fondo del papel y corrige el gradiente de forma natural.

### 002 — Pipeline de producción (`__data/normalizacion/002_background_norm_pipeline/`)
Pipeline de normalización completo basado en BackgroundNormalization:

```
resize → background_norm → brightness_contrast → histogram_peaks → normalize_levels
```

- `background_norm` — corrige el gradiente de iluminación
- `brightness_contrast` — ajuste global fino
- `histogram_peaks` — detecta picos oscuro/claro del histograma
- `normalize_levels` — ajusta punto negro y blanco según los picos detectados

**Punto de partida recomendado:** `blend_mode=1` (divide), `output_channel=0` (LAB), `background_method=1` (cierre morfológico).

---

## Notas de integración en pipeline de producción

El filtro de normalización debe aplicarse **antes** de cualquier detección de bordes o umbralización:

```
Resize → [BackgroundNormalization] → BrightnessContrast → NormalizeFromHistogram → Grayscale → Blur → Canny → ...
```
