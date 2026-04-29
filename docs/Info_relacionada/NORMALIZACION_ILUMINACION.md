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
| 10 invert_soft_light | `B_inv=1-B; (1-2·B_inv)·I²+2·B_inv·I` | Soft light con el fondo invertido. Aclara zonas oscuras más agresivamente que el soft_light estándar |

**Parámetro `blend_mode` — rango:** 0–9 inicialmente; se añadió el modo 10 (`invert_soft_light`).

---

### 1b. AutoLevels (`auto_levels_filter.py`)

Complemento de BackgroundNormalization. Estira el histograma del resultado al rango completo
(equivalente al "Auto Levels" de Photoshop).

**Proceso:**
1. Recorta percentiles bajo/alto del histograma (elimina outliers de píxeles)
2. Estira el rango restante a `[0, 255]`
3. Aplica corrección de gamma si el punto medio difiere de 128: `γ = log(0.5) / log(midpoint/255)`

**Parámetros:**

| Parámetro | Rango | Descripción |
|---|---|---|
| `clip_low` | 0–5 (×0.1%) | Percentil de recorte bajo (elimina negros extremos) |
| `clip_high` | 0–5 (×0.1%) | Percentil de recorte alto (elimina blancos extremos) |
| `midpoint` | 1–254 | Punto medio del histograma (128 = neutro, <128 = aclara, >128 = oscurece) |
| `output_channel` | 0–2 | 0=canal L, 1=gris, 2=BGR |
| `show_comparison` | 0–1 | Comparación lado a lado |

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
background_norm → [brightness_contrast] → [histogram_peaks] → [normalize_levels] → mask_page
```

Los filtros en `[corchetes]` están deshabilitados (`"enabled": 0`) en producción.
El pipeline de producción usa efectivamente sólo BackgroundNormalization → MaskOutsidePolygon.

**Parámetros de producción (2026-04-24):**

| Parámetro | Valor | Notas |
|---|---|---|
| `blend_mode` | 3 (gamma_divide) | `(I/B)^γ` — buena relación corrección/tonalidad |
| `background_method` | 0 (Gaussiano puro) | Más rápido; cierre morfológico (method=1) es más preciso pero más lento |
| `blur_radius` | 130 | Suavizado del fondo estimado |
| `strength` | 95 | Corrección casi completa |
| `gamma` | 25 | Exponente efectivo: 25/10 = 2.5 |
| `output_channel` | 0 (canal L LAB) | Conserva tonalidad cálida del papel |

**¿Por qué `gamma_divide` sobre `divide` (modo 1)?**
El modo 1 (`I/B × mean(B)`) implementa el modelo físico correcto, pero amplifica las zonas muy oscuras cuando el fondo estimado es bajo. El modo 3 (`(I/B)^γ`) aplica un exponente que comprime esta amplificación, dando un resultado visualmente más equilibrado para el corpus Heraldo donde el papel tiene tonos muy variados.

**MaskOutsidePolygon** (último paso): lee el `.crop.json` compañero de la imagen escalada
(paso 5) y pinta de blanco el área exterior al polígono con blur de borde `blur_radius=10`.
Requiere que `batch_processor` haya configurado `current_image_path`.

---

## Notas de integración en pipeline de producción

En el pipeline de producción (paso 6), la normalización se aplica **sobre la imagen escalada** (paso 5), no sobre el crop directo. El orden completo es:

```
paso4_recortado → paso5_escalado → [BackgroundNormalization → MaskOutsidePolygon] → paso6_normalizado
```

El OCR (paso 7) usa la imagen de paso5_escalado directamente, NO la normalizada en color,
porque la binarización adaptativa (Sauvola) aprovecha mejor la imagen sin la corrección de color.
