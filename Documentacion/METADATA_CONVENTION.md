# Convención de Metadata para Filtros

## 📋 Regla Simple

**Si tu filtro produce datos con coordenadas absolutas (x, y) o métricas en píxeles:**
→ **DEBE incluir un output `*_metadata` con las dimensiones de la imagen**

## ✅ ¿Cuándo usar metadata?

| Tipo de Output | ¿Necesita metadata? | Ejemplo |
|----------------|---------------------|---------|
| Líneas con coordenadas | ✅ SÍ | `{"x1": 10, "y1": 20, "x2": 100, "y2": 200}` |
| Contornos con puntos | ✅ SÍ | `{"points": [[10,20], [30,40]], "area": 500}` |
| Esquinas/Puntos | ✅ SÍ | `{"top_left": {"x": 10, "y": 5}}` |
| Bounding boxes | ✅ SÍ | `{"x": 50, "y": 60, "width": 100, "height": 80}` |
| Solo imágenes | ❌ NO | `{"output_image": np.ndarray}` |
| Datos sin coordenadas | ❌ NO | `{"color": "red", "count": 42}` |

## 📝 Formato Requerido

### Outputs

```python
OUTPUTS = {
    "lines_data": "lines",          # Tus datos
    "lines_metadata": "metadata",   # ✅ OBLIGATORIO
    "sample_image": "image"
}
```

### Metadata Mínima

```python
metadata = {
    "image_width": int(w),    # ✅ OBLIGATORIO
    "image_height": int(h),   # ✅ OBLIGATORIO
    # ... otros datos opcionales ...
}
```

## 🎯 Convención de Nombres

| Tipo de Filtro | Nombre Output | Metadata |
|----------------|---------------|----------|
| Detección de líneas | `lines_data` | `lines_metadata` |
| Detección de contornos | `contours_data` | `contours_metadata` |
| Detección de esquinas | `corners` | `corners_metadata` |
| Detección de puntos | `points_data` | `points_metadata` |

**Patrón:** `{tipo}_metadata`

## ❌ Errores Comunes

### Error 1: Metadata mezclada con datos

```python
# ❌ MAL
return {
    "corners": {
        "top_left": {"x": 10, "y": 5},
        "_image_width": 640,  # ← Mezclado con datos
        "_image_height": 480
    }
}

# ✅ BIEN
return {
    "corners": {
        "top_left": {"x": 10, "y": 5}
    },
    "corners_metadata": {
        "image_width": 640,
        "image_height": 480
    }
}
```

### Error 2: Usar prefijo `_`

```python
# ❌ MAL
metadata = {
    "_image_width": 640,
    "_image_height": 480
}

# ✅ BIEN
metadata = {
    "image_width": 640,
    "image_height": 480
}
```

### Error 3: No incluir metadata

```python
# ❌ MAL - Filtro que detecta líneas sin metadata
OUTPUTS = {
    "lines_data": "lines",
    "sample_image": "image"
}

# ✅ BIEN
OUTPUTS = {
    "lines_data": "lines",
    "lines_metadata": "metadata",  # ← Incluir
    "sample_image": "image"
}
```

## 💡 Ejemplo Completo

```python
class HoughLinesFilter(BaseFilter):
    FILTER_NAME = "HoughLines"
    
    INPUTS = {
        "edge_image": "image",
        "base_image": "image"
    }
    
    OUTPUTS = {
        "lines_data": "lines",
        "lines_metadata": "metadata",  # ✅ Metadata obligatoria
        "sample_image": "image"
    }
    
    def process(self, inputs, original_image):
        edge_img = inputs.get("edge_image")
        base_img = inputs.get("base_image", original_image)
        
        h, w = base_img.shape[:2]  # ✅ Obtener dimensiones
        
        # Detectar líneas...
        lines = cv2.HoughLinesP(edge_img, ...)
        lines_data = [{"x1": x1, "y1": y1, "x2": x2, "y2": y2} for ...]
        
        # ✅ Crear metadata
        metadata = {
            "image_width": int(w),
            "image_height": int(h),
            "total_lines": len(lines_data),
            "method": "probabilistic"
        }
        
        return {
            "lines_data": lines_data,
            "lines_metadata": metadata,  # ✅ Retornar metadata
            "sample_image": visualization
        }
```

## 🔧 Casos de Uso

### Escalar coordenadas

```python
# Filtro que escala líneas de imagen pequeña a grande
class ScaleLines(BaseFilter):
    def process(self, inputs, original_image):
        lines = inputs.get("lines_data")
        metadata = inputs.get("lines_metadata")
        
        # ✅ Usar metadata para escalar
        scale_x = original_image.shape[1] / metadata["image_width"]
        scale_y = original_image.shape[0] / metadata["image_height"]
        
        scaled_lines = []
        for line in lines:
            scaled_lines.append({
                "x1": int(line["x1"] * scale_x),
                "y1": int(line["y1"] * scale_y),
                "x2": int(line["x2"] * scale_x),
                "y2": int(line["y2"] * scale_y)
            })
        
        return {"scaled_lines": scaled_lines, ...}
```

### Validar coordenadas

```python
corners = inputs.get("corners")
metadata = inputs.get("corners_metadata")

# ✅ Validar que estén dentro de límites
for name, corner in corners.items():
    x, y = corner["x"], corner["y"]
    if not (0 <= x < metadata["image_width"]):
        raise ValueError(f"{name} x={x} fuera de límites")
    if not (0 <= y < metadata["image_height"]):
        raise ValueError(f"{name} y={y} fuera de límites")
```

### Contextualizar métricas

```python
contours = inputs.get("contours_data")
metadata = inputs.get("contours_metadata")

total_area = sum(c["area"] for c in contours)
image_area = metadata["image_width"] * metadata["image_height"]
coverage = (total_area / image_area) * 100

print(f"Los contornos cubren {coverage:.1f}% de la imagen")
print(f"Imagen de referencia: {metadata['image_width']}x{metadata['image_height']}")
```

## 🎓 Resumen

1. **Detectas líneas/contornos/puntos** → Incluye `*_metadata`
2. **Metadata mínima:** `image_width`, `image_height`
3. **Sin prefijo `_`:** `image_width` (no `_image_width`)
4. **Output separado:** No mezclar con datos
5. **Convención de nombres:** `{tipo}_metadata`

Esto permite que otros filtros escalen, validen y contextualicen tus datos correctamente.
