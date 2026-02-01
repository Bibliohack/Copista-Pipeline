# Referencia: Creación de Filtros para filter_library.py

## Plantilla

### Filtro que produce solo imágenes

```python
class MiFiltro(BaseFilter):
    FILTER_NAME = "MiFiltro"
    DESCRIPTION = "Qué hace el filtro"
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "resultado": "image",
        "sample_image": "image"  # OBLIGATORIO
    }
    
    PARAMS = {
        "mi_param": {
            "default": 50,
            "min": 0,
            "max": 100,
            "step": 5,
            "description": "Descripción del parámetro"
        }
    }
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        img = inputs.get("input_image", original_image)
        valor = self.params["mi_param"]
        
        # ... procesamiento con OpenCV ...
        
        return {
            "resultado": imagen_procesada,
            "sample_image": imagen_procesada
        }
```

### Filtro que puede omitir preview (para batch processing)

```python
def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
    img = inputs.get("input_image", original_image)
    
    # ... procesamiento de datos ...
    result = {
        "lines_data": lines_data,
        "lines_metadata": metadata
    }
    
    # Generar preview solo si es necesario
    if not self.without_preview:
        sample = self._create_visualization(...)
        result["sample_image"] = sample
    
    return result
```

### Filtro que produce datos con coordenadas

```python
class DetectLines(BaseFilter):
    FILTER_NAME = "DetectLines"
    DESCRIPTION = "Detecta líneas en la imagen"
    
    INPUTS = {
        "input_image": "image"
    }
    
    OUTPUTS = {
        "lines_data": "lines",
        "lines_metadata": "metadata",  # ✅ OBLIGATORIO para datos con coordenadas
        "sample_image": "image"
    }
    
    PARAMS = {...}
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        img = inputs.get("input_image", original_image)
        h, w = img.shape[:2]
        
        # ... detectar líneas ...
        lines_data = [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}, ...]
        
        # ✅ IMPORTANTE: Crear metadata con dimensiones
        metadata = {
            "image_width": int(w),
            "image_height": int(h),
            "total_lines": len(lines_data)
        }
        
        return {
            "lines_data": lines_data,
            "lines_metadata": metadata,  # ✅ Incluir metadata
            "sample_image": visualizacion
        }
```

## Reglas

1. **OUTPUTS siempre debe incluir `"sample_image": "image"`**
   - El filtro puede omitir la generación de sample_image si `self.without_preview == True`
   - Por defecto `without_preview = False` (compatible con param_configurator.py)

2. **Tipos de datos para INPUTS/OUTPUTS:**
   - `"image"` → numpy.ndarray (BGR o grayscale)
   - `"lines"`, `"contours"`, `"histogram"`, etc. → datos estructurados
   - `"metadata"` → diccionario con información contextual

3. **Si el filtro produce datos con coordenadas o métricas**, debe incluir un output de metadata:
   ```python
   OUTPUTS = {
       "lines_data": "lines",         # Datos con coordenadas
       "lines_metadata": "metadata",  # ✅ OBLIGATORIO: dimensiones de imagen
       "sample_image": "image"        # Visualización
   }
   
   # El metadata debe contener AL MENOS:
   metadata = {
       "image_width": int(w),
       "image_height": int(h),
       # ... otros datos específicos del filtro
   }
   ```
   
   **¿Por qué?** Permite escalar coordenadas entre resoluciones, validar límites y contextualizar métricas.

4. **Convención de nombres para metadata:**
   - Líneas: `"lines_metadata"`
   - Contornos: `"contours_metadata"`
   - Esquinas/Puntos: `"corners_metadata"` o `"points_metadata"`
   - Claves sin prefijo `_`: `"image_width"` (no `"_image_width"`)

5. **Restricción de cache:** Solo filtros con todos los outputs tipo `"image"` pueden estar antes de un checkpoint

6. **El filtro se registra automáticamente** al definir la clase

## Uso en pipeline.json

```json
{
    "mi_filtro": {
        "filter_name": "MiFiltro",
        "inputs": {
            "input_image": "grayscale.grayscale_image"
        }
    }
}
```

**Formato de referencia:** `"filter_id.nombre_output"`

Ejemplo completo con metadata:

```json
{
    "filters": {
        "resize": {
            "filter_name": "Resize",
            "inputs": {}
        },
        "canny": {
            "filter_name": "CannyEdge",
            "inputs": {
                "input_image": "resize.resized_image"
            }
        },
        "hough": {
            "filter_name": "HoughLines",
            "description": "Detecta líneas - produce lines_data y lines_metadata",
            "inputs": {
                "edge_image": "canny.edge_image",
                "base_image": "resize.resized_image"
            }
        },
        "scale_lines": {
            "filter_name": "ScaleCoordinates",
            "description": "Escala líneas a resolución original usando metadata",
            "inputs": {
                "lines_data": "hough.lines_data",
                "lines_metadata": "hough.lines_metadata"
            }
        }
    }
}
```

El orden de los filtros en el JSON define el orden de ejecución.
