# Referencia: Creación de Filtros para filter_library.py

## Plantilla

```python
class MiFiltro(BaseFilter):
    FILTER_NAME = "MiFiltro"                    # Identificador único (PascalCase)
    DESCRIPTION = "Qué hace el filtro"
    
    INPUTS = {
        "input_image": "image"                  # Entradas de otros filtros. {} si usa imagen original
    }
    
    OUTPUTS = {
        "resultado": "image",
        "sample_image": "image"                 # OBLIGATORIO: imagen para visualización
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
            "sample_image": imagen_procesada    # Lo que se muestra en el visualizador
        }
```

## Reglas

1. **OUTPUTS siempre debe incluir `"sample_image": "image"`** - Es lo que muestra el visualizador

2. **Tipos de datos para INPUTS/OUTPUTS:**
   - `"image"` → numpy.ndarray (BGR o grayscale)
   - `"lines"`, `"contours"`, `"histogram"`, etc. → datos estructurados

3. **Si el filtro produce datos (no imagen)**, sample_image debe ser una representación visual:
   ```python
   OUTPUTS = {
       "lines_data": "lines",      # Datos
       "sample_image": "image"     # Visualización de los datos
   }
   ```

4. **Restricción de cache:** Solo filtros con todos los outputs tipo `"image"` pueden estar antes de un checkpoint

5. **El filtro se registra automáticamente** al definir la clase

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

Ejemplo completo:

```json
{
    "filters": {
        "resize": {
            "filter_name": "Resize",
            "inputs": {}
        },
        "grayscale": {
            "filter_name": "Grayscale",
            "inputs": {
                "input_image": "resize.resized_image"
            }
        },
        "mi_filtro": {
            "filter_name": "MiFiltro",
            "inputs": {
                "input_image": "grayscale.grayscale_image"
            }
        }
    }
}
```

El orden de los filtros en el JSON define el orden de ejecución.
