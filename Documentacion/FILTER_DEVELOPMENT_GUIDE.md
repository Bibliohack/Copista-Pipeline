# Guía Técnica: Creación de Filtros para filter_library.py

## Resumen

Este documento explica cómo crear nuevos filtros para el sistema de procesamiento de imágenes. Cada filtro es una clase Python que hereda de `BaseFilter` y se registra automáticamente en el sistema.

---

## Arquitectura del Sistema

### Flujo de datos

```
Imagen Original
      ↓
  [Filtro resize] → outputs: {resized_image: img, sample_image: img}
      ↓
  [Filtro blur] → recibe inputs de filtros anteriores → produce outputs
      ↓
  [Filtro canny] → ...
      ↓
  Resultado Final
```

### Archivos involucrados

| Archivo | Función |
|---------|---------|
| `filter_library/` | Contiene todas las clases de filtros |
| `pipeline.json` | Define qué filtros usar y cómo conectarlos |
| `params.json` | Almacena los valores de parámetros configurados |
| `param_configurator.py` | GUI para ajustar parámetros y visualizar resultados |

---

## Estructura de un Filtro

### Plantilla básica

```python
class MiFiltro(BaseFilter):
    """Descripción breve del filtro"""
    
    # === ATRIBUTOS DE CLASE OBLIGATORIOS ===
    
    FILTER_NAME = "MiFiltro"  # Identificador único, usado en pipeline.json
    DESCRIPTION = "Descripción detallada de lo que hace el filtro"
    
    INPUTS = {
        # Entradas que este filtro necesita de filtros anteriores
        # Formato: "nombre_input": "tipo_dato"
        "input_image": "image"
    }
    
    OUTPUTS = {
        # Salidas que este filtro produce
        # Formato: "nombre_output": "tipo_dato"
        # OBLIGATORIO: siempre debe incluir "sample_image": "image"
        "mi_resultado": "image",
        "sample_image": "image"  # OBLIGATORIO para visualización
    }
    
    PARAMS = {
        # Parámetros ajustables por el usuario
        # Cada parámetro es un dict con: default, min, max, step, description
        "mi_parametro": {
            "default": 50,
            "min": 0,
            "max": 100,
            "step": 5,
            "description": "Descripción del parámetro para el usuario"
        }
    }
    
    # === MÉTODO OBLIGATORIO ===
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        """
        Procesa el filtro.
        
        Args:
            inputs: Dict con datos de filtros anteriores (según INPUTS definidos)
            original_image: Imagen original sin procesar (siempre disponible)
        
        Returns:
            Dict con todos los outputs definidos en OUTPUTS
        """
        # Obtener imagen de entrada
        img = inputs.get("input_image", original_image)
        
        # Obtener parámetros
        param_valor = self.params["mi_parametro"]
        
        # Procesar...
        resultado = self._hacer_algo(img, param_valor)
        
        # Retornar TODOS los outputs declarados en OUTPUTS
        return {
            "mi_resultado": resultado,
            "sample_image": resultado  # Para visualización
        }
```

---

[... resto del contenido del FILTER_DEVELOPMENT_GUIDE original, pero con estos cambios ...]

## Uso en pipeline.json

Una vez creado el filtro, se usa en `pipeline.json` así:

```json
{
    "filters": {
        "resize": {
            "filter_name": "Resize",
            "inputs": {}
        },
        "mi_filtro": {
            "filter_name": "MiFiltro",
            "inputs": {
                "input_image": "resize.resized_image"
            }
        },
        "otro_filtro": {
            "filter_name": "OtroFiltro",
            "inputs": {
                "input_image": "mi_filtro.mi_resultado"
            }
        }
    }
}
```

**Formato de referencia:** `"filter_id.nombre_output"`

### Características del sistema de IDs:

- **IDs semánticos**: Usa nombres descriptivos como `"resize"`, `"blur"`, `"canny"`
- **Orden implícito**: El orden en el JSON define el orden de ejecución
- **Inserción fácil**: Agregar un filtro entre otros es trivial

### Ejemplo: Insertar un filtro

```json
{
    "filters": {
        "resize": {...},
        "grayscale": {...},
        "denoise": {  // ← NUEVO - Solo lo insertas aquí
            "filter_name": "DenoiseNLMeans",
            "inputs": {"input_image": "grayscale.grayscale_image"}
        },
        "blur": {
            "inputs": {"input_image": "denoise.denoised_image"}  // ← Solo cambias esto
        }
    }
}
```

El filtro `denoise` se ejecutará entre `grayscale` y `blur` automáticamente.

---

## Checklist para Nuevo Filtro

- [ ] La clase hereda de `BaseFilter`
- [ ] `FILTER_NAME` es único y en PascalCase
- [ ] `DESCRIPTION` describe claramente la función
- [ ] `INPUTS` lista todas las entradas necesarias
- [ ] `OUTPUTS` incluye `"sample_image": "image"`
- [ ] `PARAMS` tiene `default`, `min`, `max`, `step`, `description` para cada parámetro
- [ ] `process()` retorna **todos** los outputs definidos
- [ ] `process()` maneja el caso donde `inputs` puede estar vacío
- [ ] Si produce solo imágenes, es compatible con checkpoint
- [ ] Los valores de parámetros se validan/corrigen si es necesario (ej: kernel impar)

---

## Registro Automático

**No es necesario registrar manualmente el filtro.** La clase `BaseFilter` usa `__init_subclass__` para registrar automáticamente cada subclase:

```python
class BaseFilter(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'FILTER_NAME') and cls.FILTER_NAME != "base":
            FILTER_REGISTRY[cls.FILTER_NAME] = cls
```

Simplemente define la clase en `filter_library/` y estará disponible.
