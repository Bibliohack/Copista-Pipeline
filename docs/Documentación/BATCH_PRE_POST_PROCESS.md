# Batch Processor: Scripts de Pre y Postproceso

## Concepto

El `batch_processor.py` procesa imágenes una a una, ejecutando el pipeline sobre cada archivo. Sin embargo, algunos pasos del flujo de trabajo operan a nivel de **corpus completo**: análisis estadístico, cálculo de métricas, encadenamiento entre pipelines, etc. Estos pasos no encajan en el modelo por imagen.

La funcionalidad de **preprocess** y **postprocess** permite declarar scripts externos en `batch_config.json` que se ejecutan automáticamente **antes o después** del procesamiento batch principal.

---

## Configuración en `batch_config.json`

```json
{
    "source_folder": "...",
    "targets": [...],

    "preprocess": [
        {
            "script": "../../scripts/mi_script.py",
            "input":  "ruta/de/entrada",
            "output": "ruta/de/salida",
            "params": {"clave": "valor"}
        }
    ],

    "postprocess": [
        {
            "script": "../../scripts/find_peak_proportion.py",
            "input":  "./__data/experiments/001_baseline/results/Derecha",
            "output": "./__data/calibration",
            "params": {"bin_size": 0.01, "smoothing_sigma": 1.5}
        }
    ]
}
```

Los campos `preprocess` y `postprocess` son **opcionales**. Si no están presentes, el comportamiento es idéntico al anterior.

### Campos de cada script

| Campo    | Obligatorio | Descripción |
|----------|-------------|-------------|
| `script` | Sí          | Ruta al script Python. Si es relativa, se resuelve desde la carpeta del `batch_config.json`. |
| `input`  | No          | Carpeta de entrada pasada al script como `--input`. |
| `output` | No          | Carpeta de salida pasada al script como `--output`. |
| `params` | No          | Diccionario JSON de parámetros adicionales, pasado como `--params '<JSON>'`. |

### Múltiples scripts

Se pueden declarar varios scripts en cada fase. Se ejecutan **secuencialmente** en el orden del array:

```json
"postprocess": [
    { "script": "scripts/compute_iou_metrics.py",    "input": "...", "output": "..." },
    { "script": "scripts/find_peak_proportion.py",   "input": "...", "output": "..." },
    { "script": "scripts/generate_report.py",        "input": "...", "output": "..." }
]
```

---

## Comportamiento de errores

| Fase         | Si falla un script... |
|--------------|----------------------|
| `preprocess` | **Aborta** el proceso completo. No se procesan imágenes. |
| `postprocess` | Se reporta el error pero **no afecta** los resultados ya guardados. |

---

## Interfaz estándar de los scripts

Todos los scripts del catálogo siguen la misma interfaz de línea de comandos:

```bash
python script.py --input <carpeta_entrada> --output <carpeta_salida> [--params '<JSON>']
```

- `--input` y `--output` son rutas de carpeta (no de archivo individual).
- `--params` recibe un string JSON con parámetros opcionales específicos de cada script.
- El script debe salir con código `0` en éxito y código distinto de `0` en error.

---

## Scripts disponibles

### `scripts/find_peak_proportion.py`

Analiza una carpeta de archivos JSON con polígonos detectados (`.det.json`, `.gt.json`) o con proporciones pre-calculadas, construye un histograma y determina la **proporción de papel más común** del corpus.

**Caso de uso principal**: determinar la proporción verdadera del papel tras correr el pipeline de detección de polígonos sobre el corpus completo. Esta proporción se usa luego como restricción en el pipeline de recorte.

```json
{
    "script": "../../scripts/find_peak_proportion.py",
    "input":  "./__data/experiments/001_baseline/results/Derecha",
    "output": "./__data/calibration",
    "params": {
        "bin_size":        0.01,
        "smoothing_sigma": 1.5,
        "file_pattern":    "*.json"
    }
}
```

**Parámetros**:

| Parámetro         | Default   | Descripción |
|-------------------|-----------|-------------|
| `bin_size`        | `0.01`    | Tamaño de bin del histograma (unidades de proporción). |
| `smoothing_sigma` | `1.5`     | Sigma del suavizado gaussiano antes de buscar el pico (`0` = sin suavizado). |
| `file_pattern`    | `*.json`  | Patrón glob para filtrar archivos dentro de la carpeta de entrada. |

**Salida**: `{output}/peak_proportion.json`

```json
{
  "peak_proportion": 0.737114,
  "peak_count": 9,
  "mean": 0.739948,
  "std": 0.026055,
  "min": 0.694649,
  "max": 0.817326,
  "n_bins": 13,
  "valid_proportions": 50,
  "skipped": 0,
  "errors": 0
}
```

**Formatos de JSON de entrada aceptados**:
- Con clave `polygon` (salida directa de `PolygonToGTFormat`, archivos `.det.json` o `.gt.json`)
- Con clave `proportion` (salida de `CalculatePolygonProportion` guardado como JSON)

---

## Flujo de trabajo típico: calibración de proporción

```
1. Correr batch con pipeline Heraldo_Claude (genera .det.json por imagen)
         ↓
2. postprocess: find_peak_proportion.py lee todos los .det.json
         ↓
3. Resultado: __data/calibration/peak_proportion.json
         ↓
4. El pipeline de recorte consume peak_proportion como restricción
```

Configuración de ejemplo completa:

```json
{
    "source_folder": "./__data/heraldo_raw_100/Derecha/",
    "targets": [
        {
            "filter_id":   "polygon_to_gt",
            "output_name": "gt_data",
            "destination": {
                "folder":    "./__data/experiments/001_baseline/results/Derecha/",
                "extension": "det.json"
            }
        }
    ],
    "postprocess": [
        {
            "script": "../../scripts/find_peak_proportion.py",
            "input":  "./__data/experiments/001_baseline/results/Derecha",
            "output": "./__data/calibration",
            "params": {"bin_size": 0.01, "smoothing_sigma": 1.5}
        }
    ]
}
```

---

## Crear nuevos scripts

Cualquier script Python que siga la interfaz estándar puede usarse. Estructura mínima:

```python
#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--params", default="{}")
    args = parser.parse_args()

    params = json.loads(args.params)
    input_folder  = Path(args.input)
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    # ... lógica del script ...

    # Salir con 0 en éxito, distinto de 0 en error
    sys.exit(0)

if __name__ == "__main__":
    main()
```

Los scripts van en la carpeta `scripts/` en la raíz del proyecto.
