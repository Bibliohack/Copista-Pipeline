#!/usr/bin/env python3
"""
Script: find_peak_proportion.py
===============================

Análisis de corpus: lee una carpeta de JSONs con polígonos o proporciones
pre-calculadas, construye un histograma y encuentra la proporción más común (pico).

Formatos de JSON aceptados:
  - Archivos con clave 'polygon'    (salida de PolygonToGTFormat / .det.json / .gt.json)
  - Archivos con clave 'proportion' (salida de CalculatePolygonProportion guardado)

Uso directo:
    python find_peak_proportion.py --input <carpeta> --output <carpeta>
    python find_peak_proportion.py --input <carpeta> --output <carpeta> \\
        --params '{"bin_size": 0.005, "smoothing_sigma": 2.0}'

Uso desde batch_config.json (postprocess):
    {
        "script": "../../scripts/find_peak_proportion.py",
        "input":  "__data/experiments/001_baseline/results/Derecha",
        "output": "__data/calibration",
        "params": {"bin_size": 0.01, "smoothing_sigma": 1.5}
    }

Salida:
    {output}/peak_proportion.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np


# ── Utilidades de cálculo ──────────────────────────────────────────────────────

def _dist(p1: Dict, p2: Dict) -> float:
    return math.sqrt((p2["x"] - p1["x"]) ** 2 + (p2["y"] - p1["y"]) ** 2)


def _proportion_from_polygon(polygon: List) -> Optional[float]:
    """Calcula avg(lado_izq, lado_der) / lado_superior desde [TL, TR, BR, BL]."""
    if not polygon or len(polygon) != 4 or any(p is None for p in polygon):
        return None
    if not all("x" in p and "y" in p for p in polygon):
        return None
    tl, tr, br, bl = polygon
    top_side = _dist(tl, tr)
    if top_side == 0:
        return None
    return (_dist(tl, bl) + _dist(tr, br)) / (2.0 * top_side)


def _area_from_polygon(polygon: List) -> Optional[float]:
    """Calcula el área del polígono usando la fórmula shoelace."""
    if not polygon or len(polygon) != 4 or any(p is None for p in polygon):
        return None
    if not all("x" in p and "y" in p for p in polygon):
        return None
    n = len(polygon)
    acc = 0.0
    for i in range(n):
        x1 = polygon[i]["x"]
        y1 = polygon[i]["y"]
        x2 = polygon[(i + 1) % n]["x"]
        y2 = polygon[(i + 1) % n]["y"]
        acc += x1 * y2 - x2 * y1
    return abs(acc) / 2.0


def _extract_area_fraction(data: Dict) -> Optional[float]:
    """
    Extrae la fracción area_polygon / (image_width * image_height).
    Requiere claves 'polygon', 'image_width' e 'image_height'.
    Retorna None si falta alguna clave o el área de imagen es 0.
    """
    img_w = data.get("image_width")
    img_h = data.get("image_height")
    if img_w is None or img_h is None:
        return None
    try:
        img_w = float(img_w)
        img_h = float(img_h)
    except (TypeError, ValueError):
        return None
    img_area = img_w * img_h
    if img_area <= 0:
        return None
    if "polygon" not in data:
        return None
    area = _area_from_polygon(data["polygon"])
    if area is None or area <= 0:
        return None
    return area / img_area


def _extract_proportion(data: Dict) -> Optional[float]:
    """Extrae la proporción de un dict JSON (cualquier formato soportado)."""
    if "proportion" in data and data["proportion"] is not None:
        try:
            return float(data["proportion"])
        except (TypeError, ValueError):
            pass
    if "polygon" in data:
        return _proportion_from_polygon(data["polygon"])
    return None


def _gaussian_kernel(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


# ── Lógica principal ───────────────────────────────────────────────────────────

def load_proportions(input_folder: Path, file_pattern: str):
    proportions = []
    area_fractions = []
    skipped = 0
    errors = 0

    for json_path in sorted(input_folder.glob(file_pattern)):
        if not json_path.is_file():
            continue
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            prop = _extract_proportion(data)
            if prop is not None and np.isfinite(prop) and prop > 0:
                proportions.append(prop)
            else:
                skipped += 1
            # El área se calcula independientemente (puede faltar image_width/height)
            frac = _extract_area_fraction(data)
            if frac is not None and np.isfinite(frac) and 0 < frac <= 1.0:
                area_fractions.append(frac)
        except Exception:
            errors += 1

    return proportions, area_fractions, skipped, errors


def find_peak(proportions: List[float], bin_size: float, smoothing_sigma: float) -> Dict:
    arr = np.array(proportions)
    p_min = float(arr.min())
    p_max = float(arr.max())
    n_bins = max(1, int(math.ceil((p_max - p_min) / bin_size)))

    counts, edges = np.histogram(arr, bins=n_bins, range=(p_min, p_max + 1e-9))

    if smoothing_sigma > 0 and len(counts) >= 3:
        radius = max(1, int(math.ceil(3 * smoothing_sigma)))
        kernel = _gaussian_kernel(smoothing_sigma, radius)
        smooth = np.convolve(counts.astype(np.float64), kernel, mode="same")
    else:
        smooth = counts.astype(np.float64)

    peak_bin  = int(np.argmax(smooth))
    peak_prop = float((edges[peak_bin] + edges[peak_bin + 1]) / 2.0)

    return {
        "peak_proportion": round(peak_prop, 6),
        "peak_count":      int(counts[peak_bin]),
        "mean":            round(float(arr.mean()), 6),
        "std":             round(float(arr.std()),  6),
        "min":             round(p_min, 6),
        "max":             round(p_max, 6),
        "n_bins":          n_bins,
        "valid_proportions": len(proportions),
    }


# ── Entrada del script ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Encuentra la proporción de papel más común en un corpus de polígonos JSON."
    )
    parser.add_argument("--input",  required=True, help="Carpeta con archivos JSON")
    parser.add_argument("--output", required=True, help="Carpeta donde guardar peak_proportion.json")
    parser.add_argument("--params", default="{}",
                        help='JSON con parámetros opcionales: bin_size, smoothing_sigma, file_pattern')
    args = parser.parse_args()

    try:
        params = json.loads(args.params)
    except json.JSONDecodeError as e:
        print(f"❌ --params no es JSON válido: {e}")
        sys.exit(1)

    bin_size        = float(params.get("bin_size",        0.01))
    smoothing_sigma = float(params.get("smoothing_sigma", 1.5))
    file_pattern    = str(params.get("file_pattern",      "*.json"))

    input_folder  = Path(args.input)
    output_folder = Path(args.output)

    if not input_folder.is_dir():
        print(f"❌ Carpeta de entrada no existe: {input_folder}")
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    # Cargar proporciones y fracciones de área
    print(f"  Leyendo archivos de: {input_folder}")
    proportions, area_fractions, skipped, errors = load_proportions(input_folder, file_pattern)

    total = len(proportions) + skipped + errors
    print(f"  Archivos encontrados:     {total}")
    print(f"  Proporciones válidas:     {len(proportions)}")
    print(f"  Fracciones de área válidas: {len(area_fractions)}")
    if skipped:
        print(f"  Sin proporción (saltados): {skipped}")
    if errors:
        print(f"  Errores de lectura:       {errors}")

    if len(proportions) < 2:
        print("❌ No hay suficientes datos para calcular el pico (mínimo 2)")
        sys.exit(1)

    # Calcular pico de proporción
    result = find_peak(proportions, bin_size, smoothing_sigma)
    result["skipped"]          = skipped
    result["errors"]           = errors
    result["input_folder"]     = str(input_folder)
    result["bin_size"]         = bin_size
    result["smoothing_sigma"]  = smoothing_sigma

    # Calcular estadísticas de fracción de área
    if len(area_fractions) >= 2:
        arr_af = np.array(area_fractions)
        result["peak_area_fraction"]  = round(float(arr_af.mean()), 6)
        result["area_fraction_std"]   = round(float(arr_af.std()),  6)
        result["area_fraction_min"]   = round(float(arr_af.min()),  6)
        result["area_fraction_max"]   = round(float(arr_af.max()),  6)
        result["area_fraction_count"] = len(area_fractions)
        print(f"  Fracción de área media (peak_area_fraction): {result['peak_area_fraction']}")
        print(f"  Std área: {result['area_fraction_std']}  Min: {result['area_fraction_min']}  Max: {result['area_fraction_max']}")
    else:
        print("  Advertencia: no hay suficientes datos para calcular peak_area_fraction")

    # Guardar resultado
    output_path = output_folder / "peak_proportion.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"  Proporción pico: {result['peak_proportion']}")
    print(f"  Media: {result['mean']}  |  Std: {result['std']}")
    print(f"  Guardado en: {output_path}")


if __name__ == "__main__":
    main()
