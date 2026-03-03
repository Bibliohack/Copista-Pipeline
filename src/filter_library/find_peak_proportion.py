"""
Filtro: FindPeakProportion

Lee una carpeta de archivos JSON con polígonos o proporciones pre-calculadas,
construye un histograma de proporciones y encuentra el pico (valor más común).

Dos formatos de JSON aceptados:
  - Archivos con clave 'polygon'    (salida de PolygonToGTFormat / .det.json / .gt.json)
  - Archivos con clave 'proportion' (salida de CalculatePolygonProportion guardado)

Uso típico: pipeline de calibración del corpus. Ejecutar una vez sobre el corpus
completo para determinar la proporción verdadera del papel.
"""

import json
import math
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


def _dist(p1: Dict, p2: Dict) -> float:
    return math.sqrt((p2["x"] - p1["x"]) ** 2 + (p2["y"] - p1["y"]) ** 2)


def _proportion_from_polygon(polygon: List) -> Optional[float]:
    """Calcula avg(lado_izq, lado_der) / lado_superior desde 4 esquinas [TL,TR,BR,BL]."""
    if not polygon or len(polygon) != 4 or any(p is None for p in polygon):
        return None
    if not all("x" in p and "y" in p for p in polygon):
        return None
    tl, tr, br, bl = polygon
    top_side   = _dist(tl, tr)
    right_side = _dist(tr, br)
    left_side  = _dist(tl, bl)
    if top_side == 0:
        return None
    return (left_side + right_side) / (2.0 * top_side)


def _extract_proportion(data: Dict) -> Optional[float]:
    """Extrae la proporción de un dict JSON (cualquier formato soportado)."""
    # Formato proportion_data (CalculatePolygonProportion)
    if "proportion" in data and data["proportion"] is not None:
        try:
            return float(data["proportion"])
        except (TypeError, ValueError):
            pass

    # Formato gt_data / det (PolygonToGTFormat)
    if "polygon" in data:
        return _proportion_from_polygon(data["polygon"])

    return None


def _gaussian_kernel(sigma: float, radius: int) -> np.ndarray:
    """Kernel gaussiano 1D normalizado."""
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


class FindPeakProportion(BaseFilter):
    """
    Analiza una carpeta de JSONs para encontrar la proporción más común del corpus.

    Funciona como filtro independiente: ignora la imagen procesada y lee
    directamente los archivos de la carpeta indicada por el parámetro
    `source_folder`. Puede usarse en un pipeline de calibración procesando
    una sola imagen de "disparo" para obtener el resultado.

    El pico se determina por histograma con suavizado gaussiano opcional, lo que
    lo hace robusto frente a variaciones menores de zoom o perspectiva.
    """

    FILTER_NAME = "FindPeakProportion"
    DESCRIPTION = (
        "Lee una carpeta de JSONs (polígonos o proporciones), construye un histograma "
        "de proporciones y encuentra el pico (valor más común del corpus). "
        "Parámetro requerido: source_folder."
    )

    INPUTS = {}  # No depende de la imagen del pipeline

    OUTPUTS = {
        "peak_data":    "metadata",
        "sample_image": "image",
    }

    PARAMS = {
        "source_folder": {
            "default": "",
            "description": "Carpeta con archivos JSON a analizar (ruta absoluta o relativa al CWD).",
        },
        "file_pattern": {
            "default": "*.json",
            "description": "Patrón glob para filtrar archivos dentro de source_folder.",
        },
        "bin_size": {
            "default": 0.01,
            "min": 0.001,
            "max": 0.1,
            "step": 0.005,
            "description": "Tamaño de bin del histograma (unidades de proporción).",
        },
        "smoothing_sigma": {
            "default": 1.5,
            "min": 0.0,
            "max": 5.0,
            "step": 0.5,
            "description": "Sigma del suavizado gaussiano (0 = sin suavizado).",
        },
        "histogram_height": {
            "default": 350,
            "min": 150,
            "max": 600,
            "step": 50,
            "description": "Altura de la imagen de histograma.",
        },
        "histogram_width": {
            "default": 900,
            "min": 400,
            "max": 1600,
            "step": 100,
            "description": "Ancho de la imagen de histograma.",
        },
    }

    # ── Proceso principal ──────────────────────────────────────────────────────

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        source_folder   = str(self.params.get("source_folder", "")).strip()
        file_pattern    = str(self.params.get("file_pattern",  "*.json")).strip()
        bin_size        = float(self.params["bin_size"])
        smoothing_sigma = float(self.params["smoothing_sigma"])

        # Leer proporciones de los archivos
        proportions, skipped, errors = self._load_proportions(source_folder, file_pattern)

        # Calcular histograma y pico
        peak_data = self._find_peak(proportions, bin_size, smoothing_sigma,
                                    len(proportions), skipped, errors)

        result = {"peak_data": peak_data}

        if not self.without_preview:
            result["sample_image"] = self._make_histogram(proportions, peak_data,
                                                          bin_size, smoothing_sigma)

        return result

    # ── Carga de datos ─────────────────────────────────────────────────────────

    def _load_proportions(self, source_folder: str,
                          file_pattern: str) -> Tuple[List[float], int, int]:
        """Lee todos los JSON de la carpeta y extrae proporciones."""
        proportions: List[float] = []
        skipped = 0
        errors  = 0

        if not source_folder:
            return proportions, skipped, errors

        folder = Path(source_folder)
        if not folder.is_dir():
            return proportions, skipped, errors

        for json_path in sorted(folder.glob(file_pattern)):
            if not json_path.is_file():
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                prop = _extract_proportion(data)
                if prop is not None and np.isfinite(prop) and prop > 0:
                    proportions.append(prop)
                else:
                    skipped += 1
            except Exception:
                errors += 1

        return proportions, skipped, errors

    # ── Análisis de pico ───────────────────────────────────────────────────────

    def _find_peak(self, proportions: List[float], bin_size: float,
                   smoothing_sigma: float, total_read: int,
                   skipped: int, errors: int) -> Dict:
        """Construye histograma, aplica suavizado y encuentra el pico."""
        base = {
            "total_files_read": total_read,
            "valid_proportions": len(proportions),
            "skipped": skipped,
            "errors":  errors,
            "source_folder": str(self.params.get("source_folder", "")),
        }

        if len(proportions) < 2:
            return {**base,
                    "peak_proportion": None,
                    "peak_count":      0,
                    "mean":            proportions[0] if proportions else None,
                    "std":             0.0,
                    "min":             proportions[0] if proportions else None,
                    "max":             proportions[0] if proportions else None,
                    "valid":           False}

        arr = np.array(proportions)
        p_min = float(arr.min())
        p_max = float(arr.max())

        # Construir histograma
        n_bins  = max(1, int(math.ceil((p_max - p_min) / bin_size)))
        counts, edges = np.histogram(arr, bins=n_bins, range=(p_min, p_max + 1e-9))

        # Suavizado gaussiano
        if smoothing_sigma > 0 and len(counts) >= 3:
            radius = max(1, int(math.ceil(3 * smoothing_sigma)))
            kernel = _gaussian_kernel(smoothing_sigma, radius)
            smooth = np.convolve(counts.astype(np.float64), kernel, mode="same")
        else:
            smooth = counts.astype(np.float64)

        # Pico = bin de máximo valor suavizado
        peak_bin   = int(np.argmax(smooth))
        peak_prop  = float((edges[peak_bin] + edges[peak_bin + 1]) / 2.0)
        peak_count = int(counts[peak_bin])

        return {**base,
                "peak_proportion": round(peak_prop, 6),
                "peak_count":      peak_count,
                "mean":            round(float(arr.mean()), 6),
                "std":             round(float(arr.std()), 6),
                "min":             round(p_min, 6),
                "max":             round(p_max, 6),
                "n_bins":          n_bins,
                "bin_size":        bin_size,
                "valid":           True}

    # ── Visualización ──────────────────────────────────────────────────────────

    def _make_histogram(self, proportions: List[float],
                        peak_data: Dict, bin_size: float,
                        smoothing_sigma: float) -> np.ndarray:
        """Genera imagen de histograma con el pico marcado."""
        vis_w = int(self.params["histogram_width"])
        vis_h = int(self.params["histogram_height"])

        margin_l = 70
        margin_r = 30
        margin_t = 50
        margin_b = 60
        plot_w = vis_w - margin_l - margin_r
        plot_h = vis_h - margin_t - margin_b

        img = np.full((vis_h, vis_w, 3), (30, 30, 30), dtype=np.uint8)

        if not peak_data.get("valid") or not proportions:
            msg = peak_data.get("source_folder", "")
            n   = peak_data.get("valid_proportions", 0)
            cv2.putText(img, f"Sin datos suficientes ({n} proporciones)", (20, vis_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, f"Carpeta: {msg}", (20, vis_h // 2 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            return img

        arr = np.array(proportions)
        p_min  = peak_data["min"]
        p_max  = peak_data["max"]
        n_bins = peak_data["n_bins"]

        counts, edges = np.histogram(arr, bins=n_bins, range=(p_min, p_max + 1e-9))

        # Suavizado
        if smoothing_sigma > 0 and len(counts) >= 3:
            radius = max(1, int(math.ceil(3 * smoothing_sigma)))
            kernel = _gaussian_kernel(smoothing_sigma, radius)
            smooth = np.convolve(counts.astype(np.float64), kernel, mode="same")
        else:
            smooth = counts.astype(np.float64)

        max_count = max(counts.max(), 1)
        peak_prop = peak_data["peak_proportion"]

        # ── Dibujar barras (histograma crudo) ────────────────────────────────
        bar_w = max(1, plot_w // n_bins)
        for i, c in enumerate(counts):
            bh = int(c / max_count * plot_h)
            x0 = margin_l + int(i * plot_w / n_bins)
            x1 = margin_l + int((i + 1) * plot_w / n_bins)
            y0 = margin_t + plot_h - bh
            y1 = margin_t + plot_h
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1), (100, 150, 200), -1)

        # ── Curva suavizada ──────────────────────────────────────────────────
        smooth_pts = []
        for i, s in enumerate(smooth):
            sx = margin_l + int((i + 0.5) * plot_w / n_bins)
            sy = margin_t + plot_h - int(s / max_count * plot_h)
            smooth_pts.append((sx, sy))
        if len(smooth_pts) >= 2:
            pts_arr = np.array(smooth_pts, dtype=np.int32)
            cv2.polylines(img, [pts_arr], isClosed=False, color=(0, 230, 118), thickness=2)

        # ── Línea de pico ────────────────────────────────────────────────────
        peak_x = margin_l + int((peak_prop - p_min) / max(p_max - p_min, 1e-9) * plot_w)
        peak_x = max(margin_l, min(margin_l + plot_w, peak_x))
        cv2.line(img, (peak_x, margin_t), (peak_x, margin_t + plot_h), (0, 255, 255), 2)

        # Etiqueta del pico
        label = f"PICO: {peak_prop:.4f}  (n={peak_data['peak_count']})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx = min(peak_x + 5, vis_w - tw - 5)
        cv2.rectangle(img, (lx - 2, margin_t + 2), (lx + tw + 2, margin_t + th + 8), (0, 0, 0), -1)
        cv2.putText(img, label, (lx, margin_t + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # ── Ejes ─────────────────────────────────────────────────────────────
        # Eje Y
        cv2.line(img, (margin_l, margin_t), (margin_l, margin_t + plot_h), (150, 150, 150), 1)
        # Eje X
        cv2.line(img, (margin_l, margin_t + plot_h),
                 (margin_l + plot_w, margin_t + plot_h), (150, 150, 150), 1)

        # Etiquetas eje X (5 puntos)
        for i in range(6):
            val  = p_min + i * (p_max - p_min) / 5
            xpos = margin_l + int(i * plot_w / 5)
            cv2.putText(img, f"{val:.3f}", (xpos - 18, margin_t + plot_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
            cv2.line(img, (xpos, margin_t + plot_h), (xpos, margin_t + plot_h + 4), (150, 150, 150), 1)

        # ── Título y estadísticas ─────────────────────────────────────────────
        title = "HISTOGRAMA DE PROPORCIONES DEL CORPUS"
        cv2.putText(img, title, (margin_l, margin_t - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        stats = (
            f"N={peak_data['valid_proportions']}  "
            f"media={peak_data['mean']:.4f}  "
            f"std={peak_data['std']:.4f}  "
            f"rango=[{p_min:.4f}, {p_max:.4f}]"
        )
        cv2.putText(img, stats, (margin_l, vis_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

        return img
