#!/usr/bin/env python3
"""
Evaluación IoU para experimentos de detección de bordes de página.

Compara los polígonos detectados automáticamente (.det.json) contra las
anotaciones manuales (.gt.json) usando Intersection over Union (IoU).

Modos:
    evaluate  — evalúa un experimento y genera metrics.json
    compare   — tabla comparativa de todos los experimentos evaluados

Uso:
    python iou_metrics.py evaluate <experiment_folder> [--gt <gt_folder>]
    python iou_metrics.py compare  [--experiments <experiments_folder>]

Ejemplos:
    python iou_metrics.py evaluate __data/experiments/001_baseline/
    python iou_metrics.py evaluate __data/experiments/002_mejor_canny/ --gt __data/ground_truth/
    python iou_metrics.py compare
    python iou_metrics.py compare --experiments __data/experiments/
"""

import json
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime


# ── Cálculo de IoU ────────────────────────────────────────────────────────────

def polygon_to_numpy(polygon: list):
    """Convierte lista de puntos del polígono a array numpy para cv2.
    Devuelve None si algún punto es None (detección incompleta)."""
    if not polygon:
        return None
    pts = []
    for p in polygon:
        if p is None:
            return None
        pts.append([float(p['x']), float(p['y'])])
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)


def compute_iou(poly_gt: np.ndarray, poly_det: np.ndarray) -> float:
    """Calcula IoU entre dos polígonos convexos. Devuelve 0.0 si alguno es None."""
    if poly_gt is None or poly_det is None:
        return 0.0

    area_gt  = cv2.contourArea(poly_gt)
    area_det = cv2.contourArea(poly_det)

    if area_gt == 0 or area_det == 0:
        return 0.0

    intersection_area, _ = cv2.intersectConvexConvex(poly_gt, poly_det)
    union_area = area_gt + area_det - intersection_area

    if union_area == 0:
        return 0.0

    return float(intersection_area / union_area)


# ── Lógica de evaluación ──────────────────────────────────────────────────────

def evaluate_experiment(experiment_path: Path, gt_root: Path) -> dict | None:
    """
    Evalúa un experimento completo comparando cada .det.json contra su .gt.json.

    Itera directamente los *.det.json en results/ (estructura plana, sin subsets).
    Busca el GT correspondiente en gt_root/ con el mismo stem + .gt.json.
    Sólo evalúa imágenes que tengan tanto .det.json como .gt.json.

    Returns:
        dict con resultados completos, o None si no hay pares para evaluar.
    """
    results_root = experiment_path / "results"

    if not results_root.exists():
        print(f"  No existe carpeta results/ en {experiment_path}")
        return None

    all_results = []
    evaluated = skipped = 0

    det_files = sorted(results_root.glob("*.det.json"))

    for det_path in det_files:
        stem = Path(det_path.stem).stem  # "001_derecha.det.json" → stem="001_derecha.det" → stem="001_derecha"
        gt_path = gt_root / (stem + ".gt.json")
        if not gt_path.exists():
            skipped += 1
            continue

        with open(det_path, encoding="utf-8") as f:
            det_data = json.load(f)
        with open(gt_path, encoding="utf-8") as f:
            gt_data = json.load(f)

        gt_np  = polygon_to_numpy(gt_data.get("polygon", []))
        det_np = polygon_to_numpy(det_data.get("polygon", []))
        iou    = compute_iou(gt_np, det_np)

        all_results.append({
            "image":             stem,
            "iou":               round(iou, 4),
            "all_corners_found": det_data.get("all_corners_found", False),
        })
        evaluated += 1

    print(f"  {evaluated} evaluadas, {skipped} sin GT.")

    if not all_results:
        print("  No se encontraron pares GT / DET para evaluar.")
        return None

    ious   = [r["iou"] for r in all_results]
    failed = sum(1 for r in all_results if not r["all_corners_found"])

    return {
        "experiment":    experiment_path.name,
        "evaluated_at":  datetime.now().isoformat(timespec="seconds"),
        "gt_folder":     str(gt_root),
        "summary": {
            "evaluated":        len(all_results),
            "failed_detection": failed,
            "mean_iou":         round(float(np.mean(ious)),   4),
            "median_iou":       round(float(np.median(ious)), 4),
            "std_iou":          round(float(np.std(ious)),    4),
            "min_iou":          round(float(np.min(ious)),    4),
            "max_iou":          round(float(np.max(ious)),    4),
        },
        "results": sorted(all_results, key=lambda r: r["image"]),
    }


# ── Subcomando: evaluate ──────────────────────────────────────────────────────

def cmd_evaluate(args):
    experiment_path = Path(args.experiment)
    gt_root         = Path(args.gt)

    if not experiment_path.exists():
        print(f"❌ Experimento no encontrado: {experiment_path}")
        sys.exit(1)

    if not gt_root.exists():
        print(f"❌ Carpeta de ground truth no encontrada: {gt_root}")
        sys.exit(1)

    print(f"\nEvaluando : {experiment_path.name}")
    print(f"GT root   : {gt_root}\n")

    metrics = evaluate_experiment(experiment_path, gt_root)
    if metrics is None:
        sys.exit(1)

    metrics_path = experiment_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    _print_experiment_summary(metrics)
    print(f"✅ Guardado: {metrics_path}\n")


def _print_experiment_summary(metrics: dict):
    s = metrics["summary"]
    print(f"\n{'='*52}")
    print(f"  {metrics['experiment']}")
    print(f"{'='*52}")
    print(f"  Imágenes evaluadas   : {s['evaluated']}")
    print(f"  Detecciones fallidas : {s['failed_detection']}")
    print(f"  IoU medio            : {s['mean_iou']:.4f}")
    print(f"  IoU mediana          : {s['median_iou']:.4f}")
    print(f"  IoU std              : {s['std_iou']:.4f}")
    print(f"  IoU mín / máx        : {s['min_iou']:.4f} / {s['max_iou']:.4f}")
    print(f"{'='*52}")
    print(f"  Por imagen:")
    for r in metrics["results"]:
        mark = "✓" if r["all_corners_found"] else "✗"
        print(f"    [{mark}] {r['image']}  IoU={r['iou']:.4f}")


# ── Subcomando: compare ───────────────────────────────────────────────────────

def cmd_compare(args):
    experiments_root = Path(args.experiments)

    if not experiments_root.exists():
        print(f"❌ Carpeta de experimentos no encontrada: {experiments_root}")
        sys.exit(1)

    evaluated_dirs = sorted(
        d for d in experiments_root.iterdir()
        if d.is_dir() and (d / "metrics.json").exists()
    )

    if not evaluated_dirs:
        print("❌ No se encontraron experimentos con metrics.json.")
        print("   Primero ejecuta: python iou_metrics.py evaluate <experiment_folder>")
        sys.exit(1)

    rows = []
    for exp_dir in evaluated_dirs:
        with open(exp_dir / "metrics.json", encoding="utf-8") as f:
            m = json.load(f)
        s = m["summary"]
        rows.append({
            "name":   exp_dir.name,
            "n":      s["evaluated"],
            "failed": s["failed_detection"],
            "mean":   s["mean_iou"],
            "median": s["median_iou"],
            "std":    s["std_iou"],
            "min":    s["min_iou"],
            "max":    s["max_iou"],
        })

    col_w = max(len(r["name"]) for r in rows) + 2
    header = (
        f"{'Experimento':<{col_w}} {'N':>4}  {'Fallidos':>8}  "
        f"{'Media':>7}  {'Mediana':>7}  {'Std':>6}  {'Mín':>6}  {'Máx':>6}"
    )
    sep = "─" * len(header)

    print(f"\n{'═'*len(header)}")
    print(f"  COMPARATIVA DE EXPERIMENTOS — IoU de polígono detectado vs ground truth")
    print(f"{'═'*len(header)}")
    print(header)
    print(sep)

    # Ordenar por media IoU descendente (mejores arriba)
    for r in sorted(rows, key=lambda x: x["mean"], reverse=True):
        print(
            f"{r['name']:<{col_w}} {r['n']:>4}  {r['failed']:>8}  "
            f"{r['mean']:>7.4f}  {r['median']:>7.4f}  {r['std']:>6.4f}  "
            f"{r['min']:>6.4f}  {r['max']:>6.4f}"
        )

    print(f"{'═'*len(header)}\n")


# ── Punto de entrada ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluación IoU de detección de bordes de página.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="Evalúa un experimento y genera metrics.json")
    p_eval.add_argument("experiment", help="Carpeta del experimento")
    p_eval.add_argument(
        "--gt", default="__data/ground_truth",
        help="Carpeta raíz de ground truth (default: __data/ground_truth)"
    )

    p_cmp = sub.add_parser("compare", help="Tabla comparativa de todos los experimentos")
    p_cmp.add_argument(
        "--experiments", default="__data/experiments",
        help="Carpeta raíz de experimentos (default: __data/experiments)"
    )

    args = parser.parse_args()

    if args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
