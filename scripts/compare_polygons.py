#!/usr/bin/env python3
"""
Script: compare_polygons.py
===========================

Genera imágenes de comparación visual dibujando los polígonos detectados
(GT, exp1, exp2) superpuestos sobre la imagen original.

Uso:
    python3 scripts/compare_polygons.py \\
        --images  __data/heraldo_raw_100/ \\
        --gt      __data/ground_truth/ \\
        --exp1    __data/experiments/001_baseline/results/ \\
        --exp2    __data/experiments/002_refine_polygon/results/ \\
        --output  __data/comparisons/001_vs_002/ \\
        [--filter improved|regressed|all] \\
        [--threshold 0.02]

Parámetros:
    --images     Carpeta con imágenes originales (.jpg / .jpeg / .png)
    --gt         Carpeta con ground truth (.gt.json)
    --exp1       Carpeta con resultados de experimento 1 (.det.json)
    --exp2       Carpeta con resultados de experimento 2 (.det.json) [opcional]
    --output     Carpeta donde guardar las imágenes generadas
    --filter     Qué casos mostrar: all | improved | regressed (default: all)
    --threshold  Delta IoU mínimo para improved/regressed (default: 0.02)
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np


# ── Importar funciones IoU desde src/ ─────────────────────────────────────────

def _setup_src_path():
    """Agrega src/ al sys.path para poder importar iou_metrics."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

_setup_src_path()

try:
    from iou_metrics import polygon_to_numpy, compute_iou
except ImportError:
    # Fallback: definir las funciones aquí si iou_metrics no está disponible
    warnings.warn("No se pudo importar iou_metrics. Usando implementación local.")

    def polygon_to_numpy(polygon: list):
        """Convierte lista de puntos del polígono a array numpy para cv2."""
        if not polygon:
            return None
        pts = []
        for p in polygon:
            if p is None:
                return None
            pts.append([float(p['x']), float(p['y'])])
        return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    def compute_iou(poly_gt: np.ndarray, poly_det: np.ndarray) -> float:
        """Calcula IoU entre dos polígonos convexos."""
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


# ── Constantes de visualización ───────────────────────────────────────────────

MAX_WIDTH       = 1200          # ancho máximo de la imagen de salida
COLOR_GT        = (0, 200, 0)   # verde (BGR)
COLOR_EXP1      = (255, 200, 0) # cian (BGR)
COLOR_EXP2      = (0, 80, 255)  # naranja-rojo (BGR)
THICKNESS_GT    = 3
THICKNESS_DET   = 2
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.55
FONT_THICKNESS  = 1
LINE_HEIGHT     = 22            # píxeles entre líneas del panel de info


# ── Utilidades ────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | None:
    """Carga un JSON. Devuelve None si falla."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARN: no se pudo leer {path}: {e}", file=sys.stderr)
        return None


def find_image(images_dir: Path, stem: str) -> Path | None:
    """Busca la imagen con el stem dado en .jpg, .jpeg o .png."""
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def scale_image_and_factor(img: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    """Escala la imagen para que no supere max_width. Devuelve (imagen_escalada, factor)."""
    h, w = img.shape[:2]
    if w <= max_width:
        return img, 1.0
    factor = max_width / w
    new_w  = max_width
    new_h  = int(h * factor)
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return scaled, factor


def scale_polygon(polygon_np: np.ndarray | None, factor: float) -> np.ndarray | None:
    """Escala las coordenadas del polígono por el factor dado."""
    if polygon_np is None:
        return None
    return (polygon_np * factor).astype(np.int32)


def polygon_top_left(polygon_np: np.ndarray | None) -> tuple[int, int]:
    """Devuelve el punto más cercano a la esquina superior izquierda del polígono."""
    if polygon_np is None:
        return (10, 30)
    pts = polygon_np.reshape(-1, 2)
    # Punto con mínima suma x+y (esquina superior izquierda)
    idx = np.argmin(pts[:, 0] + pts[:, 1])
    return (int(pts[idx, 0]), int(pts[idx, 1]))


def draw_polygon_with_label(
    img: np.ndarray,
    polygon_np: np.ndarray | None,
    color: tuple,
    thickness: int,
    label: str,
):
    """Dibuja el polígono y una etiqueta en su esquina superior izquierda."""
    if polygon_np is None:
        return
    pts = polygon_np.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    # Etiqueta en la esquina superior izquierda del polígono
    tx, ty = polygon_top_left(polygon_np)
    # Pequeño fondo negro para legibilidad
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    cv2.rectangle(img, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
    cv2.putText(img, label, (tx, ty), FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)


def draw_gt_corners(img: np.ndarray, polygon_np: np.ndarray | None, color: tuple):
    """Dibuja solo los puntos de las esquinas del GT (círculos con borde blanco)."""
    if polygon_np is None:
        return
    pts = polygon_np.reshape(-1, 2).astype(np.int32)
    for pt in pts:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), 8, (255, 255, 255), -1)  # borde blanco
        cv2.circle(img, (x, y), 6, color, -1)             # relleno del color GT


def draw_info_panel(img: np.ndarray, lines: list[tuple[str, tuple]]):
    """
    Dibuja un panel de información en la esquina superior izquierda de la imagen.
    lines: lista de (texto, color_bgr).
    """
    if not lines:
        return

    padding    = 6
    line_h     = LINE_HEIGHT
    panel_h    = len(lines) * line_h + 2 * padding
    max_tw     = 0
    for text, _ in lines:
        (tw, _), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        max_tw = max(max_tw, tw)
    panel_w = max_tw + 2 * padding

    # Rectángulo semitransparente
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    y = padding + line_h - 4
    for text, color in lines:
        cv2.putText(img, text, (padding, y), FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)
        y += line_h


# ── Lógica principal ───────────────────────────────────────────────────────────

def process_image(
    stem: str,
    images_dir: Path,
    gt_dir: Path,
    exp1_dir: Path,
    exp2_dir: Path | None,
    output_dir: Path,
    filter_mode: str,
    threshold: float,
) -> str:
    """
    Procesa una imagen y genera el PNG de comparación.

    Retorna:
        "generated"  — imagen generada correctamente
        "skipped"    — saltada por filtro o por archivos faltantes
        "error"      — error al procesar
    """
    # ── 1. Buscar archivos ────────────────────────────────────────────────────
    image_path = find_image(images_dir, stem)
    if image_path is None:
        print(f"  WARN [{stem}]: imagen no encontrada en {images_dir}", file=sys.stderr)
        return "skipped"

    gt_path   = gt_dir   / (stem + ".gt.json")
    det1_path = exp1_dir / (stem + ".det.json")

    if not gt_path.exists():
        print(f"  WARN [{stem}]: GT no encontrado: {gt_path}", file=sys.stderr)
        return "skipped"

    if not det1_path.exists():
        print(f"  WARN [{stem}]: det exp1 no encontrado: {det1_path}", file=sys.stderr)
        return "skipped"

    # ── 2. Cargar JSONs ───────────────────────────────────────────────────────
    gt_data   = load_json(gt_path)
    det1_data = load_json(det1_path)

    if gt_data is None or det1_data is None:
        return "error"

    det2_data = None
    if exp2_dir is not None:
        det2_path = exp2_dir / (stem + ".det.json")
        if det2_path.exists():
            det2_data = load_json(det2_path)
        else:
            print(f"  WARN [{stem}]: det exp2 no encontrado: {det2_path}", file=sys.stderr)

    # ── 3. Calcular IoU ───────────────────────────────────────────────────────
    gt_np   = polygon_to_numpy(gt_data.get("polygon", []))
    det1_np = polygon_to_numpy(det1_data.get("polygon", []))
    iou1    = compute_iou(gt_np, det1_np)

    iou2      = None
    det2_np   = None
    delta_iou = None

    if det2_data is not None:
        det2_np   = polygon_to_numpy(det2_data.get("polygon", []))
        iou2      = compute_iou(gt_np, det2_np)
        delta_iou = iou2 - iou1

    # ── 4. Aplicar filtro ─────────────────────────────────────────────────────
    if filter_mode == "improved":
        if delta_iou is None or delta_iou <= threshold:
            return "skipped"
    elif filter_mode == "regressed":
        if delta_iou is None or delta_iou >= -threshold:
            return "skipped"
    # "all" no filtra nada

    # ── 5. Cargar imagen y escalar ────────────────────────────────────────────
    img_orig = cv2.imread(str(image_path))
    if img_orig is None:
        print(f"  WARN [{stem}]: no se pudo cargar la imagen: {image_path}", file=sys.stderr)
        return "error"

    img, factor = scale_image_and_factor(img_orig, MAX_WIDTH)

    # ── 6. Escalar polígonos ──────────────────────────────────────────────────
    gt_scaled   = scale_polygon(gt_np,   factor)
    det1_scaled = scale_polygon(det1_np, factor)
    det2_scaled = scale_polygon(det2_np, factor) if det2_np is not None else None

    # ── 7. Dibujar polígonos ──────────────────────────────────────────────────
    corners1 = det1_data.get("all_corners_found", False)
    corners2 = det2_data.get("all_corners_found", False) if det2_data else False

    label1 = f"EXP1 IoU={iou1:.4f}" + ("" if corners1 else " (FAIL)")
    label2 = ""
    if iou2 is not None:
        label2 = f"EXP2 IoU={iou2:.4f}" + ("" if corners2 else " (FAIL)")

    # Orden: exp1 abajo, exp2 encima, puntos GT arriba de todo
    draw_polygon_with_label(img, det1_scaled, COLOR_EXP1, THICKNESS_DET, label1)
    draw_polygon_with_label(img, det2_scaled, COLOR_EXP2, THICKNESS_DET, label2)
    draw_gt_corners(img, gt_scaled, COLOR_GT)

    # ── 8. Panel de información ───────────────────────────────────────────────
    info_lines: list[tuple[str, tuple]] = [
        (stem, (255, 255, 255)),
        (f"EXP1 IoU = {iou1:.4f}" + ("" if corners1 else " (FAIL)"), COLOR_EXP1),
    ]
    if iou2 is not None:
        info_lines.append(
            (f"EXP2 IoU = {iou2:.4f}" + ("" if corners2 else " (FAIL)"), COLOR_EXP2)
        )
        sign = "+" if delta_iou >= 0 else ""
        info_lines.append(
            (f"Delta IoU = {sign}{delta_iou:.4f}", (200, 200, 200))
        )

    draw_info_panel(img, info_lines)

    # ── 9. Guardar ────────────────────────────────────────────────────────────
    output_path = output_dir / (stem + ".png")
    ok = cv2.imwrite(str(output_path), img)
    if not ok:
        print(f"  ERROR [{stem}]: no se pudo guardar {output_path}", file=sys.stderr)
        return "error"

    return "generated"


# ── Punto de entrada ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera imágenes de comparación de polígonos detectados vs ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--images",    required=True,  help="Carpeta con imágenes originales")
    parser.add_argument("--gt",        required=True,  help="Carpeta con ground truth (.gt.json)")
    parser.add_argument("--exp1",      required=True,  help="Carpeta con resultados de exp1 (.det.json)")
    parser.add_argument("--exp2",      default=None,   help="Carpeta con resultados de exp2 (.det.json) [opcional]")
    parser.add_argument("--output",    required=True,  help="Carpeta de salida para las imágenes generadas")
    parser.add_argument(
        "--filter",
        default="all",
        choices=["all", "improved", "regressed"],
        help="Qué casos mostrar: all | improved | regressed (default: all)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Delta IoU mínimo para considerar mejora/regresión (default: 0.02)",
    )

    args = parser.parse_args()

    images_dir = Path(args.images)
    gt_dir     = Path(args.gt)
    exp1_dir   = Path(args.exp1)
    exp2_dir   = Path(args.exp2) if args.exp2 else None
    output_dir = Path(args.output)

    # ── Validar directorios de entrada ────────────────────────────────────────
    for label, path in [("--images", images_dir), ("--gt", gt_dir), ("--exp1", exp1_dir)]:
        if not path.exists():
            print(f"ERROR: {label} no existe: {path}", file=sys.stderr)
            sys.exit(1)

    if exp2_dir is not None and not exp2_dir.exists():
        print(f"ERROR: --exp2 no existe: {exp2_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Crear directorio de salida ────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Iterar sobre todos los GT ─────────────────────────────────────────────
    gt_files = sorted(gt_dir.glob("*.gt.json"))
    if not gt_files:
        print(f"ERROR: no se encontraron archivos .gt.json en {gt_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nCompare Polygons")
    print(f"  images : {images_dir}")
    print(f"  gt     : {gt_dir}")
    print(f"  exp1   : {exp1_dir}")
    print(f"  exp2   : {exp2_dir or '(no especificado)'}")
    print(f"  output : {output_dir}")
    print(f"  filter : {args.filter}  threshold={args.threshold}")
    print(f"  GT files encontrados: {len(gt_files)}\n")

    counts = {"generated": 0, "skipped": 0, "error": 0}

    for gt_file in gt_files:
        # "001_derecha.gt.json" → stem = "001_derecha"
        stem = gt_file.name
        if stem.endswith(".gt.json"):
            stem = stem[: -len(".gt.json")]
        else:
            stem = Path(stem).stem

        result = process_image(
            stem       = stem,
            images_dir = images_dir,
            gt_dir     = gt_dir,
            exp1_dir   = exp1_dir,
            exp2_dir   = exp2_dir,
            output_dir = output_dir,
            filter_mode= args.filter,
            threshold  = args.threshold,
        )
        counts[result] += 1

    # ── Resumen final ─────────────────────────────────────────────────────────
    total = sum(counts.values())
    print(f"\n{'='*48}")
    print(f"  RESUMEN")
    print(f"{'='*48}")
    print(f"  Total GT procesados : {total}")
    print(f"  Imágenes generadas  : {counts['generated']}")
    print(f"  Saltadas (filtro/faltantes): {counts['skipped']}")
    print(f"  Errores             : {counts['error']}")
    print(f"  Salida              : {output_dir}")
    print(f"{'='*48}\n")

    if counts["error"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
