"""
Utilidades geométricas compartidas para los filtros de refinamiento de polígono.

No es un filtro: no hereda de BaseFilter ni se registra en FILTER_REGISTRY.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Intersección y polígono ────────────────────────────────────────────────────

def line_intersection(l1: Dict, l2: Dict) -> Optional[Tuple[float, float]]:
    """Intersección de dos líneas (formato x1,y1,x2,y2). None si paralelas."""
    x1, y1, x2, y2 = l1["x1"], l1["y1"], l1["x2"], l1["y2"]
    x3, y3, x4, y4 = l2["x1"], l2["y1"], l2["x2"], l2["y2"]
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def polygon_from_lines(
    top: Dict, bottom: Dict, left: Dict, right: Dict
) -> Optional[List[Tuple[float, float]]]:
    """
    Vértices [TL, TR, BR, BL] intersectando las 4 líneas.
    Retorna None si alguna intersección falla (líneas paralelas).
    """
    tl = line_intersection(top, left)
    tr = line_intersection(top, right)
    br = line_intersection(bottom, right)
    bl = line_intersection(bottom, left)
    if None in (tl, tr, br, bl):
        return None
    return [tl, tr, br, bl]


def polygon_proportion(poly: List[Tuple[float, float]]) -> Optional[float]:
    """avg(lado_izquierdo, lado_derecho) / lado_superior."""
    if not poly or len(poly) != 4:
        return None
    tl, tr, br, bl = poly
    top_side = math.dist(tl, tr)
    if top_side < 1e-6:
        return None
    return (math.dist(tl, bl) + math.dist(tr, br)) / (2.0 * top_side)


def polygon_area(poly: List[Tuple[float, float]]) -> float:
    """Área por fórmula de Gauss (shoelace)."""
    n = len(poly)
    acc = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        acc += x1 * y2 - x2 * y1
    return abs(acc) / 2.0


# ── Gestión del pool de líneas ─────────────────────────────────────────────────

def add_positions(lines: List[Dict], img_w: int, img_h: int,
                  is_horizontal: bool) -> List[Dict]:
    """
    Añade clave 'pos' a cada línea.
    Horizontal → Y evaluada en x = img_w/2.
    Vertical   → X evaluada en y = img_h/2.
    """
    result = []
    for line in lines:
        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
        if is_horizontal:
            cx = img_w / 2.0
            pos = (y1 + (cx - x1) * (y2 - y1) / (x2 - x1)
                   if abs(x2 - x1) > 1e-6 else (y1 + y2) / 2.0)
        else:
            cy = img_h / 2.0
            pos = (x1 + (cy - y1) * (x2 - x1) / (y2 - y1)
                   if abs(y2 - y1) > 1e-6 else (x1 + x2) / 2.0)
        result.append({**line, "pos": pos})
    return result


def limit_lines(lines: List[Dict], max_n: int) -> List[Dict]:
    """
    Limita el pool a max_n líneas muestreando uniformemente para preservar cobertura.
    Requiere que las líneas ya tengan clave 'pos'.
    """
    if len(lines) <= max_n:
        return lines
    sorted_lines = sorted(lines, key=lambda l: l["pos"])
    n = len(sorted_lines)
    indices = {0, n - 1}
    for i in range(1, max_n - 1):
        indices.add(round(i * (n - 1) / (max_n - 1)))
    return [sorted_lines[i] for i in sorted(indices)]


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_candidate(
    proportion: float,
    area: float,
    target_proportion: float,
    target_area: float,
    img_area: float,
    prop_weight: float,
    area_weight: float,
) -> float:
    """
    Score compuesto (0-1, mayor es mejor).

    Con target_area > 0:  area_score = 1 - error_relativo_area
    Sin target_area (=0): area_score = area / img_area  (maximizar tamaño)
    """
    prop_err   = abs(proportion - target_proportion) / max(target_proportion, 1e-9)
    prop_score = max(0.0, 1.0 - prop_err)

    if target_area > 0:
        area_err   = abs(area - target_area) / max(target_area, 1.0)
        area_score = max(0.0, 1.0 - area_err)
    else:
        area_score = min(1.0, area / max(img_area, 1.0))

    return prop_weight * prop_score + area_weight * area_score


# ── Búsqueda exhaustiva ────────────────────────────────────────────────────────

def exhaustive_search(
    h_lines: List[Dict],
    v_lines: List[Dict],
    target_proportion: float,
    target_area: float,
    img_w: int,
    img_h: int,
    prop_weight: float,
    area_weight: float,
    top_k: int = 5,
    zone_top: float = 0.0,
    zone_bottom: float = 0.0,
    zone_left: float = 0.0,
    zone_right: float = 0.0,
    min_area_fraction: float = 0.0,
) -> List[Dict]:
    """
    Prueba todas las combinaciones (top, bottom) × (left, right) y devuelve
    los top_k candidatos ordenados por score descendente.

    Complejidad: O(Nh² × Nv²). Usar limit_lines antes para controlar el tamaño.

    Parámetros de zona (0 = desactivado):
      zone_top:    fracción superior de la imagen donde puede estar la línea top
      zone_bottom: fracción inferior de la imagen donde puede estar la línea bottom
      zone_left:   fracción lateral izquierda donde puede estar la línea left
      zone_right:  fracción lateral derecha donde puede estar la línea right

    min_area_fraction: fracción mínima de img_w*img_h que debe tener el polígono (0=desactivado).

    Cuando alguna zona está activa, el score se simplifica a solo proporción.
    """
    img_area   = float(img_w * img_h)
    use_zones  = zone_top > 0 or zone_bottom > 0 or zone_left > 0 or zone_right > 0
    min_area_px = min_area_fraction * img_area if min_area_fraction > 0 else 0.0
    candidates = []

    for h_top in h_lines:
        # Filtro de zona: la línea top solo puede estar en el X% superior
        if zone_top > 0 and h_top["pos"] >= img_h * zone_top:
            continue
        for h_bot in h_lines:
            if h_bot["pos"] <= h_top["pos"]:
                continue  # top debe tener Y menor que bottom
            # Filtro de zona: la línea bottom solo puede estar en el Y% inferior
            if zone_bottom > 0 and h_bot["pos"] <= img_h * (1 - zone_bottom):
                continue
            for v_left in v_lines:
                # Filtro de zona: la línea left solo puede estar en el Z% lateral izquierdo
                if zone_left > 0 and v_left["pos"] >= img_w * zone_left:
                    continue
                for v_right in v_lines:
                    if v_right["pos"] <= v_left["pos"]:
                        continue  # left debe tener X menor que right
                    # Filtro de zona: la línea right solo puede estar en el Z% lateral derecho
                    if zone_right > 0 and v_right["pos"] <= img_w * (1 - zone_right):
                        continue

                    poly = polygon_from_lines(h_top, h_bot, v_left, v_right)
                    if poly is None:
                        continue

                    prop = polygon_proportion(poly)
                    area = polygon_area(poly)
                    if prop is None or area < 1.0:
                        continue

                    # Piso mínimo de área para descartar polígonos degenerados
                    if min_area_px > 0 and area < min_area_px:
                        continue

                    # Score: solo proporción cuando zonas activas, compuesto si no
                    if use_zones:
                        prop_err = abs(prop - target_proportion) / max(target_proportion, 1e-9)
                        sc = max(0.0, 1.0 - prop_err)
                    else:
                        sc = score_candidate(prop, area, target_proportion, target_area,
                                             img_area, prop_weight, area_weight)

                    # Calcular errores individuales para ordenación lexicográfica
                    prop_err = abs(prop - target_proportion) / max(target_proportion, 1e-9)
                    if target_area > 0:
                        area_err = abs(area - target_area) / max(target_area, 1.0)
                    else:
                        # Sin target_area: area_err inversamente proporcional al tamaño
                        # (menor área_err = mayor área relativa)
                        area_err = 1.0 - min(1.0, area / max(img_area, 1.0))

                    candidates.append({
                        "score":      sc,
                        "proportion": round(prop, 6),
                        "area":       round(area, 1),
                        "area_err":   area_err,
                        "prop_err":   prop_err,
                        "top":        h_top,
                        "bottom":     h_bot,
                        "left":       v_left,
                        "right":      v_right,
                    })

    if use_zones:
        # Con zonas activas: ordenar solo por error de proporción
        candidates.sort(key=lambda c: c["prop_err"])
    elif target_area > 0:
        # Ordenación lexicográfica: primero por error de área, luego por error de proporción
        candidates.sort(key=lambda c: (c["area_err"], c["prop_err"]))
    else:
        # Sin target_area: mantener comportamiento original (score ponderado, maximizar tamaño)
        candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_k]


# ── Conversión de formato ──────────────────────────────────────────────────────

def candidate_to_border_lines(candidate: Dict) -> Dict:
    """Convierte el mejor candidato al formato border_lines de SelectBorderLines."""
    def clean(line):
        return {"x1": line["x1"], "y1": line["y1"],
                "x2": line["x2"], "y2": line["y2"],
                "angle": line.get("angle", 0.0)}
    return {
        "top":    clean(candidate["top"]),
        "bottom": clean(candidate["bottom"]),
        "left":   clean(candidate["left"]),
        "right":  clean(candidate["right"]),
    }


def extract_target_proportion(param_value: float,
                               proportion_data: Optional[Dict]) -> Optional[float]:
    """
    Determina el target_proportion a usar:
    1. Si param_value > 0, lo usa directamente.
    2. Si proportion_data tiene 'peak_proportion' o 'proportion', lo usa.
    3. Si no hay ninguno, retorna None.
    """
    if param_value > 0:
        return float(param_value)
    if proportion_data:
        for key in ("peak_proportion", "proportion"):
            val = proportion_data.get(key)
            if val is not None:
                try:
                    f = float(val)
                    if f > 0:
                        return f
                except (TypeError, ValueError):
                    pass
    return None


# ── Visualización compartida ───────────────────────────────────────────────────

def make_refinement_sample(
    base_image: np.ndarray,
    all_h_lines: List[Dict],
    all_v_lines: List[Dict],
    candidates: List[Dict],
    vis_w: int,
) -> np.ndarray:
    """
    Imagen de muestra que muestra:
    - Todas las líneas candidatas (tenues)
    - Las 4 líneas del mejor candidato (prominentes)
    - El polígono resultante
    - Score y proporción en el encabezado
    """
    import cv2

    orig_h, orig_w = base_image.shape[:2]
    vis_h = int(vis_w * orig_h / orig_w)
    sx = vis_w / orig_w
    sy = vis_h / orig_h

    if len(base_image.shape) == 2:
        vis = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        vis = base_image.copy()
    vis = cv2.resize(vis, (vis_w, vis_h))

    def scale_pt(x, y):
        return (int(x * sx), int(y * sy))

    def draw_line(img, line, color, thickness):
        p1 = scale_pt(line["x1"], line["y1"])
        p2 = scale_pt(line["x2"], line["y2"])
        cv2.line(img, p1, p2, color, thickness)

    # Líneas candidatas (tenues)
    for line in all_h_lines:
        draw_line(vis, line, (60, 60, 120), 1)
    for line in all_v_lines:
        draw_line(vis, line, (60, 120, 60), 1)

    if not candidates:
        cv2.putText(vis, "Sin candidatos", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis

    best = candidates[0]

    # Líneas del mejor candidato
    draw_line(vis, best["top"],    (0, 255, 255), 2)
    draw_line(vis, best["bottom"], (0, 200, 200), 2)
    draw_line(vis, best["left"],   (255, 128, 0), 2)
    draw_line(vis, best["right"],  (200, 100, 0), 2)

    # Polígono
    poly = polygon_from_lines(best["top"], best["bottom"],
                               best["left"], best["right"])
    if poly:
        pts = np.array([scale_pt(x, y) for x, y in poly], dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 230, 118), thickness=3)
        for pt in pts:
            cv2.circle(vis, tuple(pt), 6, (0, 230, 118), -1)

    # Encabezado con score y proporción
    label = f"score={best['score']:.3f}  prop={best['proportion']:.4f}  area={best['area']:.0f}"
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 118), 1)

    # Candidatos alternativos (más tenues)
    for alt in candidates[1:]:
        alt_poly = polygon_from_lines(alt["top"], alt["bottom"],
                                       alt["left"], alt["right"])
        if alt_poly:
            pts = np.array([scale_pt(x, y) for x, y in alt_poly], dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(80, 80, 80), thickness=1)

    return vis
