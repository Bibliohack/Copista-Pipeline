"""
Filtro: LoadPolygonFromDet

Carga el polígono de borde de página desde un archivo .det.json compañero
de la imagen actual. Reemplaza la rama de detección automática en el pipeline
de recorte cuando la detección ya fue realizada en un paso previo.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any

from .base_filter import BaseFilter, FILTER_REGISTRY
from .polygon_to_gt_format import CORNER_ORDER, CORNER_COLORS, POLYGON_COLOR


class LoadPolygonFromDet(BaseFilter):
    """
    Lee el .det.json correspondiente a la imagen actual y emite las esquinas
    en formato compatible con la salida de CalculateQuadCorners, listo para
    ser consumido por RotateQuadCorners u otros filtros del pipeline de recorte.

    El path de la imagen actual se recibe a través del atributo
    current_image_path inyectado por PipelineProcessor antes de cada llamada.
    """

    FILTER_NAME = "LoadPolygonFromDet"
    DESCRIPTION = (
        "Carga el polígono de borde de página desde un archivo .det.json compañero "
        "de la imagen actual. Reemplaza la rama de detección automática en el pipeline "
        "de recorte."
    )

    INPUTS = {
        "base_image": "image",   # solo para visualización y metadata de dimensiones
    }

    OUTPUTS = {
        "corners":          "quad_points",   # mismo formato que CalculateQuadCorners
        "corners_metadata": "metadata",      # con image_width, image_height, etc.
        "sample_image":     "image",
    }

    PARAMS = {
        "det_dir": {
            "default": "",
            "description": (
                "Directorio donde buscar los .det.json. "
                "Vacío = mismo directorio que la imagen."
            ),
        },
        "visualization_size": {
            "default": 900,
            "min": 400,
            "max": 1920,
            "step": 100,
            "description": "Ancho de la imagen de muestra.",
        },
        "polygon_thickness": {
            "default": 3,
            "min": 1,
            "max": 8,
            "step": 1,
            "description": "Grosor del polígono en visualización.",
        },
    }

    # ── Proceso principal ──────────────────────────────────────────────────────

    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        base_image = inputs.get("base_image", original_image)
        h, w = base_image.shape[:2]

        # Localizar el archivo .det.json
        det_path = self._resolve_det_path()

        # Leer JSON
        try:
            with open(det_path, "r") as f:
                det_data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LoadPolygonFromDet: JSON inválido en {det_path}: {exc}"
            ) from exc

        # Extraer polígono — orden: top_left, top_right, bottom_right, bottom_left
        polygon = det_data.get("polygon", [])
        if len(polygon) < 4:
            raise ValueError(
                f"LoadPolygonFromDet: polígono incompleto en {det_path} "
                f"(se esperaban 4 puntos, hay {len(polygon)})"
            )

        # Escalar desde el espacio del .det.json (imagen original) al espacio de base_image
        det_w = det_data.get("image_width", w) or w
        det_h = det_data.get("image_height", h) or h
        scale_x = w / det_w
        scale_y = h / det_h

        # Construir corners dict con el tipo canónico
        corners = {}
        for i, name in enumerate(CORNER_ORDER):
            pt = polygon[i]
            if pt is not None and "x" in pt and "y" in pt:
                corners[name] = {
                    "x": float(pt["x"]) * scale_x,
                    "y": float(pt["y"]) * scale_y,
                    "type": "loaded_from_det",
                }
            else:
                corners[name] = None

        corners_metadata = {
            "image_width":  int(w),
            "image_height": int(h),
            "loaded_from":  str(det_path),
            "source":       "det_file",
        }

        result = {
            "corners":          corners,
            "corners_metadata": corners_metadata,
        }

        # Visualización
        if not self.without_preview:
            result["sample_image"] = self._make_sample(
                base_image, corners, det_path
            )

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_det_path(self) -> Path:
        """
        Determina la ruta del .det.json a partir de current_image_path
        y el parámetro det_dir.
        """
        img_path_str = self.current_image_path
        if not img_path_str:
            raise RuntimeError(
                "LoadPolygonFromDet: current_image_path no está configurado. "
                "El PipelineProcessor debe inyectarlo antes de llamar al filtro."
            )

        img_path = Path(img_path_str)
        stem = img_path.stem

        det_dir_str = str(self.params.get("det_dir", "")).strip()
        if det_dir_str:
            det_dir = Path(det_dir_str)
        else:
            det_dir = img_path.parent

        det_path = det_dir / f"{stem}.det.json"

        if not det_path.exists():
            raise FileNotFoundError(
                f"LoadPolygonFromDet: no se encontró el archivo de detección: {det_path}"
            )

        return det_path

    def _make_sample(
        self,
        base_image: np.ndarray,
        corners: Dict[str, Any],
        det_path: Path,
    ) -> np.ndarray:
        """Genera imagen de visualización con el polígono cargado."""
        vis_w = int(self.params["visualization_size"])
        thickness = int(self.params["polygon_thickness"])

        orig_h, orig_w = base_image.shape[:2]
        vis_h = int(vis_w * orig_h / orig_w)
        sx = vis_w / orig_w
        sy = vis_h / orig_h

        # Imagen de fondo
        if len(base_image.shape) == 2:
            vis = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        else:
            vis = base_image.copy()
        vis = cv2.resize(vis, (vis_w, vis_h))

        # Proyectar puntos al espacio de visualización
        vis_pts = {}
        for name in CORNER_ORDER:
            corner = corners.get(name)
            if corner is not None:
                vis_pts[name] = (
                    int(corner["x"] * sx),
                    int(corner["y"] * sy),
                )
            else:
                vis_pts[name] = None

        # Dibujar polígono si están las 4 esquinas
        valid = [vis_pts[n] for n in CORNER_ORDER if vis_pts[n] is not None]
        if len(valid) == 4:
            arr = np.array(valid, dtype=np.int32)
            cv2.polylines(vis, [arr], isClosed=True, color=POLYGON_COLOR, thickness=thickness)

        # Dibujar cada esquina con su color y etiqueta
        r = 8
        for name in CORNER_ORDER:
            pt = vis_pts.get(name)
            if pt is None:
                continue
            color = CORNER_COLORS[name]
            cv2.circle(vis, pt, r, color, -1)
            cv2.circle(vis, pt, r + 2, (0, 0, 0), 2)

            label = name
            tx = pt[0] + r + 4
            ty = pt[1] - r
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
            cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Nombre del archivo .det.json en esquina superior
        det_label = det_path.name
        cv2.putText(vis, det_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, det_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 118), 2)

        return vis
