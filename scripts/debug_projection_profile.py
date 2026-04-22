"""
debug_projection_profile.py
Procesa una imagen con la cadena de preprocesado del pipeline 005 y guarda
el sample_image del filtro ProjectionProfileBorder (con histogramas H y V).

Uso:
    python3 scripts/debug_projection_profile.py <imagen.jpg> [--out <carpeta>]

Ejemplo:
    python3 scripts/debug_projection_profile.py __data/heraldo_raw_100/022_izquierda.jpg
"""

import sys
import argparse
from pathlib import Path

# Añadir src al path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import cv2
from filter_library import (
    ResizeFilter, BrightnessContrastFilter, GrayscaleFilter,
    DetectHistogramPeaks, NormalizeFromHistogram,
    GaussianBlurFilter, DenoiseNLMeans, CannyEdgeFilter,
    ProjectionProfileBorder,
)


def run(image_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    print(f"Cargando: {image_path}")
    original = cv2.imread(str(image_path))
    if original is None:
        print(f"ERROR: no se pudo cargar {image_path}")
        sys.exit(1)

    print(f"  Tamaño original: {original.shape[1]}×{original.shape[0]}")

    # --- Cadena de preprocesado (params de experimento 005) ---

    main_resize_f = ResizeFilter({"mode": 2, "width": 4000, "height": 480, "interpolation": 1})
    resized = main_resize_f.process({"input_image": original}, original)["resized_image"]
    print(f"  main_resize: {resized.shape[1]}×{resized.shape[0]}")

    resize2_f = ResizeFilter({"mode": 2, "scale_percent": 50, "width": 500, "height": 280, "interpolation": 1})
    resized2 = resize2_f.process({"input_image": resized}, original)["resized_image"]
    print(f"  resize_for_detect_borders: {resized2.shape[1]}×{resized2.shape[0]}")

    contrast_f = BrightnessContrastFilter({"brightness": 0, "contrast": 100})
    contrasted = contrast_f.process({"input_image": resized2}, original)["adjusted_image"]

    gray_f = GrayscaleFilter({"method": 0, "weight_r": 30, "weight_g": 59, "weight_b": 11})
    grayed = gray_f.process({"input_image": contrasted}, original)["grayscale_image"]

    histo_f = DetectHistogramPeaks({
        "dark_zone_max": 120, "light_zone_min": 120,
        "visualization_mode": 2, "histogram_height": 300,
        "pixel_highlight_mode": 1, "show_counts": 1,
    })
    histo_out = histo_f.process({"input_image": contrasted}, original)
    histo_data = histo_out["histogram_data"]

    norm_f = NormalizeFromHistogram({
        "dark_target": 0, "min_target": 20, "light_target": 180,
        "use_piecewise": 1, "show_comparison": 1, "show_mapping_info": 1,
    })
    normalized = norm_f.process(
        {"input_image": contrasted, "histogram_data": histo_data}, original
    )["normalized_image"]

    blur_f = GaussianBlurFilter({"kernel_size": 1, "sigma": 0})
    blurred = blur_f.process({"input_image": normalized}, original)["blurred_image"]

    denoise_f = DenoiseNLMeans({
        "h": 10, "template_window_size": 7, "search_window_size": 21, "color_mode": 0,
    })
    denoised = denoise_f.process({"input_image": blurred}, original)["denoised_image"]

    canny_f = CannyEdgeFilter({"threshold1": 50, "threshold2": 185, "aperture_size": 3, "l2_gradient": 0})
    edge_image = canny_f.process({"input_image": denoised}, original)["edge_image"]

    # --- ProjectionProfileBorder ---
    ppb_f = ProjectionProfileBorder({
        "search_zone_h": 0.45,
        "search_zone_v": 0.45,
        "use_phase2": 1,
        "use_prewitt": 0,
        "visualization_size": 1200,
    })
    ppb_out = ppb_f.process({"edge_image": edge_image, "base_image": denoised}, original)

    # Guardar sample_image (histogramas + imagen con rectángulo detectado)
    sample = ppb_out.get("sample_image")
    if sample is not None:
        out_path = out_dir / f"{stem}_ppb_histograms.png"
        cv2.imwrite(str(out_path), sample)
        print(f"  Guardado: {out_path}")

    # También guardar la imagen Canny para referencia
    canny_out = out_dir / f"{stem}_canny.png"
    cv2.imwrite(str(canny_out), edge_image)
    print(f"  Guardado: {canny_out}")

    # Imprimir resultado
    meta = ppb_out.get("selection_metadata", {})
    print(f"\n  Resultado ProjectionProfileBorder:")
    print(f"    top={meta.get('top')}  bottom={meta.get('bottom')}")
    print(f"    left={meta.get('left')}  right={meta.get('right')}")
    print(f"    phase2_applied={meta.get('phase2_applied')}  side={meta.get('phase2_side')}")
    print(f"    imagen={meta.get('image_width')}×{meta.get('image_height')}")


def main():
    parser = argparse.ArgumentParser(description="Debug ProjectionProfileBorder en una imagen")
    parser.add_argument("image", help="Ruta a la imagen de entrada")
    parser.add_argument("--out", default="__data/debug_ppb", help="Carpeta de salida (default: __data/debug_ppb)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = _ROOT / image_path

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir

    run(image_path, out_dir)


if __name__ == "__main__":
    main()
