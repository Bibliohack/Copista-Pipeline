"""
Biblioteca de Filtros para el Sistema de Procesamiento de Imágenes
"""

from .base_filter import BaseFilter, FILTER_REGISTRY

from .resize_filter import ResizeFilter
from .grayscale_filter import GrayscaleFilter
from .gaussian_blur_filter import GaussianBlurFilter
from .canny_edge_filter import CannyEdgeFilter
from .threshold_filter import ThresholdFilter
from .histogram_filter import HistogramFilter
from .hough_lines_filter import HoughLinesFilter
from .morphology_filter import MorphologyFilter
from .contour_filter import ContourFilter
from .overlay_lines_filter import OverlayLinesFilter
from .brightness_contrast_filter import BrightnessContrastFilter
from .color_space_filter import ColorSpaceFilter
from .normalize_peaks import NormalizePeaks
from .min_arc_length import MinArcLength
from .denoise_nl_means import DenoiseNLMeans
from .threshold_advanced import ThresholdAdvanced
from .morphology_advanced import MorphologyAdvanced
from .contour_simplify import ContourSimplify
from .histogram_visualize import HistogramVisualize
from .classify_lines_by_angle import ClassifyLinesByAngle
from .select_border_lines import SelectBorderLines
from .calculate_quad_corners import CalculateQuadCorners


# Lista de todos los filtros disponibles
__all__ = [
    "BaseFilter",
    "FILTER_REGISTRY",
    "ResizeFilter",
    "GrayscaleFilter",
    "GaussianBlurFilter",
    "CannyEdgeFilter",
    "ThresholdFilter",
    "HistogramFilter",
    "HoughLinesFilter",
    "MorphologyFilter",
    "ContourFilter",
    "OverlayLinesFilter",
    "BrightnessContrastFilter",
    "ColorSpaceFilter",
    "NormalizePeaks",
    "MinArcLength",
    "DenoiseNLMeans",
    "ThresholdAdvanced",
    "MorphologyAdvanced",
    "ContourSimplify",
    "HistogramVisualize",
    "ClassifyLinesByAngle",
    "SelectBorderLines",
    "CalculateQuadCorners",
]


def get_filter(name: str) -> type:
    """Obtiene una clase de filtro por nombre"""
    return FILTER_REGISTRY.get(name)


def list_filters() -> list:
    """Lista todos los filtros disponibles"""
    return list(FILTER_REGISTRY.keys())


def get_filter_info(name: str) -> dict:
    """Obtiene información sobre un filtro"""
    filter_class = FILTER_REGISTRY.get(name)
    if filter_class:
        return {
            "name": filter_class.FILTER_NAME,
            "description": filter_class.DESCRIPTION,
            "inputs": filter_class.INPUTS,
            "outputs": filter_class.OUTPUTS,
            "params": filter_class.PARAMS
        }
    return None
