"""
Core - Componentes centrales del sistema Copista Pipeline
==========================================================

Contiene clases y utilidades compartidas entre diferentes scripts.
"""

from .pipeline_classes import PipelineProcessor, ImageBrowser, CacheManager

__all__ = [
    'PipelineProcessor',
    'ImageBrowser',
    'CacheManager'
]

__version__ = '1.0.0'
