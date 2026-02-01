"""
Filtro: BaseFilter
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from abc import ABC, abstractmethod


# Registro global de filtros
FILTER_REGISTRY = {}


class BaseFilter(ABC):
    """Clase base para todos los filtros"""
    
    FILTER_NAME: str = "base"
    DESCRIPTION: str = "Filtro base"
    INPUTS: Dict[str, str] = {}  # {nombre: tipo}
    OUTPUTS: Dict[str, str] = {"sample_image": "image"}  # siempre debe incluir sample_image
    PARAMS: Dict[str, Dict] = {}  # {nombre: {default, min, max, step, description}}
    
    def __init_subclass__(cls, **kwargs):
        """Registra automáticamente cada subclase"""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'FILTER_NAME') and cls.FILTER_NAME != "base":
            FILTER_REGISTRY[cls.FILTER_NAME] = cls
    
    def __init__(self, params: Dict[str, Any] = None, without_preview: bool = False):
        """
        Inicializa el filtro con parámetros
        
        Args:
            params: Diccionario con valores de parámetros del filtro
            without_preview: Si True, el filtro puede omitir la generación de sample_image
                           (por defecto False, para compatibilidad con param_configurator.py)
        """
        self.without_preview = without_preview
        self.params = {}
        # Cargar defaults
        for name, config in self.PARAMS.items():
            self.params[name] = config.get('default', 0)
        # Sobrescribir con params proporcionados
        if params:
            for name, value in params.items():
                if name in self.PARAMS:
                    self.params[name] = value
    
    def set_param(self, name: str, value: Any):
        """Establece un parámetro"""
        if name in self.PARAMS:
            self.params[name] = value
    
    def get_param(self, name: str) -> Any:
        """Obtiene un parámetro"""
        return self.params.get(name)
    
    def get_params(self) -> Dict[str, Any]:
        """Obtiene todos los parámetros"""
        return self.params.copy()
    
    def get_help(self) -> str:
        """Retorna ayuda sobre los parámetros del filtro"""
        lines = [
            f"\n{'='*60}",
            f"FILTRO: {self.FILTER_NAME}",
            f"{'='*60}",
            f"Descripción: {self.DESCRIPTION}",
            f"\nEntradas requeridas:",
        ]
        if self.INPUTS:
            for name, dtype in self.INPUTS.items():
                lines.append(f"  - {name}: {dtype}")
        else:
            lines.append("  (ninguna - usa imagen original)")
        
        lines.append(f"\nSalidas producidas:")
        for name, dtype in self.OUTPUTS.items():
            lines.append(f"  - {name}: {dtype}")
        
        lines.append(f"\nParámetros:")
        if self.PARAMS:
            for name, config in self.PARAMS.items():
                lines.append(f"  [{name}]")
                lines.append(f"    Descripción: {config.get('description', 'Sin descripción')}")
                lines.append(f"    Default: {config.get('default', 'N/A')}")
                lines.append(f"    Rango: [{config.get('min', 'N/A')} - {config.get('max', 'N/A')}]")
                lines.append(f"    Step: {config.get('step', 1)}")
        else:
            lines.append("  (sin parámetros)")
        
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)
    
    @abstractmethod
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        """
        Procesa el filtro.
        
        Args:
            inputs: dict con los datos de entrada de otros filtros
            original_image: imagen original sin procesar
            
        Returns:
            dict con los outputs producidos. Debe incluir 'sample_image' a menos que
            self.without_preview sea True (en cuyo caso es opcional).
        """
        pass
