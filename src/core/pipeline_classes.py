"""
Pipeline Classes - Componentes centrales del sistema
====================================================

PipelineProcessor: Gestiona el pipeline de filtros
ImageBrowser: Maneja la navegación de imágenes en carpetas
"""

import json
import sys
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict

import cv2
import numpy as np

# Importar biblioteca de filtros
from filter_library import FILTER_REGISTRY, get_filter, BaseFilter

class ImageBrowser:
    """Maneja la navegación de imágenes"""
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.images: List[Path] = []
        self.current_index = 0
        
        self.scan_folder()
    
    def scan_folder(self):
        """Escanea el folder buscando imágenes"""
        self.images = []
        
        if not self.folder_path.exists():
            print(f"ERROR: Carpeta '{self.folder_path}' no existe")
            return
        
        for file in sorted(self.folder_path.iterdir()):
            if file.is_file() and file.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                self.images.append(file)
        
        print(f"Encontradas {len(self.images)} imágenes en {self.folder_path}")
    
    def get_current_image(self) -> Optional[np.ndarray]:
        """Carga y retorna la imagen actual"""
        if not self.images:
            return None
        
        img_path = self.images[self.current_index]
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"ERROR: No se pudo cargar {img_path}")
        
        return img
    
    def get_current_path(self) -> str:
        """Retorna el path de la imagen actual"""
        if not self.images:
            return "Sin imágenes"
        return str(self.images[self.current_index])
    
    def get_current_name(self) -> str:
        """Retorna el nombre de la imagen actual"""
        if not self.images:
            return "Sin imágenes"
        return self.images[self.current_index].name
    
    def next_image(self):
        """Avanza a la siguiente imagen"""
        if self.images:
            self.current_index = (self.current_index + 1) % len(self.images)
    
    def prev_image(self):
        """Retrocede a la imagen anterior"""
        if self.images:
            self.current_index = (self.current_index - 1) % len(self.images)
    
    def get_image_count(self) -> int:
        """Retorna el número total de imágenes"""
        return len(self.images)
    
    def get_current_index(self) -> int:
        """Retorna el índice actual"""
        return self.current_index


class CacheManager:
    """Maneja el sistema de cache de filtros"""

    """
    NOTA: Esta clase por ahora solo la usa param_configurator.py, 
    la incluimos en core/pipeline_clasess.py porque 
    process_up_to_with_cache depende de CacheManager
    """
    
    # Tipos de output que se consideran "imagen" y pueden ser cacheados
    IMAGE_OUTPUT_TYPES = {"image"}
    
    def __init__(self, image_folder: Path, checkpoints_path: Path, processor=None):
        self.image_folder = image_folder
        self.cache_folder = image_folder / ".cache"
        self.checkpoints_path = checkpoints_path
        self.checkpoints: List[str] = []  # ← CAMBIO: Lista en lugar de string único
        self.processor = processor  # ← NUEVO: Necesitamos acceso al processor
        
        self.load_checkpoint()
    
    def set_processor(self, processor):
        """Establece el processor después de la inicialización"""
        self.processor = processor
    
    def load_checkpoint(self):
        """Carga la configuración de checkpoints desde JSON"""
        try:
            with open(self.checkpoints_path, 'r') as f:
                data = json.load(f, object_pairs_hook=OrderedDict)
                # ← CAMBIO: Cargar lista en lugar de string único
                self.checkpoints = data.get('checkpoints', [])
            if self.checkpoints:
                print(f"Checkpoints cargados: {', '.join(self.checkpoints)}")
        except FileNotFoundError:
            self.checkpoints = []
        except json.JSONDecodeError:
            self.checkpoints = []
    
    def save_checkpoint(self):
        """Guarda la configuración de checkpoints a JSON"""
        data = {
            "checkpoints": self.checkpoints,  # ← CAMBIO: Guardar lista
            "last_modified": datetime.now().isoformat()
        }
        with open(self.checkpoints_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    def toggle_checkpoint(self, filter_id: str) -> bool:
        """
        Agrega o quita un checkpoint de la lista.
        Retorna True si la operación fue exitosa.
        """
        if filter_id in self.checkpoints:
            # Quitar checkpoint
            self.checkpoints.remove(filter_id)
            self.save_checkpoint()
            print(f"Checkpoint removido: {filter_id}")
            return True
        else:
            # Agregar checkpoint - validar primero
            if self.processor is None:
                print("Error: Processor no disponible para validación")
                return False
            
            is_valid, error_msg = CacheManager.validate_checkpoint_filters(
                self.processor.pipeline, filter_id, self.processor.filter_order
            )
            
            if not is_valid:
                print(f"\n⚠️  No se puede establecer checkpoint aquí:")
                print(f"   {error_msg}")
                print(f"   Solo se puede hacer checkpoint en filtros donde todos los anteriores produzcan imágenes.\n")
                return False
            
            # Agregar y ordenar
            self.checkpoints.append(filter_id)
            # Ordenar por filter_order para mantener consistencia
            self.checkpoints.sort(key=lambda x: self.processor.filter_order.get(x, 999))
            self.save_checkpoint()
            print(f"Checkpoint agregado: {filter_id}")
            return True
    
    def get_last_checkpoint(self) -> Optional[str]:
        """
        Retorna el último checkpoint (el de mayor order).
        Este es el checkpoint crítico para la invalidación del cache.
        """
        if not self.checkpoints or self.processor is None:
            return None
        
        return max(self.checkpoints, 
                   key=lambda cp: self.processor.filter_order.get(cp, -1))
    
    def has_checkpoints(self) -> bool:
        """Verifica si hay checkpoints definidos"""
        return len(self.checkpoints) > 0
    
    def get_cache_path(self, filter_id: str, image_name: str) -> Path:
        """Obtiene el path del cache para una imagen y filtro específicos"""
        # Cambiar extensión a .png para el cache
        base_name = Path(image_name).stem + ".png"
        return self.cache_folder / filter_id / base_name
    
    def cache_exists(self, filter_id: str, image_name: str) -> bool:
        """Verifica si existe cache para una imagen y filtro"""
        cache_path = self.get_cache_path(filter_id, image_name)
        return cache_path.exists()
    
    def load_from_cache(self, filter_id: str, image_name: str) -> Optional[np.ndarray]:
        """Carga una imagen desde el cache"""
        cache_path = self.get_cache_path(filter_id, image_name)
        if cache_path.exists():
            img = cv2.imread(str(cache_path), cv2.IMREAD_UNCHANGED)
            return img
        return None
    
    def save_to_cache(self, filter_id: str, image_name: str, image: np.ndarray):
        """Guarda una imagen en el cache"""
        cache_path = self.get_cache_path(filter_id, image_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cache_path), image)
    
    def clear_filter_cache(self, filter_id: str):
        """Elimina todo el cache de un filtro específico"""
        cache_dir = self.cache_folder / filter_id
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Cache del filtro {filter_id} eliminado")
    
    def clear_all_cache(self):
        """Elimina todo el cache"""
        if self.cache_folder.exists():
            shutil.rmtree(self.cache_folder)
            print("Todo el cache ha sido eliminado")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Retorna estadísticas del cache"""
        stats = {}
        if self.cache_folder.exists():
            for filter_dir in self.cache_folder.iterdir():
                if filter_dir.is_dir():
                    count = len(list(filter_dir.glob('*')))
                    stats[filter_dir.name] = count
        return stats
    
    @staticmethod
    def filter_outputs_only_images(filter_class) -> bool:
        """
        Verifica si un filtro solo produce outputs de tipo imagen.
        Retorna True si todos los outputs son de tipo 'image'.
        """
        if filter_class is None:
            return False
        
        for output_name, output_type in filter_class.OUTPUTS.items():
            if output_type != "image":
                return False
        return True
    
    @staticmethod
    def validate_checkpoint_filters(pipeline: OrderedDict, checkpoint_id: str, 
                                   filter_order: Dict[str, int]) -> Tuple[bool, str]:
        """
        Valida que todos los filtros hasta el checkpoint (inclusive) solo produzcan imágenes.
        
        Returns:
            Tuple de (es_válido, mensaje_error)
        """
        checkpoint_order = filter_order.get(checkpoint_id, 999)
        
        for filter_id, filter_config in pipeline.items():
            if filter_order.get(filter_id, 999) > checkpoint_order:
                break
            
            filter_name = filter_config.get('filter_name')
            filter_class = get_filter(filter_name)
            
            if filter_class is None:
                return False, f"Filtro {filter_id} ({filter_name}) no encontrado"
            
            if not CacheManager.filter_outputs_only_images(filter_class):
                non_image_outputs = [
                    f"{name}:{dtype}" 
                    for name, dtype in filter_class.OUTPUTS.items() 
                    if dtype != "image"
                ]
                return False, f"Filtro {filter_id} ({filter_name}) produce datos no-imagen: {non_image_outputs}"
        
        return True, ""


class PipelineProcessor:
    """Procesa el pipeline de filtros"""
    
    def __init__(self, pipeline_path: str, params_path: str, without_preview: bool = False):
        self.pipeline_path = pipeline_path
        self.params_path = params_path
        self.without_preview = without_preview  # ← NUEVO
        self.pipeline: OrderedDict = OrderedDict()
        self.saved_params: Dict = {}
        self.filter_instances: Dict[str, BaseFilter] = {}
        self.filter_outputs: Dict[str, Dict[str, Any]] = {}
        self.filter_order: Dict[str, int] = {}  # Mapeo filter_id -> orden
        
        self.load_pipeline()
        self.load_params()
        self.instantiate_filters()
    
    def load_pipeline(self):
        """Carga el pipeline desde JSON preservando orden"""
        try:
            with open(self.pipeline_path, 'r') as f:
                data = json.load(f, object_pairs_hook=OrderedDict)
                self.pipeline = data.get('filters', OrderedDict())
            
            # Asignar orden implícito basado en posición
            self.filter_order = {filter_id: i 
                               for i, filter_id in enumerate(self.pipeline.keys())}
            
            print(f"Pipeline cargado: {len(self.pipeline)} filtros")
        except FileNotFoundError:
            print(f"ERROR: No se encontró {self.pipeline_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON inválido en pipeline: {e}")
            sys.exit(1)
    
    def load_params(self):
        """Carga parámetros guardados desde JSON"""
        try:
            with open(self.params_path, 'r') as f:
                data = json.load(f, object_pairs_hook=OrderedDict)
                self.saved_params = data.get('filter_params', {})
            print(f"Parámetros cargados desde {self.params_path}")
        except FileNotFoundError:
            print(f"Archivo de parámetros no encontrado, usando defaults")
            self.saved_params = {}
        except json.JSONDecodeError as e:
            print(f"ADVERTENCIA: JSON inválido en params: {e}, usando defaults")
            self.saved_params = {}
    
    def save_params(self):
        """Guarda los parámetros actuales a JSON"""
        params_data = {
            "version": "1.0",
            "description": "Parámetros guardados para los filtros del pipeline",
            "last_modified": datetime.now().isoformat(),
            "filter_params": {}
        }
        
        for filter_id, instance in self.filter_instances.items():
            params_data["filter_params"][filter_id] = {
                "filter_name": instance.FILTER_NAME,
                "params": instance.get_params()
            }
        
        with open(self.params_path, 'w') as f:
            json.dump(params_data, f, indent=4)
        
        print(f"Parámetros guardados en {self.params_path}")
    
    def instantiate_filters(self):
        """Crea instancias de todos los filtros del pipeline"""
        self.filter_instances = {}
        
        for filter_id in self.get_sorted_ids():
            filter_config = self.pipeline[filter_id]
            filter_name = filter_config['filter_name']
            filter_class = get_filter(filter_name)
            
            if filter_class is None:
                print(f"ERROR: Filtro '{filter_name}' no encontrado en biblioteca")
                continue
            
            # Cargar parámetros guardados o usar defaults
            params = None
            if filter_id in self.saved_params:
                params = self.saved_params[filter_id].get('params', {})

            self.filter_instances[filter_id] = filter_class(params, without_preview=self.without_preview)
            print(f"  Filtro {filter_id}: {filter_name} instanciado")
    
    def validate_pipeline(self) -> List[str]:
        """Valida que el pipeline tenga conexiones correctas"""
        errors = []
        
        for filter_id, filter_config in self.pipeline.items():
            filter_name = filter_config['filter_name']
            filter_class = get_filter(filter_name)
            
            if filter_class is None:
                errors.append(f"Filtro {filter_id}: '{filter_name}' no existe")
                continue
            
            inputs_config = filter_config.get('inputs', {})
            
            for input_name, source in inputs_config.items():
                # Validar formato "filter_id.output_name"
                if '.' not in source:
                    errors.append(f"Filtro {filter_id}: formato inválido '{source}', debe ser 'filter_id.output_name'")
                    continue
                
                source_id, output_name = source.split('.', 1)
                
                # Permitir referencia especial a imagen original
                if source_id == "original":
                    continue  # Válido, es la palabra reservada para imagen original
                
                # Validar que el filtro fuente existe
                if source_id not in self.pipeline:
                    errors.append(f"Filtro {filter_id}: referencia a filtro inexistente '{source_id}'")
                    continue
                
                # Validar que el filtro fuente está antes en el orden
                if self.filter_order.get(source_id, 999) >= self.filter_order.get(filter_id, 0):
                    errors.append(f"Filtro {filter_id}: referencia a filtro posterior '{source_id}'")
                
                # Validar que el filtro fuente produce ese output
                source_filter = get_filter(self.pipeline[source_id]['filter_name'])
                if source_filter and output_name not in source_filter.OUTPUTS:
                    errors.append(f"Filtro {filter_id}: filtro {source_id} no produce '{output_name}'")
        
        return errors
    
    def process_filter(self, filter_id: str, original_image: np.ndarray) -> Dict[str, Any]:
        """Procesa un filtro específico y retorna sus outputs"""
        if filter_id not in self.filter_instances:
            return {}
        
        filter_instance = self.filter_instances[filter_id]
        filter_config = self.pipeline[filter_id]
        inputs_config = filter_config.get('inputs', {})
        
        # Recolectar inputs de filtros anteriores
        inputs = {}
        for input_name, source in inputs_config.items():
            source_id, output_name = source.split('.', 1)
            
            # Manejar referencia a imagen original
            if source_id == "original":
                inputs[input_name] = original_image
            elif source_id in self.filter_outputs:
                inputs[input_name] = self.filter_outputs[source_id].get(output_name)
        
        # Procesar el filtro
        outputs = filter_instance.process(inputs, original_image)
        self.filter_outputs[filter_id] = outputs
        
        return outputs

    def process_up_to(self, target_id: str, original_image: np.ndarray) -> Dict[str, Any]:
        """
        Procesa todos los filtros hasta el ID especificado (sin cache).
        Versión simplificada sin cache para batch processing.
        
        Args:
            target_id: ID del filtro objetivo
            original_image: Imagen original
        
        Returns:
            Outputs del filtro objetivo
        """
        self.filter_outputs = {}
        
        sorted_ids = self.get_sorted_ids()
        target_order = self.filter_order.get(target_id, 999)
        
        for filter_id in sorted_ids:
            if self.filter_order.get(filter_id, 999) > target_order:
                break
            self.process_filter(filter_id, original_image)
        
        return self.filter_outputs.get(target_id, {})    

    def process_up_to_with_cache(self, target_id: str, original_image: np.ndarray,
                                  cache_manager: CacheManager, image_name: str,
                                  ignore_cache: bool = False) -> Dict[str, Any]:
        """
        Procesa todos los filtros hasta el ID especificado, usando cache si está disponible.
        VERSIÓN MULTI-CHECKPOINT: Usa el checkpoint válido más cercano anterior al target.
        
        Args:
            target_id: ID del filtro objetivo
            original_image: Imagen original
            cache_manager: Gestor de cache
            image_name: Nombre de la imagen (para cache)
            ignore_cache: Si True, ignora el cache aunque exista
        
        Returns:
            Outputs del filtro objetivo
        """
        self.filter_outputs = {}
        
        sorted_ids = self.get_sorted_ids()
        target_order = self.filter_order.get(target_id, 999)
        
        # Si no hay checkpoints o estamos ignorando cache, procesar todo normalmente
        if ignore_cache or not cache_manager.has_checkpoints():
            for filter_id in sorted_ids:
                if self.filter_order.get(filter_id, 999) > target_order:
                    break
                self.process_filter(filter_id, original_image)
                
                # Cachear si es checkpoint (solo si no estamos ignorando cache)
                if not ignore_cache and filter_id in cache_manager.checkpoints:
                    if not cache_manager.cache_exists(filter_id, image_name):
                        sample = self.filter_outputs.get(filter_id, {}).get('sample_image')
                        if sample is not None:
                            cache_manager.save_to_cache(filter_id, image_name, sample)
            
            return self.filter_outputs.get(target_id, {})
        
        # Encontrar checkpoints aplicables (anteriores al target y con cache disponible)
        applicable_checkpoints = [
            cp for cp in cache_manager.checkpoints
            if (self.filter_order.get(cp, 999) < target_order and
                cache_manager.cache_exists(cp, image_name))
        ]
        
        if not applicable_checkpoints:
            # No hay checkpoints con cache disponible, procesar desde el inicio
            for filter_id in sorted_ids:
                if self.filter_order.get(filter_id, 999) > target_order:
                    break
                self.process_filter(filter_id, original_image)
                
                # Cachear si es checkpoint
                if filter_id in cache_manager.checkpoints:
                    if not cache_manager.cache_exists(filter_id, image_name):
                        sample = self.filter_outputs.get(filter_id, {}).get('sample_image')
                        if sample is not None:
                            cache_manager.save_to_cache(filter_id, image_name, sample)
            
            return self.filter_outputs.get(target_id, {})
        
        # Usar el checkpoint más cercano (el último de la lista)
        start_checkpoint = applicable_checkpoints[-1]
        start_order = self.filter_order.get(start_checkpoint, -1)
        
        # Cargar desde cache todos los filtros hasta start_checkpoint (inclusive)
        for filter_id in sorted_ids:
            order = self.filter_order.get(filter_id, 999)
            if order > start_order:
                break
            
            # Solo cargar checkpoints que tengan cache
            if filter_id in cache_manager.checkpoints:
                cached_image = cache_manager.load_from_cache(filter_id, image_name)
                if cached_image is not None:
                    main_output_name = self._get_main_output_name(filter_id)
                    self.filter_outputs[filter_id] = {
                        "sample_image": cached_image,
                        main_output_name: cached_image
                    }
                else:
                    # Si un checkpoint no tiene cache, procesar desde aquí
                    self.process_filter(filter_id, original_image)
                    sample = self.filter_outputs.get(filter_id, {}).get('sample_image')
                    if sample is not None:
                        cache_manager.save_to_cache(filter_id, image_name, sample)
            else:
                # Filtro normal (no checkpoint), procesar normalmente
                self.process_filter(filter_id, original_image)
        
        # Procesar desde start_checkpoint+1 hasta target
        for filter_id in sorted_ids:
            order = self.filter_order.get(filter_id, 999)
            if order <= start_order:
                continue
            if order > target_order:
                break
            
            self.process_filter(filter_id, original_image)
            
            # Cachear si es checkpoint
            if filter_id in cache_manager.checkpoints:
                if not cache_manager.cache_exists(filter_id, image_name):
                    sample = self.filter_outputs.get(filter_id, {}).get('sample_image')
                    if sample is not None:
                        cache_manager.save_to_cache(filter_id, image_name, sample)
        
        return self.filter_outputs.get(target_id, {})
    
    def _get_main_output_name(self, filter_id: str) -> str:
        """Obtiene el nombre del output principal de un filtro (excluyendo sample_image)"""
        if filter_id not in self.pipeline:
            return "output"
        
        filter_name = self.pipeline[filter_id]['filter_name']
        filter_class = get_filter(filter_name)
        
        if filter_class:
            for output_name in filter_class.OUTPUTS:
                if output_name != "sample_image":
                    return output_name
        
        return "output"
    
    def get_filter_count(self) -> int:
        """Retorna el número de filtros en el pipeline"""
        return len(self.pipeline)
    
    def get_sorted_ids(self) -> List[str]:
        """Retorna los IDs de filtros ordenados"""
        return list(self.pipeline.keys())  # Ya están ordenados por OrderedDict
    
    def get_filter_instance(self, filter_id: str) -> Optional[BaseFilter]:
        """Obtiene la instancia de un filtro"""
        return self.filter_instances.get(filter_id)
    
    def get_filter_name(self, filter_id: str) -> str:
        """Obtiene el nombre de un filtro"""
        if filter_id in self.pipeline:
            return self.pipeline[filter_id].get('filter_name', 'Unknown')
        return 'Unknown'
    
    def get_filter_order(self, filter_id: str) -> int:
        """Obtiene el orden de un filtro"""
        return self.filter_order.get(filter_id, 999)
    
    def is_after(self, filter_id: str, reference_id: str) -> bool:
        """Verifica si un filtro está después de otro en el pipeline"""
        return self.get_filter_order(filter_id) > self.get_filter_order(reference_id)


