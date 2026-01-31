"""
Configurador GUI de Parámetros de Filtros
=========================================

IMPORTANTE: Esta versión usa IDs de filtros en lugar de índices numéricos.
Los filtros se ordenan por su posición en pipeline.json.

Uso:
    python param_configurator.py [ruta_imagenes] [--clear-cache]

Opciones:
    --clear-cache   Borra todo el cache antes de iniciar

Controles:
    a/d         - Imagen anterior/siguiente
    ESPACIO     - Avanzar al siguiente filtro
    BACKSPACE   - Retroceder al filtro anterior
    UP/DOWN     - Navegar parámetros del filtro actual
    LEFT/RIGHT  - Decrementar/incrementar valor del parámetro seleccionado
    PgUp/PgDown - Cambiar qué filtro estamos editando (sin cambiar visualización)
    c           - Marcar/desmarcar filtro actual como checkpoint
    s           - Guardar parámetros en params.json
    r           - Recargar parámetros desde params.json
    h           - Mostrar ayuda del filtro actual
    q/ESC       - Salir
"""

import cv2
import numpy as np
import json
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict

from filter_library import FILTER_REGISTRY, get_filter, list_filters, BaseFilter

# Importar sincronizador para validar pipeline/params al inicio
try:
    from sync_pipeline_params import PipelineParamsSync
except ImportError:
    PipelineParamsSync = None
    print("⚠️  sync_pipeline_params.py no encontrado - no se validará sincronización")


def check_python_version():
    """Verifica que la versión de Python sea adecuada"""
    if sys.version_info < (3, 7):
        print("="*60)
        print("⚠️  ADVERTENCIA: Python < 3.7 detectado")
        print(f"   Versión actual: {sys.version}")
        print("   El orden de filtros podría no preservarse correctamente.")
        print("   Se recomienda Python 3.7+")
        print("="*60)
        input("Presiona ENTER para continuar de todos modos...")


class CacheManager:
    """Maneja el sistema de cache de filtros"""
    
    # Tipos de output que se consideran "imagen" y pueden ser cacheados
    IMAGE_OUTPUT_TYPES = {"image"}
    
    def __init__(self, image_folder: Path, checkpoint_path: Path):
        self.image_folder = image_folder
        self.cache_folder = image_folder / ".cache"
        self.checkpoint_path = checkpoint_path
        self.checkpoint_filter: Optional[str] = None
        
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Carga la configuración de checkpoint desde JSON"""
        try:
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f, object_pairs_hook=OrderedDict)
                self.checkpoint_filter = data.get('checkpoint_filter')
            if self.checkpoint_filter:
                print(f"Checkpoint cargado: filtro {self.checkpoint_filter}")
        except FileNotFoundError:
            self.checkpoint_filter = None
        except json.JSONDecodeError:
            self.checkpoint_filter = None
    
    def save_checkpoint(self):
        """Guarda la configuración de checkpoint a JSON"""
        data = {
            "checkpoint_filter": self.checkpoint_filter,
            "last_modified": datetime.now().isoformat()
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    def set_checkpoint(self, filter_id: Optional[str]):
        """Establece o elimina el checkpoint"""
        old_checkpoint = self.checkpoint_filter
        self.checkpoint_filter = filter_id
        self.save_checkpoint()
        
        # Si había un checkpoint anterior diferente, eliminar todo el cache
        if old_checkpoint and old_checkpoint != filter_id:
            self.clear_all_cache()
            print(f"Cache anterior eliminado (checkpoint cambió de {old_checkpoint} a {filter_id})")
        
        if filter_id:
            print(f"Checkpoint establecido en filtro {filter_id}")
        else:
            print("Checkpoint eliminado")
    
    def get_checkpoint(self) -> Optional[str]:
        """Retorna el filtro checkpoint actual"""
        return self.checkpoint_filter
    
    def has_checkpoint(self) -> bool:
        """Verifica si hay un checkpoint definido"""
        return self.checkpoint_filter is not None
    
    def get_cache_path(self, filter_id: str, image_name: str) -> Path:
        """Obtiene el path del cache para una imagen y filtro específicos"""
        # Cambiar extensión a .png para el cache
        base_name = Path(image_name).stem + ".png"
        return self.cache_folder / filter_id / base_name
    
    def cache_exists(self, filter_id: str, image_name: str) -> bool:
        """Verifica si existe cache para una imagen y filtro"""
        cache_path = self.get_cache_path(filter_id, image_name)
        return cache_path.exists()
    
    def all_cache_exists_up_to(self, checkpoint_id: str, image_name: str, 
                               sorted_ids: List[str], filter_order: Dict[str, int]) -> bool:
        """Verifica si existe cache para todos los filtros hasta el checkpoint (inclusive)"""
        checkpoint_order = filter_order.get(checkpoint_id, 999)
        for filter_id in sorted_ids:
            if filter_order.get(filter_id, 999) > checkpoint_order:
                break
            if not self.cache_exists(filter_id, image_name):
                return False
        return True
    
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
    
    def __init__(self, pipeline_path: str, params_path: str):
        self.pipeline_path = pipeline_path
        self.params_path = params_path
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
            
            self.filter_instances[filter_id] = filter_class(params)
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
    
    def process_up_to_with_cache(self, target_id: str, original_image: np.ndarray,
                                  cache_manager: CacheManager, image_name: str,
                                  ignore_cache: bool = False) -> Dict[str, Any]:
        """
        Procesa todos los filtros hasta el ID especificado, usando cache si está disponible.
        
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
        checkpoint = cache_manager.get_checkpoint()
        checkpoint_order = self.filter_order.get(checkpoint, -1) if checkpoint else -1
        
        # Determinar si podemos usar cache
        use_cache = (
            checkpoint is not None and
            not ignore_cache and
            target_order >= checkpoint_order and
            cache_manager.all_cache_exists_up_to(checkpoint, image_name, sorted_ids, self.filter_order)
        )
        
        if use_cache:
            # Cargar desde cache TODOS los filtros hasta el checkpoint
            all_loaded = True
            for filter_id in sorted_ids:
                if self.filter_order.get(filter_id, 999) > checkpoint_order:
                    break
                
                cached_image = cache_manager.load_from_cache(filter_id, image_name)
                if cached_image is not None:
                    # Crear outputs desde cache
                    main_output_name = self._get_main_output_name(filter_id)
                    self.filter_outputs[filter_id] = {
                        "sample_image": cached_image,
                        main_output_name: cached_image
                    }
                else:
                    all_loaded = False
                    break
            
            if all_loaded:
                # Procesar solo los filtros después del checkpoint
                for filter_id in sorted_ids:
                    if self.filter_order.get(filter_id, 999) <= checkpoint_order:
                        continue
                    if self.filter_order.get(filter_id, 999) > target_order:
                        break
                    self.process_filter(filter_id, original_image)
                
                return self.filter_outputs.get(target_id, {})
        
        # Procesar normalmente (sin cache o cache no disponible)
        for filter_id in sorted_ids:
            if self.filter_order.get(filter_id, 999) > target_order:
                break
            self.process_filter(filter_id, original_image)
            
            # Si estamos dentro del rango del checkpoint y no estamos ignorando cache,
            # guardar cache para cada filtro hasta el checkpoint
            if checkpoint is not None and not ignore_cache:
                if self.filter_order.get(filter_id, 999) <= checkpoint_order:
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


class ParamConfigurator:
    """Interfaz GUI principal"""
    
    WINDOW_RESULT = "Resultado del Filtro"
    WINDOW_PARAMS = "Parametros"
    
    def __init__(self, image_folder: str, pipeline_path: str = "pipeline.json", 
                 params_path: str = "params.json", checkpoint_path: str = "checkpoint.json",
                 clear_cache: bool = False):
        
        self.image_folder = Path(image_folder)
        self.browser = ImageBrowser(image_folder)
        self.processor = PipelineProcessor(pipeline_path, params_path)
        self.cache_manager = CacheManager(self.image_folder, Path(checkpoint_path))
        
        # Limpiar cache si se solicita
        if clear_cache:
            self.cache_manager.clear_all_cache()
        
        # Estado de navegación (índices en el array ordenado, NO IDs)
        self.current_view_filter = 0  # Índice en sorted_ids
        self.current_edit_filter = 0  # Índice en sorted_ids
        self.current_param_index = 0  # Índice del parámetro seleccionado
        
        # Estado de cache
        self.ignore_cache = False
        self.params_modified_before_checkpoint = False
        
        # Cache de imagen actual
        self.current_image = None
        self.needs_reprocess = True
        
        # Configuración de ventanas
        cv2.namedWindow(self.WINDOW_RESULT, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.WINDOW_PARAMS, cv2.WINDOW_NORMAL)
        
        # Validar pipeline
        errors = self.processor.validate_pipeline()
        if errors:
            print("\n⚠️  ADVERTENCIAS EN EL PIPELINE:")
            for err in errors:
                print(f"   - {err}")
            print()
    
    def get_current_view_id(self) -> Optional[str]:
        """Obtiene el ID del filtro que estamos visualizando"""
        sorted_ids = self.processor.get_sorted_ids()
        if 0 <= self.current_view_filter < len(sorted_ids):
            return sorted_ids[self.current_view_filter]
        return None
    
    def get_current_edit_id(self) -> Optional[str]:
        """Obtiene el ID del filtro que estamos editando"""
        sorted_ids = self.processor.get_sorted_ids()
        if 0 <= self.current_edit_filter < len(sorted_ids):
            return sorted_ids[self.current_edit_filter]
        return None
    
    def get_current_filter_params(self) -> List[Tuple[str, Dict]]:
        """Obtiene los parámetros del filtro que estamos editando"""
        edit_id = self.get_current_edit_id()
        if not edit_id:
            return []
        
        instance = self.processor.get_filter_instance(edit_id)
        if instance is None:
            return []
        
        return list(instance.PARAMS.items())
    
    def render_params_window(self):
        """Renderiza la ventana de parámetros"""
        sorted_ids = self.processor.get_sorted_ids()
        if not sorted_ids:
            return
        
        view_id = self.get_current_view_id()
        edit_id = self.get_current_edit_id()
        
        if not view_id or not edit_id:
            return
        
        view_instance = self.processor.get_filter_instance(view_id)
        edit_instance = self.processor.get_filter_instance(edit_id)
        
        # Crear imagen para parámetros
        param_img = np.zeros((550, 500, 3), dtype=np.uint8)
        param_img[:] = (40, 40, 40)
        
        y = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Título
        cv2.putText(param_img, "CONFIGURADOR DE FILTROS", (10, y), font, 0.6, (255, 255, 255), 1)
        y += 25
        
        # Info de imagen
        img_info = f"Imagen: {self.browser.get_current_name()} ({self.browser.get_current_index()+1}/{self.browser.get_image_count()})"
        cv2.putText(param_img, img_info, (10, y), font, 0.4, (200, 200, 200), 1)
        y += 20
        
        # Info de checkpoint
        checkpoint = self.cache_manager.get_checkpoint()
        if checkpoint:
            checkpoint_name = self.processor.get_filter_name(checkpoint)
            checkpoint_color = (0, 165, 255)
            if self.ignore_cache:
                checkpoint_text = f"Checkpoint: {checkpoint} ({checkpoint_name}) [IGNORADO]"
                checkpoint_color = (0, 100, 255)
            else:
                cache_complete = self.cache_manager.all_cache_exists_up_to(
                    checkpoint,
                    self.browser.get_current_name(),
                    sorted_ids,
                    self.processor.filter_order
                )
                cache_status = "CACHED" if cache_complete else "SIN CACHE"
                checkpoint_text = f"Checkpoint: {checkpoint} ({checkpoint_name}) [{cache_status}]"
            cv2.putText(param_img, checkpoint_text, (10, y), font, 0.4, checkpoint_color, 1)
        else:
            cv2.putText(param_img, "Checkpoint: (ninguno)", (10, y), font, 0.4, (150, 150, 150), 1)
        y += 20
        
        # Separador
        cv2.line(param_img, (10, y), (490, y), (100, 100, 100), 1)
        y += 15
        
        # Info de visualización
        view_name = self.processor.get_filter_name(view_id)
        is_checkpoint = (view_id == checkpoint)
        view_color = (0, 165, 255) if is_checkpoint else (0, 255, 255)
        view_suffix = " [CHECKPOINT]" if is_checkpoint else ""
        cv2.putText(param_img, f"Viendo: {view_id} ({view_name}){view_suffix}", (10, y), font, 0.5, view_color, 1)
        y += 20
        
        # Info de edición
        edit_name = self.processor.get_filter_name(edit_id)
        is_edit_checkpoint = (edit_id == checkpoint)
        edit_color = (0, 165, 255) if is_edit_checkpoint else (0, 255, 0)
        edit_suffix = " [CHECKPOINT]" if is_edit_checkpoint else ""
        cv2.putText(param_img, f"Editando: {edit_id} ({edit_name}){edit_suffix}", (10, y), font, 0.5, edit_color, 1)
        y += 25
        
        # Separador
        cv2.line(param_img, (10, y), (490, y), (100, 100, 100), 1)
        y += 15
        
        # Parámetros del filtro que estamos editando
        cv2.putText(param_img, "PARAMETROS:", (10, y), font, 0.5, (255, 255, 0), 1)
        y += 20
        
        params = self.get_current_filter_params()
        
        if not params:
            cv2.putText(param_img, "  (sin parametros)", (10, y), font, 0.4, (150, 150, 150), 1)
        else:
            for i, (param_name, config) in enumerate(params):
                is_selected = (i == self.current_param_index)
                
                if is_selected:
                    color = (0, 255, 0)
                    cv2.rectangle(param_img, (5, y-12), (495, y+5), (60, 60, 60), -1)
                else:
                    color = (200, 200, 200)
                
                current_value = edit_instance.get_param(param_name)
                min_val = config.get('min', 0)
                max_val = config.get('max', 100)
                
                text = f"  {param_name}: {current_value}"
                cv2.putText(param_img, text, (10, y), font, 0.45, color, 1)
                
                # Barra de progreso
                bar_x = 250
                bar_w = 200
                bar_h = 10
                
                cv2.rectangle(param_img, (bar_x, y-8), (bar_x + bar_w, y-8 + bar_h), (80, 80, 80), -1)
                
                if max_val != min_val:
                    progress = (current_value - min_val) / (max_val - min_val)
                    progress = max(0, min(1, progress))
                    fill_w = int(bar_w * progress)
                    cv2.rectangle(param_img, (bar_x, y-8), (bar_x + fill_w, y-8 + bar_h), color, -1)
                
                range_text = f"[{min_val}-{max_val}]"
                cv2.putText(param_img, range_text, (bar_x + bar_w + 10, y), font, 0.35, (150, 150, 150), 1)
                
                y += 22
        
        # Controles
        y = 420
        cv2.line(param_img, (10, y), (490, y), (100, 100, 100), 1)
        y += 20
        
        cv2.putText(param_img, "CONTROLES:", (10, y), font, 0.45, (255, 255, 0), 1)
        y += 18
        
        controls = [
            "a/d: Imagen ant/sig",
            "ESPACIO/BACKSPACE: Filtro ant/sig (vista)",
            "PgUp/PgDown: Filtro ant/sig (edicion)",
            "UP/DOWN: Param ant/sig | LEFT/RIGHT: Valor -/+",
            "c: Marcar/desmarcar checkpoint",
            "s: Guardar | r: Recargar | h: Ayuda | q: Salir"
        ]
        
        for ctrl in controls:
            cv2.putText(param_img, ctrl, (10, y), font, 0.35, (180, 180, 180), 1)
            y += 15
        
        if self.ignore_cache and self.params_modified_before_checkpoint:
            y += 10
            cv2.putText(param_img, "! Params modificados - cache ignorado", (10, y), font, 0.4, (0, 100, 255), 1)
        
        cv2.imshow(self.WINDOW_PARAMS, param_img)
    
    def process_and_display(self):
        """Procesa el pipeline y muestra el resultado"""
        if self.current_image is None:
            self.current_image = self.browser.get_current_image()
        
        if self.current_image is None:
            print("No hay imagen para procesar")
            return
        
        view_id = self.get_current_view_id()
        if not view_id:
            print("No hay filtros en el pipeline")
            return
        
        image_name = self.browser.get_current_name()
        
        outputs = self.processor.process_up_to_with_cache(
            view_id,
            self.current_image,
            self.cache_manager,
            image_name,
            self.ignore_cache
        )
        
        sample = outputs.get('sample_image')
        
        if sample is not None:
            display = sample.copy()
            filter_name = self.processor.get_filter_name(view_id)
            
            h, w = display.shape[:2]
            
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
            
            checkpoint = self.cache_manager.get_checkpoint()
            checkpoint_indicator = ""
            if checkpoint is not None and self.processor.is_after(view_id, checkpoint) or view_id == checkpoint:
                if self.ignore_cache:
                    checkpoint_indicator = " [CACHE:IGNORADO]"
                elif self.cache_manager.cache_exists(view_id, image_name):
                    checkpoint_indicator = " [CACHED]"
                else:
                    checkpoint_indicator = " [CACHEANDO]"
            
            info_text = f"{view_id}: {filter_name}{checkpoint_indicator} | {self.browser.get_current_name()}"
            cv2.putText(display, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(self.WINDOW_RESULT, display)
        
        self.needs_reprocess = False
    
    def change_param_value(self, delta: int):
        """Cambia el valor del parámetro seleccionado"""
        params = self.get_current_filter_params()
        if not params or self.current_param_index >= len(params):
            return
        
        edit_id = self.get_current_edit_id()
        if not edit_id:
            return
        
        instance = self.processor.get_filter_instance(edit_id)
        
        param_name, config = params[self.current_param_index]
        current = instance.get_param(param_name)
        step = config.get('step', 1)
        min_val = config.get('min', 0)
        max_val = config.get('max', 100)
        
        new_value = current + (delta * step)
        new_value = max(min_val, min(max_val, new_value))
        
        if new_value != current:
            instance.set_param(param_name, new_value)
            self.needs_reprocess = True
            
            # Verificar si estamos modificando un filtro anterior o igual al checkpoint
            checkpoint = self.cache_manager.get_checkpoint()
            if checkpoint:
                if not self.processor.is_after(edit_id, checkpoint) or edit_id == checkpoint:
                    self.ignore_cache = True
                    self.params_modified_before_checkpoint = True
    
    def toggle_checkpoint(self):
        """Alterna el checkpoint en el filtro actualmente visualizado"""
        view_id = self.get_current_view_id()
        if not view_id:
            return
        
        current_checkpoint = self.cache_manager.get_checkpoint()
        
        if current_checkpoint == view_id:
            self.cache_manager.set_checkpoint(None)
        else:
            is_valid, error_msg = CacheManager.validate_checkpoint_filters(
                self.processor.pipeline, view_id, self.processor.filter_order
            )
            
            if not is_valid:
                print(f"\n⚠️  No se puede establecer checkpoint aquí:")
                print(f"   {error_msg}")
                print(f"   Solo se puede hacer checkpoint en filtros donde todos los anteriores produzcan imágenes.\n")
                return
            
            self.cache_manager.set_checkpoint(view_id)
        
        self.ignore_cache = False
        self.params_modified_before_checkpoint = False
    
    def save_params_with_warning(self) -> bool:
        """Guarda parámetros con advertencia si hay cambios que afectan al checkpoint"""
        if self.ignore_cache and self.params_modified_before_checkpoint:
            checkpoint = self.cache_manager.get_checkpoint()
            print("\n" + "="*60)
            print("⚠️  ADVERTENCIA")
            print("="*60)
            print(f"Has modificado parámetros que afectan al checkpoint ({checkpoint}).")
            print("Guardar los parámetros BORRARÁ todo el cache.")
            print("")
            response = input("¿Deseas continuar? (s/n): ").strip().lower()
            
            if response == 's':
                self.cache_manager.clear_all_cache()
                self.processor.save_params()
                self.ignore_cache = False
                self.params_modified_before_checkpoint = False
                print("✓ Parámetros guardados y cache borrado")
                return True
            else:
                print("Guardado cancelado. El cache permanece intacto.")
                return False
        else:
            self.processor.save_params()
            print("✓ Parámetros guardados")
            return True
    
    def run(self):
        """Loop principal de la aplicación"""
        print("\n" + "="*60)
        print("CONFIGURADOR DE FILTROS DE IMAGEN")
        print("="*60)
        print("Presiona 'h' para ver ayuda del filtro actual")
        print("Presiona 'c' para marcar/desmarcar checkpoint")
        print("Presiona 'q' o ESC para salir")
        print("="*60 + "\n")
        
        if self.browser.get_image_count() == 0:
            print("ERROR: No se encontraron imágenes. Saliendo...")
            return
        
        self.current_image = self.browser.get_current_image()
        
        while True:
            if self.needs_reprocess:
                self.process_and_display()
            
            self.render_params_window()
            
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            
            elif key == ord('a'):
                self.browser.prev_image()
                self.current_image = self.browser.get_current_image()
                self.needs_reprocess = True
            
            elif key == ord('d'):
                self.browser.next_image()
                self.current_image = self.browser.get_current_image()
                self.needs_reprocess = True
            
            elif key == 32:  # ESPACIO
                sorted_ids = self.processor.get_sorted_ids()
                if sorted_ids:
                    self.current_view_filter = min(self.current_view_filter + 1, len(sorted_ids) - 1)
                    self.needs_reprocess = True
            
            elif key == 8:  # BACKSPACE
                self.current_view_filter = max(0, self.current_view_filter - 1)
                self.needs_reprocess = True
            
            elif key == 82 or key == 0:  # UP
                params = self.get_current_filter_params()
                if params:
                    self.current_param_index = max(0, self.current_param_index - 1)
            
            elif key == 84 or key == 1:  # DOWN
                params = self.get_current_filter_params()
                if params:
                    self.current_param_index = min(len(params) - 1, self.current_param_index + 1)
            
            elif key == 81 or key == 2:  # LEFT
                self.change_param_value(-1)
            
            elif key == 83 or key == 3:  # RIGHT
                self.change_param_value(1)
            
            elif key == 85:  # PageUp
                self.current_edit_filter = max(0, self.current_edit_filter - 1)
                self.current_param_index = 0
            
            elif key == 86:  # PageDown
                sorted_ids = self.processor.get_sorted_ids()
                if sorted_ids:
                    max_edit = self.current_view_filter
                    self.current_edit_filter = min(self.current_edit_filter + 1, max_edit)
                    self.current_param_index = 0
            
            elif key == ord('c'):
                self.toggle_checkpoint()
                self.needs_reprocess = True
            
            elif key == ord('s'):
                self.save_params_with_warning()
            
            elif key == ord('r'):
                self.processor.load_params()
                self.processor.instantiate_filters()
                self.ignore_cache = False
                self.params_modified_before_checkpoint = False
                self.needs_reprocess = True
                print("✓ Parámetros recargados")
            
            elif key == ord('h'):
                edit_id = self.get_current_edit_id()
                if edit_id:
                    instance = self.processor.get_filter_instance(edit_id)
                    if instance:
                        print(instance.get_help())
        
        cv2.destroyAllWindows()


def main():
    check_python_version()
    
    image_folder = "."
    clear_cache = False
    
    args = sys.argv[1:]
    for arg in args:
        if arg == "--clear-cache":
            clear_cache = True
        elif not arg.startswith("-"):
            image_folder = arg
    
    script_dir = Path(__file__).parent
    pipeline_path = script_dir / "pipeline.json"
    params_path = script_dir / "params.json"
    checkpoint_path = script_dir / "checkpoint.json"
    
    print(f"Carpeta de imágenes: {image_folder}")
    print(f"Pipeline: {pipeline_path}")
    print(f"Parámetros: {params_path}")
    print(f"Checkpoint: {checkpoint_path}")
    if clear_cache:
        print("⚠️  Se borrará todo el cache al iniciar")
    
    # Validar sincronización
    if PipelineParamsSync is not None:
        print("\n🔍 Validando sincronización pipeline ↔ params...")
        sync = PipelineParamsSync(str(pipeline_path), str(params_path))
        
        if not sync.load_files():
            print("\n❌ Error al cargar archivos. Abortando.")
            sys.exit(1)
        
        sync.analyze()
        
        if not sync.validate_only():
            sys.exit(1)
        
        print("✅ Sincronización OK\n")
    
    configurator = ParamConfigurator(
        image_folder=image_folder,
        pipeline_path=str(pipeline_path),
        params_path=str(params_path),
        checkpoint_path=str(checkpoint_path),
        clear_cache=clear_cache
    )
    
    configurator.run()


if __name__ == "__main__":
    main()
