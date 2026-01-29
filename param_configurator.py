"""
Configurador GUI de Parámetros de Filtros
=========================================

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

from filter_library import FILTER_REGISTRY, get_filter, list_filters, BaseFilter

# Importar sincronizador para validar pipeline/params al inicio
try:
    from sync_pipeline_params import PipelineParamsSync
except ImportError:
    PipelineParamsSync = None
    print("⚠️  sync_pipeline_params.py no encontrado - no se validará sincronización")


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
                data = json.load(f)
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
    
    def set_checkpoint(self, filter_idx: Optional[str]):
        """Establece o elimina el checkpoint"""
        old_checkpoint = self.checkpoint_filter
        self.checkpoint_filter = filter_idx
        self.save_checkpoint()
        
        # Si había un checkpoint anterior diferente, eliminar todo el cache
        if old_checkpoint and old_checkpoint != filter_idx:
            self.clear_all_cache()
            print(f"Cache anterior eliminado (checkpoint cambió de {old_checkpoint} a {filter_idx})")
        
        if filter_idx:
            print(f"Checkpoint establecido en filtro {filter_idx}")
        else:
            print("Checkpoint eliminado")
    
    def get_checkpoint(self) -> Optional[str]:
        """Retorna el filtro checkpoint actual"""
        return self.checkpoint_filter
    
    def has_checkpoint(self) -> bool:
        """Verifica si hay un checkpoint definido"""
        return self.checkpoint_filter is not None
    
    def get_cache_path(self, filter_idx: str, image_name: str) -> Path:
        """Obtiene el path del cache para una imagen y filtro específicos"""
        # Cambiar extensión a .png para el cache
        base_name = Path(image_name).stem + ".png"
        return self.cache_folder / filter_idx / base_name
    
    def cache_exists(self, filter_idx: str, image_name: str) -> bool:
        """Verifica si existe cache para una imagen y filtro"""
        cache_path = self.get_cache_path(filter_idx, image_name)
        return cache_path.exists()
    
    def all_cache_exists_up_to(self, checkpoint_idx: str, image_name: str, sorted_indices: List[str]) -> bool:
        """Verifica si existe cache para todos los filtros hasta el checkpoint (inclusive)"""
        checkpoint_int = int(checkpoint_idx)
        for idx in sorted_indices:
            if int(idx) > checkpoint_int:
                break
            if not self.cache_exists(idx, image_name):
                return False
        return True
    
    def load_from_cache(self, filter_idx: str, image_name: str) -> Optional[np.ndarray]:
        """Carga una imagen desde el cache"""
        cache_path = self.get_cache_path(filter_idx, image_name)
        if cache_path.exists():
            img = cv2.imread(str(cache_path), cv2.IMREAD_UNCHANGED)
            return img
        return None
    
    def save_to_cache(self, filter_idx: str, image_name: str, image: np.ndarray):
        """Guarda una imagen en el cache"""
        cache_path = self.get_cache_path(filter_idx, image_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cache_path), image)
    
    def clear_filter_cache(self, filter_idx: str):
        """Elimina todo el cache de un filtro específico"""
        cache_dir = self.cache_folder / filter_idx
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Cache del filtro {filter_idx} eliminado")
    
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
    def validate_checkpoint_filters(pipeline: Dict, checkpoint_idx: str) -> Tuple[bool, str]:
        """
        Valida que todos los filtros hasta el checkpoint (inclusive) solo produzcan imágenes.
        
        Returns:
            Tuple de (es_válido, mensaje_error)
        """
        checkpoint_int = int(checkpoint_idx)
        sorted_indices = sorted(pipeline.keys(), key=int)
        
        for idx in sorted_indices:
            if int(idx) > checkpoint_int:
                break
            
            filter_config = pipeline[idx]
            filter_name = filter_config.get('filter_name')
            filter_class = get_filter(filter_name)
            
            if filter_class is None:
                return False, f"Filtro {idx} ({filter_name}) no encontrado"
            
            if not CacheManager.filter_outputs_only_images(filter_class):
                non_image_outputs = [
                    f"{name}:{dtype}" 
                    for name, dtype in filter_class.OUTPUTS.items() 
                    if dtype != "image"
                ]
                return False, f"Filtro {idx} ({filter_name}) produce datos no-imagen: {non_image_outputs}"
        
        return True, ""


class PipelineProcessor:
    """Procesa el pipeline de filtros"""
    
    def __init__(self, pipeline_path: str, params_path: str):
        self.pipeline_path = pipeline_path
        self.params_path = params_path
        self.pipeline = {}
        self.saved_params = {}
        self.filter_instances: Dict[str, BaseFilter] = {}
        self.filter_outputs: Dict[str, Dict[str, Any]] = {}
        
        self.load_pipeline()
        self.load_params()
        self.instantiate_filters()
    
    def load_pipeline(self):
        """Carga el pipeline desde JSON"""
        try:
            with open(self.pipeline_path, 'r') as f:
                data = json.load(f)
                self.pipeline = data.get('filters', {})
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
                data = json.load(f)
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
        
        for idx, instance in self.filter_instances.items():
            params_data["filter_params"][idx] = {
                "filter_name": instance.FILTER_NAME,
                "params": instance.get_params()
            }
        
        with open(self.params_path, 'w') as f:
            json.dump(params_data, f, indent=4)
        
        print(f"Parámetros guardados en {self.params_path}")
    
    def instantiate_filters(self):
        """Crea instancias de todos los filtros del pipeline"""
        self.filter_instances = {}
        
        for idx in sorted(self.pipeline.keys(), key=int):
            filter_config = self.pipeline[idx]
            filter_name = filter_config['filter_name']
            filter_class = get_filter(filter_name)
            
            if filter_class is None:
                print(f"ERROR: Filtro '{filter_name}' no encontrado en biblioteca")
                continue
            
            # Cargar parámetros guardados o usar defaults
            params = None
            if idx in self.saved_params:
                params = self.saved_params[idx].get('params', {})
            
            self.filter_instances[idx] = filter_class(params)
            print(f"  Filtro {idx}: {filter_name} instanciado")
    
    def validate_pipeline(self) -> List[str]:
        """Valida que el pipeline tenga conexiones correctas"""
        errors = []
        
        for idx in sorted(self.pipeline.keys(), key=int):
            filter_config = self.pipeline[idx]
            filter_name = filter_config['filter_name']
            filter_class = get_filter(filter_name)
            
            if filter_class is None:
                errors.append(f"Filtro {idx}: '{filter_name}' no existe")
                continue
            
            inputs_config = filter_config.get('inputs', {})
            
            for input_name, source in inputs_config.items():
                # Validar formato "filtro_idx.output_name"
                if '.' not in source:
                    errors.append(f"Filtro {idx}: formato inválido '{source}', debe ser 'N.output_name'")
                    continue
                
                source_idx, output_name = source.split('.', 1)
                
                # Validar que el filtro fuente existe
                if source_idx not in self.pipeline:
                    errors.append(f"Filtro {idx}: referencia a filtro inexistente '{source_idx}'")
                    continue
                
                # Validar que el filtro fuente produce ese output
                source_filter = get_filter(self.pipeline[source_idx]['filter_name'])
                if source_filter and output_name not in source_filter.OUTPUTS:
                    errors.append(f"Filtro {idx}: filtro {source_idx} no produce '{output_name}'")
        
        return errors
    
    def process_filter(self, idx: str, original_image: np.ndarray) -> Dict[str, Any]:
        """Procesa un filtro específico y retorna sus outputs"""
        if idx not in self.filter_instances:
            return {}
        
        filter_instance = self.filter_instances[idx]
        filter_config = self.pipeline[idx]
        inputs_config = filter_config.get('inputs', {})
        
        # Recolectar inputs de filtros anteriores
        inputs = {}
        for input_name, source in inputs_config.items():
            source_idx, output_name = source.split('.', 1)
            if source_idx in self.filter_outputs:
                inputs[input_name] = self.filter_outputs[source_idx].get(output_name)
        
        # Procesar el filtro
        outputs = filter_instance.process(inputs, original_image)
        self.filter_outputs[idx] = outputs
        
        return outputs
    
    def process_up_to_with_cache(self, target_idx: str, original_image: np.ndarray,
                                  cache_manager: CacheManager, image_name: str,
                                  ignore_cache: bool = False) -> Dict[str, Any]:
        """
        Procesa todos los filtros hasta el índice especificado, usando cache si está disponible.
        
        Args:
            target_idx: Índice del filtro objetivo
            original_image: Imagen original
            cache_manager: Gestor de cache
            image_name: Nombre de la imagen (para cache)
            ignore_cache: Si True, ignora el cache aunque exista
        
        Returns:
            Outputs del filtro objetivo
        """
        self.filter_outputs = {}
        
        sorted_indices = sorted(self.pipeline.keys(), key=int)
        target_int = int(target_idx)
        checkpoint = cache_manager.get_checkpoint()
        checkpoint_int = int(checkpoint) if checkpoint else -1
        
        # Determinar si podemos usar cache (debe existir cache para TODOS los filtros hasta checkpoint)
        use_cache = (
            checkpoint is not None and
            not ignore_cache and
            target_int >= checkpoint_int and
            cache_manager.all_cache_exists_up_to(checkpoint, image_name, sorted_indices)
        )
        
        if use_cache:
            # Cargar desde cache TODOS los filtros hasta el checkpoint
            all_loaded = True
            for idx in sorted_indices:
                idx_int = int(idx)
                if idx_int > checkpoint_int:
                    break
                
                cached_image = cache_manager.load_from_cache(idx, image_name)
                if cached_image is not None:
                    # Crear outputs desde cache
                    main_output_name = self._get_main_output_name(idx)
                    self.filter_outputs[idx] = {
                        "sample_image": cached_image,
                        main_output_name: cached_image
                    }
                else:
                    all_loaded = False
                    break
            
            if all_loaded:
                # Procesar solo los filtros después del checkpoint
                for idx in sorted_indices:
                    idx_int = int(idx)
                    if idx_int <= checkpoint_int:
                        continue  # Saltar filtros hasta el checkpoint
                    if idx_int > target_int:
                        break
                    self.process_filter(idx, original_image)
                
                return self.filter_outputs.get(target_idx, {})
        
        # Procesar normalmente (sin cache o cache no disponible)
        for idx in sorted_indices:
            if int(idx) > target_int:
                break
            self.process_filter(idx, original_image)
            
            # Si estamos dentro del rango del checkpoint y no estamos ignorando cache,
            # guardar cache para cada filtro hasta el checkpoint
            if checkpoint is not None and not ignore_cache:
                if int(idx) <= checkpoint_int:
                    if not cache_manager.cache_exists(idx, image_name):
                        sample = self.filter_outputs.get(idx, {}).get('sample_image')
                        if sample is not None:
                            cache_manager.save_to_cache(idx, image_name, sample)
        
        return self.filter_outputs.get(target_idx, {})
    
    def _get_main_output_name(self, filter_idx: str) -> str:
        """Obtiene el nombre del output principal de un filtro (excluyendo sample_image)"""
        if filter_idx not in self.pipeline:
            return "output"
        
        filter_name = self.pipeline[filter_idx]['filter_name']
        filter_class = get_filter(filter_name)
        
        if filter_class:
            for output_name in filter_class.OUTPUTS:
                if output_name != "sample_image":
                    return output_name
        
        return "output"
    
    def get_filter_count(self) -> int:
        """Retorna el número de filtros en el pipeline"""
        return len(self.pipeline)
    
    def get_sorted_indices(self) -> List[str]:
        """Retorna los índices de filtros ordenados"""
        return sorted(self.pipeline.keys(), key=int)
    
    def get_filter_instance(self, idx: str) -> Optional[BaseFilter]:
        """Obtiene la instancia de un filtro"""
        return self.filter_instances.get(idx)
    
    def get_filter_name(self, idx: str) -> str:
        """Obtiene el nombre de un filtro"""
        if idx in self.pipeline:
            return self.pipeline[idx].get('filter_name', 'Unknown')
        return 'Unknown'


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
        
        # Estado de navegación
        self.current_view_filter = 0  # Filtro que estamos visualizando
        self.current_edit_filter = 0  # Filtro cuyos parámetros estamos editando
        self.current_param_index = 0  # Índice del parámetro seleccionado
        
        # Estado de cache
        self.ignore_cache = False  # Si True, ignora el cache al procesar
        self.params_modified_before_checkpoint = False  # Rastrea si se modificaron params pre-checkpoint
        
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
    
    def get_filter_indices(self) -> List[str]:
        """Obtiene los índices de filtros ordenados"""
        return self.processor.get_sorted_indices()
    
    def get_current_filter_params(self) -> List[Tuple[str, Dict]]:
        """Obtiene los parámetros del filtro que estamos editando"""
        indices = self.get_filter_indices()
        if not indices or self.current_edit_filter >= len(indices):
            return []
        
        idx = indices[self.current_edit_filter]
        instance = self.processor.get_filter_instance(idx)
        
        if instance is None:
            return []
        
        return list(instance.PARAMS.items())
    
    def render_params_window(self):
        """Renderiza la ventana de parámetros"""
        indices = self.get_filter_indices()
        if not indices:
            return
        
        view_idx = indices[self.current_view_filter]
        edit_idx = indices[self.current_edit_filter]
        
        view_instance = self.processor.get_filter_instance(view_idx)
        edit_instance = self.processor.get_filter_instance(edit_idx)
        
        # Crear imagen para parámetros
        param_img = np.zeros((550, 500, 3), dtype=np.uint8)
        param_img[:] = (40, 40, 40)  # Fondo gris oscuro
        
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
            checkpoint_color = (0, 165, 255)  # Naranja
            if self.ignore_cache:
                checkpoint_text = f"Checkpoint: Filtro {checkpoint} ({checkpoint_name}) [IGNORADO]"
                checkpoint_color = (0, 100, 255)  # Rojo-naranja
            else:
                # Verificar si existe cache completo
                cache_complete = self.cache_manager.all_cache_exists_up_to(
                    checkpoint, 
                    self.browser.get_current_name(),
                    indices
                )
                cache_status = "CACHED" if cache_complete else "SIN CACHE"
                checkpoint_text = f"Checkpoint: Filtro {checkpoint} ({checkpoint_name}) [{cache_status}]"
            cv2.putText(param_img, checkpoint_text, (10, y), font, 0.4, checkpoint_color, 1)
        else:
            cv2.putText(param_img, "Checkpoint: (ninguno)", (10, y), font, 0.4, (150, 150, 150), 1)
        y += 20
        
        # Separador
        cv2.line(param_img, (10, y), (490, y), (100, 100, 100), 1)
        y += 15
        
        # Info de visualización
        view_name = self.processor.get_filter_name(view_idx)
        is_checkpoint = (view_idx == checkpoint)
        view_color = (0, 165, 255) if is_checkpoint else (0, 255, 255)
        view_suffix = " [CHECKPOINT]" if is_checkpoint else ""
        cv2.putText(param_img, f"Viendo: Filtro {view_idx} ({view_name}){view_suffix}", (10, y), font, 0.5, view_color, 1)
        y += 20
        
        # Info de edición
        edit_name = self.processor.get_filter_name(edit_idx)
        is_edit_checkpoint = (edit_idx == checkpoint)
        edit_color = (0, 165, 255) if is_edit_checkpoint else (0, 255, 0)
        edit_suffix = " [CHECKPOINT]" if is_edit_checkpoint else ""
        cv2.putText(param_img, f"Editando: Filtro {edit_idx} ({edit_name}){edit_suffix}", (10, y), font, 0.5, edit_color, 1)
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
                
                # Color según selección
                if is_selected:
                    color = (0, 255, 0)  # Verde si está seleccionado
                    cv2.rectangle(param_img, (5, y-12), (495, y+5), (60, 60, 60), -1)
                else:
                    color = (200, 200, 200)
                
                # Valor actual
                current_value = edit_instance.get_param(param_name)
                min_val = config.get('min', 0)
                max_val = config.get('max', 100)
                
                # Nombre y valor
                text = f"  {param_name}: {current_value}"
                cv2.putText(param_img, text, (10, y), font, 0.45, color, 1)
                
                # Barra de progreso
                bar_x = 250
                bar_w = 200
                bar_h = 10
                
                # Fondo de la barra
                cv2.rectangle(param_img, (bar_x, y-8), (bar_x + bar_w, y-8 + bar_h), (80, 80, 80), -1)
                
                # Progreso
                if max_val != min_val:
                    progress = (current_value - min_val) / (max_val - min_val)
                    progress = max(0, min(1, progress))
                    fill_w = int(bar_w * progress)
                    cv2.rectangle(param_img, (bar_x, y-8), (bar_x + fill_w, y-8 + bar_h), color, -1)
                
                # Rango
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
        
        # Advertencia si hay cambios pendientes
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
        
        indices = self.get_filter_indices()
        if not indices:
            print("No hay filtros en el pipeline")
            return
        
        # Procesar hasta el filtro que estamos visualizando
        view_idx = indices[self.current_view_filter]
        image_name = self.browser.get_current_name()
        
        outputs = self.processor.process_up_to_with_cache(
            view_idx, 
            self.current_image,
            self.cache_manager,
            image_name,
            self.ignore_cache
        )
        
        # Obtener sample_image
        sample = outputs.get('sample_image')
        
        if sample is not None:
            # Agregar info en la imagen
            display = sample.copy()
            filter_name = self.processor.get_filter_name(view_idx)
            
            h, w = display.shape[:2]
            
            # Fondo semitransparente para texto
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
            
            # Indicador de checkpoint
            checkpoint = self.cache_manager.get_checkpoint()
            checkpoint_indicator = ""
            if checkpoint is not None and int(view_idx) <= int(checkpoint):
                if self.ignore_cache:
                    checkpoint_indicator = " [CACHE:IGNORADO]"
                elif self.cache_manager.cache_exists(view_idx, image_name):
                    checkpoint_indicator = " [CACHED]"
                else:
                    checkpoint_indicator = " [CACHEANDO]"
            
            info_text = f"Filtro {view_idx}: {filter_name}{checkpoint_indicator} | {self.browser.get_current_name()}"
            cv2.putText(display, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(self.WINDOW_RESULT, display)
        
        self.needs_reprocess = False
    
    def change_param_value(self, delta: int):
        """Cambia el valor del parámetro seleccionado"""
        params = self.get_current_filter_params()
        if not params or self.current_param_index >= len(params):
            return
        
        indices = self.get_filter_indices()
        edit_idx = indices[self.current_edit_filter]
        instance = self.processor.get_filter_instance(edit_idx)
        
        param_name, config = params[self.current_param_index]
        current = instance.get_param(param_name)
        step = config.get('step', 1)
        min_val = config.get('min', 0)
        max_val = config.get('max', 100)
        
        new_value = current + (delta * step)
        new_value = max(min_val, min(max_val, new_value))
        
        # Solo proceder si el valor cambió
        if new_value != current:
            instance.set_param(param_name, new_value)
            self.needs_reprocess = True
            
            # Verificar si estamos modificando un filtro anterior o igual al checkpoint
            checkpoint = self.cache_manager.get_checkpoint()
            if checkpoint:
                checkpoint_int = int(checkpoint)
                edit_int = int(edit_idx)
                if edit_int <= checkpoint_int:
                    self.ignore_cache = True
                    self.params_modified_before_checkpoint = True
    
    def toggle_checkpoint(self):
        """Alterna el checkpoint en el filtro actualmente visualizado"""
        indices = self.get_filter_indices()
        if not indices:
            return
        
        view_idx = indices[self.current_view_filter]
        current_checkpoint = self.cache_manager.get_checkpoint()
        
        if current_checkpoint == view_idx:
            # Quitar checkpoint
            self.cache_manager.set_checkpoint(None)
        else:
            # Validar que todos los filtros hasta este punto solo produzcan imágenes
            is_valid, error_msg = CacheManager.validate_checkpoint_filters(
                self.processor.pipeline, view_idx
            )
            
            if not is_valid:
                print(f"\n⚠️  No se puede establecer checkpoint aquí:")
                print(f"   {error_msg}")
                print(f"   Solo se puede hacer checkpoint en filtros donde todos los anteriores produzcan imágenes.\n")
                return
            
            # Establecer nuevo checkpoint
            self.cache_manager.set_checkpoint(view_idx)
        
        # Resetear estado de ignore_cache
        self.ignore_cache = False
        self.params_modified_before_checkpoint = False
    
    def save_params_with_warning(self) -> bool:
        """
        Guarda parámetros con advertencia si hay cambios que afectan al checkpoint.
        Retorna True si se guardó, False si se canceló.
        """
        if self.ignore_cache and self.params_modified_before_checkpoint:
            checkpoint = self.cache_manager.get_checkpoint()
            print("\n" + "="*60)
            print("⚠️  ADVERTENCIA")
            print("="*60)
            print(f"Has modificado parámetros que afectan al checkpoint (filtro {checkpoint}).")
            print("Guardar los parámetros BORRARÁ todo el cache.")
            print("")
            response = input("¿Deseas continuar? (s/n): ").strip().lower()
            
            if response == 's':
                # Borrar todo el cache (porque ahora cacheamos todos los filtros previos)
                self.cache_manager.clear_all_cache()
                # Guardar parámetros
                self.processor.save_params()
                # Resetear estado
                self.ignore_cache = False
                self.params_modified_before_checkpoint = False
                print("✓ Parámetros guardados y cache borrado")
                return True
            else:
                print("Guardado cancelado. El cache permanece intacto.")
                return False
        else:
            # Guardar normalmente
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
        
        # Cargar primera imagen
        self.current_image = self.browser.get_current_image()
        
        while True:
            # Procesar y mostrar
            if self.needs_reprocess:
                self.process_and_display()
            
            # Renderizar ventana de parámetros
            self.render_params_window()
            
            # Esperar input
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q') or key == 27:  # q o ESC
                break
            
            elif key == ord('a'):  # Imagen anterior
                self.browser.prev_image()
                self.current_image = self.browser.get_current_image()
                self.needs_reprocess = True
            
            elif key == ord('d'):  # Imagen siguiente
                self.browser.next_image()
                self.current_image = self.browser.get_current_image()
                self.needs_reprocess = True
            
            elif key == 32:  # ESPACIO - siguiente filtro (vista)
                indices = self.get_filter_indices()
                if indices:
                    self.current_view_filter = min(self.current_view_filter + 1, len(indices) - 1)
                    self.needs_reprocess = True
            
            elif key == 8:  # BACKSPACE - filtro anterior (vista)
                self.current_view_filter = max(0, self.current_view_filter - 1)
                self.needs_reprocess = True
            
            elif key == 82 or key == 0:  # UP - parámetro anterior
                params = self.get_current_filter_params()
                if params:
                    self.current_param_index = max(0, self.current_param_index - 1)
            
            elif key == 84 or key == 1:  # DOWN - parámetro siguiente
                params = self.get_current_filter_params()
                if params:
                    self.current_param_index = min(len(params) - 1, self.current_param_index + 1)
            
            elif key == 81 or key == 2:  # LEFT - decrementar valor
                self.change_param_value(-1)
            
            elif key == 83 or key == 3:  # RIGHT - incrementar valor
                self.change_param_value(1)
            
            elif key == 85:  # PageUp - filtro anterior (edición)
                self.current_edit_filter = max(0, self.current_edit_filter - 1)
                self.current_param_index = 0
            
            elif key == 86:  # PageDown - filtro siguiente (edición)
                indices = self.get_filter_indices()
                if indices:
                    # Solo podemos editar filtros hasta el que estamos viendo
                    max_edit = self.current_view_filter
                    self.current_edit_filter = min(self.current_edit_filter + 1, max_edit)
                    self.current_param_index = 0
            
            elif key == ord('c'):  # Toggle checkpoint
                self.toggle_checkpoint()
                self.needs_reprocess = True
            
            elif key == ord('s'):  # Guardar parámetros
                self.save_params_with_warning()
            
            elif key == ord('r'):  # Recargar parámetros
                self.processor.load_params()
                self.processor.instantiate_filters()
                self.ignore_cache = False
                self.params_modified_before_checkpoint = False
                self.needs_reprocess = True
                print("✓ Parámetros recargados")
            
            elif key == ord('h'):  # Ayuda
                indices = self.get_filter_indices()
                if indices:
                    edit_idx = indices[self.current_edit_filter]
                    instance = self.processor.get_filter_instance(edit_idx)
                    if instance:
                        print(instance.get_help())
        
        cv2.destroyAllWindows()


def main():
    # Procesar argumentos
    image_folder = "."
    clear_cache = False
    
    args = sys.argv[1:]
    for arg in args:
        if arg == "--clear-cache":
            clear_cache = True
        elif not arg.startswith("-"):
            image_folder = arg
    
    # Paths de configuración
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
    
    # Validar sincronización entre pipeline.json y params.json
    if PipelineParamsSync is not None:
        print("\n🔍 Validando sincronización pipeline ↔ params...")
        sync = PipelineParamsSync(str(pipeline_path), str(params_path))
        
        if not sync.load_files():
            print("\n❌ Error al cargar archivos. Abortando.")
            sys.exit(1)
        
        sync.analyze()
        
        if not sync.validate_only():
            # Hay problemas - detener ejecución
            sys.exit(1)
        
        print("✅ Sincronización OK\n")
    
    # Crear y ejecutar configurador
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
