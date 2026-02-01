"""
Configurador GUI de Parámetros de Filtros - VERSIÓN MULTI-CHECKPOINT
====================================================================

CAMBIOS PRINCIPALES:
- Soporte para múltiples checkpoints (antes solo uno)
- Lógica ultra-simple: si modificas <= último checkpoint, borrar TODO el cache
- Toggle de checkpoints con 'c' agrega/quita del conjunto

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
    c           - Agregar/quitar filtro actual como checkpoint
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

from core import PipelineProcessor, ImageBrowser, CacheManager
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

class ParamConfigurator:
    """Interfaz GUI principal - VERSIÓN MULTI-CHECKPOINT"""
    
    WINDOW_RESULT = "Resultado del Filtro"
    WINDOW_PARAMS = "Parametros"
    
    def __init__(self, image_folder: str, pipeline_path: str = "pipeline.json", 
                 params_path: str = "params.json", checkpoints_path: str = "checkpoints.json",
                 clear_cache: bool = False):
        
        self.image_folder = Path(image_folder)
        self.browser = ImageBrowser(image_folder)
        self.processor = PipelineProcessor(pipeline_path, params_path)
        self.cache_manager = CacheManager(self.image_folder, Path(checkpoints_path), self.processor)
        
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
        """Renderiza la ventana de parámetros - VERSIÓN MULTI-CHECKPOINT"""
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
        
        # ← CAMBIO: Info de checkpoints (múltiples)
        checkpoints = self.cache_manager.checkpoints
        if checkpoints:
            # Mostrar lista de checkpoints
            checkpoint_names = [f"{cp} ({self.processor.get_filter_name(cp)})" for cp in checkpoints]
            checkpoint_text = f"Checkpoints ({len(checkpoints)}): {', '.join(checkpoint_names[:2])}"
            if len(checkpoints) > 2:
                checkpoint_text += f", +{len(checkpoints)-2}"
            
            checkpoint_color = (0, 165, 255)
            if self.ignore_cache:
                checkpoint_text += " [IGNORADOS]"
                checkpoint_color = (0, 100, 255)
            
            cv2.putText(param_img, checkpoint_text, (10, y), font, 0.35, checkpoint_color, 1)
            
            # Mostrar último checkpoint en segunda línea
            y += 18
            last_checkpoint = self.cache_manager.get_last_checkpoint()
            if last_checkpoint:
                last_checkpoint_name = self.processor.get_filter_name(last_checkpoint)
                last_text = f"  Ultimo: {last_checkpoint} ({last_checkpoint_name})"
                cv2.putText(param_img, last_text, (10, y), font, 0.35, checkpoint_color, 1)
        else:
            cv2.putText(param_img, "Checkpoints: (ninguno)", (10, y), font, 0.4, (150, 150, 150), 1)
        y += 20
        
        # Separador
        cv2.line(param_img, (10, y), (490, y), (100, 100, 100), 1)
        y += 15
        
        # ← CAMBIO: Info de visualización (indicar si es checkpoint)
        view_name = self.processor.get_filter_name(view_id)
        is_view_checkpoint = (view_id in checkpoints)
        view_color = (0, 165, 255) if is_view_checkpoint else (0, 255, 255)
        view_suffix = " [CHECKPOINT]" if is_view_checkpoint else ""
        cv2.putText(param_img, f"Viendo: {view_id} ({view_name}){view_suffix}", (10, y), font, 0.5, view_color, 1)
        y += 20
        
        # ← CAMBIO: Info de edición (indicar si es checkpoint)
        edit_name = self.processor.get_filter_name(edit_id)
        is_edit_checkpoint = (edit_id in checkpoints)
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
            "c: Agregar/quitar checkpoint",  # ← CAMBIO: Texto actualizado
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
        
        outputs = self.processor.process_up_to_with_cache( #xxx
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
            
            # ← CAMBIO: Indicador de cache más detallado
            cache_indicator = ""
            if self.cache_manager.has_checkpoints():
                if self.ignore_cache:
                    cache_indicator = " [CACHE:IGNORADO]"
                elif view_id in self.cache_manager.checkpoints:
                    if self.cache_manager.cache_exists(view_id, image_name):
                        cache_indicator = " [CACHED]"
                    else:
                        cache_indicator = " [CACHEANDO]"
                else:
                    # Verificar si hay algún checkpoint anterior con cache
                    last_checkpoint = self.cache_manager.get_last_checkpoint()
                    if last_checkpoint:
                        view_order = self.processor.get_filter_order(view_id)
                        last_checkpoint_order = self.processor.get_filter_order(last_checkpoint)
                        if view_order > last_checkpoint_order:
                            cache_indicator = " [POST-CACHE]"
            
            info_text = f"{view_id}: {filter_name}{cache_indicator} | {self.browser.get_current_name()}"
            cv2.putText(display, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(self.WINDOW_RESULT, display)
        
        self.needs_reprocess = False
    
    def change_param_value(self, delta: int):
        """Cambia el valor del parámetro seleccionado - VERSIÓN ULTRA-SIMPLE"""
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
            
            # ← CAMBIO: Lógica ultra-simple con último checkpoint
            if self.cache_manager.has_checkpoints():
                last_checkpoint = self.cache_manager.get_last_checkpoint()
                if last_checkpoint:
                    edit_order = self.processor.get_filter_order(edit_id)
                    last_checkpoint_order = self.processor.get_filter_order(last_checkpoint)
                    
                    # Si modificamos un filtro <= último checkpoint, invalidar cache
                    if edit_order <= last_checkpoint_order:
                        self.ignore_cache = True
                        self.params_modified_before_checkpoint = True
    
    def toggle_checkpoint(self):
        """Alterna checkpoint en el filtro actualmente visualizado - VERSIÓN MULTI-CHECKPOINT"""
        view_id = self.get_current_view_id()
        if not view_id:
            return
        
        # Intentar toggle
        success = self.cache_manager.toggle_checkpoint(view_id)
        
        if success:
            # Reset del estado de invalidación si quitamos el último checkpoint
            if not self.cache_manager.has_checkpoints():
                self.ignore_cache = False
                self.params_modified_before_checkpoint = False
    
    def save_params_with_warning(self) -> bool:
        """
        Guarda parámetros con advertencia si hay cambios que afectan checkpoints.
        VERSIÓN ULTRA-SIMPLE: Borra TODO el cache si se modificó <= último checkpoint.
        """
        if self.ignore_cache and self.params_modified_before_checkpoint:
            print("\n" + "="*60)
            print("⚠️  ADVERTENCIA")
            print("="*60)
            print(f"Has modificado parámetros antes o en el último checkpoint.")
            print("Guardar los parámetros BORRARÁ TODO el cache de TODOS los checkpoints.")
            print("")
            response = input("¿Deseas continuar? (s/n): ").strip().lower()
            
            if response == 's':
                self.cache_manager.clear_all_cache()  # ← Borra TODO
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
        print("CONFIGURADOR DE FILTROS DE IMAGEN - MULTI-CHECKPOINT")
        print("="*60)
        print("Presiona 'h' para ver ayuda del filtro actual")
        print("Presiona 'c' para agregar/quitar checkpoint")  # ← CAMBIO
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
    checkpoints_path = script_dir / "checkpoints.json"
    
    print(f"Carpeta de imágenes: {image_folder}")
    print(f"Pipeline: {pipeline_path}")
    print(f"Parámetros: {params_path}")
    print(f"Checkpoint: {checkpoints_path}")
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
        checkpoints_path=str(checkpoints_path),
        clear_cache=clear_cache
    )
    
    configurator.run()


if __name__ == "__main__":
    main()
