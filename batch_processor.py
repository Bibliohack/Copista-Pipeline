#!/usr/bin/env python3
"""
Batch Processor - Procesamiento por lotes sin GUI
==================================================

Procesa múltiples imágenes aplicando el pipeline configurado y guardando
outputs específicos según targets definidos en batch_config.json.

Uso:
    python batch_processor.py [--config batch_config.json] [--overwrite]

Características:
    - Sin GUI (modo headless)
    - Sin cache (procesa todo desde cero)
    - Optimizado (without_preview=True)
    - Multi-target (guarda múltiples outputs)
    - Validación previa de configuración
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import OrderedDict
import cv2
import numpy as np
from datetime import datetime
from tqdm import tqdm  # Para progress bar

from core import PipelineProcessor, ImageBrowser

# Importar sincronizador para validación
try:
    from sync_pipeline_params import PipelineParamsSync
except ImportError:
    PipelineParamsSync = None
    print("⚠️  sync_pipeline_params.py no encontrado - no se validará sincronización")

# Importar biblioteca de filtros para introspección
from filter_library import get_filter


class BatchConfig:
    """Maneja la configuración del procesamiento por lotes"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Dict = {}
        self.source_folder: Path = Path(".")
        self.targets: List[Dict] = []
        self.log_file: Optional[Path] = None
        
        self.load()
    
    def load(self):
        """Carga la configuración desde JSON"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f, object_pairs_hook=OrderedDict)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON inválido en {self.config_path}: {e}")
        
        # Extraer campos
        self.source_folder = Path(self.config.get("source_folder", "."))
        self.targets = self.config.get("targets", [])
        
        log_file_str = self.config.get("log_file")
        if log_file_str:
            self.log_file = Path(log_file_str)
        
        # Validar estructura básica
        if not self.targets:
            raise ValueError("La configuración debe tener al menos un target")
        
        print(f"✓ Configuración cargada: {len(self.targets)} target(s)")
    
    def validate_structure(self):
        """Valida la estructura de cada target"""
        errors = []
        
        for i, target in enumerate(self.targets):
            # Validar campos requeridos
            if "filter_id" not in target:
                errors.append(f"Target {i}: falta campo 'filter_id'")
            
            if "output_name" not in target:
                errors.append(f"Target {i}: falta campo 'output_name'")
            
            if "destination" not in target:
                errors.append(f"Target {i}: falta campo 'destination'")
            elif not isinstance(target["destination"], dict):
                errors.append(f"Target {i}: 'destination' debe ser un objeto")
        
        if errors:
            raise ValueError("Errores en estructura de targets:\n  " + "\n  ".join(errors))
        
        print(f"✓ Estructura de targets validada")


class BatchValidator:
    """Valida la configuración del batch contra el pipeline"""
    
    def __init__(self, batch_config: BatchConfig, processor: PipelineProcessor):
        self.batch_config = batch_config
        self.processor = processor
    
    def validate_all(self) -> bool:
        """Ejecuta todas las validaciones"""
        print("\n" + "="*60)
        print("VALIDANDO CONFIGURACIÓN DE BATCH")
        print("="*60)
        
        # 1. Validar que source_folder existe y tiene imágenes
        if not self._validate_source_folder():
            return False
        
        # 2. Validar cada target
        if not self._validate_targets():
            return False
        
        # 3. Crear carpetas de destino
        if not self._create_destination_folders():
            return False
        
        print("="*60)
        print("✅ VALIDACIÓN EXITOSA")
        print("="*60 + "\n")
        return True
    
    def _validate_source_folder(self) -> bool:
        """Valida que la carpeta fuente existe y tiene imágenes"""
        source = self.batch_config.source_folder
        
        if not source.exists():
            print(f"❌ ERROR: Carpeta fuente no existe: {source}")
            return False
        
        if not source.is_dir():
            print(f"❌ ERROR: La ruta fuente no es una carpeta: {source}")
            return False
        
        # Contar imágenes
        browser = ImageBrowser(str(source))
        img_count = browser.get_image_count()
        
        if img_count == 0:
            print(f"❌ ERROR: No se encontraron imágenes en: {source}")
            return False
        
        print(f"✓ Carpeta fuente: {source} ({img_count} imágenes)")
        return True
    
    def _validate_targets(self) -> bool:
        """Valida cada target contra el pipeline"""
        errors = []
        warnings = []
        
        for i, target in enumerate(self.batch_config.targets):
            filter_id = target["filter_id"]
            output_name = target["output_name"]
            
            # 1. Verificar que filter_id existe en pipeline
            if filter_id not in self.processor.pipeline:
                errors.append(f"Target {i}: filter_id '{filter_id}' no existe en pipeline.json")
                continue
            
            # 2. Obtener clase del filtro
            filter_config = self.processor.pipeline[filter_id]
            filter_name = filter_config.get("filter_name")
            filter_class = get_filter(filter_name)
            
            if filter_class is None:
                errors.append(f"Target {i}: filtro '{filter_name}' no encontrado en biblioteca")
                continue
            
            # 3. Verificar que output_name existe en OUTPUTS del filtro
            if output_name not in filter_class.OUTPUTS:
                available = ", ".join(filter_class.OUTPUTS.keys())
                errors.append(f"Target {i}: output '{output_name}' no existe en filtro '{filter_id}'")
                errors.append(f"         Outputs disponibles: {available}")
                continue
            
            # 4. Verificar coherencia de extensión con tipo de output
            output_type = filter_class.OUTPUTS[output_name]
            dest = target["destination"]
            extension = dest.get("extension", "").lower()
            
            if output_type == "image":
                if extension and extension not in ["png", "jpg", "jpeg", "bmp", "tiff"]:
                    warnings.append(f"Target {i}: extensión '{extension}' inusual para imagen (recomendado: png, jpg)")
            elif extension and extension not in ["json", "txt"]:
                warnings.append(f"Target {i}: extensión '{extension}' inusual para datos (recomendado: json)")
            
            print(f"  ✓ Target {i}: {filter_id}.{output_name} ({output_type})")
        
        # Mostrar warnings
        if warnings:
            print("\n⚠️  ADVERTENCIAS:")
            for warning in warnings:
                print(f"  {warning}")
        
        # Mostrar errores
        if errors:
            print("\n❌ ERRORES EN TARGETS:")
            for error in errors:
                print(f"  {error}")
            return False
        
        return True
    
    def _create_destination_folders(self) -> bool:
        """Crea las carpetas de destino si no existen"""
        folders_created = []
        errors = []
        
        for target in self.batch_config.targets:
            dest = target["destination"]
            folder = dest.get("folder")
            
            if folder:
                folder_path = Path(folder)
                
                try:
                    folder_path.mkdir(parents=True, exist_ok=True)
                    if folder_path not in folders_created:
                        folders_created.append(folder_path)
                except Exception as e:
                    errors.append(f"No se pudo crear carpeta '{folder}': {e}")
        
        if errors:
            print("\n❌ ERRORES AL CREAR CARPETAS:")
            for error in errors:
                print(f"  {error}")
            return False
        
        if folders_created:
            print(f"\n✓ Carpetas de destino verificadas/creadas: {len(folders_created)}")
            for folder in folders_created:
                print(f"  - {folder}")
        
        return True

class OutputSaver:
    """Maneja el guardado de diferentes tipos de outputs"""
    
    @staticmethod
    def save(data: Any, output_path: Path, output_type: str) -> bool:
        """
        Guarda un output según su tipo.
        
        Args:
            data: Datos a guardar
            output_path: Path completo del archivo destino
            output_type: Tipo de output (image, lines, metadata, etc.)
        
        Returns:
            True si se guardó exitosamente
        """
        try:
            if output_type == "image":
                return OutputSaver._save_image(data, output_path)
            elif output_type == "float":
                return OutputSaver._save_float(data, output_path)
            else:
                # Por defecto, intentar guardar como JSON
                return OutputSaver._save_json(data, output_path)
        except Exception as e:
            print(f"  ❌ Error al guardar {output_path}: {e}")
            return False
    
    @staticmethod
    def _save_image(img: np.ndarray, path: Path) -> bool:
        """Guarda una imagen"""
        if img is None:
            print(f"  ⚠️  Imagen None, no se guardó: {path}")
            return False
        
        path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(path), img)
        
        if not success:
            print(f"  ❌ cv2.imwrite falló: {path}")
            return False
        
        return True
    
    @staticmethod
    def _save_json(data: Any, path: Path) -> bool:
        """Guarda datos como JSON"""
        if data is None:
            print(f"  ⚠️  Datos None, no se guardó: {path}")
            return False
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return True
    
    @staticmethod
    def _save_float(value: float, path: Path) -> bool:
        """Guarda un valor float como texto o JSON"""
        if value is None:
            print(f"  ⚠️  Valor None, no se guardó: {path}")
            return False
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Si la extensión es JSON, guardar como JSON
        if path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump({"value": float(value)}, f, indent=2)
        else:
            # Guardar como texto simple
            with open(path, 'w') as f:
                f.write(str(value))
        
        return True

class BatchProcessor:
    """Procesador por lotes principal"""
    
    def __init__(self, batch_config_path: str = "batch_config.json",
                 pipeline_path: str = "pipeline.json",
                 params_path: str = "params.json"):
        
        self.batch_config_path = batch_config_path
        self.pipeline_path = pipeline_path
        self.params_path = params_path
        
        # Componentes
        self.batch_config: Optional[BatchConfig] = None
        self.processor: Optional[PipelineProcessor] = None
        self.browser: Optional[ImageBrowser] = None
        self.validator: Optional[BatchValidator] = None
        
        # Estadísticas
        self.stats = {
            "total_images": 0,
            "processed": 0,
            "errors": 0,
            "skipped": 0
        }
    
    def initialize(self) -> bool:
        """Inicializa todos los componentes y ejecuta validaciones"""
        print("\n" + "="*60)
        print("BATCH PROCESSOR - INICIALIZACIÓN")
        print("="*60 + "\n")
        
        # 1. Validar sincronización pipeline ↔ params
        if not self._validate_pipeline_params_sync():
            return False
        
        # 2. Cargar configuración batch
        try:
            self.batch_config = BatchConfig(self.batch_config_path)
            self.batch_config.validate_structure()
        except Exception as e:
            print(f"❌ ERROR al cargar configuración batch: {e}")
            return False
        
        # 3. Crear processor (SIN preview para optimización)
        try:
            self.processor = PipelineProcessor(
                self.pipeline_path,
                self.params_path,
                without_preview=True  # ← Clave para batch!
            )
            print(f"✓ Pipeline cargado con {self.processor.get_filter_count()} filtros (without_preview=True)")
        except Exception as e:
            print(f"❌ ERROR al crear processor: {e}")
            return False
        
        # 4. Crear browser
        try:
            self.browser = ImageBrowser(str(self.batch_config.source_folder))
            self.stats["total_images"] = self.browser.get_image_count()
        except Exception as e:
            print(f"❌ ERROR al crear browser: {e}")
            return False
        
        # 5. Validar targets contra pipeline
        self.validator = BatchValidator(self.batch_config, self.processor)
        if not self.validator.validate_all():
            return False
        
        print("✅ Inicialización completada exitosamente\n")
        return True
    
    def _validate_pipeline_params_sync(self) -> bool:
        """Valida sincronización entre pipeline.json y params.json"""
        if PipelineParamsSync is None:
            print("⚠️  Saltando validación de sincronización (sync_pipeline_params.py no disponible)\n")
            return True
        
        print("🔍 Validando sincronización pipeline ↔ params...")
        sync = PipelineParamsSync(self.pipeline_path, self.params_path)
        
        if not sync.load_files():
            print("\n❌ Error al cargar archivos. Abortando.")
            return False
        
        sync.analyze()
        
        if not sync.validate_only():
            print("\n❌ Sincronización fallida. Ejecuta:")
            print("   python sync_pipeline_params.py")
            print("para corregir los problemas.\n")
            return False
        
        print("✅ Sincronización OK\n")
        return True
            
    def process_all(self, overwrite: bool = False):
        """Procesa todas las imágenes según los targets configurados"""
        if self.browser is None or self.batch_config is None:
            print("❌ ERROR: BatchProcessor no inicializado correctamente")
            return
        
        print("\n" + "="*60)
        print("PROCESANDO IMÁGENES")
        print("="*60)
        print(f"Total de imágenes: {self.stats['total_images']}")
        print(f"Targets configurados: {len(self.batch_config.targets)}")
        print(f"Modo: {'Sobrescribir' if overwrite else 'Saltar existentes'}")
        print("="*60 + "\n")
        
        # Determinar el último filtro necesario entre todos los targets
        last_filter_needed = self._get_last_filter_needed()
        print(f"ℹ️  Último filtro a procesar: {last_filter_needed}")
        print(f"   (optimización: no se procesarán filtros posteriores)\n")
        
        # Progress bar
        with tqdm(total=self.stats["total_images"], desc="Procesando", unit="img") as pbar:
            for i in range(self.stats["total_images"]):
                img_name = self.browser.get_current_name()
                
                try:
                    # Procesar esta imagen
                    success = self._process_single_image(img_name, last_filter_needed, overwrite)
                    
                    if success:
                        self.stats["processed"] += 1
                    else:
                        self.stats["skipped"] += 1
                
                except Exception as e:
                    print(f"\n❌ ERROR procesando {img_name}: {e}")
                    self.stats["errors"] += 1
                
                # Avanzar a siguiente imagen
                self.browser.next_image()
                pbar.update(1)
        
        # Mostrar resumen
        self._print_summary()
    
    def _get_last_filter_needed(self) -> str:
        """
        Determina el último filtro que necesita procesarse entre todos los targets.
        Esto optimiza el procesamiento evitando ejecutar filtros innecesarios.
        """
        max_order = -1
        last_filter = None
        
        for target in self.batch_config.targets:
            filter_id = target["filter_id"]
            order = self.processor.get_filter_order(filter_id)
            
            if order > max_order:
                max_order = order
                last_filter = filter_id
        
        return last_filter
    
    def _process_single_image(self, img_name: str, last_filter_id: str, overwrite: bool) -> bool:
        """
        Procesa una imagen individual y guarda todos sus targets.
        
        Returns:
            True si se procesó (aunque sea parcialmente), False si se saltó completamente
        """
        # Cargar imagen
        img = self.browser.get_current_image()
        if img is None:
            print(f"  ⚠️  No se pudo cargar: {img_name}")
            return False
        
        # Verificar si debemos saltar esta imagen
        if not overwrite and self._all_targets_exist(img_name):
            # Todos los targets ya existen, saltar
            return False
        
        # Procesar pipeline hasta el último filtro necesario
        try:
            self.processor.process_up_to(last_filter_id, img)
        except Exception as e:
            print(f"\n  ❌ Error en pipeline para {img_name}: {e}")
            return False
        
        # Guardar cada target
        saved_count = 0
        for target in self.batch_config.targets:
            if self._save_target(img_name, target, overwrite):
                saved_count += 1
        
        return saved_count > 0
    
    def _all_targets_exist(self, img_name: str) -> bool:
        """Verifica si todos los targets de esta imagen ya existen"""
        for target in self.batch_config.targets:
            output_path = self._build_output_path(img_name, target)
            if not output_path.exists():
                return False
        return True
    
    def _save_target(self, img_name: str, target: Dict, overwrite: bool) -> bool:
        """
        Guarda un target específico.
        
        Returns:
            True si se guardó exitosamente
        """
        filter_id = target["filter_id"]
        output_name = target["output_name"]
        
        # Construir path de salida
        output_path = self._build_output_path(img_name, target)
        
        # Verificar si existe y no debemos sobrescribir
        if not overwrite and output_path.exists():
            return False  # Saltado silenciosamente
        
        # Obtener el output del filtro procesado
        filter_outputs = self.processor.filter_outputs.get(filter_id, {})
        data = filter_outputs.get(output_name)
        
        if data is None:
            print(f"  ⚠️  Output '{output_name}' no encontrado en filtro '{filter_id}'")
            return False
        
        # Obtener tipo de output
        output_type = self._get_output_type(filter_id, output_name)
        
        # Guardar
        return OutputSaver.save(data, output_path, output_type)
    
    def _build_output_path(self, img_name: str, target: Dict) -> Path:
        """
        Construye el path completo del archivo de salida.
        
        Formato: {folder}/{prefix}{nombre_base}{suffix}.{extension}
        """
        dest = target["destination"]
        
        # Extraer componentes
        folder = dest.get("folder", ".")
        prefix = dest.get("prefix", "")
        suffix = dest.get("suffix", f"_{target['filter_id']}")  # Sufijo por defecto
        extension = dest.get("extension", "png")
        
        # Nombre base (sin extensión original)
        base_name = Path(img_name).stem
        
        # Construir nombre final
        file_name = f"{prefix}{base_name}{suffix}.{extension}"
        
        return Path(folder) / file_name
    
    def _get_output_type(self, filter_id: str, output_name: str) -> str:
        """Obtiene el tipo de un output consultando la clase del filtro"""
        filter_config = self.processor.pipeline[filter_id]
        filter_name = filter_config["filter_name"]
        filter_class = get_filter(filter_name)
        
        if filter_class and output_name in filter_class.OUTPUTS:
            return filter_class.OUTPUTS[output_name]
        
        return "unknown"
    
    def _print_summary(self):
        """Imprime resumen final del procesamiento"""
        print("\n" + "="*60)
        print("RESUMEN DE PROCESAMIENTO")
        print("="*60)
        print(f"Total imágenes:     {self.stats['total_images']}")
        print(f"Procesadas:         {self.stats['processed']}")
        print(f"Saltadas:           {self.stats['skipped']}")
        print(f"Errores:            {self.stats['errors']}")
        print("="*60)
        
        if self.stats['errors'] == 0 and self.stats['processed'] > 0:
            print("✅ Procesamiento completado exitosamente")
        elif self.stats['errors'] > 0:
            print("⚠️  Procesamiento completado con errores")
        else:
            print("ℹ️  No se procesaron imágenes nuevas")
        print("="*60 + "\n")

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Procesamiento por lotes de imágenes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python batch_processor.py
  python batch_processor.py --config mi_config.json
  python batch_processor.py --overwrite
        """
    )
    parser.add_argument('--config', default='batch_config.json',
                       help='Ruta al archivo de configuración (default: batch_config.json)')
    parser.add_argument('--pipeline', default='pipeline.json',
                       help='Ruta a pipeline.json (default: pipeline.json)')
    parser.add_argument('--params', default='params.json',
                       help='Ruta a params.json (default: params.json)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Sobrescribir archivos existentes')
    
    args = parser.parse_args()
    
    # Crear processor
    processor = BatchProcessor(
        batch_config_path=args.config,
        pipeline_path=args.pipeline,
        params_path=args.params
    )
    
    # Inicializar
    if not processor.initialize():
        print("\n❌ Inicialización fallida. Abortando.")
        sys.exit(1)
    
    # Procesar
    processor.process_all(overwrite=args.overwrite)


if __name__ == "__main__":
    main()

