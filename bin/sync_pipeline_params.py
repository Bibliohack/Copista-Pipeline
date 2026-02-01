#!/usr/bin/env python3
"""
Sincronizador de pipeline.json y params.json
=============================================

Este script detecta y corrige inconsistencias entre pipeline.json y params.json:
- Filtros con mismo ID pero diferente filter_name (cambio de tipo de filtro)
- Filtros eliminados del pipeline (params huérfanos)
- Filtros nuevos sin parámetros guardados

Uso:
    python sync_pipeline_params.py                    # Modo interactivo
    python sync_pipeline_params.py --validate-only     # Solo validar (para param_configurator)
    python sync_pipeline_params.py --auto-clean        # Limpiar huérfanos automáticamente

Modos:
    Interactivo: Te pregunta qué hacer con cada problema
    Validate-only: Solo reporta problemas y sale con código de error
    Auto-clean: Elimina huérfanos y acepta nuevos filtros automáticamente
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

# Importar get_filter para validar nombres de filtros
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root / "src"))
try:
    from filter_library import get_filter
except ImportError:
    print("ERROR: No se puede importar filter_library.py")
    print("Asegúrate de que filter_library.py está en el mismo directorio.")
    sys.exit(1)


class PipelineParamsSync:
    """Sincronizador de pipeline y parámetros"""
    
    def __init__(self, pipeline_path: str = "pipeline.json", 
                 params_path: str = "params.json"):
        self.pipeline_path = Path(pipeline_path)
        self.params_path = Path(params_path)
        self.pipeline = OrderedDict()
        self.params = {}
        self.issues = []
        
    def load_files(self) -> bool:
        """Carga los archivos JSON. Retorna True si todo OK."""
        # Cargar pipeline.json
        if not self.pipeline_path.exists():
            print(f"❌ ERROR: No existe {self.pipeline_path}")
            return False
        
        try:
            with open(self.pipeline_path, 'r') as f:
                data = json.load(f, object_pairs_hook=OrderedDict)
                self.pipeline = data.get('filters', OrderedDict())
            print(f"✓ Pipeline cargado: {len(self.pipeline)} filtros")
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: JSON inválido en pipeline.json: {e}")
            return False
        
        # Cargar params.json (puede no existir)
        if self.params_path.exists():
            try:
                with open(self.params_path, 'r') as f:
                    data = json.load(f, object_pairs_hook=OrderedDict)
                    self.params = data.get('filter_params', {})
                print(f"✓ Parámetros cargados: {len(self.params)} filtros")
            except json.JSONDecodeError as e:
                print(f"❌ ERROR: JSON inválido en params.json: {e}")
                return False
        else:
            print(f"⚠  {self.params_path} no existe, se creará nuevo")
            self.params = {}
        
        return True
    
    def analyze(self) -> List[Dict]:
        """
        Analiza las diferencias entre pipeline y params.
        Retorna lista de issues encontrados.
        """
        self.issues = []
        
        # Issue types y su criticidad:
        # - 'mismatch': Mismo ID, diferente filtro [BLOQUEANTE]
        # - 'orphan': Parámetros sin filtro correspondiente [ADVERTENCIA]
        # - 'new': Filtro sin parámetros guardados [INFO - OK]
        
        pipeline_ids = set(self.pipeline.keys())
        params_ids = set(self.params.keys())
        
        # 1. Detectar mismatches (mismo ID, diferente filter_name)
        for filter_id in pipeline_ids & params_ids:
            pipeline_filter = self.pipeline[filter_id]['filter_name']
            params_filter = self.params[filter_id].get('filter_name', '')
            
            if pipeline_filter != params_filter:
                self.issues.append({
                    'type': 'mismatch',
                    'filter_id': filter_id,
                    'pipeline_filter': pipeline_filter,
                    'params_filter': params_filter,
                    'params_data': self.params[filter_id].get('params', {})
                })
        
        # 2. Detectar huérfanos (parámetros sin filtro)
        orphans = params_ids - pipeline_ids
        for filter_id in orphans:
            self.issues.append({
                'type': 'orphan',
                'filter_id': filter_id,
                'params_filter': self.params[filter_id].get('filter_name', 'unknown'),
                'params_data': self.params[filter_id].get('params', {})
            })
        
        # 3. Detectar nuevos (filtros sin parámetros)
        new_filters = pipeline_ids - params_ids
        for filter_id in new_filters:
            self.issues.append({
                'type': 'new',
                'filter_id': filter_id,
                'pipeline_filter': self.pipeline[filter_id]['filter_name']
            })
        
        return self.issues
    
    def has_blocking_issues(self) -> bool:
        """
        Verifica si hay issues que bloquean la ejecución.
        Solo MISMATCH es bloqueante. NEW y ORPHAN son advertencias.
        """
        for issue in self.issues:
            if issue['type'] == 'mismatch':
                return True
        return False
    
    def get_blocking_issues(self) -> List[Dict]:
        """Retorna solo los issues bloqueantes"""
        return [i for i in self.issues if i['type'] == 'mismatch']
    
    def get_warning_issues(self) -> List[Dict]:
        """Retorna issues no bloqueantes (advertencias)"""
        return [i for i in self.issues if i['type'] == 'orphan']
    
    def get_info_issues(self) -> List[Dict]:
        """Retorna issues informativos (OK)"""
        return [i for i in self.issues if i['type'] == 'new']
    
    def print_report(self):
        """Imprime reporte de issues encontrados"""
        if not self.issues:
            print("\n" + "="*60)
            print("✅ TODO OK - Pipeline y parámetros están sincronizados")
            print("="*60)
            return
        
        blocking = self.get_blocking_issues()
        warnings = self.get_warning_issues()
        info = self.get_info_issues()
        
        print("\n" + "="*60)
        if blocking:
            print(f"❌ ERRORES BLOQUEANTES: {len(blocking)}")
        elif warnings:
            print(f"⚠️  ADVERTENCIAS: {len(warnings)}")
        if info:
            print(f"ℹ️  INFORMACIÓN: {len(info)}")
        print("="*60)
        
        # Agrupar por tipo
        from collections import defaultdict
        by_type = defaultdict(list)
        for issue in self.issues:
            by_type[issue['type']].append(issue)
        
        # Mismatches (BLOQUEANTES)
        if 'mismatch' in by_type:
            print(f"\n❌ CAMBIOS DE FILTRO - BLOQUEANTE ({len(by_type['mismatch'])}):")
            print("   Estos requieren corrección antes de continuar.")
            for issue in by_type['mismatch']:
                print(f"  ID '{issue['filter_id']}':")
                print(f"    Pipeline: {issue['pipeline_filter']}")
                print(f"    Params:   {issue['params_filter']}")
        
        # Huérfanos (ADVERTENCIA)
        if 'orphan' in by_type:
            print(f"\n⚠️  PARÁMETROS HUÉRFANOS - ADVERTENCIA ({len(by_type['orphan'])}):")
            print("   Estos pueden limpiarse pero no bloquean ejecución.")
            for issue in by_type['orphan']:
                print(f"  ID '{issue['filter_id']}': {issue['params_filter']}")
                print(f"    (este filtro ya no está en pipeline.json)")
        
        # Nuevos (INFORMACIÓN)
        if 'new' in by_type:
            print(f"\nℹ️  FILTROS NUEVOS - OK ({len(by_type['new'])}):")
            print("   Estos usarán valores por defecto. Esto es normal.")
            for issue in by_type['new']:
                print(f"  ID '{issue['filter_id']}': {issue['pipeline_filter']}")
        
        print("="*60)
    
    def interactive_fix(self) -> bool:
        """Modo interactivo: pregunta al usuario cómo resolver cada issue"""
        if not self.issues:
            return True
        
        blocking = self.get_blocking_issues()
        warnings = self.get_warning_issues()
        info = self.get_info_issues()
        
        print("\n" + "="*60)
        print("MODO INTERACTIVO - Resolución de problemas")
        print("="*60 + "\n")
        
        if blocking:
            print(f"❌ {len(blocking)} error(es) crítico(s) que DEBEN resolverse")
        if warnings:
            print(f"⚠️  {len(warnings)} advertencia(s) opcional(es)")
        if info:
            print(f"ℹ️  {len(info)} filtro(s) nuevo(s) (OK)")
        
        modified = False
        
        # Procesar cada issue
        for i, issue in enumerate(self.issues, 1):
            print(f"\n[{i}/{len(self.issues)}] ", end="")
            
            if issue['type'] == 'mismatch':
                print(f"CAMBIO DE FILTRO en ID '{issue['filter_id']}'")
                print(f"  Pipeline: {issue['pipeline_filter']}")
                print(f"  Params:   {issue['params_filter']}")
                print("\nOpciones:")
                print("  1) Usar parámetros guardados para el NUEVO filtro (si son compatibles)")
                print("  2) Descartar parámetros antiguos (usar defaults del nuevo filtro)")
                print("  3) Cancelar (mantener como está)")
                
                choice = input("\nElige opción (1/2/3): ").strip()
                
                if choice == '1':
                    # Intentar mantener parámetros si son compatibles
                    new_filter_class = get_filter(issue['pipeline_filter'])
                    if new_filter_class:
                        # Filtrar solo parámetros válidos
                        valid_params = {
                            k: v for k, v in issue['params_data'].items()
                            if k in new_filter_class.PARAMS
                        }
                        self.params[issue['filter_id']] = {
                            'filter_name': issue['pipeline_filter'],
                            'params': valid_params
                        }
                        print(f"  ✓ Parámetros compatibles transferidos: {list(valid_params.keys())}")
                        modified = True
                    else:
                        print(f"  ⚠️  Filtro {issue['pipeline_filter']} no encontrado")
                
                elif choice == '2':
                    del self.params[issue['filter_id']]
                    print(f"  ✓ Parámetros antiguos descartados")
                    modified = True
            
            elif issue['type'] == 'orphan':
                print(f"PARÁMETROS HUÉRFANOS en ID '{issue['filter_id']}'")
                print(f"  Filtro: {issue['params_filter']}")
                print(f"  (ya no existe en pipeline.json)")
                print("\nOpciones:")
                print("  1) Eliminar estos parámetros")
                print("  2) Mantener (por si lo vuelves a usar)")
                
                choice = input("\nElige opción (1/2): ").strip()
                
                if choice == '1':
                    del self.params[issue['filter_id']]
                    print(f"  ✓ Parámetros eliminados")
                    modified = True
            
            elif issue['type'] == 'new':
                print(f"FILTRO NUEVO en ID '{issue['filter_id']}'")
                print(f"  Filtro: {issue['pipeline_filter']}")
                print(f"  ✅ Esto está OK. Usará valores por defecto.")
                print(f"  Podrás configurar sus parámetros en param_configurator.py")
                print("\nPresiona Enter para continuar...")
                input()
        
        if modified:
            self.save_params()
            print("\n✅ Cambios guardados en params.json")
        else:
            print("\n⚠️  No se hicieron cambios")
        
        return True
    
    def auto_clean(self) -> bool:
        """Modo automático: elimina huérfanos, acepta nuevos"""
        if not self.issues:
            return True
        
        print("\n" + "="*60)
        print("MODO AUTO-CLEAN")
        print("="*60)
        
        modified = False
        
        for issue in self.issues:
            if issue['type'] == 'orphan':
                print(f"  🗑️  Eliminando parámetros huérfanos: ID '{issue['filter_id']}' ({issue['params_filter']})")
                del self.params[issue['filter_id']]
                modified = True
            
            elif issue['type'] == 'new':
                print(f"  ✓ Filtro nuevo aceptado: ID '{issue['filter_id']}' ({issue['pipeline_filter']})")
            
            elif issue['type'] == 'mismatch':
                print(f"  ⚠️  ADVERTENCIA: Cambio de filtro en ID '{issue['filter_id']}'")
                print(f"      {issue['params_filter']} → {issue['pipeline_filter']}")
                print(f"      Eliminando parámetros antiguos")
                del self.params[issue['filter_id']]
                modified = True
        
        if modified:
            self.save_params()
            print("\n✅ Limpieza completada y guardada")
        
        return True
    
    def save_params(self):
        """Guarda params.json con la estructura correcta"""
        data = {
            "version": "1.0",
            "description": "Parámetros guardados para los filtros del pipeline",
            "last_modified": datetime.now().isoformat(),
            "filter_params": self.params
        }
        
        with open(self.params_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"✓ Guardado: {self.params_path}")
    
    def validate_only(self) -> bool:
        """
        Modo validación: solo verifica y retorna True si está OK o solo hay advertencias.
        SOLO bloquea si hay errores críticos (mismatch).
        Usado por param_configurator.py
        """
        if not self.issues:
            return True
        
        blocking = self.get_blocking_issues()
        warnings = self.get_warning_issues()
        info = self.get_info_issues()
        
        # Si solo hay info o warnings, está OK para continuar
        if not blocking:
            if warnings or info:
                print("\n" + "="*60)
                print("⚠️  AVISOS DETECTADOS (no bloqueantes)")
                print("="*60)
                
                if warnings:
                    print(f"\n{len(warnings)} advertencia(s):")
                    for w in warnings:
                        print(f"  • Parámetros huérfanos en ID '{w['filter_id']}'")
                    print("\nPuedes limpiar esto ejecutando:")
                    print("  python sync_pipeline_params.py --auto-clean")
                
                if info:
                    print(f"\n{len(info)} filtro(s) nuevo(s) detectado(s):")
                    for i in info:
                        print(f"  • ID '{i['filter_id']}': {i['pipeline_filter']} (usará defaults)")
                
                print("\n✅ Puedes continuar. param_configurator.py funcionará normalmente.")
                print("="*60)
            return True
        
        # Hay errores bloqueantes
        print("\n" + "="*60)
        print("❌ VALIDACIÓN FALLIDA - ERRORES CRÍTICOS")
        print("="*60)
        print(f"Se encontraron {len(blocking)} errores bloqueantes:")
        
        for err in blocking:
            print(f"\n  • ID '{err['filter_id']}': Cambio de filtro")
            print(f"    Era: {err['params_filter']}")
            print(f"    Ahora: {err['pipeline_filter']}")
        
        print("\n⚠️  Estos errores DEBEN corregirse antes de continuar.")
        print("\nEjecuta este comando para corregirlos:")
        print("  python sync_pipeline_params.py")
        print("\nLuego vuelve a ejecutar param_configurator.py")
        print("="*60)
        
        return False


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Sincroniza pipeline.json y params.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python sync_pipeline_params.py                  # Modo interactivo
  python sync_pipeline_params.py --validate-only  # Solo validar
  python sync_pipeline_params.py --auto-clean     # Limpiar automáticamente
        """
    )
    parser.add_argument('--validate-only', action='store_true',
                       help='Solo validar (para uso desde param_configurator)')
    parser.add_argument('--auto-clean', action='store_true',
                       help='Limpiar huérfanos automáticamente')
    parser.add_argument('--pipeline', default='pipeline.json',
                       help='Path a pipeline.json (default: pipeline.json)')
    parser.add_argument('--params', default='params.json',
                       help='Path a params.json (default: params.json)')
    
    args = parser.parse_args()
    
    # Crear sincronizador
    sync = PipelineParamsSync(args.pipeline, args.params)
    
    # Cargar archivos
    if not sync.load_files():
        sys.exit(1)
    
    # Analizar
    print("\n🔍 Analizando diferencias...")
    sync.analyze()
    
    # Imprimir reporte
    sync.print_report()
    
    # Ejecutar según modo
    if args.validate_only:
        # Modo validación (para param_configurator)
        if sync.validate_only():
            print("\n✅ Validación exitosa")
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif args.auto_clean:
        # Modo auto-clean
        sync.auto_clean()
        sys.exit(0)
    
    else:
        # Modo interactivo
        if not sync.issues:
            sys.exit(0)
        
        print("\n¿Deseas corregir estos problemas? (s/n): ", end="")
        response = input().strip().lower()
        
        if response == 's':
            sync.interactive_fix()
            sys.exit(0)
        else:
            print("Cancelado. No se hicieron cambios.")
            sys.exit(1)


if __name__ == "__main__":
    main()
