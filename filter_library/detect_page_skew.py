"""
Filtro: DetectPageSkew
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from .base_filter import BaseFilter, FILTER_REGISTRY


class DetectPageSkew(BaseFilter):
    """Detecta el ángulo de inclinación de la página mediante clustering de líneas"""
    
    FILTER_NAME = "DetectPageSkew"
    DESCRIPTION = "Detecta el ángulo de inclinación de la página analizando clusters dominantes de líneas de Hough. Busca grupos mayoritarios de líneas horizontales y verticales con inclinación similar."
    
    INPUTS = {
        "lines_data": "lines",
        "base_image": "image"
    }
    
    OUTPUTS = {
        "skew_angle": "float",
        "skew_metadata": "metadata",
        "sample_image": "image"
    }
    
    PARAMS = {
        "angle_bin_size": {
            "default": 5,
            "min": 1,
            "max": 20,
            "step": 1,
            "description": "Tamaño del bin de ángulos (en décimas de grado). Menor = más preciso pero más fragmentado."
        },
        "min_cluster_size": {
            "default": 5,
            "min": 1,
            "max": 50,
            "step": 1,
            "description": "Mínimo de líneas para considerar un cluster válido."
        },
        "h_v_tolerance": {
            "default": 30,
            "min": 10,
            "max": 45,
            "step": 5,
            "description": "Tolerancia (grados) para separar líneas horizontales de verticales."
        },
        "perpendicularity_tolerance": {
            "default": 5,
            "min": 1,
            "max": 15,
            "step": 1,
            "description": "Tolerancia (grados) para validar que clusters H y V sean perpendiculares."
        },
        "use_horizontal_only": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Usar solo horizontales para calcular skew (0=No, 1=Sí)."
        },
        "use_vertical_only": {
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "description": "Usar solo verticales para calcular skew (0=No, 1=Sí)."
        },
        "visualization_mode": {
            "default": 0,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": "Visualización: 0=Histograma, 1=Líneas coloreadas, 2=Ambos."
        }
    }
    
    
    def _get_line_angle(self, line: Dict) -> float:
        """Calcula el ángulo de una línea en grados (0-180)."""
        if "angle" in line:
            return line["angle"]
        
        x1, y1 = line["x1"], line["y1"]
        x2, y2 = line["x2"], line["y2"]
        
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle < 0:
            angle += 180
        return angle
    
    def _normalize_angle_to_skew(self, angle: float) -> float:
        """
        Normaliza el ángulo al rango de skew (-45 a +45 grados).
        0° = horizontal perfecto
        90° = vertical perfecto
        
        Para skew de página:
        - Líneas horizontales cerca de 0° o 180° → skew pequeño
        - Líneas verticales cerca de 90° → skew pequeño
        """
        # Convertir a rango -90 a +90
        if angle > 90:
            angle = angle - 180
        
        # Para ángulos cerca de 90° (verticales), convertir a referencia horizontal
        # Una línea vertical con skew de 1° está en 91° o 89°
        # La queremos mapear a ±1°
        if angle > 45:
            angle = 90 - angle  # 91° → -1°, 89° → 1°
        elif angle < -45:
            angle = -90 - angle  # -89° → -1°, -91° → 1°
        
        return angle
    
    def _is_near_horizontal(self, angle: float, tolerance: float) -> bool:
        """Determina si una línea es aproximadamente horizontal."""
        return angle < tolerance or angle > (180 - tolerance)
    
    def _is_near_vertical(self, angle: float, tolerance: float) -> bool:
        """Determina si una línea es aproximadamente vertical."""
        return abs(angle - 90) < tolerance
    
    def _find_dominant_cluster(self, angles: List[float], bin_size_tenths: int, 
                               min_cluster_size: int) -> Tuple[float, int, float]:
        """
        Encuentra el cluster dominante de ángulos.
        
        Returns:
            (angle_mean, count, confidence)
        """
        if not angles:
            return (0.0, 0, 0.0)
        
        # Convertir bin_size de décimas a grados
        bin_size = bin_size_tenths / 10.0
        
        # Crear bins para histograma
        bins = np.arange(-90, 90 + bin_size, bin_size)
        hist, edges = np.histogram(angles, bins=bins)
        
        # Encontrar el bin más poblado
        max_idx = np.argmax(hist)
        max_count = hist[max_idx]
        
        if max_count < min_cluster_size:
            return (0.0, 0, 0.0)
        
        # Obtener todas las líneas en ese bin
        bin_start = edges[max_idx]
        bin_end = edges[max_idx + 1]
        
        cluster_angles = [a for a in angles if bin_start <= a < bin_end]
        
        # Calcular promedio del cluster
        angle_mean = np.mean(cluster_angles)
        
        # Calcular confianza (% de líneas en el cluster dominante)
        confidence = (max_count / len(angles)) * 100
        
        return (angle_mean, max_count, confidence)
    
    def _create_histogram_visualization(self, h_angles: List[float], v_angles: List[float],
                                       h_cluster_angle: float, v_cluster_angle: float,
                                       width: int = 800, height: int = 400) -> np.ndarray:
        """Crea visualización de histograma de ángulos."""
        vis = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Configuración
        margin = 50
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin
        
        # Combinar todos los ángulos
        all_angles = h_angles + v_angles
        if not all_angles:
            return vis
        
        # Crear histograma
        bins = np.arange(-90, 91, 1)
        hist, edges = np.histogram(all_angles, bins=bins)
        
        # Normalizar para visualización
        max_count = np.max(hist) if np.max(hist) > 0 else 1
        
        # Dibujar barras
        bar_width = plot_width / len(hist)
        for i, count in enumerate(hist):
            if count == 0:
                continue
            
            bar_height = int((count / max_count) * plot_height)
            x = int(margin + i * bar_width)
            y = margin + plot_height - bar_height
            
            # Color según tipo de línea
            angle_center = (edges[i] + edges[i + 1]) / 2
            if abs(angle_center) < 30:  # Horizontal
                color = (100, 150, 255)  # Naranja claro
            elif abs(angle_center - 90) < 30 or abs(angle_center + 90) < 30:  # Vertical
                color = (255, 150, 100)  # Azul claro
            else:
                color = (200, 200, 200)  # Gris
            
            cv2.rectangle(vis, (x, y), (x + int(bar_width), margin + plot_height), color, -1)
        
        # Dibujar ejes
        cv2.line(vis, (margin, margin + plot_height), (width - margin, margin + plot_height), 
                (0, 0, 0), 2)
        cv2.line(vis, (margin, margin), (margin, margin + plot_height), (0, 0, 0), 2)
        
        # Marcar clusters dominantes
        if h_cluster_angle is not None:
            x = int(margin + ((h_cluster_angle + 90) / 180) * plot_width)
            cv2.line(vis, (x, margin), (x, margin + plot_height), (0, 0, 255), 2)
            cv2.putText(vis, f"H: {h_cluster_angle:.2f}°", (x + 5, margin + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if v_cluster_angle is not None:
            x = int(margin + ((v_cluster_angle + 90) / 180) * plot_width)
            cv2.line(vis, (x, margin), (x, margin + plot_height), (255, 0, 0), 2)
            cv2.putText(vis, f"V: {v_cluster_angle:.2f}°", (x + 5, margin + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Etiquetas de ejes
        cv2.putText(vis, "-90°", (margin - 30, margin + plot_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(vis, "0°", (margin + plot_width // 2 - 10, margin + plot_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(vis, "90°", (width - margin - 20, margin + plot_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return vis
    
    def _create_lines_visualization(self, base_img: np.ndarray, lines_data: List[Dict],
                                   h_cluster_lines: List[Dict], v_cluster_lines: List[Dict],
                                   skew_angle: float) -> np.ndarray:
        """Crea visualización de líneas coloreadas por cluster."""
        if len(base_img.shape) == 2:
            vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            vis = base_img.copy()
        
        h, w = vis.shape[:2]
        
        # Crear conjunto de líneas en clusters para búsqueda rápida
        h_cluster_set = set()
        v_cluster_set = set()
        
        for line in h_cluster_lines:
            key = (line["x1"], line["y1"], line["x2"], line["y2"])
            h_cluster_set.add(key)
        
        for line in v_cluster_lines:
            key = (line["x1"], line["y1"], line["x2"], line["y2"])
            v_cluster_set.add(key)
        
        # Dibujar todas las líneas
        for line in lines_data:
            x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
            key = (x1, y1, x2, y2)
            
            if key in h_cluster_set:
                color = (0, 255, 0)  # Verde - horizontal dominante
                thickness = 2
            elif key in v_cluster_set:
                color = (255, 0, 0)  # Azul - vertical dominante
                thickness = 2
            else:
                color = (128, 128, 128)  # Gris - outliers
                thickness = 1
            
            cv2.line(vis, (x1, y1), (x2, y2), color, thickness)
        
        # Agregar información de skew
        info_text = f"Skew detectado: {skew_angle:.2f} grados"
        cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 0), 1)
        
        # Leyenda
        y_offset = 60
        cv2.rectangle(vis, (10, y_offset), (30, y_offset + 15), (0, 255, 0), -1)
        cv2.putText(vis, f"Horizontales: {len(h_cluster_lines)}", (35, y_offset + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.rectangle(vis, (10, y_offset), (30, y_offset + 15), (255, 0, 0), -1)
        cv2.putText(vis, f"Verticales: {len(v_cluster_lines)}", (35, y_offset + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.rectangle(vis, (10, y_offset), (30, y_offset + 15), (128, 128, 128), -1)
        cv2.putText(vis, f"Otros: {len(lines_data) - len(h_cluster_lines) - len(v_cluster_lines)}", 
                   (35, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        lines_data = inputs.get("lines_data", [])
        base_img = inputs.get("base_image", original_image)
        
        if not lines_data:
            # Sin líneas, retornar skew 0
            return {
                "skew_angle": 0.0,
                "skew_metadata": {
                    "error": "No lines provided",
                    "total_lines": 0
                },
                "sample_image": base_img if len(base_img.shape) == 3 else cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
            }
        
        # Obtener parámetros
        bin_size = self.params["angle_bin_size"]
        min_cluster = self.params["min_cluster_size"]
        h_v_tol = self.params["h_v_tolerance"]
        perp_tol = self.params["perpendicularity_tolerance"]
        use_h_only = bool(self.params["use_horizontal_only"])
        use_v_only = bool(self.params["use_vertical_only"])
        vis_mode = self.params["visualization_mode"]
        
        # Extraer ángulos de todas las líneas
        all_angles = [self._get_line_angle(line) for line in lines_data]
        
        # Separar en horizontales y verticales
        h_lines = []
        v_lines = []
        
        for i, line in enumerate(lines_data):
            angle = all_angles[i]
            if self._is_near_horizontal(angle, h_v_tol):
                h_lines.append(line)
            elif self._is_near_vertical(angle, h_v_tol):
                v_lines.append(line)
        
        # Normalizar ángulos a rango de skew (-45 a +45)
        h_angles_norm = [self._normalize_angle_to_skew(self._get_line_angle(l)) for l in h_lines]
        v_angles_norm = [self._normalize_angle_to_skew(self._get_line_angle(l)) for l in v_lines]
        
        # Encontrar clusters dominantes
        h_angle, h_count, h_conf = self._find_dominant_cluster(h_angles_norm, bin_size, min_cluster)
        v_angle, v_count, v_conf = self._find_dominant_cluster(v_angles_norm, bin_size, min_cluster)
        
        # Calcular skew final
        skew_angle = 0.0
        is_perpendicular = False
        
        if use_h_only:
            if h_count > 0:
                skew_angle = h_angle
        elif use_v_only:
            if v_count > 0:
                skew_angle = v_angle
        else:
            # Usar ambos con validación de perpendicularidad
            if h_count > 0 and v_count > 0:
                # Verificar perpendicularidad
                angle_diff = abs(abs(h_angle - v_angle) - 90)
                is_perpendicular = angle_diff < perp_tol
                
                if is_perpendicular:
                    # Promedio ponderado por confianza
                    total_weight = h_conf + v_conf
                    if total_weight > 0:
                        skew_angle = (h_angle * h_conf + v_angle * v_conf) / total_weight
                else:
                    # No son perpendiculares, usar el más confiable
                    skew_angle = h_angle if h_conf >= v_conf else v_angle
            elif h_count > 0:
                skew_angle = h_angle
            elif v_count > 0:
                skew_angle = v_angle
        
        # Preparar metadata
        metadata = {
            "total_lines": len(lines_data),
            "horizontal_lines": len(h_lines),
            "vertical_lines": len(v_lines),
            "other_lines": len(lines_data) - len(h_lines) - len(v_lines),
            "horizontal_cluster": {
                "angle": float(h_angle),
                "count": int(h_count),
                "confidence": float(h_conf)
            },
            "vertical_cluster": {
                "angle": float(v_angle),
                "count": int(v_count),
                "confidence": float(v_conf)
            },
            "is_perpendicular": is_perpendicular,
            "final_skew_angle": float(skew_angle)
        }
        
        # Identificar líneas de cada cluster para visualización
        h_cluster_lines = []
        v_cluster_lines = []
        
        if h_count > 0:
            h_cluster_lines = [
                line for i, line in enumerate(h_lines)
                if abs(h_angles_norm[i] - h_angle) < (bin_size / 10.0)
            ]
        
        if v_count > 0:
            v_cluster_lines = [
                line for i, line in enumerate(v_lines)
                if abs(v_angles_norm[i] - v_angle) < (bin_size / 10.0)
            ]
        
        # Crear visualización
        if vis_mode == 0:
            # Solo histograma
            sample = self._create_histogram_visualization(
                h_angles_norm, v_angles_norm, h_angle if h_count > 0 else None, 
                v_angle if v_count > 0 else None
            )
        elif vis_mode == 1:
            # Solo líneas coloreadas
            sample = self._create_lines_visualization(
                base_img, lines_data, h_cluster_lines, v_cluster_lines, skew_angle
            )
        else:
            # Ambos - split screen
            hist_vis = self._create_histogram_visualization(
                h_angles_norm, v_angles_norm, h_angle if h_count > 0 else None,
                v_angle if v_count > 0 else None
            )
            lines_vis = self._create_lines_visualization(
                base_img, lines_data, h_cluster_lines, v_cluster_lines, skew_angle
            )
            
            # Redimensionar para combinar
            h_img = base_img.shape[0]
            w_img = base_img.shape[1]
            
            # Resize histogram para que coincida en altura
            hist_vis_resized = cv2.resize(hist_vis, (w_img, h_img // 3))
            
            # Combinar verticalmente
            sample = np.vstack([hist_vis_resized, lines_vis])
        
        return {
            "skew_angle": float(skew_angle),
            "skew_metadata": metadata,
            "sample_image": sample
        }
