"""
Filtro: CalculateRotationFromLines
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_filter import BaseFilter, FILTER_REGISTRY


class CalculateRotationFromLines(BaseFilter):
    """Calcula el ángulo de rotación de la página a partir de líneas horizontales y/o verticales"""
    
    FILTER_NAME = "CalculateRotationFromLines"
    DESCRIPTION = "Calcula el ángulo de rotación analizando clusters dominantes de líneas horizontales y/o verticales. Genera visualización con grilla rotada e histograma de ángulos."
    
    INPUTS = {
        "horizontal_lines": "lines",
        "vertical_lines": "lines",
        "lines_metadata": "metadata",
        "base_image": "image"
    }
    
    OUTPUTS = {
        "rotation_angle": "float",
        "rotation_metadata": "metadata",
        "grid_preview": "image",
        "histogram_preview": "image",
        "sample_image": "image"
    }
    
    PARAMS = {
        "use_mode": {
            "default": 1,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": "Modo: 0=Solo horizontales, 1=Solo verticales, 2=Ambas (promedio ponderado)."
        },
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
        "perpendicularity_tolerance": {
            "default": 5,
            "min": 1,
            "max": 15,
            "step": 1,
            "description": "Tolerancia (grados) para validar que clusters H y V sean perpendiculares (solo modo 2)."
        },
        "grid_spacing": {
            "default": 30,
            "min": 10,
            "max": 100,
            "step": 5,
            "description": "Espaciado entre líneas de la grilla en píxeles."
        },
        "grid_thickness": {
            "default": 1,
            "min": 1,
            "max": 5,
            "step": 1,
            "description": "Grosor de las líneas de la grilla."
        },
        "grid_color_r": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de la grilla - Rojo."
        },
        "grid_color_g": {
            "default": 255,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de la grilla - Verde."
        },
        "grid_color_b": {
            "default": 0,
            "min": 0,
            "max": 255,
            "step": 5,
            "description": "Color de la grilla - Azul."
        },
        "grid_alpha": {
            "default": 70,
            "min": 0,
            "max": 100,
            "step": 5,
            "description": "Transparencia de la grilla (0-100)."
        },
        "preview_mode": {
            "default": 2,
            "min": 0,
            "max": 2,
            "step": 1,
            "description": "Vista previa: 0=Solo grilla, 1=Solo histograma, 2=Ambas (split)."
        }
    }
    
    def _normalize_angle_to_skew(self, angle: float) -> float:
        """
        Normaliza el ángulo al rango de skew (-45 a +45 grados).
        0° = horizontal perfecto
        90° = vertical perfecto
        """
        # Convertir a rango -90 a +90
        if angle > 90:
            angle = angle - 180
        
        # Para ángulos cerca de 90° (verticales), convertir a referencia horizontal
        if angle > 45:
            angle = 90 - angle
        elif angle < -45:
            angle = -90 - angle
        
        return angle
    
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
    
    def _create_grid_preview(self, base_img: np.ndarray, rotation_angle: float,
                            spacing: int, thickness: int, color: Tuple[int, int, int],
                            alpha: int) -> np.ndarray:
        """Crea visualización con grilla rotada"""
        
        if len(base_img.shape) == 2:
            vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            vis = base_img.copy()
        
        h, w = vis.shape[:2]
        
        # Crear imagen para la grilla
        grid = np.zeros_like(vis)
        
        # Centro de la imagen
        cx, cy = w // 2, h // 2
        
        # Calcular cuántas líneas necesitamos (con margen)
        max_dim = int(np.sqrt(w**2 + h**2))
        num_lines = (max_dim // spacing) + 2
        
        # Convertir ángulo a radianes (negativo porque OpenCV usa coordenadas invertidas en Y)
        angle_rad = np.radians(-rotation_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Dibujar líneas verticales (rotadas)
        for i in range(-num_lines, num_lines + 1):
            offset = i * spacing
            
            # Punto inicial y final en el sistema rotado
            x_offset = offset * cos_a
            y_offset = offset * sin_a
            
            # Extender la línea en dirección perpendicular
            perp_x = -sin_a * max_dim
            perp_y = cos_a * max_dim
            
            x1 = int(cx + x_offset - perp_x)
            y1 = int(cy + y_offset - perp_y)
            x2 = int(cx + x_offset + perp_x)
            y2 = int(cy + y_offset + perp_y)
            
            cv2.line(grid, (x1, y1), (x2, y2), color, thickness)
        
        # Dibujar líneas horizontales (rotadas)
        # Estas son perpendiculares a las verticales
        angle_perp_rad = angle_rad + np.pi / 2
        cos_b = np.cos(angle_perp_rad)
        sin_b = np.sin(angle_perp_rad)
        
        for i in range(-num_lines, num_lines + 1):
            offset = i * spacing
            
            x_offset = offset * cos_b
            y_offset = offset * sin_b
            
            perp_x = -sin_b * max_dim
            perp_y = cos_b * max_dim
            
            x1 = int(cx + x_offset - perp_x)
            y1 = int(cy + y_offset - perp_y)
            x2 = int(cx + x_offset + perp_x)
            y2 = int(cy + y_offset + perp_y)
            
            cv2.line(grid, (x1, y1), (x2, y2), color, thickness)
        
        # Mezclar con transparencia
        alpha_factor = alpha / 100.0
        result = cv2.addWeighted(vis, 1.0, grid, alpha_factor, 0)
        
        # Agregar información
        info_text = f"Rotacion: {rotation_angle:.2f} grados"
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Fondo para texto
        overlay = result.copy()
        cv2.rectangle(overlay, (8, 8), (12 + text_size[0], 38), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
        
        cv2.putText(result, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return result
    
    def _create_histogram_preview(self, h_angles: List[float], v_angles: List[float],
                                  h_cluster_angle: float, h_cluster_count: int,
                                  v_cluster_angle: float, v_cluster_count: int,
                                  rotation_angle: float, use_mode: int,
                                  width: int = 800, height: int = 400) -> np.ndarray:
        """Crea visualización de histograma de ángulos"""
        
        vis = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        margin = 50
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin
        
        # Combinar ángulos según modo
        if use_mode == 0:  # Solo horizontales
            all_angles = h_angles
            title = "Histograma - Solo Horizontales"
        elif use_mode == 1:  # Solo verticales
            all_angles = v_angles
            title = "Histograma - Solo Verticales"
        else:  # Ambas
            all_angles = h_angles + v_angles
            title = "Histograma - Horizontales + Verticales"
        
        if not all_angles:
            cv2.putText(vis, "Sin datos de angulos", (width // 2 - 100, height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
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
            
            # Color según tipo
            angle_center = (edges[i] + edges[i + 1]) / 2
            if abs(angle_center) < 30:  # Horizontal
                color = (100, 150, 255)
            elif abs(angle_center - 90) < 30 or abs(angle_center + 90) < 30:  # Vertical
                color = (255, 150, 100)
            else:
                color = (200, 200, 200)
            
            cv2.rectangle(vis, (x, y), (x + int(bar_width), margin + plot_height), color, -1)
        
        # Dibujar ejes
        cv2.line(vis, (margin, margin + plot_height), (width - margin, margin + plot_height), 
                (0, 0, 0), 2)
        cv2.line(vis, (margin, margin), (margin, margin + plot_height), (0, 0, 0), 2)
        
        # Marcar cluster horizontal si existe
        if h_cluster_count > 0:
            x = int(margin + ((h_cluster_angle + 90) / 180) * plot_width)
            cv2.line(vis, (x, margin), (x, margin + plot_height), (255, 0, 0), 2)
            cv2.putText(vis, f"H: {h_cluster_angle:.2f}°", (x + 5, margin + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Marcar cluster vertical si existe
        if v_cluster_count > 0:
            x = int(margin + ((v_cluster_angle + 90) / 180) * plot_width)
            cv2.line(vis, (x, margin), (x, margin + plot_height), (0, 0, 255), 2)
            cv2.putText(vis, f"V: {v_cluster_angle:.2f}°", (x + 5, margin + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Marcar rotación final
        x = int(margin + ((rotation_angle + 90) / 180) * plot_width)
        cv2.line(vis, (x, margin), (x, margin + plot_height), (0, 255, 0), 3)
        cv2.putText(vis, f"Rotacion: {rotation_angle:.2f}°", (x + 5, margin + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Etiquetas de ejes
        cv2.putText(vis, "-90°", (margin - 30, margin + plot_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(vis, "0°", (margin + plot_width // 2 - 10, margin + plot_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(vis, "90°", (width - margin - 20, margin + plot_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Título
        cv2.putText(vis, title, (margin, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return vis
    
    def process(self, inputs: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        h_lines = inputs.get("horizontal_lines", [])
        v_lines = inputs.get("vertical_lines", [])
        lines_metadata = inputs.get("lines_metadata", {})
        base_img = inputs.get("base_image", original_image)
        
        h, w = base_img.shape[:2]
        
        # Obtener parámetros
        use_mode = self.params["use_mode"]
        bin_size = self.params["angle_bin_size"]
        min_cluster = self.params["min_cluster_size"]
        perp_tol = self.params["perpendicularity_tolerance"]
        
        grid_spacing = self.params["grid_spacing"]
        grid_thickness = self.params["grid_thickness"]
        grid_alpha = self.params["grid_alpha"]
        preview_mode = self.params["preview_mode"]
        
        grid_color = (
            self.params["grid_color_b"],
            self.params["grid_color_g"],
            self.params["grid_color_r"]
        )
        
        # Extraer y normalizar ángulos
        h_angles = [self._normalize_angle_to_skew(line["angle"]) for line in h_lines]
        v_angles = [self._normalize_angle_to_skew(line["angle"]) for line in v_lines]
        
        # Encontrar clusters dominantes
        h_angle, h_count, h_conf = self._find_dominant_cluster(h_angles, bin_size, min_cluster)
        v_angle, v_count, v_conf = self._find_dominant_cluster(v_angles, bin_size, min_cluster)
        
        # Calcular rotación según modo
        rotation_angle = 0.0
        is_perpendicular = False
        calculation_mode = ""
        
        if use_mode == 0:  # Solo horizontales
            if h_count > 0:
                rotation_angle = h_angle
                calculation_mode = "horizontal_only"
        elif use_mode == 1:  # Solo verticales
            if v_count > 0:
                rotation_angle = v_angle
                calculation_mode = "vertical_only"
        else:  # Ambas
            if h_count > 0 and v_count > 0:
                # Verificar perpendicularidad
                angle_diff = abs(abs(h_angle - v_angle) - 90)
                is_perpendicular = angle_diff < perp_tol
                
                if is_perpendicular:
                    # Promedio ponderado por confianza
                    total_weight = h_conf + v_conf
                    if total_weight > 0:
                        rotation_angle = (h_angle * h_conf + v_angle * v_conf) / total_weight
                    calculation_mode = "both_perpendicular"
                else:
                    # No son perpendiculares, usar el más confiable
                    rotation_angle = h_angle if h_conf >= v_conf else v_angle
                    calculation_mode = "both_non_perpendicular"
            elif h_count > 0:
                rotation_angle = h_angle
                calculation_mode = "horizontal_fallback"
            elif v_count > 0:
                rotation_angle = v_angle
                calculation_mode = "vertical_fallback"
        
        # Preparar metadata
        rotation_metadata = {
            "image_width": int(w),
            "image_height": int(h),
            "rotation_angle": float(rotation_angle),
            "use_mode": ["horizontal_only", "vertical_only", "both"][use_mode],
            "calculation_mode": calculation_mode,
            "horizontal_lines_count": len(h_lines),
            "vertical_lines_count": len(v_lines),
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
            "is_perpendicular": is_perpendicular
        }
        
        result = {
            "rotation_angle": float(rotation_angle),
            "rotation_metadata": rotation_metadata
        }
        
        # Solo generar visualización si no estamos en modo without_preview
        if not self.without_preview:
            # Crear vista de grilla
            grid_preview = self._create_grid_preview(
                base_img, rotation_angle, grid_spacing, grid_thickness, 
                grid_color, grid_alpha
            )
            
            # Crear vista de histograma
            histogram_preview = self._create_histogram_preview(
                h_angles, v_angles, h_angle, h_count, v_angle, v_count,
                rotation_angle, use_mode
            )
            
            result["grid_preview"] = grid_preview
            result["histogram_preview"] = histogram_preview
            
            # Sample image según preview_mode
            if preview_mode == 0:
                result["sample_image"] = grid_preview
            elif preview_mode == 1:
                result["sample_image"] = histogram_preview
            else:  # preview_mode == 2: Split
                # Redimensionar para combinar
                h_grid, w_grid = grid_preview.shape[:2]
                h_hist, w_hist = histogram_preview.shape[:2]
                
                # Ajustar histograma al ancho de la grilla
                hist_resized = cv2.resize(histogram_preview, (w_grid, h_hist * w_grid // w_hist))
                
                # Combinar verticalmente
                result["sample_image"] = np.vstack([hist_resized, grid_preview])
        
        return result
