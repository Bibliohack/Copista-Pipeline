#!/usr/bin/env python3
"""
Herramienta de anotación de ground truth para bordes de página.

Permite recorrer una carpeta de imágenes y marcar manualmente el polígono
(4 esquinas en sentido horario desde sup-izq) que delimita el borde físico
del papel. Guarda las coordenadas en archivos .gt.json junto a cada imagen.

Uso:
    python ground_truth_annotator.py <carpeta_de_imagenes>

Controles:
    Click izquierdo    — colocar punto (hasta 4, en orden)
    Arrastrar izq.     — mover un punto ya colocado
    Rueda del ratón    — zoom in/out centrado en cursor
    Arrastrar der.     — paneo de la imagen
    R                  — resetear zoom
    ← / →              — imagen anterior / siguiente
    S                  — guardar
"""

import json
import os
import sys
from pathlib import Path

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
POINT_LABELS = ["Sup-Izq", "Sup-Der", "Inf-Der", "Inf-Izq"]
POINT_COLORS = ["yellow", "cyan", "orange", "magenta"]
POINT_RADIUS = 7
HIT_THRESHOLD = 14
ZOOM_STEP = 1.25
ZOOM_MIN = 0.5    # relativo al encaje inicial
ZOOM_MAX = 16.0   # relativo al encaje inicial


class AnnotationTool:
    def __init__(self, root, folder_path):
        self.root = root
        self.root.title("Anotador Ground Truth — Bordes de Página")

        self.folder = Path(folder_path)
        self.images = sorted(
            p for p in self.folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not self.images:
            messagebox.showerror("Error", "No se encontraron imágenes JPG/PNG en la carpeta.")
            root.destroy()
            return

        self.current_index = 0

        # Puntos en coordenadas de la imagen original
        self.points = []
        self.drag_index = None

        # Transformada de vista: canvas = orig * S + (tx, ty)
        self.S = 1.0       # escala actual
        self.tx = 0.0      # traslación x
        self.ty = 0.0      # traslación y
        self.base_S = 1.0  # escala de encaje inicial (referencia para zoom %)

        # Estado de paneo
        self._pan_start = None
        self._pan_tx0 = 0.0
        self._pan_ty0 = 0.0

        self.orig_width = 0
        self.orig_height = 0
        self._orig_image = None
        self._tk_image = None
        self._pending_gt = None

        self._build_ui()
        self.root.update_idletasks()
        self._load_image(0)

    # ── Construcción de la UI ─────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Barra superior
        top = tk.Frame(self.root, pady=4)
        top.grid(row=0, column=0, sticky="ew", padx=6)

        self.btn_prev = tk.Button(top, text="◀ Anterior", width=12, command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT, padx=2)

        self.lbl_status = tk.Label(top, text="", font=("Arial", 11, "bold"))
        self.lbl_status.pack(side=tk.LEFT, expand=True)

        self.btn_next = tk.Button(top, text="Siguiente ▶", width=12, command=self.next_image)
        self.btn_next.pack(side=tk.LEFT, padx=2)

        # Canvas
        self.canvas = tk.Canvas(self.root, bg="#2b2b2b", cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew")

        # Eventos: puntos y arrastre
        self.canvas.bind("<ButtonPress-1>",   self.on_click)
        self.canvas.bind("<B1-Motion>",        self.on_drag)
        self.canvas.bind("<ButtonRelease-1>",  self.on_release)

        # Eventos: paneo con botón derecho
        self.canvas.bind("<ButtonPress-3>",   self.on_pan_start)
        self.canvas.bind("<B3-Motion>",        self.on_pan_move)
        self.canvas.bind("<ButtonRelease-3>",  self.on_pan_end)

        # Eventos: zoom con rueda
        self.canvas.bind("<MouseWheel>", self.on_zoom)   # Windows / macOS
        self.canvas.bind("<Button-4>",   self.on_zoom)   # Linux scroll up
        self.canvas.bind("<Button-5>",   self.on_zoom)   # Linux scroll down

        # Barra inferior
        bot = tk.Frame(self.root, pady=4)
        bot.grid(row=2, column=0, sticky="ew", padx=6)

        self.lbl_hint = tk.Label(bot, text="", fg="#888888", anchor="w")
        self.lbl_hint.pack(side=tk.LEFT, expand=True, fill=tk.X)

        tk.Button(bot, text="Reset zoom", width=10, command=self.reset_zoom).pack(side=tk.LEFT, padx=2)
        tk.Button(bot, text="Limpiar",    width=10, command=self.clear_points).pack(side=tk.LEFT, padx=2)
        tk.Button(
            bot, text="Guardar", width=10,
            bg="#2e7d32", fg="white", activebackground="#388e3c",
            command=self.save
        ).pack(side=tk.LEFT, padx=2)

        # Atajos de teclado
        self.root.bind("<Left>",      lambda e: self.prev_image())
        self.root.bind("<Right>",     lambda e: self.next_image())
        self.root.bind("<s>",         lambda e: self.save())
        self.root.bind("<r>",         lambda e: self.reset_zoom())
        self.root.bind("<Configure>", self._on_resize)

    # ── Carga de imagen ───────────────────────────────────────────────────────

    def _load_image(self, index):
        self.current_index = index
        self.points = []
        self.drag_index = None

        path = self.images[index]
        img = Image.open(path)
        self.orig_width, self.orig_height = img.size
        self._orig_image = img

        gt_path = path.with_name(path.stem + '.gt.json')
        if gt_path.exists():
            with open(gt_path) as f:
                data = json.load(f)
            self._pending_gt = data.get('polygon', [])
        else:
            self._pending_gt = None

        self._reset_transform()
        self._update_display()
        self._update_status()

    def _reset_transform(self):
        """Calcula la transformada de encaje (zoom 100%)."""
        self.root.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 900, 650

        S = min(cw / self.orig_width, ch / self.orig_height)
        self.base_S = S
        self.S = S
        self.tx = (cw - self.orig_width  * S) / 2
        self.ty = (ch - self.orig_height * S) / 2

    # ── Conversión de coordenadas ─────────────────────────────────────────────

    def _orig_to_canvas(self, ox, oy):
        return ox * self.S + self.tx, oy * self.S + self.ty

    def _canvas_to_orig(self, cx, cy):
        return (cx - self.tx) / self.S, (cy - self.ty) / self.S

    def _clamp_orig(self, ox, oy):
        return (
            max(0.0, min(float(self.orig_width),  ox)),
            max(0.0, min(float(self.orig_height), oy)),
        )

    # ── Renderizado ───────────────────────────────────────────────────────────

    def _update_display(self):
        self.root.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 900, 650

        # Rectángulo visible en coordenadas de imagen original
        x0 = max(0.0,                    (-self.tx) / self.S)
        y0 = max(0.0,                    (-self.ty) / self.S)
        x1 = min(float(self.orig_width),  (cw - self.tx) / self.S)
        y1 = min(float(self.orig_height), (ch - self.ty) / self.S)

        if x1 <= x0 or y1 <= y0:
            return

        # Tamaño en píxeles de canvas que ocupa ese recorte
        disp_w = round((x1 - x0) * self.S)
        disp_h = round((y1 - y0) * self.S)
        if disp_w < 1 or disp_h < 1:
            return

        crop = self._orig_image.crop((int(x0), int(y0), int(x1), int(y1)))
        img_resized = crop.resize((disp_w, disp_h), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(img_resized)

        # Posición en canvas de la esquina sup-izq del recorte
        img_canvas_x = int(max(0.0, self.tx))
        img_canvas_y = int(max(0.0, self.ty))

        self.canvas.delete("all")
        self.canvas.create_image(img_canvas_x, img_canvas_y, anchor=tk.NW, image=self._tk_image)

        # Convertir puntos pendientes (coords originales) al estado interno
        if self._pending_gt is not None:
            self.points = [(float(p['x']), float(p['y'])) for p in self._pending_gt]
            self._pending_gt = None

        self._draw_polygon()

    # ── Dibujo del polígono ───────────────────────────────────────────────────

    def _draw_polygon(self):
        self.canvas.delete("polygon")
        self.canvas.delete("point")

        n = len(self.points)
        if n == 0:
            return

        canvas_pts = [self._orig_to_canvas(ox, oy) for ox, oy in self.points]

        # Líneas
        for i in range(n - 1):
            x1, y1 = canvas_pts[i]
            x2, y2 = canvas_pts[i + 1]
            self.canvas.create_line(x1, y1, x2, y2, fill="#00e676", width=2, tags="polygon")
        if n == 4:
            x1, y1 = canvas_pts[3]
            x2, y2 = canvas_pts[0]
            self.canvas.create_line(x1, y1, x2, y2, fill="#00e676", width=2, tags="polygon")

        # Puntos con etiqueta
        r = POINT_RADIUS
        for i, (x, y) in enumerate(canvas_pts):
            color = POINT_COLORS[i]
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=color, outline="white", width=2, tags="point"
            )
            self.canvas.create_text(
                x + r + 4, y - r - 2,
                text=POINT_LABELS[i], fill=color,
                font=("Arial", 9, "bold"), anchor="w", tags="point"
            )

    # ── Barra de estado ───────────────────────────────────────────────────────

    def _update_status(self):
        path = self.images[self.current_index]
        gt_path = path.with_name(path.stem + '.gt.json')
        mark = "  ✓" if gt_path.exists() else ""
        zoom_pct = round(self.S / self.base_S * 100)
        self.lbl_status.config(
            text=f"{path.name}{mark}   [{self.current_index + 1} / {len(self.images)}]   {zoom_pct}%"
        )

        n = len(self.points)
        if n < 4:
            self.lbl_hint.config(
                text=f"Click para marcar punto {n + 1}/4: {POINT_LABELS[n]}",
                fg="#aaaaaa"
            )
        else:
            self.lbl_hint.config(
                text="Polígono completo. Arrastra los puntos para ajustar.",
                fg="#aaaaaa"
            )

        self.btn_prev.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_index < len(self.images) - 1 else tk.DISABLED)

    # ── Eventos: puntos ───────────────────────────────────────────────────────

    def on_click(self, event):
        if len(self.points) < 4:
            ox, oy = self._clamp_orig(*self._canvas_to_orig(event.x, event.y))
            self.points.append((ox, oy))
            self._draw_polygon()
            self._update_status()
        else:
            self.drag_index = self._hit_point(event.x, event.y)

    def on_drag(self, event):
        if self.drag_index is not None:
            ox, oy = self._clamp_orig(*self._canvas_to_orig(event.x, event.y))
            self.points[self.drag_index] = (ox, oy)
            self._draw_polygon()

    def on_release(self, event):
        self.drag_index = None

    def _hit_point(self, cx, cy):
        for i, (ox, oy) in enumerate(self.points):
            px, py = self._orig_to_canvas(ox, oy)
            if abs(px - cx) <= HIT_THRESHOLD and abs(py - cy) <= HIT_THRESHOLD:
                return i
        return None

    # ── Eventos: paneo ────────────────────────────────────────────────────────

    def on_pan_start(self, event):
        self._pan_start = (event.x, event.y)
        self._pan_tx0 = self.tx
        self._pan_ty0 = self.ty
        self.canvas.config(cursor="fleur")

    def on_pan_move(self, event):
        if self._pan_start:
            self.tx = self._pan_tx0 + (event.x - self._pan_start[0])
            self.ty = self._pan_ty0 + (event.y - self._pan_start[1])
            self._update_display()
            self._update_status()

    def on_pan_end(self, event):
        self._pan_start = None
        self.canvas.config(cursor="crosshair")

    # ── Eventos: zoom ─────────────────────────────────────────────────────────

    def on_zoom(self, event):
        if event.num == 4 or getattr(event, 'delta', 0) > 0:
            dz = ZOOM_STEP
        else:
            dz = 1.0 / ZOOM_STEP

        new_S = self.S * dz
        if not (ZOOM_MIN <= new_S / self.base_S <= ZOOM_MAX):
            return

        # Zoom centrado en la posición del cursor
        self.tx = event.x - (event.x - self.tx) * dz
        self.ty = event.y - (event.y - self.ty) * dz
        self.S = new_S

        self._update_display()
        self._update_status()

    def reset_zoom(self, event=None):
        self._reset_transform()
        self._update_display()
        self._update_status()

    # ── Redimensionado de ventana ─────────────────────────────────────────────

    def _on_resize(self, event):
        if event.widget != self.root or self._orig_image is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        # Mantener factor de zoom y centro de vista
        zoom_factor = self.S / self.base_S
        cx_orig = (cw / 2 - self.tx) / self.S
        cy_orig = (ch / 2 - self.ty) / self.S

        self.base_S = min(cw / self.orig_width, ch / self.orig_height)
        self.S = self.base_S * zoom_factor
        self.tx = cw / 2 - cx_orig * self.S
        self.ty = ch / 2 - cy_orig * self.S

        self._update_display()
        self._update_status()

    # ── Acciones ──────────────────────────────────────────────────────────────

    def clear_points(self):
        self.points = []
        self._draw_polygon()
        self._update_status()

    def save(self):
        if len(self.points) != 4:
            messagebox.showwarning("Faltan puntos",
                                   "Debes marcar exactamente 4 puntos antes de guardar.")
            return

        path = self.images[self.current_index]
        gt_path = path.with_name(path.stem + '.gt.json')

        data = {
            "image_file":   path.name,
            "image_width":  self.orig_width,
            "image_height": self.orig_height,
            "polygon": [{"x": round(ox), "y": round(oy)} for ox, oy in self.points]
        }

        with open(gt_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self._update_status()
        self.lbl_hint.config(text=f"Guardado: {gt_path.name}", fg="#69f0ae")
        self.root.after(2500, self._update_status)

    def prev_image(self):
        if self.current_index > 0:
            self._load_image(self.current_index - 1)

    def next_image(self):
        if self.current_index < len(self.images) - 1:
            self._load_image(self.current_index + 1)


# ── Punto de entrada ──────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Uso: python ground_truth_annotator.py <carpeta_de_imagenes>")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' no es una carpeta válida.")
        sys.exit(1)

    root = tk.Tk()
    root.geometry("1280x860")
    root.minsize(640, 480)
    AnnotationTool(root, folder)
    root.mainloop()


if __name__ == "__main__":
    main()
