## Regla Única del Cache

Si se modifica cualquier filtro en una posición MENOR o IGUAL (antes o el mismo) al último checkpoint:

 - Se marca ignore_cache = True y el script pasa a ignorar el cache "provisionalmente".
 - Si cerramos el script (q), sin guardar antes, entonces ningún cache será afectado y los parámetros modificados se perderán (esto es intencional y sirve para hacer pruebas sin afectar nada guardado).
 - Si guardamos (s): se borra TODO el cache de TODOS los checkpoints. El cache se irá regenerando a medida que avancemos por los filtros del pipeline.
 - No importa si los filtros modificados afectan o no a los checkpoints (es decir, si están encadenados o no sus inputs y outputs), alcanza con saber que están antes del último checkpoint.

Si se modifica un filtro en una posición MAYOR (posterior) al último checkpoint:

 - No pasa nada con el cache incluso si guardas las modificaciones de parámentros (s). Este es el comportamiento esperable del cache, que evita que se vuelvan a procesar los filtros "pesados" si ya fueron procesados previamente. Estos filtros pesados el usuario deberá posicionarlos estratégicamente al comienzo del pipeline para poder manipular los parámetros de la mayoría de los filtros sin afectar constantemente el cache.

## Explicación detallada

### **Caso A: Modificas filtro <= último checkpoint**

1. **Durante edición:**
   - Se marca `ignore_cache = True`
   - El script ignora el cache **provisionalmente** mientras editas
   - Ves los cambios en tiempo real sin usar cache

2. **Si cierras sin guardar (tecla `q`):**
   - ❌ Parámetros modificados se pierden
   - ✅ Cache permanece intacto
   - ✅ Útil para experimentar sin consecuencias

3. **Si guardas (tecla `s`):**
   - ⚠️ Advertencia mostrada al usuario
   - 🗑️ Se borra **TODO** el cache de **TODOS** los checkpoints
   - 🔄 Cache se regenera automáticamente al navegar imágenes
   - ✅ Parámetros guardados en `params.json`

4. **Independencia de encadenamiento:**
   - ❌ NO importa si los filtros están conectados por inputs/outputs
   - ✅ Solo importa la **posición** en el pipeline

---

### **Caso B: Modificas filtro > último checkpoint**

1. **Durante edición y al guardar:**
   - ✅ Cache NO se afecta en absoluto
   - ✅ Parámetros se guardan normalmente
   - ✅ Cache sigue válido y operativo

2. **Propósito del diseño:**
   - Filtros "pesados" (resize, denoise) van al **inicio**
   - Esos son los checkpoints
   - Filtros "ligeros" (ajustes, visualización) van **después**
   - Puedes ajustar parámetros ligeros sin perder cache pesado

---

## Ejemplo Práctico

```
Pipeline:
├─ filtro0: Resize (1920x1080 → 640x480) ✓ checkpoint [PESADO]
├─ filtro1: Denoise                      ✓ checkpoint [PESADO]
├─ filtro2: Grayscale
├─ filtro3: Blur
└─ filtro4: Canny
                                         ↑ último checkpoint = filtro1
```

**Escenario 1:** Ajustas parámetros de `Blur` (filtro3)
- ✅ filtro3 > filtro1 → cache NO se toca
- Cambias kernel_size de 5 a 11 → guardas → cache intacto
- Al cambiar de imagen, Resize y Denoise usan cache (rápido)

**Escenario 2:** Ajustas parámetros de `Resize` (filtro0)
- ⚠️ filtro0 <= filtro1 → cache se invalidará al guardar
- Cambias scale de 50% a 30% → guardas → TODO el cache se borra
- Al cambiar de imagen, se reprocesa todo (lento la primera vez)

### Ejemplo de checkpoints.json

```json
{
  "checkpoints": [
    "resize",
    "denoise"
  ],
  "last_modified": "2025-01-31T10:30:00"
}
```

---


