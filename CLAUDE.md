## Generación de reportes

Cuando el usuario pida generar un reporte:

1. Leer el template en `/home/claudeuser/reportes/reporte-template.md`
2. Copiar con el nombre: `{nombre-proyecto} - {YYYY-MM-DD} - {persona}.md`
   - El nombre del proyecto y la persona están en el template
   - La fecha es la del día actual (disponible en el contexto como `currentDate`)
3. Guardar en `/home/claudeuser/reportes/`
4. Completar las secciones:
   - **Tareas planificadas:** tareas del plan formal del proyecto (si existe)
   - **Nuevas tareas:** entre 4 y 8 tareas importantes realizadas en la sesión, explicadas en 1-2 líneas cada una. Marcar como `[x]` si se completaron.
   - **Bloqueos:** impedimentos actuales que frenan el avance
5. No incluir más de 8 tareas en total para mantener el reporte conciso.
