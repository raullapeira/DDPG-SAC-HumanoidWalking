# Convenciones del proyecto

## Carpetas de GIFs de simulación

Al crear o modificar un script de entrenamiento, la variable `_MEDIA_DIR` donde se guardan los GIFs de evaluación **siempre** debe seguir el formato:

```
media/DD_MM_YYYY_detalle_ultimo_cambio/
```

Ejemplos reales del proyecto:
- `media/26_05_2026_fighting/`
- `media/27_05_2026_parallel_race_non_stop/`

La fecha se genera automáticamente con:
```python
_TODAY = datetime.date.today().strftime("%d_%m_%Y")
_MEDIA_DIR = os.path.join(_HERE, "media", f"{_TODAY}_detalle_ultimo_cambio")
```

El sufijo debe describir el cambio más relevante de esa sesión de entrenamiento, no el nombre genérico del script.
