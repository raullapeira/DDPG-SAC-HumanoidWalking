# Convenciones del proyecto

## Carpetas de GIFs de simulación y de checkpoints

Al crear o modificar un script de entrenamiento, tanto la variable `_MEDIA_DIR` (GIFs de evaluación) como las carpetas de `checkpoints/` **siempre** deben llevar la fecha primero y en formato `YYYY_MM_DD` (para que ordenen cronológicamente al listar el directorio):

```
media/YYYY_MM_DD_detalle_ultimo_cambio/
checkpoints/YYYY_MM_DD_detalle_del_experimento/...
```

Ejemplos reales del proyecto:
- `media/2026_09_19_freno_auto_stop_dist_desde_cero/`
- `checkpoints/2026_09_19_versus_stop_dist/{r1,r2}/`
- `checkpoints/2026_09_20_full_fight_acercar_parar_pegar/{leg_r1,leg_r2,arm_r1,arm_r2}/`

La fecha se genera automáticamente con:
```python
_TODAY = datetime.date.today().strftime("%Y_%m_%d")
_MEDIA_DIR = os.path.join(_HERE, "media", f"{_TODAY}_detalle_ultimo_cambio")
```

El sufijo debe describir el cambio más relevante de esa sesión de entrenamiento, no el nombre genérico del script.

Para la carpeta de `checkpoints/`, la fecha debe ser la de **inicio** del entrenamiento, no la del día en que se relanza el script (si ya existe una carpeta con checkpoints guardados bajo ese sufijo, hay que reutilizarla para poder resumir entre días en vez de crear una nueva cada vez — ver `fighting.py` o `full_fight.py` para el patrón con `glob`).
