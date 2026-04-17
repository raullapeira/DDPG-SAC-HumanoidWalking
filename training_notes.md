# Notas de entrenamiento SAC Alpha

## cogv3 — primer walking conseguido

**Fecha:** 2026-04-17  
**Script:** `main_sac_alpha_cogv3.py`  
**Env:** `envs/alpha_env.py`  
**XML:** `robot/configs/v2/alpha_single.xml`  
**Checkpoints:** `checkpoints/sac_alpha_cogv3/`

### Hitos

| Step | Reward medio | Comportamiento |
|------|-------------|----------------|
| ~169K | ~45-65 | Robot cae en ~18 pasos, apenas se sostiene |
| ~590K | ~2500 | **Robot camina 1000 pasos (episodio completo)** |

### Parámetros del entorno cuando empezó a caminar (~590K steps)

```python
_CTRL_COST_WEIGHT    = 0.01
_FORWARD_WEIGHT      = 5.0
_ALIVE_BONUS         = 1.0
_UPRIGHT_WEIGHT      = 0.3
_LATERAL_COST_WEIGHT = 0.15
_YAW_COST_WEIGHT     = 1.0

_FOOT_HEIGHT_WEIGHT  = 2.0
_FRONT_LIFT_PENALTY  = 1.0
_STANCE_PENALTY      = -0.5
_SLOW_PENALTY        = -2.0

_ANKLE_COST_WEIGHT   = 2.0
_FEET_COST_WEIGHT    = 4.0
_FOOT_FLAT_WEIGHT    = 8.0
_COM_SUPPORT_WEIGHT  = 8.0
_SINGLE_SUPP_BONUS   = 0.3

_STANCE_Z = 0.04
```

### Métricas de pies en ~590K steps

```
tilt L avg ~0.075-0.081  max ~0.285-0.312
tilt R avg ~0.091-0.094  max ~0.294-0.312
alt  L avg ~0.045-0.046  max ~0.078-0.106
alt  R avg ~0.047-0.048  max ~0.083-0.105
```

- Tilt medio bajo (~0.08) → pie mayormente plano durante la marcha
- Tilt máx ~0.30 → sigue usando algo la puntera en el despegue (toe push-off residual)
- Altura media ~0.046 → swing pequeño pero existe
- Altura máx ~0.10 → el pie de swing se levanta ~10 cm

### Pendiente

- Reducir tilt máx por debajo de 0.20 → subir `_FOOT_FLAT_WEIGHT` de 8.0 a 10-12
- Probar en robot real con este checkpoint (~590K)
- Continuar entrenamiento hasta 2M steps para ver si mejora sola
