import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

_XML_PATH = os.path.join(
    os.path.dirname(__file__), "robot", "configs", "fighting", "alpha_versus.xml"
)

_FALL_Z        = 0.12
_TILT_Z        = 0.7
_FRAME_SKIP    = 5
_ACTION_REPEAT = 4
_MAX_STEPS     = 500      # episodios cortos: aqui solo se entrena el combate

# Distancia (raiz-raiz, XY) de arranque. IMPORTANTE: este valor NO es el mismo
# que _STOP_DIST en full_fight_env.py — ese se eligio con margen para frenar
# durante el acercamiento, pero aqui no hay acercamiento, spawean ya quietos,
# asi que puede (y debe) ser mucho mas ajustado.
# OJO: una medicion anterior (0.072m de alcance -> spawn 0.12) usaba el
# origen del cuerpo "Hand", que es el CODO — ese cuerpo es el antebrazo entero
# (8.5cm), el puño llega bastante mas lejos. A 0.12m, con el brazo al frente,
# los brazos de ambos se solapaban y el choque era una "explosion" fisica, no
# un golpe. Comprobado con un robot golpeando y el otro en guardia (con el
# giro "Arm" ya fijo, ver _ROLL_LOCK): a 0.16m el puño conecta con hombro y
# brazo del rival; a 0.20m ya no llega.
_SPAWN_DIST    = 0.16

# Pesos de recompensa del combate. _ALIVE_BONUS/_UPRIGHT_W rebajados respecto a
# full_fight_env.py: con la postura ahora mucho mas estable (ver mas abajo),
# un episodio entero de pie sin pelear acumularia ~1.3*500=650 de recompensa
# "gratis" — mucho mas que un golpe suelto (5.0*hit) o incluso el KO (20).
# Asi apenas hay incentivo para arriesgarse a golpear. Bajamos lo pasivo y
# subimos lo que depende de pelear de verdad.
_HIT_WEIGHT    =  8.0     # solo se aplica al golpe que puntua (ver ciclo mas abajo)
_RECV_WEIGHT   =  3.0     # penalizacion por cada golpe del rival que puntua

# ── Ciclo de golpe: cargar -> golpear -> retirar -> volver a golpear ─────────
# Antes cualquier contacto con velocidad puntuaba en cada step, asi que lo
# mas rentable era plantar el puño en el centro de gravedad del rival y
# quedarse ahi apretando. Ahora cada brazo tiene un estado "cargado": solo un
# brazo cargado puede puntuar golpe, al golpear se descarga, y se vuelve a
# cargar al RECOGER el brazo (angulo del propio hombro, no distancia al
# rival: el rival se mueve, y con la distancia el brazo izquierdo nunca
# llegaba a "alejarse" lo suficiente aunque se retirase). Medido: los servos
# son lentos — en ciclos de 0.4s el hombro oscila ~0.5<->1.04 rad, en 0.8s
# ~0.32<->1.25; los contactos se dan con el hombro en ~0.9-1.2 rad.
# Distancia puño->centro de gravedad del rival: guardia ~0.16m, contacto ~0.11m.
_STRIKE_BONUS    = 10.0   # por cada golpe con el brazo cargado
_STRIKE_HIT_MIN  = 0.015  # hit por mano (filtrado por velocidad) minimo para contar como golpe
_REARM_SHOULDER  = 0.6    # rad: el hombro debe bajar de aqui para recargar (0=abajo, 1.57=al frente)
_CLINCH_DIST     = 0.13   # puño mas cerca que esto del centro de gravedad del rival...
_CLINCH_GRACE    = 3      # ...durante mas de estos steps (~0.3s) sin recoger el brazo = clinch
_CLINCH_PENALTY  = 0.5    # por step y por mano en clinch
_SHOULDER_QPOS   = np.array([7 + 5, 7 + 13, 30 + 5, 30 + 13])   # [r1_izq, r1_dcho, r2_izq, r2_dcho]
_ALIVE_BONUS   =  0.3
_UPRIGHT_W     =  0.1
_ARM_CTRL_COST =  0.01
_FALL_PENALTY  = -20.0
_KO_BONUS      =  30.0
# El KO solo cuenta si el que cae RECIBIO un golpe real hace poco (contacto de
# puño del rival en los ultimos _KO_HIT_WINDOW steps, ~0.1s cada uno). Si se
# cae solo (p.ej. agitando los brazos — comprobado: con brazos aleatorios el
# que se agita se tumba solo en 13-64 steps) no hay bonus para el rival; la
# penalizacion por caerse si se aplica siempre.
_KO_HIT_WINDOW    = 10
_KO_HIT_MIN       = 0.03   # hit (ya filtrado por velocidad) minimo para "recibio golpe"; un jab real da ~0.03-0.08
_SENSOR_CLIP   = 20.0

# Recompensa por mover la mano rapido en la linea mano-rival — lanzar Y
# retirar, no solo lanzar (ver nota junto a r1_punch_speed mas abajo sobre por
# que retirar tambien cuenta). Restringido a esa linea concreta, no a
# "moverse rapido en cualquier direccion" — eso ya lo probamos, aprenden a
# flapear de lado sin avanzar de verdad.
_PUNCH_SPEED_WEIGHT = 3.0

# Velocidad minima de la mano (m/s, medida por diferencias finitas de posicion
# a lo largo de un step de ~0.1s) para que el contacto cuente como golpe real
# en vez de apoyo estatico. Con un asalto de ~7cm de alcance total, una mano
# que recorra >=4cm en ese step ya cuenta entero; por debajo, el hit se escala
# proporcionalmente (a velocidad 0, el hit no cuenta nada).
_MIN_STRIKE_SPEED = 0.4

# Recompensa DENSA por ACERCAR la mano al centro de masas del rival
# (subtree_com de MuJoCo — pondera la masa real de cada eslabon, no solo la
# posicion de la raiz). El hit y la velocidad de golpe son señales escasas
# (solo disparan al conectar o al moverse rapido en la linea exacta); esto da
# gradiente en TODO momento, incluso sin llegar a tocar.
#
# OJO — primera version media la distancia ABSOLUTA (mas cerca = mas premio
# cada step). Eso reintroducia el mismo fallo que ya habiamos arreglado para
# el hit: en cuanto el robot encontraba una postura cercana, quedarse quieto
# ahi cobraba el maximo para siempre sin necesidad de seguir moviendose —
# confirmado a los 150k steps, el azul dejaba los brazos doblados delante,
# parados. Ahora se premia la MEJORA dentro de cada step (distancia al
# empezar menos distancia al acabar, nunca negativa) — estar quieto da
# exactamente 0, solo cobra acercarse de verdad paso a paso.
_COM_PROXIMITY_WEIGHT = 6.0

# NOTA sobre el "congelado" de piernas — historial de intentos:
# 1) Teletransportar qpos/qvel de las piernas a la postura neutra CADA
#    sub-paso fisico (haciendolas rigidas de verdad) generaba una caida
#    pasiva por si sola en ~50 pasos, y encima con pinta de "tabla recta
#    cayendo" (nada de flexion natural) — el propio teletransporte peleaba
#    con el solver de contactos e inyectaba un pequeño impulso cada sub-paso.
# 2) Compensarlo con un muelle horizontal que recentraba la raiz ayudaba pero
#    era fragil (la zona estable dependia de forma no lineal/caotica del
#    valor de la constante) y no evitaba caidas ocasionales sin ningun golpe.
# 3) Solucion real: NO teletransportar nada — dejar el ctrl de las piernas
#    fijo en la postura neutra y que el propio servo PD (kp del actuador) la
#    sostenga con su rigidez natural, sin forzar qpos/qvel. Probado: deriva
#    de apenas 0.07-0.13m en 500 pasos con movimiento de brazos ACTIVO y
#    aleatorio (antes: 0.4m+ o caida en ~50 pasos), sin ninguna caida pasiva
#    en 5 seeds. No hace falta muelle de recentrado en absoluto.

# qpos: [r1_freejoint(7) | r1_joints(16) | r2_freejoint(7) | r2_joints(16)] = 46
# qvel: [r1_freejoint(6) | r1_joints(16) | r2_freejoint(6) | r2_joints(16)] = 44
_LEG_IDX = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
_ARM_IDX = np.array([5, 6, 7, 13, 14, 15], dtype=int)

_R1_LEG_CTRL = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
_R2_LEG_CTRL = np.array([16, 17, 18, 19, 20, 24, 25, 26, 27, 28], dtype=int)

_R1_LEG_QPOS = 7  + _LEG_IDX
_R2_LEG_QPOS = 30 + _LEG_IDX

# Cada brazo es [Shoulder, Arm, Hand]. Mapeado por cinematica directa:
#  - Arm (giro/"roll") en 0 deja el brazo en cruz, horizontal hacia el lado, y
#    en esa postura el Shoulder solo hace girar el brazo sobre su propio eje:
#    la mano no se mueve NADA.
#  - Con Arm en su limite (-1.57 brazos izquierdos, +1.57 derechos: son
#    espejo) el brazo queda bajado y el Shoulder lo barre en arco:
#    0=abajo, 1.57=RECTO HACIA DELANTE, 3.14=arriba.
# La postura de puñetazo exigia que la politica clavase Arm justo en el
# extremo del rango (accion = ±1 exacto), cosa que la tanh de SAC casi nunca
# alcanza — por eso nunca llegaba a poner el brazo hacia delante, se quedaba
# en la zona central (brazo en cruz) agitando el hombro sin avanzar.
# Solucion: Arm fijo en su valor correcto (como las piernas) y la politica
# controla solo Shoulder (lanzar) y Hand (antebrazo) de cada brazo.
_R1_ROLL_CTRL = np.array([6, 14], dtype=int)
_R2_ROLL_CTRL = np.array([22, 30], dtype=int)
_R1_ROLL_QPOS = 7  + np.array([6, 14])
_R2_ROLL_QPOS = 30 + np.array([6, 14])
_ROLL_LOCK    = np.array([-1.57, 1.57])   # [izquierdo, derecho]

_R1_ACT_CTRL = np.array([5, 7, 13, 15], dtype=int)    # Shoulder, Hand (izq, dcho)
_R2_ACT_CTRL = np.array([21, 23, 29, 31], dtype=int)
ARM_ACT_DIM  = 4

FIGHT_OBS_DIM = 30   # 6+6+5+3+3+3+2 + 2 (brazo izq/dcho cargado)


class CloseCombatEnv(gym.Env):
    """Entorno de SOLO combate — separado del entrenamiento de caminar a proposito.

    Los robots arrancan ya a _SPAWN_DIST entre si, de pie, con las piernas
    congeladas en una postura neutra fija desde el primer step (la misma
    tecnica validada en full_fight_env.py — nada de "media zancada" ni de
    cargar una politica de walking externa: las piernas no hacen falta que
    se muevan, asi que ni se intentan mover).

    Solo los brazos son entrenables. Cada caida termina el episodio (KO) —
    aqui SIEMPRE es de verdad un KO, no hay fase de acercamiento que pueda
    confundirse con el combate.

    step(actions) recibe 8 valores: [arm_a1(4)|arm_a2(4)], cada uno
    [Shoulder_izq, Hand_izq, Shoulder_dcho, Hand_dcho] (el giro "Arm" va fijo,
    ver _ROLL_LOCK).
    reset/step devuelven (fight_obs_r1, fight_obs_r2).

    Sensores de fuerza en los puños (alpha_versus.xml):
      data.sensordata[0] = r1_lh_touch
      data.sensordata[1] = r1_rh_touch
      data.sensordata[2] = r2_lh_touch
      data.sensordata[3] = r2_rh_touch
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, xml_path=None):
        self.render_mode = render_mode

        xml_path = os.path.abspath(xml_path if xml_path is not None else _XML_PATH)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self._r1_arm_low  = self.model.actuator_ctrlrange[_R1_ACT_CTRL, 0].copy()
        self._r1_arm_high = self.model.actuator_ctrlrange[_R1_ACT_CTRL, 1].copy()
        self._r2_arm_low  = self.model.actuator_ctrlrange[_R2_ACT_CTRL, 0].copy()
        self._r2_arm_high = self.model.actuator_ctrlrange[_R2_ACT_CTRL, 1].copy()
        self._leg_low  = self.model.actuator_ctrlrange[_R1_LEG_CTRL, 0].copy()
        self._leg_high = self.model.actuator_ctrlrange[_R1_LEG_CTRL, 1].copy()

        _bid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
        self._r1_id = _bid("r1_root")
        self._r2_id = _bid("r2_root")
        self._r1_lh_id = _bid("r1_Left_Hand_link")
        self._r1_rh_id = _bid("r1_Right_Hand_link")
        self._r2_lh_id = _bid("r2_Left_Hand_link")
        self._r2_rh_id = _bid("r2_Right_Hand_link")
        # Centro del puño = centro del sitio del sensor de contacto de cada mano.
        _sid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
        self._fist_sites = np.array([_sid("r1_lh_site"), _sid("r1_rh_site"),
                                     _sid("r2_lh_site"), _sid("r2_rh_site")])
        self._armed       = np.ones(4, dtype=bool)
        self._close_steps = np.zeros(4, dtype=int)

        # Postura neutra de referencia (la del modelo en reposo) — se congelan
        # ahi las piernas de cada episodio, sin depender de en que fase de
        # zancada estuvieran (nunca han llegado a andar en este entorno).
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._r1_neutral_leg_qpos = self.data.qpos[_R1_LEG_QPOS].copy()
        self._r2_neutral_leg_qpos = self.data.qpos[_R2_LEG_QPOS].copy()
        self._r1_leg_ctrl_hold = np.clip(self._r1_neutral_leg_qpos, self._leg_low, self._leg_high)
        self._r2_leg_ctrl_hold = np.clip(self._r2_neutral_leg_qpos, self._leg_low, self._leg_high)

        self._r1_init_qpos = self.data.qpos[:23].copy()
        self._r2_init_qpos = self.data.qpos[23:].copy()
        self._rng = np.random.default_rng()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2 * ARM_ACT_DIM,), dtype=np.float32)
        obs_limit = np.full(FIGHT_OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_limit, high=obs_limit, dtype=np.float32)

        self._renderer   = None
        self._step_count = 0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _denorm_arm(self, a, low, high):
        center = (high + low) / 2.0
        half   = (high - low) / 2.0
        return np.clip(center + a * half, low, high)

    def _get_fight_obs(self):
        qpos    = self.data.qpos.flat.copy()
        qvel    = self.data.qvel.flat.copy()
        sensors = np.clip(self.data.sensordata[:4].copy(), 0, _SENSOR_CLIP) / _SENSOR_CLIP

        r1_pos = qpos[0:3];  r2_pos = qpos[23:26]
        r1_vel = qvel[0:3];  r2_vel = qvel[22:25]

        obs_r1 = np.concatenate([
            qpos[7:23][_ARM_IDX],   qvel[6:22][_ARM_IDX],
            qpos[2:7],              qvel[3:6],
            r2_pos - r1_pos,        r2_vel - r1_vel,
            sensors[0:2],
            self._armed[0:2].astype(np.float64),
        ]).astype(np.float32)

        obs_r2 = np.concatenate([
            qpos[30:46][_ARM_IDX],  qvel[28:44][_ARM_IDX],
            qpos[25:30],            qvel[25:28],
            r1_pos - r2_pos,        r1_vel - r2_vel,
            sensors[2:4],
            self._armed[2:4].astype(np.float64),
        ]).astype(np.float32)

        return obs_r1, obs_r2

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0]  =  _SPAWN_DIST / 2.0
        self.data.qpos[23] = -_SPAWN_DIST / 2.0
        # Piernas directamente en la postura neutra congelada — nunca llegan
        # a andar en este entorno, asi que no tiene sentido perturbarlas.
        self.data.qpos[_R1_LEG_QPOS] = self._r1_neutral_leg_qpos
        self.data.qpos[_R2_LEG_QPOS] = self._r2_neutral_leg_qpos
        # Brazos ya en su plano de puñetazo (giro "Arm" fijo, ver _ROLL_LOCK);
        # con Shoulder=0 arrancan bajados, en guardia.
        self.data.qpos[_R1_ROLL_QPOS] = _ROLL_LOCK
        self.data.qpos[_R2_ROLL_QPOS] = _ROLL_LOCK
        mujoco.mj_forward(self.model, self.data)
        return self.begin_fight(), {}

    def begin_fight(self):
        """Arranca el asalto desde el estado fisico ACTUAL de self.data, sin
        tocarlo (reset() lo usa tras colocar a los robots; walk_and_fight.py
        lo llama justo cuando llegan andando a distancia de combate)."""
        self._step_count = 0
        # Ultimo step en que cada robot RECIBIO un golpe del rival (ver _KO_HIT_WINDOW).
        self._r1_last_hit_recv = -10**9
        self._r2_last_hit_recv = -10**9
        # Estado del ciclo de golpe por mano: [r1_izq, r1_dcho, r2_izq, r2_dcho].
        self._armed       = np.ones(4, dtype=bool)
        self._close_steps = np.zeros(4, dtype=int)
        return self._get_fight_obs()

    def step(self, actions):
        actions = np.clip(actions, -1.0, 1.0).astype(np.float64)
        arm_a1 = actions[0:ARM_ACT_DIM]
        arm_a2 = actions[ARM_ACT_DIM:2 * ARM_ACT_DIM]

        total_sensors = np.zeros(4)
        r1_lh_start = self.data.xpos[self._r1_lh_id].copy()
        r1_rh_start = self.data.xpos[self._r1_rh_id].copy()
        r2_lh_start = self.data.xpos[self._r2_lh_id].copy()
        r2_rh_start = self.data.xpos[self._r2_rh_id].copy()
        r1_root_start = self.data.xpos[self._r1_id].copy()
        r2_root_start = self.data.xpos[self._r2_id].copy()

        # Distancia mano-CoM del rival AL INICIO del step — para premiar solo
        # la MEJORA (ver mas abajo, junto a r1_com_progress): si premiamos la
        # distancia absoluta, quedarse quieto en la mejor postura encontrada
        # cobra el maximo para siempre, sin necesidad de seguir moviendose
        # (mismo fallo que ya arreglamos para el hit, reintroducido aqui).
        r2_com_start = self.data.subtree_com[self._r2_id].copy()
        r1_com_start = self.data.subtree_com[self._r1_id].copy()
        r1_com_dist_start = min(
            float(np.linalg.norm(r1_lh_start - r2_com_start)),
            float(np.linalg.norm(r1_rh_start - r2_com_start)),
        )
        r2_com_dist_start = min(
            float(np.linalg.norm(r2_lh_start - r1_com_start)),
            float(np.linalg.norm(r2_rh_start - r1_com_start)),
        )

        # Velocidad de PICO por sub-paso fisico (no la media de todo el step)
        # — el propio choque frena la mano justo al contactar, asi que medir
        # la velocidad media de todo el step haria que un golpe real se
        # auto-penalizara (la desaceleracion del impacto tapa la velocidad de
        # swing que hubo justo antes de tocar).
        #
        # IMPORTANTE — geometria real del brazo: comprobamos con un demo
        # animado que el UNICO camino con alcance real es hombro~90°+codo~90°,
        # un barrido en gancho/cruzado, NO un jab recto sobre la linea
        # raiz-raiz (con el codo recto el hombro no mueve la mano ni un mm,
        # gira sobre su propio eje). Proyectar la velocidad sobre esa linea
        # recta (unit_12 fijo) penalizaba precisamente el unico movimiento que
        # conecta de verdad. Ahora proyectamos sobre la direccion desde la
        # PROPIA MANO hacia la raiz del rival, recalculada cada sub-paso — asi
        # cualquier trayectoria que de verdad acorte distancia cuenta, sea
        # recta, en gancho o cruzada, sin asumir una linea de ataque concreta.
        def _dir_to(target, origin):
            v = target - origin
            n = np.linalg.norm(v)
            return v / n if n > 0.01 else np.zeros(3)

        r1_lh_peak = r1_rh_peak = r2_lh_peak = r2_rh_peak = 0.0
        r1_lh_prev = r1_lh_start.copy(); r1_rh_prev = r1_rh_start.copy()
        r2_lh_prev = r2_lh_start.copy(); r2_rh_prev = r2_rh_start.copy()
        _sub_dt = self.model.opt.timestep

        for _ in range(_ACTION_REPEAT):
            ctrl = np.zeros(self.model.nu, dtype=np.float64)
            ctrl[_R1_LEG_CTRL] = self._r1_leg_ctrl_hold
            ctrl[_R2_LEG_CTRL] = self._r2_leg_ctrl_hold
            ctrl[_R1_ROLL_CTRL] = _ROLL_LOCK
            ctrl[_R2_ROLL_CTRL] = _ROLL_LOCK
            ctrl[_R1_ACT_CTRL] = self._denorm_arm(arm_a1, self._r1_arm_low, self._r1_arm_high)
            ctrl[_R2_ACT_CTRL] = self._denorm_arm(arm_a2, self._r2_arm_low, self._r2_arm_high)
            self.data.ctrl[:] = ctrl

            for _ in range(_FRAME_SKIP):
                mujoco.mj_step(self.model, self.data)
                _r1_lh_now = self.data.xpos[self._r1_lh_id]
                _r1_rh_now = self.data.xpos[self._r1_rh_id]
                _r2_lh_now = self.data.xpos[self._r2_lh_id]
                _r2_rh_now = self.data.xpos[self._r2_rh_id]
                _r1_root_now = self.data.xpos[self._r1_id]
                _r2_root_now = self.data.xpos[self._r2_id]

                r1_lh_peak = max(r1_lh_peak, float(np.dot((_r1_lh_now - r1_lh_prev) / _sub_dt, _dir_to(_r2_root_now, _r1_lh_now))))
                r1_rh_peak = max(r1_rh_peak, float(np.dot((_r1_rh_now - r1_rh_prev) / _sub_dt, _dir_to(_r2_root_now, _r1_rh_now))))
                r2_lh_peak = max(r2_lh_peak, float(np.dot((_r2_lh_now - r2_lh_prev) / _sub_dt, _dir_to(_r1_root_now, _r2_lh_now))))
                r2_rh_peak = max(r2_rh_peak, float(np.dot((_r2_rh_now - r2_rh_prev) / _sub_dt, _dir_to(_r1_root_now, _r2_rh_now))))
                r1_lh_prev = _r1_lh_now.copy(); r1_rh_prev = _r1_rh_now.copy()
                r2_lh_prev = _r2_lh_now.copy(); r2_rh_prev = _r2_rh_now.copy()
                total_sensors += self.data.sensordata[:4].copy()
                # Piernas sostenidas por el servo PD (ctrl fijo en la postura
                # neutra) — sin teletransportar qpos/qvel. Ver nota mas arriba:
                # esto es mas estable que congelarlas del todo.
            mujoco.mj_forward(self.model, self.data)

        qpos = self.data.qpos.flat.copy()
        avg_sensors = total_sensors / (_ACTION_REPEAT * _FRAME_SKIP)

        r1_upz = 1.0 - 2.0 * (qpos[4]**2 + qpos[5]**2)
        r2_upz = 1.0 - 2.0 * (qpos[27]**2 + qpos[28]**2)
        r1_fell = bool(qpos[2]  < _FALL_Z or r1_upz < _TILT_Z)
        r2_fell = bool(qpos[25] < _FALL_Z or r2_upz < _TILT_Z)

        # Velocidad de las manos (por diferencias finitas de posicion a lo
        # largo de este step) — la necesitamos ANTES de calcular el hit, para
        # exigir que el golpe venga acompañado de velocidad real.
        dt = _ACTION_REPEAT * _FRAME_SKIP * self.model.opt.timestep

        r1_lh_vel = (self.data.xpos[self._r1_lh_id] - r1_lh_start) / dt
        r1_rh_vel = (self.data.xpos[self._r1_rh_id] - r1_rh_start) / dt
        r2_lh_vel = (self.data.xpos[self._r2_lh_id] - r2_lh_start) / dt
        r2_rh_vel = (self.data.xpos[self._r2_rh_id] - r2_rh_start) / dt

        # HIT "de verdad" solo si la mano que golpea se estaba moviendo rapido
        # HACIA EL RIVAL en ese instante — si no, un robot puede quedarse
        # apoyado en quieto contra el rival y cobrar "golpe" continuo por pura
        # presion estatica, sin arriesgarse a nada. Verificado que esto pasaba:
        # en un entrenamiento largo, hits_r1/hits_r2 se quedaban clavados en el
        # mismo valor durante cientos de episodios seguidos, sin caidas ni
        # KOs — apoyo estatico, no pelea.
        # IMPORTANTE #1: el gate tiene que ser la velocidad PROYECTADA hacia el
        # rival (igual que r1_punch_speed), no la velocidad total — con la
        # velocidad total bastaba con agitar el puño de lado a lado (o hacia
        # cualquier lado) para "desbloquear" el golpe sin lanzar nada de
        # verdad hacia delante.
        # IMPORTANTE #2: el gate usa la velocidad de PICO por sub-paso fisico
        # (r1_lh_peak etc., calculada dentro del bucle de arriba), NO la media
        # de todo el step (r1_lh_vel). Verificado con contactos brutos: SI
        # habia colisiones mano-mano reales (~19% de los pasos), pero el hit
        # seguia saliendo ~0 — el propio impacto frena la mano justo al
        # contactar, así que la velocidad MEDIA de ese step (que incluye el
        # frenazo del choque) se quedaba por debajo del umbral aunque la mano
        # SI iba lanzada justo antes de tocar. El pico por sub-paso captura
        # la velocidad de swing de verdad, no la contaminada por el propio golpe.
        r1_lh_gate = min(1.0, r1_lh_peak / _MIN_STRIKE_SPEED)
        r1_rh_gate = min(1.0, r1_rh_peak / _MIN_STRIKE_SPEED)
        r2_lh_gate = min(1.0, r2_lh_peak / _MIN_STRIKE_SPEED)
        r2_rh_gate = min(1.0, r2_rh_peak / _MIN_STRIKE_SPEED)

        # Hit por mano [r1_izq, r1_dcho, r2_izq, r2_dcho]: contacto * velocidad hacia el rival.
        hand_hits = np.clip(avg_sensors, 0, _SENSOR_CLIP) / _SENSOR_CLIP \
                  * np.array([r1_lh_gate, r1_rh_gate, r2_lh_gate, r2_rh_gate])
        r1_hit = float(hand_hits[0:2].sum())
        r2_hit = float(hand_hits[2:4].sum())

        # Ciclo de golpe (ver _STRIKE_BONUS): un golpe solo puntua si el brazo
        # estaba cargado; al golpear se descarga y solo recarga al recoger el
        # brazo (hombro < _REARM_SHOULDER). Seguir apretando contra el rival
        # sin recoger el brazo penaliza.
        fist_pos = self.data.site_xpos[self._fist_sites]
        com_r1   = self.data.subtree_com[self._r1_id]
        com_r2   = self.data.subtree_com[self._r2_id]
        fist_d   = np.linalg.norm(fist_pos - np.array([com_r2, com_r2, com_r1, com_r1]), axis=1)
        retracted = self.data.qpos[_SHOULDER_QPOS] < _REARM_SHOULDER

        strikes = self._armed & (hand_hits >= _STRIKE_HIT_MIN)
        self._armed[strikes] = False
        self._armed[retracted] = True
        pressing = (fist_d < _CLINCH_DIST) & ~retracted
        self._close_steps = np.where(pressing, self._close_steps + 1, 0)
        clinch = self._close_steps > _CLINCH_GRACE

        r1_strikes   = int(strikes[0:2].sum());  r2_strikes   = int(strikes[2:4].sum())
        r1_strike_hit = float(hand_hits[0:2][strikes[0:2]].sum())
        r2_strike_hit = float(hand_hits[2:4][strikes[2:4]].sum())
        r1_clinch = int(clinch[0:2].sum());       r2_clinch = int(clinch[2:4].sum())

        # Valor absoluto, no solo la componente hacia el rival: antes retirar
        # el brazo no daba ni premio ni castigo, asi que lanzar una vez y
        # quedarse quieto ahi salia mas a cuenta que retirar y volver a
        # intentarlo (retirar cuesta esfuerzo de control sin compensacion) —
        # se quedaban "atascados" tras el primer intento. Premiando tambien
        # la retirada rapida (misma linea mano-rival, cualquier sentido) sale
        # a cuenta el ciclo completo lanzar -> retirar -> lanzar otra vez, como
        # en un combate real. El GATE del golpe (mas arriba) sigue siendo solo
        # hacia delante — retirar rapido no cuenta como golpe, solo como
        # "buen ritmo".
        r1_punch_speed = abs(float(np.dot(r1_lh_vel, _dir_to(r2_root_start, r1_lh_start)))) \
                       + abs(float(np.dot(r1_rh_vel, _dir_to(r2_root_start, r1_rh_start))))
        r2_punch_speed = abs(float(np.dot(r2_lh_vel, _dir_to(r1_root_start, r2_lh_start)))) \
                       + abs(float(np.dot(r2_rh_vel, _dir_to(r1_root_start, r2_rh_start))))

        # Distancia mano-COM del rival: recompensa DENSA basada en la MEJORA
        # dentro de este step (distancia al empezar menos distancia al
        # acabar), NO en la distancia absoluta — con la distancia absoluta,
        # quedarse quieto en la mejor postura encontrada cobraba el maximo
        # para siempre sin necesidad de seguir moviendose (verificado: a los
        # 150k steps el azul dejaba los brazos doblados delante, parados, por
        # esto exactamente). Con la mejora, estar quieto da 0 — solo cobra
        # acercarse de verdad, paso a paso. No gated por velocidad minima a
        # proposito (a diferencia del hit): sirve para guiar el descubrimiento
        # de la ventana de alcance incluso con movimientos lentos/pequeños.
        r2_com = self.data.subtree_com[self._r2_id]
        r1_com = self.data.subtree_com[self._r1_id]
        d_r1_lh_com = float(np.linalg.norm(self.data.xpos[self._r1_lh_id] - r2_com))
        d_r1_rh_com = float(np.linalg.norm(self.data.xpos[self._r1_rh_id] - r2_com))
        d_r2_lh_com = float(np.linalg.norm(self.data.xpos[self._r2_lh_id] - r1_com))
        d_r2_rh_com = float(np.linalg.norm(self.data.xpos[self._r2_rh_id] - r1_com))
        r1_com_dist_end = min(d_r1_lh_com, d_r1_rh_com)
        r2_com_dist_end = min(d_r2_lh_com, d_r2_rh_com)
        r1_com_progress = max(0.0, r1_com_dist_start - r1_com_dist_end)
        r2_com_progress = max(0.0, r2_com_dist_start - r2_com_dist_end)

        # "Ha recibido golpe" = el puño del rival tocó Y se estaba moviendo hacia
        # él (hit ya filtrado por velocidad). Con fuerza bruta valia cualquier
        # contacto pasivo: si uno se caia solo ENCIMA de los puños en guardia
        # del otro, contaba como golpe recibido y el rival cobraba KO.
        if r2_hit >= _KO_HIT_MIN:
            self._r1_last_hit_recv = self._step_count
        if r1_hit >= _KO_HIT_MIN:
            self._r2_last_hit_recv = self._step_count
        r1_ko_by_hit = r1_fell and (self._step_count - self._r1_last_hit_recv) <= _KO_HIT_WINDOW
        r2_ko_by_hit = r2_fell and (self._step_count - self._r2_last_hit_recv) <= _KO_HIT_WINDOW

        reward_1 = (
            _STRIKE_BONUS * r1_strikes + _HIT_WEIGHT * r1_strike_hit
            - _RECV_WEIGHT * r2_strikes
            - _CLINCH_PENALTY * r1_clinch
            + _ALIVE_BONUS + _UPRIGHT_W * r1_upz
            + _PUNCH_SPEED_WEIGHT * r1_punch_speed
            + _COM_PROXIMITY_WEIGHT * r1_com_progress
            - _ARM_CTRL_COST * float(np.sum(arm_a1**2))
            + (_KO_BONUS     if r2_ko_by_hit and not r1_fell else 0.0)
            + (_FALL_PENALTY if r1_fell else 0.0)
        )
        reward_2 = (
            _STRIKE_BONUS * r2_strikes + _HIT_WEIGHT * r2_strike_hit
            - _RECV_WEIGHT * r1_strikes
            - _CLINCH_PENALTY * r2_clinch
            + _PUNCH_SPEED_WEIGHT * r2_punch_speed
            + _COM_PROXIMITY_WEIGHT * r2_com_progress
            + _ALIVE_BONUS + _UPRIGHT_W * r2_upz
            - _ARM_CTRL_COST * float(np.sum(arm_a2**2))
            + (_KO_BONUS     if r1_ko_by_hit and not r2_fell else 0.0)
            + (_FALL_PENALTY if r2_fell else 0.0)
        )

        self._step_count += 1
        # Cualquier caida termina el asalto, pero solo r*_ko_by_hit (caida tras
        # recibir golpe) da el bonus de KO al rival.
        terminated = bool(r1_fell or r2_fell)
        truncated  = self._step_count >= _MAX_STEPS

        info = {
            "r1_fell": r1_fell,   "r2_fell": r2_fell,
            "r1_hit":  r1_hit,    "r2_hit":  r2_hit,
            "r1_punch_speed": r1_punch_speed, "r2_punch_speed": r2_punch_speed,
            "r1_com_dist": r1_com_dist_end, "r2_com_dist": r2_com_dist_end,
            "r1_ko_by_hit": r1_ko_by_hit, "r2_ko_by_hit": r2_ko_by_hit,
            "r1_strikes": r1_strikes, "r2_strikes": r2_strikes,
            "r1_clinch": r1_clinch,   "r2_clinch": r2_clinch,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_fight_obs(), (reward_1, reward_2), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model)
        self._renderer.update_scene(self.data)
        if self.render_mode == "human":
            import cv2
            frame = self._renderer.render()
            cv2.imshow("CloseCombat", frame[:, :, ::-1])
            cv2.waitKey(1)
        elif self.render_mode == "rgb_array":
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
