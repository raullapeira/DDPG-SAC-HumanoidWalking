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
_MAX_STEPS     = 1000

# Distancia (raiz-raiz, XY) a la que termina el acercamiento y empieza el combate.
# 0.12-0.15m se quedaban cortos: los robots se cruzaban/pasaban de largo el uno
# junto al otro y tropezaban antes de que el freno surtiera efecto. Con mas
# margen (0.20m) hay tiempo de sobra para detectar y frenar antes de tocarse.
_STOP_DIST     = 0.20

# Para pasar a combate no basta con estar cerca: si alguno de los dos ya viene
# cayendose (p.ej. un traspies durante el acercamiento que de casualidad lo
# acerca al rival en ese mismo instante), NO debe contar como "han llegado y
# van a pelear" — eso generaba falsos KOs por caidas del acercamiento, no del
# combate. Exigimos ademas que ambos esten bien erguidos.
_TRANSITION_MIN_UPZ = 0.9

# ── Pesos de recompensa de ACERCAMIENTO — copia exacta de versus_env.py ───────
_CTRL_COST_WEIGHT      = 0.01
_FORWARD_WEIGHT        = 5.0
_ALIVE_BONUS           = 1.0
_UPRIGHT_WEIGHT        = 0.3
_LATERAL_COST_WEIGHT   = 0.15
_YAW_COST_WEIGHT       = 1.0
_FOOT_HEIGHT_WEIGHT    = 2.0
_FRONT_LIFT_PENALTY    = 1.0
_STANCE_PENALTY        = -0.5
_SLOW_PENALTY          = -2.0
_ANKLE_COST_WEIGHT     = 2.0
_FEET_COST_WEIGHT      = 6.0
_FOOT_FLAT_REAR_WEIGHT  = 14.0
_FOOT_FLAT_FRONT_WEIGHT =  5.0
_COM_SUPPORT_WEIGHT    = 8.0
_SINGLE_SUPP_BONUS     = 0.3
_STANCE_Z              = 0.04

# ── Pesos de recompensa de COMBATE — copia exacta de close_combat_env.py ──────
_HIT_WEIGHT    =  5.0
_RECV_WEIGHT   =  3.0
_ARM_CTRL_COST =  0.01
_FALL_PENALTY  = -20.0
_KO_BONUS      =  20.0
_SENSOR_CLIP   = 20.0

# qpos: [r1_freejoint(7) | r1_joints(16) | r2_freejoint(7) | r2_joints(16)] = 46
# qvel: [r1_freejoint(6) | r1_joints(16) | r2_freejoint(6) | r2_joints(16)] = 44
_LEG_IDX     = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
_ARM_IDX     = np.array([5, 6, 7, 13, 14, 15], dtype=int)
_R1_LEG_CTRL = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
_R1_ARM_CTRL = np.array([5, 6, 7, 13, 14, 15], dtype=int)
_R2_LEG_CTRL = np.array([16, 17, 18, 19, 20, 24, 25, 26, 27, 28], dtype=int)
_R2_ARM_CTRL = np.array([21, 22, 23, 29, 30, 31], dtype=int)

WALK_OBS_DIM  = 34   # identico a versus_env: 31 walking + vector al oponente
FIGHT_OBS_DIM = 28   # identico a close_combat_env


class FullFightEnv(gym.Env):
    """Pipeline completo de un asalto en un solo episodio.

    Fase 1 — ACERCAMIENTO: piernas controladas por una politica entrenable
    (SAC), igual que versus_env.py, con el mismo freno automatico por
    distancia. Los brazos estan en neutro.

    Fase 2 — COMBATE: en cuanto los troncos llegan a <= _STOP_DIST, las
    piernas se CONGELAN en la postura exacta que tenian en ese instante (no
    hace falta que sigan intentando moverse) y los brazos pasan a estar
    controlados por una politica de combate entrenable (otra red), con
    recompensa por golpear/recibir/derribar via sensores de contacto en los
    puños (igual que close_combat_env.py).

    El episodio TERMINA (terminated=True, no solo truncated) en cuanto un
    robot cae estando YA en fase de combate — eso es el KO de ese asalto.
    Una caida durante la fase de acercamiento solo resetea a ese robot
    individualmente y el episodio sigue (como en versus_env.py).

    step(actions) recibe 32 valores: [leg_a1(10)|arm_a1(6)|leg_a2(10)|arm_a2(6)].
    Los leg_a* se ignoran una vez ha empezado el combate; los arm_a* se
    ignoran mientras no ha empezado (brazos en neutro).

    reset/step devuelven ((walk_obs_r1, walk_obs_r2), (fight_obs_r1, fight_obs_r2)).
    info["fighting"] indica en que fase se ha ejecutado ESTE step.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, xml_path=None):
        self.render_mode = render_mode

        xml_path = os.path.abspath(xml_path if xml_path is not None else _XML_PATH)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self._leg_low  = self.model.actuator_ctrlrange[_R1_LEG_CTRL, 0].copy()
        self._leg_high = self.model.actuator_ctrlrange[_R1_LEG_CTRL, 1].copy()
        self._r1_arm_low  = self.model.actuator_ctrlrange[_R1_ARM_CTRL, 0].copy()
        self._r1_arm_high = self.model.actuator_ctrlrange[_R1_ARM_CTRL, 1].copy()
        self._r2_arm_low  = self.model.actuator_ctrlrange[_R2_ARM_CTRL, 0].copy()
        self._r2_arm_high = self.model.actuator_ctrlrange[_R2_ARM_CTRL, 1].copy()

        _bid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
        self._r1_id    = _bid("r1_root")
        self._r2_id    = _bid("r2_root")
        self._r1_lf_id = _bid("r1_Left_Feet_link")
        self._r1_rf_id = _bid("r1_Right_Feet_link")
        self._r2_lf_id = _bid("r2_Left_Feet_link")
        self._r2_rf_id = _bid("r2_Right_Feet_link")

        self._r1_lank  = 7  + 3
        self._r1_rank  = 7  + 11
        self._r1_lfeet = 7  + 4
        self._r1_rfeet = 7  + 12
        self._r2_lank  = 30 + 3
        self._r2_rank  = 30 + 11
        self._r2_lfeet = 30 + 4
        self._r2_rfeet = 30 + 12

        self._r1_leg_qpos = 7  + _LEG_IDX
        self._r1_leg_qvel = 6  + _LEG_IDX
        self._r2_leg_qpos = 30 + _LEG_IDX
        self._r2_leg_qvel = 28 + _LEG_IDX
        self._r1_arm_qpos = 7  + _ARM_IDX
        self._r1_arm_qvel = 6  + _ARM_IDX
        self._r2_arm_qpos = 30 + _ARM_IDX
        self._r2_arm_qvel = 28 + _ARM_IDX

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._r1_init_qpos = self.data.qpos[:23].copy()
        self._r2_init_qpos = self.data.qpos[23:].copy()
        self._rng = np.random.default_rng()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(32,), dtype=np.float32)
        walk_limit  = np.full(WALK_OBS_DIM,  np.inf, dtype=np.float32)
        fight_limit = np.full(FIGHT_OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=-walk_limit,  high=walk_limit,  dtype=np.float32),
            spaces.Box(low=-fight_limit, high=fight_limit, dtype=np.float32),
        ))

        self._renderer   = None
        self._step_count = 0
        self._fighting    = False
        self._r1_leg_freeze_qpos = None
        self._r2_leg_freeze_qpos = None
        self._r1_leg_ctrl_hold   = None
        self._r2_leg_ctrl_hold   = None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _denorm_leg(self, a):
        half = (self._leg_high - self._leg_low) / 2.0
        return np.clip(a * half, self._leg_low, self._leg_high)

    def _denorm_arm(self, a, low, high):
        center = (high + low) / 2.0
        half   = (high - low) / 2.0
        return np.clip(center + a * half, low, high)

    def _get_walk_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        r1_x, r1_y, r1_z = qpos[0],  qpos[1],  qpos[2]
        r2_x, r2_y, r2_z = qpos[23], qpos[24], qpos[25]
        obs_r1 = np.concatenate([
            qpos[2:7],  qvel[0:6],
            qpos[7:23][_LEG_IDX],  qvel[6:22][_LEG_IDX],
            [r2_x - r1_x, r2_y - r1_y, r2_z - r1_z],
        ]).astype(np.float32)
        obs_r2 = np.concatenate([
            qpos[25:30], qvel[22:28],
            qpos[30:46][_LEG_IDX], qvel[28:44][_LEG_IDX],
            [r1_x - r2_x, r1_y - r2_y, r1_z - r2_z],
        ]).astype(np.float32)
        return obs_r1, obs_r2

    def _get_fight_obs(self):
        """[arm_qpos(6) | arm_qvel(6) | orient+z(5) | angvel(3) | to_opp(3) | opp_vel_rel(3) | touch(2)]"""
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
        ]).astype(np.float32)

        obs_r2 = np.concatenate([
            qpos[30:46][_ARM_IDX],  qvel[28:44][_ARM_IDX],
            qpos[25:30],            qvel[25:28],
            r1_pos - r2_pos,        r1_vel - r2_vel,
            sensors[2:4],
        ]).astype(np.float32)

        return obs_r1, obs_r2

    def _get_obs(self):
        return self._get_walk_obs(), self._get_fight_obs()

    def _reset_robot(self, robot):
        if robot == 1:
            self.data.qpos[:23]  = self._r1_init_qpos.copy()
            self.data.qvel[:22]  = 0.0
            self.data.qpos[7:23] += self._rng.uniform(-0.05, 0.05, 16)
        else:
            self.data.qpos[23:]   = self._r2_init_qpos.copy()
            self.data.qvel[22:]   = 0.0
            self.data.qpos[30:46] += self._rng.uniform(-0.05, 0.05, 16)

    def _approach_reward(self, qpos, qvel, a, qpos_off, qvel_off,
                         lf_id, rf_id, com_id, lank, rank, lfeet, rfeet,
                         approach_vel, robot_pos_2d, unit_to_opp_2d, dist_2d):
        """Reward de acercamiento — copia exacta de versus_env.py._robot_reward."""
        x_velocity = float(qvel[qvel_off])
        y_velocity = float(qvel[qvel_off + 1])
        yaw_vel    = float(qvel[qvel_off + 5])

        lf_pos = self.data.xpos[lf_id]
        rf_pos = self.data.xpos[rf_id]
        lf_z = float(lf_pos[2]); lf_xy = lf_pos[:2]
        rf_z = float(rf_pos[2]); rf_xy = rf_pos[:2]

        lf_stance = lf_z < _STANCE_Z
        rf_stance = rf_z < _STANCE_Z

        lf_proj = float(np.dot(lf_xy - robot_pos_2d, unit_to_opp_2d))
        rf_proj = float(np.dot(rf_xy - robot_pos_2d, unit_to_opp_2d))
        rear_is_left = lf_proj < rf_proj
        rear_z  = lf_z if rear_is_left else rf_z
        front_z = rf_z if rear_is_left else lf_z

        qx = float(qpos[qpos_off + 4])
        qy = float(qpos[qpos_off + 5])
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

        forward_reward     = _FORWARD_WEIGHT * max(0.0, approach_vel)
        slow_penalty       = _SLOW_PENALTY if (approach_vel < 0.02 and dist_2d > _STOP_DIST) else 0.0

        upright_reward     = _UPRIGHT_WEIGHT * up_z
        ctrl_cost          = _CTRL_COST_WEIGHT * float(np.sum(np.square(a)))
        lateral_cost       = _LATERAL_COST_WEIGHT * y_velocity ** 2
        yaw_cost           = _YAW_COST_WEIGHT * yaw_vel ** 2
        foot_height_reward = _FOOT_HEIGHT_WEIGHT * max(0.0, rear_z - _STANCE_Z)
        front_lift_penalty = _FRONT_LIFT_PENALTY * max(0.0, front_z - 0.05)
        stance_penalty     = _STANCE_PENALTY if (lf_stance and rf_stance) else 0.0
        single_supp_bonus  = _SINGLE_SUPP_BONUS if (lf_stance != rf_stance) else 0.0

        com   = self.data.subtree_com[com_id]
        com_x, com_y = float(com[0]), float(com[1])
        com_support_reward = 0.0
        if lf_stance and not rf_stance:
            dx = com_x - float(lf_pos[0]); dy = com_y - float(lf_pos[1])
            com_support_reward = -_COM_SUPPORT_WEIGHT * (dx*dx + dy*dy) ** 0.5
        elif rf_stance and not lf_stance:
            dx = com_x - float(rf_pos[0]); dy = com_y - float(rf_pos[1])
            com_support_reward = -_COM_SUPPORT_WEIGHT * (dx*dx + dy*dy) ** 0.5

        ankle_cost = 0.0
        if lf_stance:
            ankle_cost += _ANKLE_COST_WEIGHT * float(qpos[lank])  ** 2
            ankle_cost += _FEET_COST_WEIGHT  * float(qpos[lfeet]) ** 2
        if rf_stance:
            ankle_cost += _ANKLE_COST_WEIGHT * float(qpos[rank])  ** 2
            ankle_cost += _FEET_COST_WEIGHT  * float(qpos[rfeet]) ** 2

        lf_mat  = self.data.xmat[lf_id].reshape(3, 3)
        rf_mat  = self.data.xmat[rf_id].reshape(3, 3)
        lf_tilt = 1.0 - float(np.max(np.abs(lf_mat[2, :])))
        rf_tilt = 1.0 - float(np.max(np.abs(rf_mat[2, :])))
        lf_fw = _FOOT_FLAT_REAR_WEIGHT  if rear_is_left else _FOOT_FLAT_FRONT_WEIGHT
        rf_fw = _FOOT_FLAT_REAR_WEIGHT  if not rear_is_left else _FOOT_FLAT_FRONT_WEIGHT
        foot_flat_cost = 0.0
        if lf_stance:
            foot_flat_cost += lf_fw * lf_tilt
        if rf_stance:
            foot_flat_cost += rf_fw * rf_tilt

        reward = (
            forward_reward + _ALIVE_BONUS + upright_reward
            + foot_height_reward + com_support_reward + single_supp_bonus
            - ctrl_cost - lateral_cost - yaw_cost
            - ankle_cost - foot_flat_cost - front_lift_penalty
            + stance_penalty + slow_penalty
        )
        return float(reward)

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        self.data.qpos[7:23]  += rng.uniform(-0.05, 0.05, 16)
        self.data.qpos[30:46] += rng.uniform(-0.05, 0.05, 16)
        self._step_count = 0
        self._fighting = False
        self._r1_leg_freeze_qpos = None
        self._r2_leg_freeze_qpos = None
        self._r1_leg_ctrl_hold   = None
        self._r2_leg_ctrl_hold   = None
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, actions):
        actions = np.clip(actions, -1.0, 1.0).astype(np.float64)
        leg_a1 = actions[0:10]
        arm_a1 = actions[10:16]
        leg_a2 = actions[16:26]
        arm_a2 = actions[26:32]

        # Fase con la que se ejecuta ESTE step (decidida al final del anterior).
        fighting_now = self._fighting
        total_sensors = np.zeros(4)

        for _ in range(_ACTION_REPEAT):
            ctrl = np.zeros(self.model.nu, dtype=np.float64)
            if fighting_now:
                ctrl[_R1_LEG_CTRL] = self._r1_leg_ctrl_hold
                ctrl[_R2_LEG_CTRL] = self._r2_leg_ctrl_hold
                ctrl[_R1_ARM_CTRL] = self._denorm_arm(arm_a1, self._r1_arm_low, self._r1_arm_high)
                ctrl[_R2_ARM_CTRL] = self._denorm_arm(arm_a2, self._r2_arm_low, self._r2_arm_high)
            else:
                ctrl[_R1_LEG_CTRL] = self._denorm_leg(leg_a1)
                ctrl[_R2_LEG_CTRL] = self._denorm_leg(leg_a2)
            self.data.ctrl[:] = ctrl

            for _ in range(_FRAME_SKIP):
                mujoco.mj_step(self.model, self.data)
                if fighting_now:
                    total_sensors += self.data.sensordata[:4].copy()
                    # Piernas realmente congeladas — no dependemos solo del PD.
                    self.data.qpos[self._r1_leg_qpos] = self._r1_leg_freeze_qpos
                    self.data.qvel[self._r1_leg_qvel] = 0.0
                    self.data.qpos[self._r2_leg_qpos] = self._r2_leg_freeze_qpos
                    self.data.qvel[self._r2_leg_qvel] = 0.0
                else:
                    # Brazos en neutro mientras se acercan.
                    self.data.qpos[self._r1_arm_qpos] = 0.0
                    self.data.qvel[self._r1_arm_qvel] = 0.0
                    self.data.qpos[self._r2_arm_qpos] = 0.0
                    self.data.qvel[self._r2_arm_qvel] = 0.0
                    # Freno automatico al llegar a distancia de combate. Frenamos
                    # TODO el freejoint (lineal + angular), no solo el avance —
                    # si no, el robot puede llegar con un residuo de giro/balanceo
                    # del propio andar y, con las piernas ya congeladas (sin
                    # control activo de equilibrio), ese residuo basta para
                    # tumbarlo solo en un par de segundos, sin que nadie le
                    # haya llegado a golpear.
                    r1_xy = self.data.xpos[self._r1_id][:2]
                    r2_xy = self.data.xpos[self._r2_id][:2]
                    if float(np.linalg.norm(r1_xy - r2_xy)) <= _STOP_DIST:
                        self.data.qvel[0:6]   = 0.0
                        self.data.qvel[22:28] = 0.0
            mujoco.mj_forward(self.model, self.data)

        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        r1_pos = self.data.xpos[self._r1_id].copy()
        r2_pos = self.data.xpos[self._r2_id].copy()
        dir_12_xy = r2_pos[:2] - r1_pos[:2]
        dist_2d   = float(np.linalg.norm(dir_12_xy))
        unit_12   = dir_12_xy / dist_2d if dist_2d > 0.01 else np.array([0.0, 0.0])

        r1_upz  = 1.0 - 2.0 * (qpos[4] ** 2 + qpos[5] ** 2)
        r2_upz  = 1.0 - 2.0 * (qpos[27] ** 2 + qpos[28] ** 2)
        r1_fell = bool(qpos[2]  < _FALL_Z or r1_upz < _TILT_Z)
        r2_fell = bool(qpos[25] < _FALL_Z or r2_upz < _TILT_Z)

        if fighting_now:
            avg_sensors = total_sensors / (_ACTION_REPEAT * _FRAME_SKIP)
            r1_hit = float(np.sum(np.clip(avg_sensors[0:2], 0, _SENSOR_CLIP)) / _SENSOR_CLIP)
            r2_hit = float(np.sum(np.clip(avg_sensors[2:4], 0, _SENSOR_CLIP)) / _SENSOR_CLIP)

            reward_1 = (
                _HIT_WEIGHT * r1_hit - _RECV_WEIGHT * r2_hit
                + _ALIVE_BONUS + _UPRIGHT_WEIGHT * r1_upz
                - _ARM_CTRL_COST * float(np.sum(arm_a1 ** 2))
                + (_KO_BONUS     if r2_fell and not r1_fell else 0.0)
                + (_FALL_PENALTY if r1_fell else 0.0)
            )
            reward_2 = (
                _HIT_WEIGHT * r2_hit - _RECV_WEIGHT * r1_hit
                + _ALIVE_BONUS + _UPRIGHT_WEIGHT * r2_upz
                - _ARM_CTRL_COST * float(np.sum(arm_a2 ** 2))
                + (_KO_BONUS     if r1_fell and not r2_fell else 0.0)
                + (_FALL_PENALTY if r2_fell else 0.0)
            )
            extra = {"r1_hit": r1_hit, "r2_hit": r2_hit}
        else:
            approach_r1 = float(np.dot(np.array([qvel[0], qvel[1]]),  unit_12))
            approach_r2 = float(np.dot(np.array([qvel[22], qvel[23]]), -unit_12))
            reward_1 = self._approach_reward(
                qpos, qvel, leg_a1, 0, 0,
                self._r1_lf_id, self._r1_rf_id, self._r1_id,
                self._r1_lank, self._r1_rank, self._r1_lfeet, self._r1_rfeet,
                approach_r1, r1_pos[:2], unit_12, dist_2d,
            )
            reward_2 = self._approach_reward(
                qpos, qvel, leg_a2, 23, 22,
                self._r2_lf_id, self._r2_rf_id, self._r2_id,
                self._r2_lank, self._r2_rank, self._r2_lfeet, self._r2_rfeet,
                approach_r2, r2_pos[:2], -unit_12, dist_2d,
            )
            extra = {"r1_hit": 0.0, "r2_hit": 0.0}

            if r1_fell:
                self._reset_robot(1)
            if r2_fell:
                self._reset_robot(2)
            if r1_fell or r2_fell:
                mujoco.mj_forward(self.model, self.data)

        # Transicion a modo combate PARA EL SIGUIENTE step. En vez de congelar
        # las piernas en la postura que tuvieran a media zancada (que casi
        # nunca es un apoyo estable — el propio reward de caminar penaliza
        # tener los dos pies apoyados a la vez, así que ese instante "bueno"
        # practicamente no se da nunca durante el andar), las llevamos a una
        # postura neutra FIJA y conocida (la misma con la que arranca cada
        # episodio) — así el combate arranca siempre desde una base estable,
        # y una caida durante el combate es de verdad por el impacto recibido.
        if (not fighting_now) and dist_2d <= _STOP_DIST \
                and not r1_fell and not r2_fell \
                and r1_upz >= _TRANSITION_MIN_UPZ and r2_upz >= _TRANSITION_MIN_UPZ:
            self._fighting = True
            self._r1_leg_freeze_qpos = self._r1_init_qpos[7:23][_LEG_IDX].copy()
            self._r2_leg_freeze_qpos = self._r2_init_qpos[7:23][_LEG_IDX].copy()
            self._r1_leg_ctrl_hold   = np.clip(self._r1_leg_freeze_qpos, self._leg_low, self._leg_high)
            self._r2_leg_ctrl_hold   = np.clip(self._r2_leg_freeze_qpos, self._leg_low, self._leg_high)

        self._step_count += 1
        # KO: el episodio termina si la caida ocurre YA en fase de combate.
        terminated = bool(fighting_now and (r1_fell or r2_fell))
        truncated  = self._step_count >= _MAX_STEPS

        info = {
            "fighting": fighting_now,
            "dist":     dist_2d,
            "r1_fell":  r1_fell, "r2_fell": r2_fell,
            **extra,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), (reward_1, reward_2), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model)
        self._renderer.update_scene(self.data)
        if self.render_mode == "human":
            import cv2
            frame = self._renderer.render()
            cv2.imshow("FullFight", frame[:, :, ::-1])
            cv2.waitKey(1)
        elif self.render_mode == "rgb_array":
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
