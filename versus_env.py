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

# Distancia (raíz-raíz, XY) a la que frenamos en seco a ambos robots para que
# no sigan cerrando distancia hasta chocar/tropezar. 0.12-0.15m se quedaban
# cortos: los robots se cruzaban/pasaban de largo el uno junto al otro y
# tropezaban antes de que el freno surtiera efecto. Con mas margen (0.20m)
# hay tiempo de sobra para detectar y frenar antes de que se toquen.
_STOP_DIST     = 0.20

# Pesos de recompensa — copia exacta de walking_env.py
_CTRL_COST_WEIGHT      = 0.01
_FORWARD_WEIGHT        = 5.0     # aplicado sobre approach_velocity, no world-X
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

# qpos: [r1_freejoint(7) | r1_joints(16) | r2_freejoint(7) | r2_joints(16)] = 46
# qvel: [r1_freejoint(6) | r1_joints(16) | r2_freejoint(6) | r2_joints(16)] = 44
_LEG_IDX     = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
_ARM_IDX     = np.array([5, 6, 7, 13, 14, 15], dtype=int)
_R1_LEG_CTRL = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
_R2_LEG_CTRL = np.array([16, 17, 18, 19, 20, 24, 25, 26, 27, 28], dtype=int)


class VersusEnv(gym.Env):
    """Dos robots Alpha enfrentados: r1 en x=+1.0 mira -X, r2 en x=-1.0 mira +X.
    Cada robot aprende a andar HACIA el oponente.

    Correcciones respecto a parallel_running / fighting_env antiguo:
      1. Forward reward = approach_velocity (proyección de vel sobre dirección al rival).
         Esto es correcto para CUALQUIER orientación — no depende de world-X.
      2. Pie trasero/delantero determinado por proyección sobre dirección al rival:
         el pie más avanzado HACIA el oponente es el delantero.
         Esto es correcto para r1 (-X) y r2 (+X) sin código especial.
      3. slow_penalty cuando approach_velocity < 0.02, no cuando qvel[0] < 0.02.
      4. obs 34 dims = 31 walking + [dx, dy, dz al oponente].
      5. Reset individual por robot — el otro no se interrumpe.

    step() devuelve info["r1_fell"] / info["r2_fell"] para done por robot.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, xml_path=None):
        self.render_mode = render_mode

        xml_path = os.path.abspath(xml_path if xml_path is not None else _XML_PATH)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self._r1_ctrl_low  = self.model.actuator_ctrlrange[_R1_LEG_CTRL, 0].copy()
        self._r1_ctrl_high = self.model.actuator_ctrlrange[_R1_LEG_CTRL, 1].copy()

        n_leg   = len(_LEG_IDX)
        obs_dim = self._get_obs()[0].shape[0]   # 34

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2 * n_leg,), dtype=np.float32
        )
        obs_limit = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_limit, high=obs_limit, dtype=np.float32
        )

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

        self._r1_arm_qpos = 7  + _ARM_IDX
        self._r1_arm_qvel = 6  + _ARM_IDX
        self._r2_arm_qpos = 30 + _ARM_IDX
        self._r2_arm_qvel = 28 + _ARM_IDX

        # Posición inicial para reset individual
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._r1_init_qpos = self.data.qpos[:23].copy()
        self._r2_init_qpos = self.data.qpos[23:].copy()
        self._rng = np.random.default_rng()

        self._renderer   = None
        self._step_count = 0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _denorm(self, action):
        half = (self._r1_ctrl_high - self._r1_ctrl_low) / 2.0
        return np.clip(action * half, self._r1_ctrl_low, self._r1_ctrl_high)

    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        r1_x, r1_y, r1_z = qpos[0],  qpos[1],  qpos[2]
        r2_x, r2_y, r2_z = qpos[23], qpos[24], qpos[25]
        obs_r1 = np.concatenate([
            qpos[2:7],  qvel[0:6],
            qpos[7:23][_LEG_IDX],  qvel[6:22][_LEG_IDX],
            [r2_x - r1_x, r2_y - r1_y, r2_z - r1_z],   # vector al oponente
        ]).astype(np.float32)
        obs_r2 = np.concatenate([
            qpos[25:30], qvel[22:28],
            qpos[30:46][_LEG_IDX], qvel[28:44][_LEG_IDX],
            [r1_x - r2_x, r1_y - r2_y, r1_z - r2_z],   # vector al oponente
        ]).astype(np.float32)
        return obs_r1, obs_r2

    def _reset_robot(self, robot):
        if robot == 1:
            self.data.qpos[:23]  = self._r1_init_qpos.copy()
            self.data.qvel[:22]  = 0.0
            self.data.qpos[7:23] += self._rng.uniform(-0.05, 0.05, 16)
        else:
            self.data.qpos[23:]   = self._r2_init_qpos.copy()
            self.data.qvel[22:]   = 0.0
            self.data.qpos[30:46] += self._rng.uniform(-0.05, 0.05, 16)

    def _robot_reward(self, qpos, qvel, a, qpos_off, qvel_off,
                      lf_id, rf_id, com_id, lank, rank, lfeet, rfeet,
                      approach_vel, robot_pos_2d, unit_to_opp_2d, dist_2d):
        """Reward idéntico a walking_env pero con approach_velocity en lugar de world-X.
        El pie trasero/delantero se determina por proyección sobre la dirección al oponente,
        lo que es correcto para CUALQUIER orientación del robot."""
        x_velocity = float(qvel[qvel_off])       # vx mundo (solo para diagnóstico)
        y_velocity = float(qvel[qvel_off + 1])
        yaw_vel    = float(qvel[qvel_off + 5])

        lf_pos = self.data.xpos[lf_id]
        rf_pos = self.data.xpos[rf_id]
        lf_z = float(lf_pos[2]); lf_xy = lf_pos[:2]
        rf_z = float(rf_pos[2]); rf_xy = rf_pos[:2]

        lf_stance = lf_z < _STANCE_Z
        rf_stance = rf_z < _STANCE_Z

        # Pie delantero = el más avanzado hacia el oponente
        # Proyectamos la posición de cada pie (relativa al robot) sobre unit_to_opp
        lf_proj = float(np.dot(lf_xy - robot_pos_2d, unit_to_opp_2d))
        rf_proj = float(np.dot(rf_xy - robot_pos_2d, unit_to_opp_2d))
        rear_is_left = lf_proj < rf_proj   # True: lf es trasero; False: rf es trasero
        rear_z  = lf_z if rear_is_left else rf_z
        front_z = rf_z if rear_is_left else lf_z

        qx = float(qpos[qpos_off + 4])
        qy = float(qpos[qpos_off + 5])
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

        # ── Reward principal: avanzar hacia el oponente ───────────────────────
        forward_reward     = _FORWARD_WEIGHT * max(0.0, approach_vel)
        # Dentro de _STOP_DIST no penalizamos ir despacio/parado — ya han llegado.
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
        # Peso asimétrico según qué pie es trasero (correcto para ambas orientaciones)
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
        return float(reward), {
            "approach_vel":   approach_vel,
            "x_velocity":     x_velocity,
            "lf_tilt":        lf_tilt,        "rf_tilt":      rf_tilt,
            "lf_z":           lf_z,           "rf_z":         rf_z,
            "forward_reward": forward_reward,
            "rear_is_left":   rear_is_left,
            # diagnósticos cadera lateral (abre piernas = bug de espejo)
            "lat_thigh_L":    float(qpos[qpos_off + 7]),
            "lat_thigh_R":    float(qpos[qpos_off + 15]),
        }

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        self.data.qpos[7:23]  += rng.uniform(-0.05, 0.05, 16)
        self.data.qpos[30:46] += rng.uniform(-0.05, 0.05, 16)
        self._step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, actions):
        actions = np.clip(actions, -1.0, 1.0)
        a1 = actions[:10]
        a2 = actions[10:]

        for _ in range(_ACTION_REPEAT):
            ctrl = np.zeros(self.model.nu, dtype=np.float64)
            ctrl[_R1_LEG_CTRL] = self._denorm(a1)
            ctrl[_R2_LEG_CTRL] = self._denorm(a2)
            self.data.ctrl[:] = ctrl
            for _ in range(_FRAME_SKIP):
                mujoco.mj_step(self.model, self.data)
                # Freno automático: en cuanto los troncos están a <= _STOP_DIST,
                # anulamos la velocidad horizontal de cada raíz en CADA substep.
                # Así dejan de acercarse en el momento exacto en que llegan a rango,
                # en vez de esperar a que la política haya aprendido a frenar sola
                # (lo que antes les dejaba seguir con inercia hasta tropezar/chocar).
                r1_xy = self.data.xpos[self._r1_id][:2]
                r2_xy = self.data.xpos[self._r2_id][:2]
                if float(np.linalg.norm(r1_xy - r2_xy)) <= _STOP_DIST:
                    self.data.qvel[0:6]   = 0.0
                    self.data.qvel[22:28] = 0.0
            self.data.qpos[self._r1_arm_qpos] = 0.0
            self.data.qvel[self._r1_arm_qvel] = 0.0
            self.data.qpos[self._r2_arm_qpos] = 0.0
            self.data.qvel[self._r2_arm_qvel] = 0.0
            mujoco.mj_forward(self.model, self.data)

        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        # Posiciones de los cuerpos raíz (world frame)
        r1_pos = self.data.xpos[self._r1_id].copy()
        r2_pos = self.data.xpos[self._r2_id].copy()

        # Vector unitario 2D de cada robot hacia el otro
        dir_12_xy = r2_pos[:2] - r1_pos[:2]
        dist_2d   = float(np.linalg.norm(dir_12_xy))
        if dist_2d > 0.01:
            unit_12 = dir_12_xy / dist_2d
        else:
            unit_12 = np.array([0.0, 0.0])

        # Velocidad de aproximación: proyección de vel sobre la dirección al oponente
        r1_vel_xy = np.array([float(qvel[0]),  float(qvel[1])])
        r2_vel_xy = np.array([float(qvel[22]), float(qvel[23])])
        approach_r1 = float(np.dot(r1_vel_xy,  unit_12))
        approach_r2 = float(np.dot(r2_vel_xy, -unit_12))

        r1_upz  = 1.0 - 2.0 * (qpos[4] ** 2 + qpos[5] ** 2)
        r2_upz  = 1.0 - 2.0 * (qpos[27] ** 2 + qpos[28] ** 2)
        r1_fell = bool(qpos[2]  < _FALL_Z or r1_upz < _TILT_Z)
        r2_fell = bool(qpos[25] < _FALL_Z or r2_upz < _TILT_Z)

        reward_1, info1 = self._robot_reward(
            qpos, qvel, a1, 0, 0,
            self._r1_lf_id, self._r1_rf_id, self._r1_id,
            self._r1_lank, self._r1_rank, self._r1_lfeet, self._r1_rfeet,
            approach_r1, r1_pos[:2], unit_12, dist_2d,
        )
        reward_2, info2 = self._robot_reward(
            qpos, qvel, a2, 23, 22,
            self._r2_lf_id, self._r2_rf_id, self._r2_id,
            self._r2_lank, self._r2_rank, self._r2_lfeet, self._r2_rfeet,
            approach_r2, r2_pos[:2], -unit_12, dist_2d,
        )

        # Reset individual del robot caído
        if r1_fell:
            self._reset_robot(1)
        if r2_fell:
            self._reset_robot(2)
        if r1_fell or r2_fell:
            mujoco.mj_forward(self.model, self.data)

        self._step_count += 1
        terminated = False
        truncated  = self._step_count >= _MAX_STEPS

        info = {
            "r1_fell":          r1_fell,                     "r2_fell":          r2_fell,
            "dist":             dist_2d,
            "stopped":          dist_2d <= _STOP_DIST,
            "r1_approach_vel":  info1["approach_vel"],        "r2_approach_vel":  info2["approach_vel"],
            "r1_x_velocity":    info1["x_velocity"],          "r2_x_velocity":    info2["x_velocity"],
            "r1_lf_tilt":       info1["lf_tilt"],             "r1_rf_tilt":       info1["rf_tilt"],
            "r2_lf_tilt":       info2["lf_tilt"],             "r2_rf_tilt":       info2["rf_tilt"],
            # diagnósticos espejo
            "r1_fwd_rew":       info1["forward_reward"],      "r2_fwd_rew":       info2["forward_reward"],
            "r1_lat_L":         info1["lat_thigh_L"],         "r1_lat_R":         info1["lat_thigh_R"],
            "r2_lat_L":         info2["lat_thigh_L"],         "r2_lat_R":         info2["lat_thigh_R"],
            "r1_rear_is_left":  info1["rear_is_left"],        "r2_rear_is_left":  info2["rear_is_left"],
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
            cv2.imshow("Versus", frame[:, :, ::-1])
            cv2.waitKey(1)
        elif self.render_mode == "rgb_array":
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
