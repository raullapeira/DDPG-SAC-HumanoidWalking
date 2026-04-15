import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

_XML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "robot", "alpha_single.xml"
)

_INIT_Z = 0.298
_FALL_Z = 0.12
_TILT_Z = 0.7

_FRAME_SKIP    = 5
_ACTION_REPEAT = 4

_CTRL_COST_WEIGHT    = 0.01
_FORWARD_WEIGHT      = 5.0
_ALIVE_BONUS         = 1.0
_UPRIGHT_WEIGHT      = 0.3
_LATERAL_COST_WEIGHT = 0.15
_YAW_COST_WEIGHT     = 1.0

_FOOT_HEIGHT_WEIGHT  = 2.0   # pie trasero se levanta durante el swing
_FRONT_LIFT_PENALTY  = 1.0   # penaliza levantar el pie delantero antes de tiempo
_STANCE_PENALTY      = -0.5  # penaliza doble apoyo prolongado
_SLOW_PENALTY        = -2.0  # penaliza velocidad casi nula

# v14: sin push_off (causaba propulsión por inclinación de tobillo)
# Nuevo: penalizar tobillo en pie de apoyo, COG sobre pie de apoyo en 2D
_ANKLE_COST_WEIGHT   = 2.0   # penaliza ank_fwd^2 cuando el pie está apoyado
_FEET_COST_WEIGHT    = 4.0   # penaliza puntera (Feet joint) en pie de apoyo
_FOOT_FLAT_WEIGHT    = 8.0   # penaliza inclinación real del pie respecto al suelo
_COM_SUPPORT_WEIGHT  = 8.0   # distancia 2D del COG al pie de apoyo (era 4.0, solo Y)
_SINGLE_SUPP_BONUS   = 0.3   # bonus por apoyo simple claro (un pie levantado >_SWING_Z)

# Umbral de altura del pie (Left/Right_Feet_link) para considerar apoyo vs. swing
_STANCE_Z = 0.04   # pie por debajo → apoyado; por encima → en el aire


class AlphaEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, xml_path=None):
        self.render_mode = render_mode

        xml_path = os.path.abspath(xml_path if xml_path is not None else _XML_PATH)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self._ctrl_low  = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        # Índices de actuadores: piernas (10) y brazos (6)
        self._leg_ctrl_idx = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
        self._arm_ctrl_idx = np.array([5, 6, 7, 13, 14, 15], dtype=int)
        self._leg_obs_idx  = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
        # Índices en qpos/qvel para fijar brazos
        self._arm_qpos_idx = np.array([7+5, 7+6, 7+7, 7+13, 7+14, 7+15], dtype=int)
        self._arm_qvel_idx = np.array([6+5, 6+6, 6+7, 6+13, 6+14, 6+15], dtype=int)

        n_act = len(self._leg_ctrl_idx)
        n_obs = self._get_obs().shape[0]

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32
        )
        obs_limit = np.full(n_obs, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_limit, high=obs_limit, dtype=np.float32
        )

        self._renderer = None
        self._prev_action = np.zeros(len(self._leg_ctrl_idx), dtype=np.float32)

        # Cuerpos de los pies (más precisos que las espinillas para detectar apoyo)
        self._lf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Left_Feet_link")
        self._rf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Right_Feet_link")

        # Índices de ángulo de tobillo adelante/atrás en qpos[7:]
        # qpos[7+3]  = Left_Ankle_link_joint  (L_ank_fwd, MuJoCo izq = físico der)
        # qpos[7+11] = Right_Ankle_link_joint (R_ank_fwd, MuJoCo der = físico izq)
        # qpos[7+4]  = Left_Feet_link_joint   (puntera izq)
        # qpos[7+12] = Right_Feet_link_joint  (puntera der)
        self._lank_fwd_idx  = 7 + 3
        self._rank_fwd_idx  = 7 + 11
        self._lfeet_idx     = 7 + 4
        self._rfeet_idx     = 7 + 12

    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        leg_qpos = qpos[7:][self._leg_obs_idx]
        leg_qvel = qvel[6:][self._leg_obs_idx]
        # 5 (pose torso) + 6 (vel torso) + 10 + 10 = 31D
        return np.concatenate([qpos[2:7], qvel[0:6], leg_qpos, leg_qvel]).astype(np.float32)

    def _denorm_action(self, action):
        low  = self._ctrl_low[self._leg_ctrl_idx]
        high = self._ctrl_high[self._leg_ctrl_idx]
        half = (high - low) / 2.0
        return np.clip(action * half, low, high)

    def _torso_up_z(self):
        qw, qx, qy, qz = self.data.qpos[3:7]
        return 1.0 - 2.0 * (qx * qx + qy * qy)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        rng = np.random.default_rng(seed)
        n_joints = self.model.nq - 7
        self.data.qpos[7:] += rng.uniform(-0.05, 0.05, n_joints)

        self._prev_action = np.zeros(len(self._leg_ctrl_idx), dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        for _ in range(_ACTION_REPEAT):
            full_ctrl = np.zeros(self.model.nu, dtype=np.float64)
            full_ctrl[self._leg_ctrl_idx] = self._denorm_action(action)
            self.data.ctrl[:] = full_ctrl
            for _ in range(_FRAME_SKIP):
                mujoco.mj_step(self.model, self.data)
            self.data.qpos[self._arm_qpos_idx] = 0.0
            self.data.qvel[self._arm_qvel_idx] = 0.0
            mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()

        root_z = self.data.qpos[2]
        up_z   = self._torso_up_z()
        terminated = bool(root_z < _FALL_Z or up_z < _TILT_Z)

        x_velocity = self.data.qvel[0]
        y_velocity = self.data.qvel[1]

        # ── Posiciones de los pies ────────────────────────────────────────────
        lf_pos = self.data.xpos[self._lf_id]
        rf_pos = self.data.xpos[self._rf_id]
        lf_z, lf_x = float(lf_pos[2]), float(lf_pos[0])
        rf_z, rf_x = float(rf_pos[2]), float(rf_pos[0])

        lf_stance = lf_z < _STANCE_Z   # pie izq (MuJoCo) apoyado
        rf_stance = rf_z < _STANCE_Z   # pie der (MuJoCo) apoyado

        # Pie trasero y delantero por posición X
        if lf_x < rf_x:
            rear_z, front_z = lf_z, rf_z
        else:
            rear_z, front_z = rf_z, lf_z

        # ── Rewards base ──────────────────────────────────────────────────────
        forward_reward = _FORWARD_WEIGHT * max(0.0, x_velocity)
        alive_bonus    = _ALIVE_BONUS
        upright_reward = _UPRIGHT_WEIGHT * up_z

        ctrl_cost    = _CTRL_COST_WEIGHT * np.sum(np.square(action))
        lateral_cost = _LATERAL_COST_WEIGHT * y_velocity ** 2
        yaw_cost     = _YAW_COST_WEIGHT * self.data.qvel[5] ** 2

        # Levantamiento del pie trasero (swing)
        foot_height_reward = _FOOT_HEIGHT_WEIGHT * max(0.0, rear_z - _STANCE_Z)

        # Penalizar levantar el pie delantero antes de apoyar
        front_lift_penalty = _FRONT_LIFT_PENALTY * max(0.0, front_z - 0.05)

        # Penalizar doble apoyo prolongado
        double_support = lf_stance and rf_stance
        stance_penalty = _STANCE_PENALTY if double_support else 0.0

        # Velocidad mínima
        slow_penalty = _SLOW_PENALTY if x_velocity < 0.02 else 0.0

        # ── Bonus por apoyo simple claro ──────────────────────────────────────
        single_support = lf_stance != rf_stance   # exactamente un pie en el aire
        single_supp_bonus = _SINGLE_SUPP_BONUS if single_support else 0.0

        # ── COG sobre el pie de apoyo (X + Y) ────────────────────────────────
        com_x = float(self.data.subtree_com[1][0])
        com_y = float(self.data.subtree_com[1][1])
        com_support_reward = 0.0
        if lf_stance and not rf_stance:
            # izq apoyado, der en swing → COG sobre pie izq
            dx = com_x - float(lf_pos[0])
            dy = com_y - float(lf_pos[1])
            com_support_reward = -_COM_SUPPORT_WEIGHT * (dx*dx + dy*dy) ** 0.5
        elif rf_stance and not lf_stance:
            # der apoyado, izq en swing → COG sobre pie der
            dx = com_x - float(rf_pos[0])
            dy = com_y - float(rf_pos[1])
            com_support_reward = -_COM_SUPPORT_WEIGHT * (dx*dx + dy*dy) ** 0.5

        # ── Penalizar tobillo y puntera en pie de apoyo ───────────────────────
        # Ankle: impide inclinación del tobillo para propulsarse
        # Feet:  impide usar la puntera para empujar (el joint Feet no existe en el real)
        l_ank_fwd  = float(self.data.qpos[self._lank_fwd_idx])
        r_ank_fwd  = float(self.data.qpos[self._rank_fwd_idx])
        l_feet_fwd = float(self.data.qpos[self._lfeet_idx])
        r_feet_fwd = float(self.data.qpos[self._rfeet_idx])
        ankle_cost = 0.0
        if lf_stance:
            ankle_cost += _ANKLE_COST_WEIGHT * l_ank_fwd ** 2
            ankle_cost += _FEET_COST_WEIGHT  * l_feet_fwd ** 2
        if rf_stance:
            ankle_cost += _ANKLE_COST_WEIGHT * r_ank_fwd ** 2
            ankle_cost += _FEET_COST_WEIGHT  * r_feet_fwd ** 2

        # ── Penalizar inclinación real del pie respecto al suelo ──────────────
        # xmat[body_id] = rotation matrix (row-major). La fila 2 da los componentes
        # world-Z de cada eje local. Si el pie está plano, algún eje local apunta
        # exactamente a world-Z → max(abs(fila2)) = 1.0 → tilt = 0.
        lf_mat   = self.data.xmat[self._lf_id].reshape(3, 3)
        rf_mat   = self.data.xmat[self._rf_id].reshape(3, 3)
        lf_tilt  = 1.0 - float(np.max(np.abs(lf_mat[2, :])))
        rf_tilt  = 1.0 - float(np.max(np.abs(rf_mat[2, :])))
        foot_flat_cost = 0.0
        if lf_stance:
            foot_flat_cost += _FOOT_FLAT_WEIGHT * lf_tilt
        if rf_stance:
            foot_flat_cost += _FOOT_FLAT_WEIGHT * rf_tilt

        # ── Reward total ──────────────────────────────────────────────────────
        reward = (
            forward_reward
            + alive_bonus
            + upright_reward
            + foot_height_reward
            + com_support_reward
            + single_supp_bonus
            - ctrl_cost
            - lateral_cost
            - yaw_cost
            - ankle_cost
            - foot_flat_cost
            - front_lift_penalty
            + stance_penalty
            + slow_penalty
        )

        self._prev_action = action.copy()
        truncated = False

        info = {
            "x_velocity":     x_velocity,
            "rear_z":         rear_z,
            "front_z":        front_z,
            "com_support":    com_support_reward,
            "ankle_cost":     ankle_cost,
            "foot_flat_cost": foot_flat_cost,
            "single_support": single_support,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model)
        self._renderer.update_scene(self.data)
        if self.render_mode == "human":
            import cv2
            frame = self._renderer.render()
            cv2.imshow("Alpha", frame[:, :, ::-1])
            cv2.waitKey(1)
        elif self.render_mode == "rgb_array":
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
