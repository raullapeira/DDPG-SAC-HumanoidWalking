import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

_XML_PATH = os.path.join(
    os.path.dirname(__file__), "robot", "configs", "fighting", "alpha_fight.xml"
)

_FALL_Z        = 0.12
_TILT_Z        = 0.7
_FRAME_SKIP    = 5
_ACTION_REPEAT = 4
_MAX_STEPS     = 1000

# Pesos de recompensa — copia exacta de walking_env.py
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

# qpos: [r1_freejoint(7) | r1_joints(16) | r2_freejoint(7) | r2_joints(16)] = 46
# qvel: [r1_freejoint(6) | r1_joints(16) | r2_freejoint(6) | r2_joints(16)] = 44
_LEG_IDX     = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
_ARM_IDX     = np.array([5, 6, 7, 13, 14, 15], dtype=int)
_R1_LEG_CTRL = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
_R2_LEG_CTRL = np.array([16, 17, 18, 19, 20, 24, 25, 26, 27, 28], dtype=int)


class FightingEnv(gym.Env):
    """Dos robots Alpha en la misma escena, cada uno aprende a andar hacia +X.
    Cuando un robot cae se resetea INDIVIDUALMENTE — el otro sigue sin interrupciones.
    El episodio solo termina por truncación (max_steps).
    step() devuelve info["r1_fell"] / info["r2_fell"] para que el bucle de
    entrenamiento use done correcto por robot en el replay buffer.
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
        obs_dim = self._get_obs()[0].shape[0]   # 31, igual que walking_env

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

        # Índices absolutos en qpos/qvel para ankle y punta de pie
        # r1 joint block: qpos[7:23],  r2 joint block: qpos[30:46]
        self._r1_lank  = 7  + 3
        self._r1_rank  = 7  + 11
        self._r1_lfeet = 7  + 4
        self._r1_rfeet = 7  + 12
        self._r2_lank  = 30 + 3
        self._r2_rank  = 30 + 11
        self._r2_lfeet = 30 + 4
        self._r2_rfeet = 30 + 12

        # Índices para fijar brazos en neutro
        self._r1_arm_qpos = 7  + _ARM_IDX   # [12,13,14,20,21,22]
        self._r1_arm_qvel = 6  + _ARM_IDX   # [11,12,13,19,20,21]
        self._r2_arm_qpos = 30 + _ARM_IDX   # [35,36,37,43,44,45]
        self._r2_arm_qvel = 28 + _ARM_IDX   # [33,34,35,41,42,43]

        # Posición inicial de cada robot (para reset individual)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._r1_init_qpos = self.data.qpos[:23].copy()   # freejoint + 16 joints
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
        # r1: qpos[2:7] = (z, qw, qx, qy, qz), qvel[0:6] = (vx,vy,vz,wx,wy,wz)
        obs_r1 = np.concatenate([
            qpos[2:7],  qvel[0:6],
            qpos[7:23][_LEG_IDX],  qvel[6:22][_LEG_IDX],
        ]).astype(np.float32)
        # r2: qpos[25:30] = (z, qw, qx, qy, qz), qvel[22:28] = (vx,vy,vz,wx,wy,wz)
        obs_r2 = np.concatenate([
            qpos[25:30], qvel[22:28],
            qpos[30:46][_LEG_IDX], qvel[28:44][_LEG_IDX],
        ]).astype(np.float32)
        return obs_r1, obs_r2

    def _reset_robot(self, robot):
        """Teleporta un robot a su posición inicial con pequeña perturbación aleatoria.
        El otro robot no se toca — continúa desde su estado actual."""
        if robot == 1:
            self.data.qpos[:23]  = self._r1_init_qpos.copy()
            self.data.qvel[:22]  = 0.0
            self.data.qpos[7:23] += self._rng.uniform(-0.05, 0.05, 16)
        else:
            self.data.qpos[23:]   = self._r2_init_qpos.copy()
            self.data.qvel[22:]   = 0.0
            self.data.qpos[30:46] += self._rng.uniform(-0.05, 0.05, 16)

    def _robot_reward(self, qpos, qvel, a, qpos_off, qvel_off,
                      lf_id, rf_id, com_id, lank, rank, lfeet, rfeet):
        """Reward idéntico a walking_env para un robot dado por sus offsets."""
        x_velocity = float(qvel[qvel_off])
        y_velocity = float(qvel[qvel_off + 1])
        yaw_vel    = float(qvel[qvel_off + 5])

        lf_pos = self.data.xpos[lf_id]
        rf_pos = self.data.xpos[rf_id]
        lf_z, lf_x = float(lf_pos[2]), float(lf_pos[0])
        rf_z, rf_x = float(rf_pos[2]), float(rf_pos[0])

        lf_stance = lf_z < _STANCE_Z
        rf_stance = rf_z < _STANCE_Z
        rear_z, front_z = (lf_z, rf_z) if lf_x < rf_x else (rf_z, lf_z)

        qx = float(qpos[qpos_off + 4])   # qx del cuaternión del torso
        qy = float(qpos[qpos_off + 5])
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

        forward_reward     = _FORWARD_WEIGHT * max(0.0, x_velocity)
        upright_reward     = _UPRIGHT_WEIGHT * up_z
        ctrl_cost          = _CTRL_COST_WEIGHT * float(np.sum(np.square(a)))
        lateral_cost       = _LATERAL_COST_WEIGHT * y_velocity ** 2
        yaw_cost           = _YAW_COST_WEIGHT * yaw_vel ** 2
        foot_height_reward = _FOOT_HEIGHT_WEIGHT * max(0.0, rear_z - _STANCE_Z)
        front_lift_penalty = _FRONT_LIFT_PENALTY * max(0.0, front_z - 0.05)
        stance_penalty     = _STANCE_PENALTY if (lf_stance and rf_stance) else 0.0
        slow_penalty       = _SLOW_PENALTY   if x_velocity < 0.02        else 0.0
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
        lf_fw   = _FOOT_FLAT_REAR_WEIGHT  if lf_x < rf_x else _FOOT_FLAT_FRONT_WEIGHT
        rf_fw   = _FOOT_FLAT_REAR_WEIGHT  if rf_x < lf_x else _FOOT_FLAT_FRONT_WEIGHT
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
            "x_velocity": x_velocity,
            "lf_tilt": lf_tilt, "rf_tilt": rf_tilt,
            "lf_z": lf_z,       "rf_z": rf_z,
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
            self.data.qpos[self._r1_arm_qpos] = 0.0
            self.data.qvel[self._r1_arm_qvel] = 0.0
            self.data.qpos[self._r2_arm_qpos] = 0.0
            self.data.qvel[self._r2_arm_qvel] = 0.0
            mujoco.mj_forward(self.model, self.data)

        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        r1_upz  = 1.0 - 2.0 * (qpos[4] ** 2 + qpos[5] ** 2)
        r2_upz  = 1.0 - 2.0 * (qpos[27] ** 2 + qpos[28] ** 2)
        r1_fell = bool(qpos[2]  < _FALL_Z or r1_upz < _TILT_Z)
        r2_fell = bool(qpos[25] < _FALL_Z or r2_upz < _TILT_Z)

        reward_1, info1 = self._robot_reward(
            qpos, qvel, a1, 0, 0,
            self._r1_lf_id, self._r1_rf_id, self._r1_id,
            self._r1_lank, self._r1_rank, self._r1_lfeet, self._r1_rfeet,
        )
        reward_2, info2 = self._robot_reward(
            qpos, qvel, a2, 23, 22,
            self._r2_lf_id, self._r2_rf_id, self._r2_id,
            self._r2_lank, self._r2_rank, self._r2_lfeet, self._r2_rfeet,
        )

        # Reset individual del robot caído — el otro sigue sin interrupciones
        if r1_fell:
            self._reset_robot(1)
        if r2_fell:
            self._reset_robot(2)
        if r1_fell or r2_fell:
            mujoco.mj_forward(self.model, self.data)

        self._step_count += 1
        # El episodio NUNCA termina por caída — solo por truncación
        terminated = False
        truncated  = self._step_count >= _MAX_STEPS

        info = {
            "r1_fell": r1_fell,                   "r2_fell": r2_fell,
            "r1_x_velocity": info1["x_velocity"], "r2_x_velocity": info2["x_velocity"],
            "r1_lf_tilt":    info1["lf_tilt"],    "r1_rf_tilt":    info1["rf_tilt"],
            "r2_lf_tilt":    info2["lf_tilt"],    "r2_rf_tilt":    info2["rf_tilt"],
        }

        if self.render_mode == "human":
            self.render()

        # obs calculado DESPUÉS del reset: cada robot ve su nueva posición inicial
        return self._get_obs(), (reward_1, reward_2), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model)
        self._renderer.update_scene(self.data)
        if self.render_mode == "human":
            import cv2
            frame = self._renderer.render()
            cv2.imshow("Parallel Race", frame[:, :, ::-1])
            cv2.waitKey(1)
        elif self.render_mode == "rgb_array":
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
