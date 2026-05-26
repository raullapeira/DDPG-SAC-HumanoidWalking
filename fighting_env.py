import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

_XML_PATH = os.path.join(
    os.path.dirname(__file__), "robot", "configs", "fighting", "alpha_fight.xml"
)

_INIT_Z   = 0.298
_FALL_Z   = 0.12
_TILT_Z   = 0.7
_FRAME_SKIP    = 5
_ACTION_REPEAT = 4
_MAX_STEPS     = 1000

# qpos layout: [r1_freejoint(7) | r1_joints(16) | r2_freejoint(7) | r2_joints(16)]
# qvel layout: [r1_freejoint(6) | r1_joints(16) | r2_freejoint(6) | r2_joints(16)]
_R1_QPOS = 0    # r1 freejoint start in qpos
_R2_QPOS = 23   # r2 freejoint start in qpos  (7 + 16)
_R1_QVEL = 0    # r1 freejoint start in qvel
_R2_QVEL = 22   # r2 freejoint start in qvel  (6 + 16)

# Leg joint indices within each robot's joint block (same for r1 and r2)
_LEG_IDX = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
# Arm joint indices within each robot's joint block
_ARM_IDX = np.array([5, 6, 7, 13, 14, 15], dtype=int)

# r1 ctrl indices in the full 32-actuator array
_R1_LEG_CTRL = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
_R1_ARM_CTRL = np.array([5, 6, 7, 13, 14, 15], dtype=int)
# r2 ctrl indices (offset by 16)
_R2_LEG_CTRL = np.array([16, 17, 18, 19, 20, 24, 25, 26, 27, 28], dtype=int)
_R2_ARM_CTRL = np.array([21, 22, 23, 29, 30, 31], dtype=int)

# qpos indices for ankle-fwd and feet joints, relative to each robot's joint block start
# r1 joint block starts at qpos[7], r2 joint block starts at qpos[30]
_LANK_J  = 3    # Left_Ankle  joint index within robot block
_RANK_J  = 11   # Right_Ankle joint index within robot block
_LFEET_J = 4    # Left_Feet   joint index within robot block
_RFEET_J = 12   # Right_Feet  joint index within robot block

_ALIVE_BONUS                = 1.0
_UPRIGHT_WEIGHT             = 0.3
_FORWARD_TOWARD_OPP_WEIGHT  = 5.0   # igual que _FORWARD_WEIGHT en walking_env
_SLOW_PENALTY               = -2.0  # igual que walking_env: penaliza velocidad casi nula
_WIN_BONUS                  = 50.0
_CTRL_COST                  = 0.01


class FightingEnv(gym.Env):
    """
    Two Alpha humanoid robots in the same MuJoCo scene learning to knock each other down.

    step() accepts a concatenated action vector of shape (20,):
        actions[:10]  -> r1 leg joints (normalised -1..1)
        actions[10:]  -> r2 leg joints (normalised -1..1)

    Returns:
        obs   : tuple (obs_r1, obs_r2)  each shape (34,)
        reward: tuple (r1, r2)
        terminated: bool  (one or both robots fell)
        truncated:  bool  (max steps reached)
        info: dict
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, xml_path=None):
        self.render_mode = render_mode

        xml_path = os.path.abspath(xml_path if xml_path is not None else _XML_PATH)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # ctrl ranges for r1 leg actuators (same physical values for r2, just offset)
        self._r1_ctrl_low  = self.model.actuator_ctrlrange[_R1_LEG_CTRL, 0].copy()
        self._r1_ctrl_high = self.model.actuator_ctrlrange[_R1_LEG_CTRL, 1].copy()

        n_leg = len(_LEG_IDX)   # 10 per robot
        obs_dim = self._get_obs()[0].shape[0]  # 34

        # Each agent acts on its own 10 leg joints, normalised -1..1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2 * n_leg,), dtype=np.float32
        )
        obs_limit = np.full(obs_dim, np.inf, dtype=np.float32)
        # observation_space describes a single robot's obs (training loop uses both)
        self.observation_space = spaces.Box(
            low=-obs_limit, high=obs_limit, dtype=np.float32
        )

        # Body IDs
        _bid = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self._r1_id    = _bid("r1_root")
        self._r2_id    = _bid("r2_root")
        self._r1_lf_id = _bid("r1_Left_Feet_link")
        self._r1_rf_id = _bid("r1_Right_Feet_link")
        self._r2_lf_id = _bid("r2_Left_Feet_link")
        self._r2_rf_id = _bid("r2_Right_Feet_link")

        # qpos absolute indices for ankle/feet of each robot
        r1_jblock = _R1_QPOS + 7   # = 7
        r2_jblock = _R2_QPOS + 7   # = 30
        self._r1_lank  = r1_jblock + _LANK_J
        self._r1_rank  = r1_jblock + _RANK_J
        self._r1_lfeet = r1_jblock + _LFEET_J
        self._r1_rfeet = r1_jblock + _RFEET_J
        self._r2_lank  = r2_jblock + _LANK_J
        self._r2_rank  = r2_jblock + _RANK_J
        self._r2_lfeet = r2_jblock + _LFEET_J
        self._r2_rfeet = r2_jblock + _RFEET_J

        # arm qpos/qvel indices to lock each step
        self._r1_arm_qpos = r1_jblock + _ARM_IDX
        self._r1_arm_qvel = _R1_QVEL + 6 + _ARM_IDX
        self._r2_arm_qpos = r2_jblock + _ARM_IDX
        self._r2_arm_qvel = _R2_QVEL + 6 + _ARM_IDX

        self._renderer  = None
        self._step_count = 0
        self._prev_dist  = 1.0   # initial separation between robots

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _up_z(qw, qx, qy, qz):
        return 1.0 - 2.0 * (qx * qx + qy * qy)

    def _denorm(self, action):
        half = (self._r1_ctrl_high - self._r1_ctrl_low) / 2.0
        return np.clip(action * half, self._r1_ctrl_low, self._r1_ctrl_high)

    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        # r1 root position and orientation
        r1_x, r1_y, r1_z = qpos[0], qpos[1], qpos[2]
        # r2 root position
        r2_x, r2_y, r2_z = qpos[23], qpos[24], qpos[25]

        # Each robot obs: [z, qw,qx,qy,qz](5) + [vx,vy,vz,wx,wy,wz](6)
        #                 + leg_qpos(10) + leg_qvel(10) + [dx,dy,dz to opponent](3) = 34
        r1_own = np.concatenate([
            qpos[2:7],                          # r1: z + quat
            qvel[0:6],                           # r1: lin+ang vel
            qpos[7:23][_LEG_IDX],               # r1: leg joint angles
            qvel[6:22][_LEG_IDX],               # r1: leg joint velocities
            [r2_x - r1_x, r2_y - r1_y, r2_z - r1_z],  # vector to opponent
        ])
        r2_own = np.concatenate([
            qpos[25:30],                         # r2: z + quat
            qvel[22:28],                          # r2: lin+ang vel
            qpos[30:46][_LEG_IDX],              # r2: leg joint angles
            qvel[28:44][_LEG_IDX],              # r2: leg joint velocities
            [r1_x - r2_x, r1_y - r2_y, r1_z - r2_z],  # vector to opponent
        ])
        return r1_own.astype(np.float32), r2_own.astype(np.float32)

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        rng = np.random.default_rng(seed)
        n_joints = self.model.nq - 14   # total joints minus two freejoints (7 each)
        # small random perturbation to all joint angles
        self.data.qpos[7:23]  += rng.uniform(-0.05, 0.05, 16)
        self.data.qpos[30:46] += rng.uniform(-0.05, 0.05, 16)

        self._step_count = 0
        self._prev_dist  = float(np.linalg.norm(
            self.data.qpos[0:3] - self.data.qpos[23:26]
        ))

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        return obs, {}

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
            # lock arms for both robots
            self.data.qpos[self._r1_arm_qpos] = 0.0
            self.data.qvel[self._r1_arm_qvel] = 0.0
            self.data.qpos[self._r2_arm_qpos] = 0.0
            self.data.qvel[self._r2_arm_qvel] = 0.0
            mujoco.mj_forward(self.model, self.data)

        qpos = self.data.qpos.flat.copy()

        r1_z  = float(qpos[2])
        r1_qw, r1_qx, r1_qy, r1_qz = qpos[3], qpos[4], qpos[5], qpos[6]
        r1_upz = self._up_z(r1_qw, r1_qx, r1_qy, r1_qz)
        r1_fell = bool(r1_z < _FALL_Z or r1_upz < _TILT_Z)

        r2_z  = float(qpos[25])
        r2_qw, r2_qx, r2_qy, r2_qz = qpos[26], qpos[27], qpos[28], qpos[29]
        r2_upz = self._up_z(r2_qw, r2_qx, r2_qy, r2_qz)
        r2_fell = bool(r2_z < _FALL_Z or r2_upz < _TILT_Z)

        terminated = r1_fell or r2_fell

        # ── posiciones y velocidades de torso ────────────────────────────────
        r1_pos = self.data.xpos[self._r1_id]
        r2_pos = self.data.xpos[self._r2_id]
        curr_dist = float(np.linalg.norm(r1_pos - r2_pos))
        self._prev_dist = curr_dist

        # Velocidad de aproximación al oponente: componente de qvel en la dirección al rival.
        # Espejo exacto de x_velocity en walking_env, pero orientado hacia el oponente.
        r1_vel_xy = np.array([self.data.qvel[0],  self.data.qvel[1]])
        r2_vel_xy = np.array([self.data.qvel[22], self.data.qvel[23]])
        dir_12 = np.array([r2_pos[0] - r1_pos[0], r2_pos[1] - r1_pos[1]])
        dist_2d = float(np.linalg.norm(dir_12))
        if dist_2d > 0.01:
            unit_12 = dir_12 / dist_2d
            approach_vel_r1 = float(np.dot(r1_vel_xy,  unit_12))
            approach_vel_r2 = float(np.dot(r2_vel_xy, -unit_12))   # r2 va en sentido opuesto
        else:
            approach_vel_r1 = approach_vel_r2 = 0.0

        # ── rewards (misma estructura que walking_env) ────────────────────────
        def agent_reward(fell, opp_fell, upz, a, approach_vel):
            r  = _ALIVE_BONUS
            r += _UPRIGHT_WEIGHT * upz
            # forward_toward_opp: equivalente a forward_reward de walking, solo premia avanzar
            r += _FORWARD_TOWARD_OPP_WEIGHT * max(0.0, approach_vel)
            # slow_penalty: copia exacta de walking_env — obliga a moverse
            r += _SLOW_PENALTY if approach_vel < 0.02 else 0.0
            r -= _CTRL_COST * np.sum(a ** 2)
            if opp_fell and not fell:
                r += _WIN_BONUS
            elif fell and not opp_fell:
                r -= _WIN_BONUS
            return float(r)

        reward_1 = agent_reward(r1_fell, r2_fell, r1_upz, a1, approach_vel_r1)
        reward_2 = agent_reward(r2_fell, r1_fell, r2_upz, a2, approach_vel_r2)

        self._step_count += 1
        truncated = (self._step_count >= _MAX_STEPS)

        obs = self._get_obs()

        info = {
            "r1_fell": r1_fell,       "r2_fell": r2_fell,
            "dist": curr_dist,
            "r1_upz": r1_upz,         "r2_upz": r2_upz,
            "approach_vel_r1": approach_vel_r1,
            "approach_vel_r2": approach_vel_r2,
        }

        if self.render_mode == "human":
            self.render()

        return obs, (reward_1, reward_2), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model)
        self._renderer.update_scene(self.data)
        if self.render_mode == "human":
            import cv2
            frame = self._renderer.render()
            cv2.imshow("Fighting", frame[:, :, ::-1])
            cv2.waitKey(1)
        elif self.render_mode == "rgb_array":
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
