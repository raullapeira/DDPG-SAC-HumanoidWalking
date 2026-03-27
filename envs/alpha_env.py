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
_ACTION_REPEAT = 4   # 🔥 antes 8 → ahora 100ms

_CTRL_COST_WEIGHT   = 0.01
_FORWARD_WEIGHT     = 5.0
_ALIVE_BONUS        = 0.8
_UPRIGHT_WEIGHT     = 0.3
_LATERAL_COST_WEIGHT = 0.8
_YAW_COST_WEIGHT     = 1.0

_FOOT_HEIGHT_WEIGHT  = 3.0
_PUSH_OFF_WEIGHT     = 2.0   # 🔥 nuevo
_FRONT_LIFT_PENALTY  = 1.0   # 🔥 nuevo
_STANCE_PENALTY      = -0.3  # 🔥 suavizado


class AlphaEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, xml_path=None):
        self.render_mode = render_mode

        xml_path = os.path.abspath(xml_path if xml_path is not None else _XML_PATH)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self._ctrl_low  = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self._ctrl_half = (self._ctrl_high - self._ctrl_low) / 2.0

        n_act = self.model.nu
        n_obs = self._get_obs().shape[0]

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32
        )
        obs_limit = np.full(n_obs, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_limit, high=obs_limit, dtype=np.float32
        )

        self._prev_action = np.zeros(n_act, dtype=np.float32)
        self._renderer = None

        self._left_leg_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Left_Leg_Link")
        self._right_leg_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Right_Leg_link")

    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        return np.concatenate([qpos[2:], qvel]).astype(np.float32)

    def _denorm_action(self, action):
        return np.clip(action * self._ctrl_half, self._ctrl_low, self._ctrl_high)

    def _torso_up_z(self):
        qw, qx, qy, qz = self.data.qpos[3:7]
        return 1.0 - 2.0 * (qx * qx + qy * qy)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        rng = np.random.default_rng(seed)
        n_joints = self.model.nq - 7
        self.data.qpos[7:] += rng.uniform(-0.05, 0.05, n_joints)

        self._prev_action = np.zeros(self.model.nu, dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        for _ in range(_ACTION_REPEAT):
            self.data.ctrl[:] = self._denorm_action(action)
            for _ in range(_FRAME_SKIP):
                mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        root_z = self.data.qpos[2]
        up_z   = self._torso_up_z()

        terminated = bool(root_z < _FALL_Z or up_z < _TILT_Z)

        x_velocity = self.data.qvel[0]
        y_velocity = self.data.qvel[1]

        # =========================
        # 🦶 DETECCIÓN DE PIERNAS
        # =========================
        left_pos  = self.data.xpos[self._left_leg_id]
        right_pos = self.data.xpos[self._right_leg_id]

        left_x, left_z   = left_pos[0], left_pos[2]
        right_x, right_z = right_pos[0], right_pos[2]

        if left_x < right_x:
            rear_z  = left_z
            front_z = right_z
            rear_vx = self.data.cvel[self._left_leg_id][3]
        else:
            rear_z  = right_z
            front_z = left_z
            rear_vx = self.data.cvel[self._right_leg_id][3]

        # =========================
        # 🎯 REWARDS
        # =========================
        forward_reward = _FORWARD_WEIGHT * max(0.0, x_velocity)
        alive_bonus    = _ALIVE_BONUS
        upright_reward = _UPRIGHT_WEIGHT * up_z

        ctrl_cost    = _CTRL_COST_WEIGHT * np.sum(np.square(action))
        lateral_cost = _LATERAL_COST_WEIGHT * y_velocity ** 2
        yaw_cost     = _YAW_COST_WEIGHT * self.data.qvel[5] ** 2

        # 🔥 solo pierna trasera
        foot_height_reward = _FOOT_HEIGHT_WEIGHT * max(0.0, rear_z - 0.06)

        # 🔥 push-off real
        push_off_reward = _PUSH_OFF_WEIGHT * max(0.0, -rear_vx)

        # 🔥 penaliza levantar la delantera
        front_lift_penalty = _FRONT_LIFT_PENALTY * max(0.0, front_z - 0.08)

        # 🔥 stance suave
        double_support = (left_z < 0.06 and right_z < 0.06)
        stance_penalty = _STANCE_PENALTY if double_support else 0.0
        slow_penalty   = -2.0 if x_velocity < 0.02 else 0.0

        reward = (
            forward_reward
            + alive_bonus
            + upright_reward
            + foot_height_reward
            + push_off_reward
            - ctrl_cost
            - lateral_cost
            - yaw_cost
            - front_lift_penalty
            + stance_penalty
            + slow_penalty
        )

        self._prev_action = action.copy()

        truncated = False

        info = {
            "x_velocity": x_velocity,
            "rear_z": rear_z,
            "front_z": front_z,
            "push_off": push_off_reward,
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