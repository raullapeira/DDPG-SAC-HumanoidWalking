import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

_XML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "robot", "alpha_single.xml"
)

_INIT_Z = 0.298
_FALL_Z = 0.12       # terminate if root drops below this height
_TILT_Z = 0.7        # terminate if torso tilts > ~45° from vertical (cos threshold)

_FRAME_SKIP    = 5   # physics steps per control step: 5 * 0.005s = 25ms
_ACTION_REPEAT = 8   # control steps per policy step:  8 * 25ms   = 200ms

_CTRL_COST_WEIGHT   = 0.01
_SMOOTH_WEIGHT      = 0.5   # penalty for large changes between consecutive actions
_FORWARD_WEIGHT     = 3.0
_ALIVE_BONUS        = 1.0
_UPRIGHT_WEIGHT     = 3.0
_FALL_PENALTY       = -50.0


class AlphaEnv(gym.Env):
    """
    Single-robot walking environment for the Alpha humanoid.

    Observation (43-dim):
        qpos[2:]  - root z-height, quaternion (4), 16 joint angles  = 21
        qvel[:]   - 6 root velocities + 16 joint velocities          = 22

    Action (16-dim):
        Normalised target joint positions in [-1, 1], centered on the
        neutral standing pose (all joints at 0), scaled by half the ctrlrange.
        action=0 maps to joint angle 0 (natural standing).

    Policy step = ACTION_REPEAT * FRAME_SKIP * timestep
                = 8 * 5 * 0.005s = 200ms
    Each policy decision maps directly to one servo command on the real robot
    with 200ms transition duration.

    Reward:
        forward_vel   - root x-velocity (capped at 0.5 m/s, coupled with up_z)
        alive_bonus   - constant bonus each step the robot stays up
        upright       - bonus for keeping torso vertical
        ctrl_cost     - penalty for large absolute actions
        smooth_cost   - penalty for large changes between consecutive actions
        slow_penalty  - penalty for not moving forward
        fall_penalty  - large one-time penalty on fall/tilt termination
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        xml_path = os.path.abspath(_XML_PATH)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self._ctrl_low  = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self._ctrl_half = (self._ctrl_high - self._ctrl_low) / 2.0

        n_act = self.model.nu   # 16
        n_obs = self._get_obs().shape[0]  # 43

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32
        )
        obs_limit = np.full(n_obs, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_limit, high=obs_limit, dtype=np.float32
        )

        self._prev_action = np.zeros(n_act, dtype=np.float32)
        self._renderer = None

    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        return np.concatenate([qpos[2:], qvel]).astype(np.float32)

    def _denorm_action(self, action):
        """Map normalised action [-1,1] centered on neutral pose (q=0)."""
        return np.clip(action * self._ctrl_half, self._ctrl_low, self._ctrl_high)

    def _torso_up_z(self):
        """Z-component of the torso up-vector in world frame from quaternion."""
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

        # Hold this action for ACTION_REPEAT control steps (= 200ms total)
        # Each control step = FRAME_SKIP physics steps (= 25ms)
        for _ in range(_ACTION_REPEAT):
            self.data.ctrl[:] = self._denorm_action(action)
            for _ in range(_FRAME_SKIP):
                mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        root_z = self.data.qpos[2]
        up_z   = self._torso_up_z()

        terminated = bool(root_z < _FALL_Z or up_z < _TILT_Z)

        x_velocity     = self.data.qvel[0]
        v_clipped      = np.clip(x_velocity, 0.0, 0.5)
        forward_reward = _FORWARD_WEIGHT * v_clipped * up_z
        alive_bonus    = _ALIVE_BONUS
        upright_reward = _UPRIGHT_WEIGHT * up_z
        ctrl_cost      = _CTRL_COST_WEIGHT * np.sum(np.square(action))
        smooth_cost    = _SMOOTH_WEIGHT * np.sum(np.square(action - self._prev_action))
        fall_penalty   = _FALL_PENALTY if terminated else 0.0
        slow_penalty   = -1.0 if x_velocity < 0.02 else 0.0

        reward = (forward_reward + alive_bonus + upright_reward
                  - ctrl_cost - smooth_cost + fall_penalty + slow_penalty)

        self._prev_action = action.copy()

        truncated = False

        info = {
            "x_velocity":      x_velocity,
            "root_z":          root_z,
            "torso_up_z":      up_z,
            "reward_forward":  forward_reward,
            "reward_survive":  alive_bonus,
            "reward_upright":  upright_reward,
            "reward_ctrl":     -ctrl_cost,
            "reward_smooth":   -smooth_cost,
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
