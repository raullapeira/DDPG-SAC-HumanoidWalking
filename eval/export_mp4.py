"""
Exports an MP4 of the robot running a trained checkpoint at real-time speed.
Each frame = one 200ms policy step.

Usage:
    python eval/export_mp4.py --ckpt checkpoints/sac_alpha/sac2_checkpoint_500000.pt --step 500000
"""
import sys, os, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import numpy as np
import torch
import torch.nn as nn
import cv2
import gymnasium as gym
import mujoco
from envs.alpha_env import AlphaEnv, _FRAME_SKIP, _ACTION_REPEAT

SECONDS    = 120
WIDTH      = 640
HEIGHT     = 480
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Render at every control step (25ms) for smooth transitions
CTRL_MS    = _FRAME_SKIP * 5                 # 25ms per control step
POLICY_MS  = _ACTION_REPEAT * CTRL_MS        # 200ms per policy step
FPS        = round(1000 / CTRL_MS)           # 40fps — smooth real-time playback


class Actor(nn.Module):
    def __init__(self, s, a, m):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
        self.mu  = nn.Linear(256, a)
        self.log_std = nn.Linear(256, a)
        self.max_action = m
    def act(self, state):
        with torch.no_grad():
            return torch.tanh(self.mu(self.net(state))) * self.max_action


def draw_overlay(frame_bgr, move, ep, x_vel, up_z, joint_angles, joint_names):
    """Draw HUD: frame counter, velocities, and joint angle bars."""
    h, w = frame_bgr.shape[:2]

    # Top bar
    cv2.rectangle(frame_bgr, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame_bgr,
        f"Step 500k  |  ep {ep}  |  move {move} ({move*POLICY_MS}ms)"
        f"  |  x_vel {x_vel:+.2f} m/s  |  up {up_z:.2f}",
        (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # Bottom panel: joint angle bars
    panel_h = 120
    panel_top = h - panel_h
    cv2.rectangle(frame_bgr, (0, panel_top), (w, h), (20, 20, 20), -1)

    n = len(joint_angles)
    bar_w = w // n
    max_angle_deg = 90.0

    for i, (angle, name) in enumerate(zip(joint_angles, joint_names)):
        angle_deg = np.degrees(angle)
        x0 = i * bar_w + 2
        x1 = (i + 1) * bar_w - 2
        mid_y = panel_top + panel_h // 2

        # Zero line
        cv2.line(frame_bgr, (x0, mid_y), (x1, mid_y), (80, 80, 80), 1)

        # Bar fill
        norm = np.clip(angle_deg / max_angle_deg, -1.0, 1.0)
        bar_h = int(abs(norm) * (panel_h // 2 - 15))
        color = (0, 200, 100) if norm >= 0 else (0, 100, 200)
        if norm >= 0:
            cv2.rectangle(frame_bgr, (x0+2, mid_y - bar_h), (x1-2, mid_y), color, -1)
        else:
            cv2.rectangle(frame_bgr, (x0+2, mid_y), (x1-2, mid_y + bar_h), color, -1)

        # Joint name (short)
        short = name[-4:] if len(name) > 4 else name
        cv2.putText(frame_bgr, short, (x0+1, h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)

    return frame_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    out_mp4 = os.path.join(_HERE, "..", "media", f"alpha_step_{args.step:07d}.mp4")
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)

    env = gym.wrappers.TimeLimit(AlphaEnv(), max_episode_steps=10_000)
    s = env.observation_space.shape[0]
    a = env.action_space.shape[0]
    m = float(env.action_space.high[0])

    actor = Actor(s, a, m).to(DEVICE)
    ckpt  = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    # Joint names from model
    raw_env = env.unwrapped
    joint_names = [mujoco.mj_id2name(raw_env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                   or f"j{i}" for i in range(raw_env.model.nu)]

    renderer = mujoco.Renderer(raw_env.model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.20]
    cam.distance  = 1.1
    cam.azimuth   = 150
    cam.elevation = -18

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, FPS, (WIDTH, HEIGHT))

    n_policy_steps = SECONDS * round(1000 / POLICY_MS)   # policy decisions in SECONDS
    obs, _ = env.reset(seed=0)
    raw_env_data = raw_env.data
    ep = 1
    move = 0
    total_frames = 0

    print(f"Rendering at {FPS}fps ({CTRL_MS}ms/frame, {POLICY_MS}ms/policy step)...")

    for _ in range(n_policy_steps):
        # Ask policy for action once
        st = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        action = actor.act(st).cpu().numpy()[0]
        ctrl = raw_env.unwrapped._denorm_action(action) if hasattr(raw_env, 'unwrapped') \
               else raw_env._denorm_action(action)

        term = trunc = False
        info = {}

        # Step through each control step, rendering each one (smooth transitions)
        for ctrl_i in range(_ACTION_REPEAT):
            raw_env_data.ctrl[:] = raw_env._denorm_action(action)
            for _ in range(_FRAME_SKIP):
                mujoco.mj_step(raw_env.model, raw_env_data)

            cam.lookat[0] = float(raw_env_data.qpos[0])  # follow robot in X
            cam.lookat[1] = float(raw_env_data.qpos[1])  # follow robot in Y
            renderer.update_scene(raw_env_data, camera=cam)
            frame_rgb = renderer.render().copy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            x_vel  = float(raw_env_data.qvel[0])
            up_z   = raw_env._torso_up_z()
            joint_angles = raw_env_data.qpos[7:].copy()

            draw_overlay(frame_bgr, move, ep, x_vel, up_z, joint_angles, joint_names)
            writer.write(frame_bgr)
            total_frames += 1

            # Check termination mid-step
            if raw_env_data.qpos[2] < 0.12 or up_z < 0.7:
                term = True
                break

        move += 1
        obs = raw_env._get_obs()

        if term or trunc:
            obs, _ = env.reset(seed=ep)
            ep += 1
            move = 0

        if total_frames % 40 == 0:
            print(f"  frame {total_frames}  ep {ep}  move {move}"
                  f"  x_vel={float(raw_env_data.qvel[0]):+.2f}"
                  f"  up={raw_env._torso_up_z():.2f}")

    writer.release()
    env.close()
    print(f"\nMP4 saved -> {out_mp4}")


if __name__ == "__main__":
    main()
