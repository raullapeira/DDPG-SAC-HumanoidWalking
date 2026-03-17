"""
Evolution GIF: renders the deterministic SAC policy at each available
checkpoint and stitches them into a single GIF with checkpoint labels.

Usage (from DDPG-SAC-HumanoidWalking/):
    python eval_gif.py
"""
import sys, os, glob, re
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import numpy as np
import torch
import torch.nn as nn
import imageio
from PIL import Image, ImageDraw, ImageFont
import gymnasium as gym
import mujoco

from envs.alpha_env import AlphaEnv

# ── config ────────────────────────────────────────────────────────────────────
SECONDS_PER_CLIP = 2       # how many seconds to record per checkpoint
FPS              = 30
WIDTH, HEIGHT    = 640, 480
CHECKPOINT_DIR   = os.path.join(_HERE, "..", "checkpoints", "sac_alpha")
OUTPUT_GIF       = os.path.join(_HERE, "..", "media", "alpha_evolution.gif")
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mu      = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def act(self, state):
        with torch.no_grad():
            x = self.net(state)
            mu = self.mu(x)
            return torch.tanh(mu) * self.max_action


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    img  = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    # semi-transparent black bar at top
    draw.rectangle([0, 0, WIDTH, 34], fill=(0, 0, 0, 180))
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 6), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def record_clip(actor, env, renderer, cam, n_frames):
    obs, _ = env.reset(seed=42)
    frames = []
    step_interval = max(1, int(round(1.0 / (FPS * env.unwrapped.model.opt.timestep
                                            * env.unwrapped._frame_skip_val()))))
    physics_step = 0

    # We'll step the env and capture at ~FPS rate
    # Each env.step() = _FRAME_SKIP physics steps
    env_steps_per_frame = max(1, int(round(
        1.0 / (FPS * env.unwrapped.model.opt.timestep * 5))))  # FRAME_SKIP=5

    for _ in range(n_frames):
        # advance a few env steps between frames
        for _ in range(env_steps_per_frame):
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            action  = actor.act(state_t).cpu().numpy()[0]
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=42)

        renderer.update_scene(env.unwrapped.data, camera=cam)
        frames.append(renderer.render().copy())

    return frames


def main():
    # find all alpha checkpoints (created by our new run)
    pattern = os.path.join(CHECKPOINT_DIR, "sac2_checkpoint_*.pt")  # CHECKPOINT_DIR = checkpoints/sac_alpha/
    all_ckpts = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1))
    )

    if not all_ckpts:
        print("No checkpoints found. Train first.")
        return

    # filter: only checkpoints that match our new network size
    # (new ones are ~3.7 MB; old Humanoid-v5 ones are ~7.1 MB)
    alpha_ckpts = [p for p in all_ckpts if os.path.getsize(p) < 5_000_000]

    if not alpha_ckpts:
        print("No Alpha checkpoints found yet (all found are Humanoid-v5 size).")
        return

    print(f"Found {len(alpha_ckpts)} Alpha checkpoints:")
    for p in alpha_ckpts:
        step = int(re.search(r"(\d+)\.pt$", p).group(1))
        print(f"  {step:>7,} steps  ({os.path.getsize(p)/1e6:.1f} MB)  {p}")

    # build env + renderer once
    env = gym.wrappers.TimeLimit(AlphaEnv(), max_episode_steps=10_000)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # patch _frame_skip_val onto unwrapped env for convenience
    env.unwrapped._frame_skip_val = lambda: 5

    renderer = mujoco.Renderer(env.unwrapped.model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.20]
    cam.distance  = 0.9
    cam.azimuth   = 150
    cam.elevation = -18

    n_frames = SECONDS_PER_CLIP * FPS   # frames per clip

    all_frames = []

    for ckpt_path in alpha_ckpts:
        step = int(re.search(r"(\d+)\.pt$", ckpt_path).group(1))
        label = f"Step {step:,}"
        print(f"\nRecording {label} ...")

        actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()

        frames = record_clip(actor, env, renderer, cam, n_frames)
        frames = [add_label(f, label) for f in frames]
        all_frames.extend(frames)
        print(f"  {len(frames)} frames captured")

    env.close()

    print(f"\nSaving GIF → {OUTPUT_GIF}  ({len(all_frames)} total frames)")
    imageio.mimsave(OUTPUT_GIF, all_frames, fps=FPS, loop=0)
    print("Done.")


if __name__ == "__main__":
    main()
