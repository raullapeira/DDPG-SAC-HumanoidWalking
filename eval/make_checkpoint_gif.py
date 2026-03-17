"""
Generates a short GIF for a single checkpoint. Called automatically by the training
script after each checkpoint save, or manually:
    python eval/make_checkpoint_gif.py --ckpt checkpoints/sac_alpha/sac2_checkpoint_50000.pt --step 50000
"""
import sys, os, argparse
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

SECONDS = 10
WIDTH   = 640
HEIGHT  = 480
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def add_label(frame, text):
    img  = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, WIDTH, 34], fill=(0, 0, 0))
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 6), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  required=True)
    parser.add_argument("--step",  type=int, required=True)
    args = parser.parse_args()

    out_gif = os.path.join(_HERE, "..", "media", f"alpha_step_{args.step:07d}.gif")

    env = gym.wrappers.TimeLimit(AlphaEnv(), max_episode_steps=10_000)
    s = env.observation_space.shape[0]
    a = env.action_space.shape[0]
    m = float(env.action_space.high[0])

    actor = Actor(s, a, m).to(DEVICE)
    ckpt  = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    renderer = mujoco.Renderer(env.unwrapped.model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.20]
    cam.distance  = 0.9
    cam.azimuth   = 150
    cam.elevation = -18

    from envs.alpha_env import _FRAME_SKIP, _ACTION_REPEAT
    step_ms = _FRAME_SKIP * _ACTION_REPEAT * 5   # ms per policy step (5ms timestep)
    gif_fps = round(1000 / step_ms)              # real-time fps: 1000/200 = 5fps
    n_frames = SECONDS * gif_fps                 # 1 frame per policy step

    obs, _ = env.reset(seed=0)
    frames = []
    ep = 1
    move = 0  # movement counter (each = one 200ms policy step)

    for _ in range(n_frames):
        st = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        action = actor.act(st).cpu().numpy()[0]
        obs, r, term, trunc, info = env.step(action)
        move += 1
        if term or trunc:
            obs, _ = env.reset(seed=ep)
            ep += 1
            move = 0

        renderer.update_scene(env.unwrapped.data, camera=cam)
        frame = renderer.render().copy()
        label = (f"Train {args.step:,}  |  ep {ep}  |  move {move} ({move*step_ms}ms)"
                 f"  |  x_vel {info['x_velocity']:+.2f}  |  up {info['torso_up_z']:.2f}")
        frames.append(add_label(frame, label))

    env.close()
    imageio.mimsave(out_gif, frames, fps=gif_fps, loop=0)
    print(f"GIF saved -> {out_gif}", flush=True)


if __name__ == "__main__":
    main()
