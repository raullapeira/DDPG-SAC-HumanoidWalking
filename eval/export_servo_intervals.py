"""
Exports GIFs showing the robot behaviour when servo positions are held for
fixed intervals: 50ms, 100ms and 200ms.

Each control step = 25ms (FRAME_SKIP=5 x 0.005s timestep).
  50ms  = hold each action for 2 control steps  -> fps=20
  100ms = hold each action for 4 control steps  -> fps=10
  200ms = hold each action for 8 control steps  -> fps=5

The policy is queried once per servo interval, then the same action is
replayed for N control steps — exactly what the real robot would do.

Usage:
    python eval/export_servo_intervals.py --ckpt checkpoints/sac_alpha/sac2_checkpoint_750000.pt
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

SECONDS = 8        # duration of each GIF in real-world seconds
WIDTH   = 640
HEIGHT  = 480
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INTERVALS = [
    # (label,  hold_steps, fps,  filename_tag)
    ("50ms",   2,          20,   "50ms"),
    ("100ms",  4,          10,   "100ms"),
    ("200ms",  8,           5,   "200ms"),
]


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
    draw.rectangle([0, 0, WIDTH, 44], fill=(0, 0, 0))
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 6),  text[0], fill=(255, 255, 255), font=font)
    draw.text((10, 26), text[1], fill=(200, 200, 100), font=font)
    return np.array(img)


def run_interval(actor, env, renderer, cam, hold_steps, fps, label, ckpt_step):
    """
    Run one GIF at a given servo interval.
    Policy is queried once every `hold_steps` control steps.
    One rendered frame is produced per servo interval (= real-time playback).
    """
    n_frames = SECONDS * fps   # total frames = real seconds * frames-per-second

    obs, _ = env.reset(seed=0)
    frames = []
    ep = 1
    total_reward = 0.0
    falls = 0

    for fi in range(n_frames):
        # Query policy once per servo interval
        st = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        action = actor.act(st).cpu().numpy()[0]

        # Hold that action for `hold_steps` control steps
        for _ in range(hold_steps):
            obs, r, term, trunc, info = env.step(action)
            total_reward += r
            if term or trunc:
                obs, _ = env.reset(seed=ep)
                ep += 1
                falls += 1
                break   # start next servo interval fresh

        # Render one frame (represents this servo interval)
        renderer.update_scene(env.unwrapped.data, camera=cam)
        frame = renderer.render().copy()

        line1 = (f"Step {ckpt_step:,}  |  servo interval: {label}"
                 f"  |  ep {ep}  |  falls {falls}")
        line2 = (f"x_vel {info['x_velocity']:+.2f} m/s"
                 f"  |  up_z {info['torso_up_z']:.2f}"
                 f"  |  reward {total_reward:.0f}")
        frames.append(add_label(frame, [line1, line2]))

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    ckpt_step = int(os.path.basename(args.ckpt).split("_")[-1].replace(".pt", ""))

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

    for label, hold_steps, fps, tag in INTERVALS:
        print(f"Recording {label} interval ({hold_steps} control steps/frame, fps={fps}) ...")
        env.reset(seed=0)   # same starting state for every interval
        frames = run_interval(actor, env, renderer, cam, hold_steps, fps, label, ckpt_step)

        out = os.path.join(_HERE, "..", "media",
                           f"alpha_step_{ckpt_step:07d}_servo_{tag}.gif")
        imageio.mimsave(out, frames, fps=fps, loop=0)
        print(f"  Saved -> {out}  ({len(frames)} frames)")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
