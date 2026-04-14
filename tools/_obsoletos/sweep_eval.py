"""
Evaluates all Alpha checkpoints (5 episodes each) and reports reward + x-velocity.
"""
import sys, os, re, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from envs.alpha_env import AlphaEnv

EVAL_EPISODES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_CKPT_DIR = os.path.join(_HERE, "..", "checkpoints", "sac_alpha")


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


ckpts = sorted(
    [p for p in glob.glob(os.path.join(_CKPT_DIR, "sac2_checkpoint_*.pt"))
     if os.path.getsize(p) < 5_000_000 and re.search(r"_(\d+)\.pt$", p)],
    key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1))
)

if not ckpts:
    print("No Alpha checkpoints found.")
    sys.exit(1)

env = gym.wrappers.TimeLimit(AlphaEnv(), max_episode_steps=1000)
s = env.observation_space.shape[0]
a = env.action_space.shape[0]
m = float(env.action_space.high[0])

print(f"{'Step':>12}  {'Avg Reward':>12}  {'Avg x-vel':>10}  {'Avg Ep Len':>10}")
print("-" * 52)

results = []
for ckpt_path in ckpts:
    step = int(re.search(r"(\d+)\.pt$", ckpt_path).group(1))
    actor = Actor(s, a, m).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    rewards, xvels, lengths = [], [], []
    for ep in range(EVAL_EPISODES):
        obs, _ = env.reset(seed=ep)
        ep_r, ep_xv, steps = 0, 0, 0
        done = False
        while not done:
            st = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            action = actor.act(st).cpu().numpy()[0]
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            ep_xv += info.get("x_velocity", 0)
            steps += 1
            done = term or trunc
        rewards.append(ep_r)
        xvels.append(ep_xv / max(steps, 1))
        lengths.append(steps)

    avg_r  = np.mean(rewards)
    avg_xv = np.mean(xvels)
    avg_l  = np.mean(lengths)
    results.append((step, avg_r, avg_xv, avg_l))
    print(f"{step:>12,}  {avg_r:>12.1f}  {avg_xv:>+10.3f}  {avg_l:>10.0f}", flush=True)

env.close()

best = max(results, key=lambda x: x[1])
print("-" * 52)
print(f"\nBest checkpoint: step {best[0]:,}  →  avg reward {best[1]:.1f}  |  x-vel {best[2]:+.3f} m/s")
