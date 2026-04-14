import sys, os, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import torch, torch.nn as nn, gymnasium as gym
from envs.alpha_env import AlphaEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, s, a, m):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
        self.mu = nn.Linear(256, a)
        self.log_std = nn.Linear(256, a)
        self.max_action = m
    def act(self, s):
        with torch.no_grad():
            return torch.tanh(self.mu(self.net(s))) * self.max_action

env = gym.wrappers.TimeLimit(AlphaEnv(), max_episode_steps=1000)
s = env.observation_space.shape[0]
a = env.action_space.shape[0]
m = float(env.action_space.high[0])

actor = Actor(s, a, m).to(device)
ckpt = torch.load(os.path.join(_HERE, "..", "checkpoints", "sac_alpha", "sac2_checkpoint_1000000.pt"), map_location=device, weights_only=False)
actor.load_state_dict(ckpt["actor"])
actor.eval()

rewards, xvels, lengths = [], [], []
for ep in range(10):
    obs, _ = env.reset(seed=ep)
    ep_r, ep_xv, steps = 0, 0, 0
    done = False
    while not done:
        st = torch.FloatTensor(obs).unsqueeze(0).to(device)
        action = actor.act(st).cpu().numpy()[0]
        obs, r, term, trunc, info = env.step(action)
        ep_r += r
        ep_xv += info.get("x_velocity", 0)
        steps += 1
        done = term or trunc
    rewards.append(ep_r)
    xvels.append(ep_xv / max(steps, 1))
    lengths.append(steps)
    print(f"  Ep {ep+1:2d}: reward={ep_r:7.1f}  avg_x_vel={ep_xv/max(steps,1):+.3f} m/s  steps={steps}", flush=True)

print(f"\nSUMMARY  avg_reward={np.mean(rewards):.1f}  avg_x_vel={np.mean(xvels):+.3f} m/s  avg_ep_len={np.mean(lengths):.0f}")
env.close()
