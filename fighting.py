import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

_CKPT_R1   = os.path.join(_HERE, "checkpoints", "versus", "r1")
_CKPT_R2   = os.path.join(_HERE, "checkpoints", "versus", "r2")
os.makedirs(_CKPT_R1, exist_ok=True)
os.makedirs(_CKPT_R2, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import csv
import datetime
import subprocess
from torch.utils.tensorboard import SummaryWriter
from versus_env import VersusEnv

_CSV_PATH  = os.path.join(_HERE, "training_log_versus.csv")
_TODAY     = datetime.date.today().strftime("%d_%m_%Y")
_MEDIA_DIR = os.path.join(_HERE, "media", f"{_TODAY}_fighting_versus")
os.makedirs(_MEDIA_DIR, exist_ok=True)

_GIF_SCRIPT = os.path.join(_HERE, "tools", "simu_a_real", "make_versus_gif.py")

if not os.path.exists(_CSV_PATH):
    with open(_CSV_PATH, mode="w", newline="") as f:
        csv.writer(f).writerow(["step", "episode", "reward_r1", "reward_r2",
                                 "falls_r1", "falls_r2"])

# ── Hyperparameters ────────────────────────────────────────────────────────────
LEARNING_RATE   = 3e-4
GAMMA           = 0.99
TAU             = 0.005
BUFFER_SIZE     = int(1e6)
BATCH_SIZE      = 256
LEARNING_STARTS = 1000
TOTAL_TIMESTEPS = 3_000_000
SAVE_INTERVAL   = 50_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter("runs/versus")


# ── Redes (misma arquitectura que walking.py) ─────────────────────────────────

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

    def forward(self, state):
        x = self.net(state)
        return self.mu(x), torch.clamp(self.log_std(x), -20, 2)

    def sample(self, state):
        mu, log_std = self.forward(state)
        std    = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x_t    = normal.rsample()
        y_t    = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t).sum(1, keepdim=True)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        sa = state_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(sa, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(sa, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)


class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(device),
            torch.FloatTensor(np.array(a)).to(device),
            torch.FloatTensor(np.array(r)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(ns)).to(device),
            torch.FloatTensor(np.array(d)).unsqueeze(1).to(device),
        )

    def size(self):
        return len(self.buffer)


def sac_update(actor, critic, critic_target, actor_opt, critic_opt, alpha_opt,
               log_alpha, target_entropy, buffer):
    if buffer.size() <= LEARNING_STARTS:
        return None

    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

    new_actions, log_pi = actor.sample(states)
    q1n, q2n = critic(states, new_actions)

    alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
    alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
    alpha = log_alpha.exp()

    actor_loss = (alpha * log_pi - torch.min(q1n, q2n)).mean()
    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

    with torch.no_grad():
        na, nlog = actor.sample(next_states)
        q1t, q2t = critic_target(next_states, na)
        q_target = rewards + (1 - dones) * GAMMA * (torch.min(q1t, q2t) - alpha * nlog)

    q1, q2 = critic(states, actions)
    critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)
    critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

    for p, tp in zip(critic.parameters(), critic_target.parameters()):
        tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

    return actor_loss.item(), critic_loss.item(), alpha.item()


def save_checkpoint(ckpt_dir, step, actor, critic, critic_target, log_alpha,
                    actor_opt, critic_opt, alpha_opt):
    torch.save({
        "actor": actor.state_dict(),            "critic": critic.state_dict(),
        "critic_target": critic_target.state_dict(),
        "log_alpha": log_alpha,
        "actor_opt": actor_opt.state_dict(),    "critic_opt": critic_opt.state_dict(),
        "alpha_opt": alpha_opt.state_dict(),
    }, os.path.join(ckpt_dir, f"ckpt_{step}.pt"))


def load_latest(ckpt_dir, actor, critic, critic_target, log_alpha,
                actor_opt, critic_opt, alpha_opt):
    files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
        key=lambda f: int(f.split("_")[-1].replace(".pt", ""))
    )
    if not files:
        return 0
    ckpt = torch.load(os.path.join(ckpt_dir, files[-1]),
                      map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    critic_target.load_state_dict(ckpt["critic_target"])
    log_alpha.data.copy_(ckpt["log_alpha"].to(device))
    actor_opt.load_state_dict(ckpt["actor_opt"])
    critic_opt.load_state_dict(ckpt["critic_opt"])
    alpha_opt.load_state_dict(ckpt["alpha_opt"])
    return int(files[-1].split("_")[-1].replace(".pt", ""))


# ── Entorno y agentes ─────────────────────────────────────────────────────────

env = VersusEnv()
obs_dim    = env.observation_space.shape[0]   # 34 (31 walking + 3 vector al oponente)
action_dim = 10
max_action = 1.0

actor_r1  = Actor(obs_dim, action_dim, max_action).to(device)
critic_r1 = Critic(obs_dim, action_dim).to(device)
critic_r1_target = Critic(obs_dim, action_dim).to(device)
critic_r1_target.load_state_dict(critic_r1.state_dict())
actor_r1_opt  = optim.Adam(actor_r1.parameters(),  lr=LEARNING_RATE)
critic_r1_opt = optim.Adam(critic_r1.parameters(), lr=LEARNING_RATE)
log_alpha_r1  = torch.zeros(1, requires_grad=True, device=device)
alpha_r1_opt  = optim.Adam([log_alpha_r1], lr=LEARNING_RATE)
target_ent_r1 = -float(action_dim)
buffer_r1     = ReplayBuffer()

actor_r2  = Actor(obs_dim, action_dim, max_action).to(device)
critic_r2 = Critic(obs_dim, action_dim).to(device)
critic_r2_target = Critic(obs_dim, action_dim).to(device)
critic_r2_target.load_state_dict(critic_r2.state_dict())
actor_r2_opt  = optim.Adam(actor_r2.parameters(),  lr=LEARNING_RATE)
critic_r2_opt = optim.Adam(critic_r2.parameters(), lr=LEARNING_RATE)
log_alpha_r2  = torch.zeros(1, requires_grad=True, device=device)
alpha_r2_opt  = optim.Adam([log_alpha_r2], lr=LEARNING_RATE)
target_ent_r2 = -float(action_dim)
buffer_r2     = ReplayBuffer()

step_r1 = load_latest(_CKPT_R1, actor_r1, critic_r1, critic_r1_target,
                       log_alpha_r1, actor_r1_opt, critic_r1_opt, alpha_r1_opt)
step_r2 = load_latest(_CKPT_R2, actor_r2, critic_r2, critic_r2_target,
                       log_alpha_r2, actor_r2_opt, critic_r2_opt, alpha_r2_opt)
global_step = max(step_r1, step_r2)

if global_step > 0:
    print(f"Resumiendo desde step {global_step}")
else:
    print("Entrenamiento desde cero — robots enfrentados, resets independientes, obs_dim=34")

# ── Bucle de entrenamiento ────────────────────────────────────────────────────

episode = 0

while global_step < TOTAL_TIMESTEPS:
    (obs_r1, obs_r2), _ = env.reset()
    ep_r1 = ep_r2 = 0.0
    falls_r1 = falls_r2 = 0
    done = False

    _ep_lf1, _ep_rf1, _ep_lf2, _ep_rf2 = [], [], [], []
    _ep_app1, _ep_app2   = [], []   # approach velocity — debe ser >0 si anda hacia el rival
    _ep_vx1,  _ep_vx2    = [], []   # vx mundo — diagnóstico de bugs de espejo
    _ep_fwd1, _ep_fwd2   = [], []   # componente forward_reward
    _ep_lat_l1, _ep_lat_r1 = [], []
    _ep_lat_l2, _ep_lat_r2 = [], []
    _ep_dist             = []       # distancia entre robots

    while not done:
        global_step += 1

        t1 = torch.FloatTensor(obs_r1).unsqueeze(0).to(device)
        t2 = torch.FloatTensor(obs_r2).unsqueeze(0).to(device)

        with torch.no_grad():
            a1, _ = actor_r1.sample(t1)
            a2, _ = actor_r2.sample(t2)
        a1 = a1.cpu().numpy()[0]
        a2 = a2.cpu().numpy()[0]

        (next_r1, next_r2), (r1, r2), _terminated, truncated, info = env.step(
            np.concatenate([a1, a2])
        )
        done = truncated

        ep_r1 += r1; ep_r2 += r2
        if info["r1_fell"]: falls_r1 += 1
        if info["r2_fell"]: falls_r2 += 1
        _ep_lf1.append(info["r1_lf_tilt"]); _ep_rf1.append(info["r1_rf_tilt"])
        _ep_lf2.append(info["r2_lf_tilt"]); _ep_rf2.append(info["r2_rf_tilt"])
        _ep_app1.append(info["r1_approach_vel"]); _ep_app2.append(info["r2_approach_vel"])
        _ep_vx1.append(info["r1_x_velocity"]);    _ep_vx2.append(info["r2_x_velocity"])
        _ep_fwd1.append(info["r1_fwd_rew"]);       _ep_fwd2.append(info["r2_fwd_rew"])
        _ep_lat_l1.append(info["r1_lat_L"]);       _ep_lat_r1.append(info["r1_lat_R"])
        _ep_lat_l2.append(info["r2_lat_L"]);       _ep_lat_r2.append(info["r2_lat_R"])
        _ep_dist.append(info["dist"])

        done_r1 = info["r1_fell"] or truncated
        done_r2 = info["r2_fell"] or truncated
        buffer_r1.put((obs_r1, a1, r1, next_r1, float(done_r1)))
        buffer_r2.put((obs_r2, a2, r2, next_r2, float(done_r2)))
        obs_r1, obs_r2 = next_r1, next_r2

        res1 = sac_update(actor_r1, critic_r1, critic_r1_target,
                          actor_r1_opt, critic_r1_opt, alpha_r1_opt,
                          log_alpha_r1, target_ent_r1, buffer_r1)
        res2 = sac_update(actor_r2, critic_r2, critic_r2_target,
                          actor_r2_opt, critic_r2_opt, alpha_r2_opt,
                          log_alpha_r2, target_ent_r2, buffer_r2)

        if res1:
            al1, cl1, alpha1 = res1
            writer.add_scalar("R1/actor_loss",  al1,    global_step)
            writer.add_scalar("R1/critic_loss", cl1,    global_step)
            writer.add_scalar("R1/alpha",       alpha1, global_step)
        if res2:
            al2, cl2, alpha2 = res2
            writer.add_scalar("R2/actor_loss",  al2,    global_step)
            writer.add_scalar("R2/critic_loss", cl2,    global_step)
            writer.add_scalar("R2/alpha",       alpha2, global_step)

        if global_step % SAVE_INTERVAL == 0:
            ckpt_r1_path = os.path.join(_CKPT_R1, f"ckpt_{global_step}.pt")
            ckpt_r2_path = os.path.join(_CKPT_R2, f"ckpt_{global_step}.pt")
            save_checkpoint(_CKPT_R1, global_step,
                            actor_r1, critic_r1, critic_r1_target,
                            log_alpha_r1, actor_r1_opt, critic_r1_opt, alpha_r1_opt)
            save_checkpoint(_CKPT_R2, global_step,
                            actor_r2, critic_r2, critic_r2_target,
                            log_alpha_r2, actor_r2_opt, critic_r2_opt, alpha_r2_opt)
            print(f"Checkpoints guardados en step {global_step}")

            subprocess.Popen(
                [sys.executable, _GIF_SCRIPT,
                 "--ckpt_r1", ckpt_r1_path,
                 "--ckpt_r2", ckpt_r2_path,
                 "--step",    str(global_step),
                 "--out_dir", _MEDIA_DIR],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    def _avg(lst): return sum(lst) / len(lst) if lst else 0.0

    writer.add_scalar("Reward/r1",   ep_r1,    global_step)
    writer.add_scalar("Reward/r2",   ep_r2,    global_step)
    writer.add_scalar("Falls/r1",    falls_r1, global_step)
    writer.add_scalar("Falls/r2",    falls_r2, global_step)
    writer.add_scalar("Dist/avg",    _avg(_ep_dist), global_step)
    writer.add_scalar("Foot/r1_lf_tilt_avg", _avg(_ep_lf1), global_step)
    writer.add_scalar("Foot/r1_rf_tilt_avg", _avg(_ep_rf1), global_step)
    writer.add_scalar("Foot/r2_lf_tilt_avg", _avg(_ep_lf2), global_step)
    writer.add_scalar("Foot/r2_rf_tilt_avg", _avg(_ep_rf2), global_step)
    # Velocidad de aproximación — métrica principal del versus
    writer.add_scalar("Diag/r1_approach_avg", _avg(_ep_app1),   global_step)
    writer.add_scalar("Diag/r2_approach_avg", _avg(_ep_app2),   global_step)
    # vx mundo — diagnóstico de bugs de espejo (r1 debe ser negativo, r2 positivo)
    writer.add_scalar("Diag/r1_vx_avg",       _avg(_ep_vx1),    global_step)
    writer.add_scalar("Diag/r2_vx_avg",       _avg(_ep_vx2),    global_step)
    writer.add_scalar("Diag/r1_fwd_rew_avg",  _avg(_ep_fwd1),   global_step)
    writer.add_scalar("Diag/r2_fwd_rew_avg",  _avg(_ep_fwd2),   global_step)
    writer.add_scalar("Diag/r1_lat_thigh_L",  _avg(_ep_lat_l1), global_step)
    writer.add_scalar("Diag/r1_lat_thigh_R",  _avg(_ep_lat_r1), global_step)
    writer.add_scalar("Diag/r2_lat_thigh_L",  _avg(_ep_lat_l2), global_step)
    writer.add_scalar("Diag/r2_lat_thigh_R",  _avg(_ep_lat_r2), global_step)

    lat_warn = ""
    for name, vals in [("r1L", _ep_lat_l1), ("r1R", _ep_lat_r1),
                        ("r2L", _ep_lat_l2), ("r2R", _ep_lat_r2)]:
        if abs(_avg(vals)) > 0.4:
            lat_warn += f" ⚠{name}={_avg(vals):.2f}"

    print(
        f"Ep {episode:4d} | step {global_step:7d} | "
        f"R1 {ep_r1:7.1f} (falls {falls_r1:3d})  "
        f"R2 {ep_r2:7.1f} (falls {falls_r2:3d}) | "
        f"app r1={_avg(_ep_app1):.2f} r2={_avg(_ep_app2):.2f} | "
        f"dist={_avg(_ep_dist):.2f}"
        + (f" |{lat_warn}" if lat_warn else "")
    )

    with open(_CSV_PATH, mode="a", newline="") as f:
        csv.writer(f).writerow([global_step, episode, ep_r1, ep_r2,
                                 falls_r1, falls_r2])

    episode += 1

env.close()
writer.close()
