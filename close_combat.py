import sys
import os
import glob
import datetime
_HERE  = os.path.dirname(os.path.abspath(__file__))
_TODAY = datetime.date.today().strftime("%Y_%m_%d")
sys.path.insert(0, _HERE)

# Entrenamiento SOLO de combate (brazos), separado del de caminar a propósito:
# los robots arrancan ya a distancia de combate con las piernas congeladas en
# postura neutra (ver close_combat_env.py) — cada step desde el primero aporta
# señal útil a los brazos, sin gastar episodios en la fase de acercamiento.
_RUN_TAG       = "close_combat_ciclo_golpe"   # golpe solo puntua con el brazo cargado; recarga al recoger el brazo; quedarse apretando penaliza
_CKPT_BASE     = os.path.join(_HERE, "checkpoints")
_existing_ckpt = sorted(glob.glob(os.path.join(_CKPT_BASE, f"*_{_RUN_TAG}", "arm_r1", "*.pt")))
if _existing_ckpt:
    _RUN_DATE = os.path.basename(os.path.dirname(os.path.dirname(_existing_ckpt[-1]))).replace(f"_{_RUN_TAG}", "")
else:
    _RUN_DATE = _TODAY

_CKPT_ARM_R1 = os.path.join(_CKPT_BASE, f"{_RUN_DATE}_{_RUN_TAG}", "arm_r1")
_CKPT_ARM_R2 = os.path.join(_CKPT_BASE, f"{_RUN_DATE}_{_RUN_TAG}", "arm_r2")
os.makedirs(_CKPT_ARM_R1, exist_ok=True)
os.makedirs(_CKPT_ARM_R2, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import csv
import subprocess
from torch.utils.tensorboard import SummaryWriter
from close_combat_env import CloseCombatEnv, FIGHT_OBS_DIM, ARM_ACT_DIM

_CSV_PATH  = os.path.join(_HERE, "training_log_close_combat_ciclo_golpe.csv")
_MEDIA_DIR = os.path.join(_HERE, "media", f"{_TODAY}_{_RUN_TAG}")
os.makedirs(_MEDIA_DIR, exist_ok=True)

_GIF_SCRIPT = os.path.join(_HERE, "tools", "simu_a_real", "make_close_combat_gif.py")

if not os.path.exists(_CSV_PATH):
    with open(_CSV_PATH, mode="w", newline="") as f:
        csv.writer(f).writerow(["step", "episode", "reward_r1", "reward_r2",
                                 "falls_r1", "falls_r2", "hits_r1", "hits_r2", "winner",
                                 "strikes_r1", "strikes_r2", "clinch_r1", "clinch_r2"])

# ── Hyperparameters ────────────────────────────────────────────────────────────
LEARNING_RATE   = 3e-4
GAMMA           = 0.99
TAU             = 0.005
BUFFER_SIZE     = int(5e5)   # episodios cortos, no hace falta un buffer enorme
BATCH_SIZE      = 256
LEARNING_STARTS = 1000
TOTAL_TIMESTEPS = 2_000_000
SAVE_INTERVAL   = 50_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter("runs/close_combat_solo")


# ── Redes (misma arquitectura que fighting.py / full_fight.py) ────────────────

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
        "actor": actor.state_dict(),          "critic": critic.state_dict(),
        "critic_target": critic_target.state_dict(),
        "log_alpha": log_alpha,
        "actor_opt": actor_opt.state_dict(),  "critic_opt": critic_opt.state_dict(),
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

env = CloseCombatEnv()
obs_dim    = FIGHT_OBS_DIM   # 28
arm_dim    = ARM_ACT_DIM   # Shoulder + Hand por brazo; el giro "Arm" va fijo
max_action = 1.0

arm_r1  = Actor(obs_dim, arm_dim, max_action).to(device)
critic_r1 = Critic(obs_dim, arm_dim).to(device)
critic_r1_target = Critic(obs_dim, arm_dim).to(device)
critic_r1_target.load_state_dict(critic_r1.state_dict())
arm_r1_opt    = optim.Adam(arm_r1.parameters(),  lr=LEARNING_RATE)
critic_r1_opt = optim.Adam(critic_r1.parameters(), lr=LEARNING_RATE)
log_alpha_r1  = torch.zeros(1, requires_grad=True, device=device)
alpha_r1_opt  = optim.Adam([log_alpha_r1], lr=LEARNING_RATE)
target_ent_r1 = -float(arm_dim)
buffer_r1     = ReplayBuffer()

arm_r2  = Actor(obs_dim, arm_dim, max_action).to(device)
critic_r2 = Critic(obs_dim, arm_dim).to(device)
critic_r2_target = Critic(obs_dim, arm_dim).to(device)
critic_r2_target.load_state_dict(critic_r2.state_dict())
arm_r2_opt    = optim.Adam(arm_r2.parameters(),  lr=LEARNING_RATE)
critic_r2_opt = optim.Adam(critic_r2.parameters(), lr=LEARNING_RATE)
log_alpha_r2  = torch.zeros(1, requires_grad=True, device=device)
alpha_r2_opt  = optim.Adam([log_alpha_r2], lr=LEARNING_RATE)
target_ent_r2 = -float(arm_dim)
buffer_r2     = ReplayBuffer()

step_r1 = load_latest(_CKPT_ARM_R1, arm_r1, critic_r1, critic_r1_target,
                      log_alpha_r1, arm_r1_opt, critic_r1_opt, alpha_r1_opt)
step_r2 = load_latest(_CKPT_ARM_R2, arm_r2, critic_r2, critic_r2_target,
                      log_alpha_r2, arm_r2_opt, critic_r2_opt, alpha_r2_opt)
global_step = max(step_r1, step_r2)

if global_step > 0:
    print(f"Resumiendo desde step {global_step}")
else:
    print("Entrenamiento de combate desde cero — solo brazos, piernas congeladas en postura neutra")

# ── Bucle de entrenamiento ────────────────────────────────────────────────────

episode = 0

while global_step < TOTAL_TIMESTEPS:
    (obs_r1, obs_r2), _ = env.reset()

    ep_r1 = ep_r2 = 0.0
    falls_r1 = falls_r2 = 0
    ep_hits_r1, ep_hits_r2 = [], []
    strikes_r1 = strikes_r2 = clinch_r1 = clinch_r2 = 0
    winner = 0   # 0 = nadie, 1/2 = KO de r1/r2 por golpe, -1/-2 = r1/r2 se cae solo
    done = False

    while not done:
        global_step += 1

        t1 = torch.FloatTensor(obs_r1).unsqueeze(0).to(device)
        t2 = torch.FloatTensor(obs_r2).unsqueeze(0).to(device)
        with torch.no_grad():
            arm_a1, _ = arm_r1.sample(t1)
            arm_a2, _ = arm_r2.sample(t2)
        arm_a1 = arm_a1.cpu().numpy()[0]
        arm_a2 = arm_a2.cpu().numpy()[0]

        action = np.concatenate([arm_a1, arm_a2])
        (next_r1, next_r2), (r1, r2), terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ep_r1 += r1; ep_r2 += r2
        if info["r1_fell"]: falls_r1 += 1
        if info["r2_fell"]: falls_r2 += 1
        ep_hits_r1.append(info["r1_hit"])
        ep_hits_r2.append(info["r2_hit"])
        strikes_r1 += info["r1_strikes"]; strikes_r2 += info["r2_strikes"]
        clinch_r1  += info["r1_clinch"];  clinch_r2  += info["r2_clinch"]

        done_flag = float(terminated or truncated)
        buffer_r1.put((obs_r1, arm_a1, r1, next_r1, done_flag))
        buffer_r2.put((obs_r2, arm_a2, r2, next_r2, done_flag))
        obs_r1, obs_r2 = next_r1, next_r2

        if terminated:
            if info["r1_fell"] and not info["r2_fell"]:
                winner = 2 if info["r1_ko_by_hit"] else -1
            elif info["r2_fell"] and not info["r1_fell"]:
                winner = 1 if info["r2_ko_by_hit"] else -2

        res1 = sac_update(arm_r1, critic_r1, critic_r1_target,
                          arm_r1_opt, critic_r1_opt, alpha_r1_opt,
                          log_alpha_r1, target_ent_r1, buffer_r1)
        res2 = sac_update(arm_r2, critic_r2, critic_r2_target,
                          arm_r2_opt, critic_r2_opt, alpha_r2_opt,
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
            ckpt_r1_path = os.path.join(_CKPT_ARM_R1, f"ckpt_{global_step}.pt")
            ckpt_r2_path = os.path.join(_CKPT_ARM_R2, f"ckpt_{global_step}.pt")
            save_checkpoint(_CKPT_ARM_R1, global_step, arm_r1, critic_r1, critic_r1_target,
                            log_alpha_r1, arm_r1_opt, critic_r1_opt, alpha_r1_opt)
            save_checkpoint(_CKPT_ARM_R2, global_step, arm_r2, critic_r2, critic_r2_target,
                            log_alpha_r2, arm_r2_opt, critic_r2_opt, alpha_r2_opt)
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

    avg_hit1 = _avg(ep_hits_r1)
    avg_hit2 = _avg(ep_hits_r2)

    writer.add_scalar("Reward/r1",   ep_r1,    global_step)
    writer.add_scalar("Reward/r2",   ep_r2,    global_step)
    writer.add_scalar("Falls/r1",    falls_r1, global_step)
    writer.add_scalar("Falls/r2",    falls_r2, global_step)
    writer.add_scalar("Hits/r1_avg", avg_hit1, global_step)
    writer.add_scalar("Strikes/r1", strikes_r1, global_step)
    writer.add_scalar("Strikes/r2", strikes_r2, global_step)
    writer.add_scalar("Clinch/r1",  clinch_r1,  global_step)
    writer.add_scalar("Clinch/r2",  clinch_r2,  global_step)
    writer.add_scalar("Hits/r2_avg", avg_hit2, global_step)
    writer.add_scalar("Winner",      winner,   global_step)

    print(
        f"Ep {episode:4d} | step {global_step:7d} | "
        f"R1 {ep_r1:7.1f} (golpes {strikes_r1:3d}, clinch {clinch_r1:3d})  "
        f"R2 {ep_r2:7.1f} (golpes {strikes_r2:3d}, clinch {clinch_r2:3d}) | "
        f"resultado={ {1: 'KO r1', 2: 'KO r2', -1: 'r1 se cae solo', -2: 'r2 se cae solo'}.get(winner, '-') }"
    )

    with open(_CSV_PATH, mode="a", newline="") as f:
        csv.writer(f).writerow([global_step, episode, ep_r1, ep_r2,
                                 falls_r1, falls_r2, avg_hit1, avg_hit2, winner,
                                 strikes_r1, strikes_r2, clinch_r1, clinch_r2])
    episode += 1

env.close()
writer.close()
