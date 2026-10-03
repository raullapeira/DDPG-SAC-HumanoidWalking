import sys
import os
import glob
import datetime
_HERE  = os.path.dirname(os.path.abspath(__file__))
_TODAY = datetime.date.today().strftime("%Y_%m_%d")
sys.path.insert(0, _HERE)

# La carpeta de checkpoints lleva la fecha de INICIO de este entrenamiento (no
# la del dia en que se relanza el script) — igual que en fighting.py. Fecha
# primero (YYYY_MM_DD) para que las carpetas ordenen cronologicamente.
_RUN_TAG       = "full_fight_freeze_pose_neutra"   # piernas se congelan en postura neutra fija (no a media zancada) al empezar el combate
_CKPT_BASE     = os.path.join(_HERE, "checkpoints")
_existing_ckpt = sorted(glob.glob(os.path.join(_CKPT_BASE, f"*_{_RUN_TAG}", "leg_r1", "*.pt")))
if _existing_ckpt:
    _RUN_DATE = os.path.basename(os.path.dirname(os.path.dirname(_existing_ckpt[-1]))).replace(f"_{_RUN_TAG}", "")
else:
    _RUN_DATE = _TODAY

_CKPT_LEG_R1 = os.path.join(_CKPT_BASE, f"{_RUN_DATE}_{_RUN_TAG}", "leg_r1")
_CKPT_LEG_R2 = os.path.join(_CKPT_BASE, f"{_RUN_DATE}_{_RUN_TAG}", "leg_r2")
_CKPT_ARM_R1 = os.path.join(_CKPT_BASE, f"{_RUN_DATE}_{_RUN_TAG}", "arm_r1")
_CKPT_ARM_R2 = os.path.join(_CKPT_BASE, f"{_RUN_DATE}_{_RUN_TAG}", "arm_r2")
for _d in (_CKPT_LEG_R1, _CKPT_LEG_R2, _CKPT_ARM_R1, _CKPT_ARM_R2):
    os.makedirs(_d, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import csv
import subprocess
from torch.utils.tensorboard import SummaryWriter
from full_fight_env import FullFightEnv

_CSV_PATH  = os.path.join(_HERE, "training_log_full_fight.csv")
_MEDIA_DIR = os.path.join(_HERE, "media", f"{_TODAY}_{_RUN_TAG}")
os.makedirs(_MEDIA_DIR, exist_ok=True)

_GIF_SCRIPT = os.path.join(_HERE, "tools", "simu_a_real", "make_full_fight_gif.py")

if not os.path.exists(_CSV_PATH):
    with open(_CSV_PATH, mode="w", newline="") as f:
        csv.writer(f).writerow([
            "step", "episode",
            "reward_leg_r1", "reward_leg_r2", "reward_arm_r1", "reward_arm_r2",
            "falls_r1", "falls_r2", "reached_combat", "winner",
        ])

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
writer = SummaryWriter("runs/full_fight")


# ── Redes (misma arquitectura que fighting.py / close_combat.py) ──────────────

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


def load_ckpt_file(path, actor, critic, critic_target, log_alpha,
                   actor_opt, critic_opt, alpha_opt):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    critic_target.load_state_dict(ckpt["critic_target"])
    log_alpha.data.copy_(ckpt["log_alpha"].to(device))
    actor_opt.load_state_dict(ckpt["actor_opt"])
    critic_opt.load_state_dict(ckpt["critic_opt"])
    alpha_opt.load_state_dict(ckpt["alpha_opt"])


def load_latest(ckpt_dir, actor, critic, critic_target, log_alpha,
                actor_opt, critic_opt, alpha_opt):
    files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
        key=lambda f: int(f.split("_")[-1].replace(".pt", ""))
    )
    if not files:
        return 0
    load_ckpt_file(os.path.join(ckpt_dir, files[-1]), actor, critic, critic_target,
                   log_alpha, actor_opt, critic_opt, alpha_opt)
    return int(files[-1].split("_")[-1].replace(".pt", ""))


# ── Entorno y agentes ─────────────────────────────────────────────────────────

env = FullFightEnv()
WALK_OBS_DIM  = 34
FIGHT_OBS_DIM = 28
LEG_DIM = 10
ARM_DIM = 6
max_action = 1.0


def make_agent(obs_dim, action_dim):
    actor  = Actor(obs_dim, action_dim, max_action).to(device)
    critic = Critic(obs_dim, action_dim).to(device)
    critic_target = Critic(obs_dim, action_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    actor_opt  = optim.Adam(actor.parameters(),  lr=LEARNING_RATE)
    critic_opt = optim.Adam(critic.parameters(), lr=LEARNING_RATE)
    log_alpha  = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt  = optim.Adam([log_alpha], lr=LEARNING_RATE)
    target_ent = -float(action_dim)
    buffer     = ReplayBuffer()
    return actor, critic, critic_target, actor_opt, critic_opt, log_alpha, alpha_opt, target_ent, buffer


(leg_actor_r1, leg_critic_r1, leg_critic_r1_target, leg_actor_r1_opt, leg_critic_r1_opt,
 leg_log_alpha_r1, leg_alpha_r1_opt, leg_target_ent_r1, buffer_leg_r1) = make_agent(WALK_OBS_DIM, LEG_DIM)
(leg_actor_r2, leg_critic_r2, leg_critic_r2_target, leg_actor_r2_opt, leg_critic_r2_opt,
 leg_log_alpha_r2, leg_alpha_r2_opt, leg_target_ent_r2, buffer_leg_r2) = make_agent(WALK_OBS_DIM, LEG_DIM)
(arm_actor_r1, arm_critic_r1, arm_critic_r1_target, arm_actor_r1_opt, arm_critic_r1_opt,
 arm_log_alpha_r1, arm_alpha_r1_opt, arm_target_ent_r1, buffer_arm_r1) = make_agent(FIGHT_OBS_DIM, ARM_DIM)
(arm_actor_r2, arm_critic_r2, arm_critic_r2_target, arm_actor_r2_opt, arm_critic_r2_opt,
 arm_log_alpha_r2, arm_alpha_r2_opt, arm_target_ent_r2, buffer_arm_r2) = make_agent(FIGHT_OBS_DIM, ARM_DIM)

step_leg_r1 = load_latest(_CKPT_LEG_R1, leg_actor_r1, leg_critic_r1, leg_critic_r1_target,
                          leg_log_alpha_r1, leg_actor_r1_opt, leg_critic_r1_opt, leg_alpha_r1_opt)
step_leg_r2 = load_latest(_CKPT_LEG_R2, leg_actor_r2, leg_critic_r2, leg_critic_r2_target,
                          leg_log_alpha_r2, leg_actor_r2_opt, leg_critic_r2_opt, leg_alpha_r2_opt)
step_arm_r1 = load_latest(_CKPT_ARM_R1, arm_actor_r1, arm_critic_r1, arm_critic_r1_target,
                          arm_log_alpha_r1, arm_actor_r1_opt, arm_critic_r1_opt, arm_alpha_r1_opt)
step_arm_r2 = load_latest(_CKPT_ARM_R2, arm_actor_r2, arm_critic_r2, arm_critic_r2_target,
                          arm_log_alpha_r2, arm_actor_r2_opt, arm_critic_r2_opt, arm_alpha_r2_opt)

# Si las piernas arrancan de cero en ESTE experimento, aprovechamos la politica
# de acercamiento+frenado ya validada en versus_stop_dist (mismo obs_dim=34,
# accion_dim=10) en vez de reentrenarla desde cero otra vez.
if step_leg_r1 == 0:
    _warm = sorted(glob.glob(os.path.join(_CKPT_BASE, "*_versus_stop_dist", "r1", "*.pt")))
    if _warm:
        load_ckpt_file(_warm[-1], leg_actor_r1, leg_critic_r1, leg_critic_r1_target,
                       leg_log_alpha_r1, leg_actor_r1_opt, leg_critic_r1_opt, leg_alpha_r1_opt)
        print(f"  leg r1: warm-start desde versus_stop_dist ({_warm[-1]})")
if step_leg_r2 == 0:
    _warm = sorted(glob.glob(os.path.join(_CKPT_BASE, "*_versus_stop_dist", "r2", "*.pt")))
    if _warm:
        load_ckpt_file(_warm[-1], leg_actor_r2, leg_critic_r2, leg_critic_r2_target,
                       leg_log_alpha_r2, leg_actor_r2_opt, leg_critic_r2_opt, leg_alpha_r2_opt)
        print(f"  leg r2: warm-start desde versus_stop_dist ({_warm[-1]})")

global_step = max(step_leg_r1, step_leg_r2, step_arm_r1, step_arm_r2)

if global_step > 0:
    print(f"Resumiendo desde step {global_step}")
else:
    print("Entrenamiento desde cero — acercamiento + freno + combate en un solo episodio")

# ── Bucle de entrenamiento ────────────────────────────────────────────────────

episode = 0

while global_step < TOTAL_TIMESTEPS:
    (walk_obs, fight_obs), _ = env.reset()
    walk_r1, walk_r2   = walk_obs
    fight_r1, fight_r2 = fight_obs

    ep_reward_leg_r1 = ep_reward_leg_r2 = 0.0
    ep_reward_arm_r1 = ep_reward_arm_r2 = 0.0
    falls_r1 = falls_r2 = 0
    reached_combat = False
    winner = 0   # 0 = nadie (timeout), 1 = gana r1 (tira a r2), 2 = gana r2
    done = False

    while not done:
        global_step += 1

        t1w = torch.FloatTensor(walk_r1).unsqueeze(0).to(device)
        t2w = torch.FloatTensor(walk_r2).unsqueeze(0).to(device)
        t1f = torch.FloatTensor(fight_r1).unsqueeze(0).to(device)
        t2f = torch.FloatTensor(fight_r2).unsqueeze(0).to(device)

        with torch.no_grad():
            leg_a1, _ = leg_actor_r1.sample(t1w)
            leg_a2, _ = leg_actor_r2.sample(t2w)
            arm_a1, _ = arm_actor_r1.sample(t1f)
            arm_a2, _ = arm_actor_r2.sample(t2f)
        leg_a1 = leg_a1.cpu().numpy()[0]; leg_a2 = leg_a2.cpu().numpy()[0]
        arm_a1 = arm_a1.cpu().numpy()[0]; arm_a2 = arm_a2.cpu().numpy()[0]

        action = np.concatenate([leg_a1, arm_a1, leg_a2, arm_a2])
        (next_walk, next_fight), (r1, r2), terminated, truncated, info = env.step(action)
        next_walk_r1, next_walk_r2   = next_walk
        next_fight_r1, next_fight_r2 = next_fight
        done = terminated or truncated

        if info["r1_fell"]: falls_r1 += 1
        if info["r2_fell"]: falls_r2 += 1

        if info["fighting"]:
            reached_combat = True
            ep_reward_arm_r1 += r1; ep_reward_arm_r2 += r2
            done_flag = float(terminated or truncated)
            buffer_arm_r1.put((fight_r1, arm_a1, r1, next_fight_r1, done_flag))
            buffer_arm_r2.put((fight_r2, arm_a2, r2, next_fight_r2, done_flag))

            res1 = sac_update(arm_actor_r1, arm_critic_r1, arm_critic_r1_target,
                              arm_actor_r1_opt, arm_critic_r1_opt, arm_alpha_r1_opt,
                              arm_log_alpha_r1, arm_target_ent_r1, buffer_arm_r1)
            res2 = sac_update(arm_actor_r2, arm_critic_r2, arm_critic_r2_target,
                              arm_actor_r2_opt, arm_critic_r2_opt, arm_alpha_r2_opt,
                              arm_log_alpha_r2, arm_target_ent_r2, buffer_arm_r2)
            if res1:
                al1, cl1, alpha1 = res1
                writer.add_scalar("ArmR1/actor_loss",  al1,    global_step)
                writer.add_scalar("ArmR1/critic_loss", cl1,    global_step)
                writer.add_scalar("ArmR1/alpha",       alpha1, global_step)
            if res2:
                al2, cl2, alpha2 = res2
                writer.add_scalar("ArmR2/actor_loss",  al2,    global_step)
                writer.add_scalar("ArmR2/critic_loss", cl2,    global_step)
                writer.add_scalar("ArmR2/alpha",       alpha2, global_step)

            if terminated:
                if info["r1_fell"] and not info["r2_fell"]:
                    winner = 2
                elif info["r2_fell"] and not info["r1_fell"]:
                    winner = 1
        else:
            ep_reward_leg_r1 += r1; ep_reward_leg_r2 += r2
            done_r1 = info["r1_fell"] or truncated
            done_r2 = info["r2_fell"] or truncated
            buffer_leg_r1.put((walk_r1, leg_a1, r1, next_walk_r1, float(done_r1)))
            buffer_leg_r2.put((walk_r2, leg_a2, r2, next_walk_r2, float(done_r2)))

            res1 = sac_update(leg_actor_r1, leg_critic_r1, leg_critic_r1_target,
                              leg_actor_r1_opt, leg_critic_r1_opt, leg_alpha_r1_opt,
                              leg_log_alpha_r1, leg_target_ent_r1, buffer_leg_r1)
            res2 = sac_update(leg_actor_r2, leg_critic_r2, leg_critic_r2_target,
                              leg_actor_r2_opt, leg_critic_r2_opt, leg_alpha_r2_opt,
                              leg_log_alpha_r2, leg_target_ent_r2, buffer_leg_r2)
            if res1:
                al1, cl1, alpha1 = res1
                writer.add_scalar("LegR1/actor_loss",  al1,    global_step)
                writer.add_scalar("LegR1/critic_loss", cl1,    global_step)
                writer.add_scalar("LegR1/alpha",       alpha1, global_step)
            if res2:
                al2, cl2, alpha2 = res2
                writer.add_scalar("LegR2/actor_loss",  al2,    global_step)
                writer.add_scalar("LegR2/critic_loss", cl2,    global_step)
                writer.add_scalar("LegR2/alpha",       alpha2, global_step)

        walk_r1, walk_r2   = next_walk_r1, next_walk_r2
        fight_r1, fight_r2 = next_fight_r1, next_fight_r2

        if global_step % SAVE_INTERVAL == 0:
            save_checkpoint(_CKPT_LEG_R1, global_step, leg_actor_r1, leg_critic_r1, leg_critic_r1_target,
                            leg_log_alpha_r1, leg_actor_r1_opt, leg_critic_r1_opt, leg_alpha_r1_opt)
            save_checkpoint(_CKPT_LEG_R2, global_step, leg_actor_r2, leg_critic_r2, leg_critic_r2_target,
                            leg_log_alpha_r2, leg_actor_r2_opt, leg_critic_r2_opt, leg_alpha_r2_opt)
            save_checkpoint(_CKPT_ARM_R1, global_step, arm_actor_r1, arm_critic_r1, arm_critic_r1_target,
                            arm_log_alpha_r1, arm_actor_r1_opt, arm_critic_r1_opt, arm_alpha_r1_opt)
            save_checkpoint(_CKPT_ARM_R2, global_step, arm_actor_r2, arm_critic_r2, arm_critic_r2_target,
                            arm_log_alpha_r2, arm_actor_r2_opt, arm_critic_r2_opt, arm_alpha_r2_opt)
            print(f"Checkpoints guardados en step {global_step}")

            ckpt_leg_r1_path = os.path.join(_CKPT_LEG_R1, f"ckpt_{global_step}.pt")
            ckpt_leg_r2_path = os.path.join(_CKPT_LEG_R2, f"ckpt_{global_step}.pt")
            ckpt_arm_r1_path = os.path.join(_CKPT_ARM_R1, f"ckpt_{global_step}.pt")
            ckpt_arm_r2_path = os.path.join(_CKPT_ARM_R2, f"ckpt_{global_step}.pt")
            subprocess.Popen(
                [sys.executable, _GIF_SCRIPT,
                 "--ckpt_leg_r1", ckpt_leg_r1_path,
                 "--ckpt_leg_r2", ckpt_leg_r2_path,
                 "--ckpt_arm_r1", ckpt_arm_r1_path,
                 "--ckpt_arm_r2", ckpt_arm_r2_path,
                 "--step",        str(global_step),
                 "--out_dir",     _MEDIA_DIR],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    writer.add_scalar("Reward/leg_r1", ep_reward_leg_r1, global_step)
    writer.add_scalar("Reward/leg_r2", ep_reward_leg_r2, global_step)
    writer.add_scalar("Reward/arm_r1", ep_reward_arm_r1, global_step)
    writer.add_scalar("Reward/arm_r2", ep_reward_arm_r2, global_step)
    writer.add_scalar("Falls/r1", falls_r1, global_step)
    writer.add_scalar("Falls/r2", falls_r2, global_step)
    writer.add_scalar("Combat/reached", float(reached_combat), global_step)
    writer.add_scalar("Combat/winner",  winner, global_step)

    print(
        f"Ep {episode:4d} | step {global_step:7d} | "
        f"leg r1={ep_reward_leg_r1:6.1f} r2={ep_reward_leg_r2:6.1f} | "
        f"arm r1={ep_reward_arm_r1:6.1f} r2={ep_reward_arm_r2:6.1f} | "
        f"combate={'si' if reached_combat else 'no'} | "
        f"ganador={'r1' if winner == 1 else ('r2' if winner == 2 else '-')} | "
        f"falls r1={falls_r1} r2={falls_r2}"
    )

    with open(_CSV_PATH, mode="a", newline="") as f:
        csv.writer(f).writerow([
            global_step, episode,
            ep_reward_leg_r1, ep_reward_leg_r2, ep_reward_arm_r1, ep_reward_arm_r2,
            falls_r1, falls_r2, int(reached_combat), winner,
        ])

    episode += 1

env.close()
writer.close()
