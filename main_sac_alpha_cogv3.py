import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_CKPT_DIR = os.path.join(_HERE, "checkpoints", "sac_alpha_cogv3")
os.makedirs(_CKPT_DIR, exist_ok=True)

_XML_V2 = os.path.join(_HERE, "robot", "configs", "v2", "alpha_single.xml")

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
from envs.alpha_env import AlphaEnv

import csv
import subprocess
import datetime

_CSV_PATH  = os.path.join(_HERE, "training_log_cogv3.csv")
_TODAY     = datetime.date.today().strftime("%d_%m_%Y")
_MEDIA_DIR = os.path.join(_HERE, "media", f"{_TODAY}_cogv3_pie_plano")
os.makedirs(_MEDIA_DIR, exist_ok=True)

_GIF_SCRIPT     = os.path.join(_HERE, "tools", "training", "make_checkpoint_gif_v2.py")
_GIF_COM_SCRIPT = os.path.join(_HERE, "tools", "training", "make_com_gif_transparent.py")

if not os.path.exists(_CSV_PATH):
    with open(_CSV_PATH, mode="w", newline="") as f:
        csv.writer(f).writerow(["step", "episode", "reward"])

LEARNING_RATE  = 3e-4
GAMMA          = 0.99    # factor de descuento: recompensas futuras valen GAMMA^t menos
TAU            = 0.005   # mezcla lenta del critic objetivo: θ' ← τθ + (1-τ)θ'
BUFFER_SIZE    = int(1e6)
BATCH_SIZE     = 256
LEARNING_STARTS = 1000   # pasos de exploración aleatoria antes de empezar a entrenar
TOTAL_TIMESTEPS = 2_000_000
SAVE_INTERVAL   = 50000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter("runs_v2/sac_alpha_cogv3")


class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE):
        # deque con maxlen descarta automáticamente la transición más antigua cuando está lleno
        self.buffer = deque(maxlen=max_size)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # unsqueeze(1) convierte rewards y dones de shape (N,) a (N,1) para operar con Q-values
        return (torch.FloatTensor(np.array(states)).to(device),
                torch.FloatTensor(np.array(actions)).to(device),
                torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
                torch.FloatTensor(np.array(next_states)).to(device),
                torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device))

    def size(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mu      = nn.Linear(256, action_dim)   # media de la distribución gaussiana
        self.log_std = nn.Linear(256, action_dim)   # log de la desviación estándar
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        # Clamp del log_std para evitar std demasiado pequeña (<e^-20) o grande (>e^2)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mu, log_std.exp()

    def sample(self, state):
        mu, std = self.forward(state)
        normal  = torch.distributions.Normal(mu, std)
        x_t     = normal.rsample()          # muestra con reparametrización (permite gradientes)
        y_t     = torch.tanh(x_t)           # squash a (-1, 1)
        action  = y_t * self.max_action
        # log_prob de la distribución gaussiana antes del squash
        log_prob = normal.log_prob(x_t).sum(1, keepdim=True)
        # Corrección de Jacobiano por la transformación tanh: log|det(dy/dx)| = Σ log(1 - tanh²)
        # Sin esta corrección, log_prob estaría mal calibrado y alpha aprendería entropía incorrecta
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    # Doble critic (Q1, Q2) para reducir sobreestimación del valor (truco de TD3/SAC)
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)   # concatenamos estado y acción como entrada
        return self.q1(sa), self.q2(sa)


env = gym.wrappers.TimeLimit(
    AlphaEnv(xml_path=_XML_V2), max_episode_steps=1000
)
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

actor  = Actor(state_dim, action_dim, max_action).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

critic = Critic(state_dim, action_dim).to(device)
critic_target = Critic(state_dim, action_dim).to(device)
critic_target.load_state_dict(critic.state_dict())   # critic objetivo empieza igual que el principal
critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

# SAC con alpha automático: alpha regula el peso de la entropía en el objetivo
# target_entropy es el valor deseado de entropía; la heurística -action_dim funciona bien en práctica
target_entropy = -action_dim
# Usamos log_alpha en vez de alpha directamente para garantizar alpha > 0 siempre
log_alpha      = torch.zeros(1, requires_grad=True, device=device)
alpha_optimizer = optim.Adam([log_alpha], lr=LEARNING_RATE)

replay_buffer = ReplayBuffer()
global_step   = 0
episode       = 0

# ── Resume del checkpoint más reciente si existe ──────────────────────────────
_ckpts = sorted(
    [f for f in os.listdir(_CKPT_DIR) if f.endswith(".pt")],
    key=lambda f: int(f.split("_")[-1].replace(".pt", ""))
)
if _ckpts:
    _resume_path = os.path.join(_CKPT_DIR, _ckpts[-1])
    print(f"Resumiendo desde {_resume_path} ...")
    _ckpt = torch.load(_resume_path, map_location=device, weights_only=False)
    actor.load_state_dict(_ckpt["actor"])
    critic.load_state_dict(_ckpt["critic"])
    critic_target.load_state_dict(_ckpt["critic_target"])
    # log_alpha guardado como tensor; hay que recrear el optimizador apuntando al nuevo tensor
    log_alpha = _ckpt["log_alpha"].to(device).requires_grad_(True)
    alpha_optimizer  = optim.Adam([log_alpha], lr=LEARNING_RATE)
    alpha_optimizer.load_state_dict(_ckpt["alpha_optimizer"])
    actor_optimizer.load_state_dict(_ckpt["actor_optimizer"])
    critic_optimizer.load_state_dict(_ckpt["critic_optimizer"])
    global_step = int(_ckpts[-1].split("_")[-1].replace(".pt", ""))
    print(f"Continuando desde step {global_step}")
else:
    print("Sin checkpoint previo — entrenamiento desde cero (cogv3)")
# ─────────────────────────────────────────────────────────────────────────────

while global_step < TOTAL_TIMESTEPS:
    state, _ = env.reset()
    episode_reward = 0
    done = False

    # acumuladores de debug por episodio
    _ep_lf_tilt, _ep_rf_tilt = [], []
    _ep_lf_z,    _ep_rf_z    = [], []

    while not done:
        global_step += 1
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _ = actor.sample(state_tensor)
        action = action.cpu().numpy()[0]

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        _ep_lf_tilt.append(info["lf_tilt"])
        _ep_rf_tilt.append(info["rf_tilt"])
        _ep_lf_z.append(info["lf_z"])
        _ep_rf_z.append(info["rf_z"])

        episode_reward += reward
        # Guardamos la transición (s, a, r, s', done) en el replay buffer
        replay_buffer.put((state, action, reward, next_state, float(done)))
        state = next_state

        if replay_buffer.size() > LEARNING_STARTS:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            # ── 1. Actualizar alpha (temperatura de entropía) ─────────────────
            new_actions, log_pi = actor.sample(states)
            q1_new, q2_new = critic(states, new_actions)
            q_min = torch.min(q1_new, q2_new)   # usamos el Q mínimo para evitar sobreestimación

            # alpha_loss: ajusta alpha para que la entropía media se acerque a target_entropy
            # Si log_pi >> -target_entropy → política demasiado determinista → alpha sube
            alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            alpha = log_alpha.exp()   # alpha siempre positivo gracias a exp

            # ── 2. Actualizar actor ───────────────────────────────────────────
            # El actor maximiza Q - alpha * log_pi (Q + entropía ponderada)
            actor_loss = (alpha * log_pi - q_min).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # ── 3. Calcular target Q (con critic objetivo y sin gradientes) ───
            with torch.no_grad():
                next_actions, next_log_pi = actor.sample(next_states)
                q1_next, q2_next = critic_target(next_states, next_actions)
                # Target de Bellman con regularización de entropía: Q - alpha * log_pi
                q_target = torch.min(q1_next, q2_next) - alpha * next_log_pi
                # (1-done) cancela el valor futuro si el episodio terminó
                target_q = rewards + (1 - dones) * GAMMA * q_target

            # ── 4. Actualizar critic ──────────────────────────────────────────
            q1, q2 = critic(states, actions)
            # Entrenamos ambas cabezas Q contra el mismo target; reducir las dos reduce sesgo
            critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # ── 5. Actualización suave del critic objetivo (Polyak) ───────────
            # θ'_nuevo = TAU * θ_principal + (1-TAU) * θ'_viejo
            # TAU pequeño (0.005) hace que el objetivo sea estable y reduzca divergencia
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            writer.add_scalar("Loss/actor",  actor_loss.item(),  global_step)
            writer.add_scalar("Loss/critic", critic_loss.item(), global_step)
            writer.add_scalar("Loss/alpha",  alpha_loss.item(),  global_step)
            writer.add_scalar("Alpha",        alpha.item(),       global_step)

        if global_step % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(_CKPT_DIR, f"sac2_checkpoint_{global_step}.pt")
            torch.save({
                "actor":            actor.state_dict(),
                "critic":           critic.state_dict(),
                "critic_target":    critic_target.state_dict(),
                "log_alpha":        log_alpha,
                "alpha_optimizer":  alpha_optimizer.state_dict(),
                "actor_optimizer":  actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
            }, ckpt_path)
            print(f"Checkpoint guardado en step {global_step}")

            # Lanzamos los GIFs en procesos separados para no bloquear el entrenamiento
            _gif_log = open(os.path.join(_MEDIA_DIR, f"gif_{global_step}.log"), "w")
            subprocess.Popen(
                [sys.executable, _GIF_SCRIPT,
                 "--ckpt", ckpt_path,
                 "--step", str(global_step),
                 "--out_dir", _MEDIA_DIR],
                stdout=_gif_log, stderr=_gif_log,
            )
            _com_log = open(os.path.join(_MEDIA_DIR, f"gif_com_{global_step}.log"), "w")
            subprocess.Popen(
                [sys.executable, _GIF_COM_SCRIPT,
                 "--ckpt", ckpt_path,
                 "--out", os.path.join(_MEDIA_DIR, f"com_step_{global_step:07d}.gif")],
                stdout=_com_log, stderr=_com_log,
            )

    # ── stats de debug del episodio ──────────────────────────────────────────
    avg_lf_tilt = sum(_ep_lf_tilt) / len(_ep_lf_tilt) if _ep_lf_tilt else 0.0
    avg_rf_tilt = sum(_ep_rf_tilt) / len(_ep_rf_tilt) if _ep_rf_tilt else 0.0
    max_lf_tilt = max(_ep_lf_tilt) if _ep_lf_tilt else 0.0
    max_rf_tilt = max(_ep_rf_tilt) if _ep_rf_tilt else 0.0
    avg_lf_z    = sum(_ep_lf_z)    / len(_ep_lf_z)    if _ep_lf_z    else 0.0
    avg_rf_z    = sum(_ep_rf_z)    / len(_ep_rf_z)    if _ep_rf_z    else 0.0
    max_lf_z    = max(_ep_lf_z)    if _ep_lf_z    else 0.0
    max_rf_z    = max(_ep_rf_z)    if _ep_rf_z    else 0.0

    writer.add_scalar("Foot/lf_tilt_avg", avg_lf_tilt, global_step)
    writer.add_scalar("Foot/rf_tilt_avg", avg_rf_tilt, global_step)
    writer.add_scalar("Foot/lf_tilt_max", max_lf_tilt, global_step)
    writer.add_scalar("Foot/rf_tilt_max", max_rf_tilt, global_step)
    writer.add_scalar("Foot/lf_z_avg",    avg_lf_z,    global_step)
    writer.add_scalar("Foot/rf_z_avg",    avg_rf_z,    global_step)
    writer.add_scalar("Foot/lf_z_max",    max_lf_z,    global_step)
    writer.add_scalar("Foot/rf_z_max",    max_rf_z,    global_step)

    writer.add_scalar("Reward/episode", episode_reward, global_step)
    print(
        f"Ep {episode:4d} | step {global_step:7d} | reward {episode_reward:7.2f}"
        f" | tilt L {avg_lf_tilt:.3f}(max {max_lf_tilt:.3f})"
        f" R {avg_rf_tilt:.3f}(max {max_rf_tilt:.3f})"
        f" | alt  L {avg_lf_z:.3f}(max {max_lf_z:.3f})"
        f" R {avg_rf_z:.3f}(max {max_rf_z:.3f})"
    )

    with open(_CSV_PATH, mode="a", newline="") as f:
        csv.writer(f).writerow([global_step, episode, episode_reward])

    episode += 1

env.close()
writer.close()
