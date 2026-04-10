import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_CKPT_DIR = os.path.join(_HERE, "checkpoints", "sac_alpha")
os.makedirs(_CKPT_DIR, exist_ok=True)

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
from envs.alpha_env import AlphaEnv
import subprocess

import csv

_CSV_PATH = os.path.join(_HERE, "training_log.csv")

# Crear archivo con cabecera si no existe
if not os.path.exists(_CSV_PATH):
    with open(_CSV_PATH, mode="w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["step", "episode", "reward"])

LEARNING_RATE = 3e-4  # Learning rate for all optimizers
GAMMA = 0.99  # Discount factor for future rewards
TAU = 0.005  # Soft update rate for target critic
BUFFER_SIZE = int(1e6)  # Max size of replay buffer
BATCH_SIZE = 256  # Batch size used during training
LEARNING_STARTS = 1000  # Wait until this many steps before starting updates
TOTAL_TIMESTEPS = 1_000_000  # Total steps to train for
SAVE_INTERVAL = 50000  # How often to save checkpoints

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
writer = SummaryWriter("runs2/sac_sb3_style")  # TensorBoard logger

# Replay Buffer class to store past experiences
class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)  # Fixed-size queue

    def put(self, transition):
        self.buffer.append(transition)  # Add new experience to buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # Random sample from buffer
        states, actions, rewards, next_states, dones = zip(*batch)  # Unpack transitions
        return (torch.FloatTensor(np.array(states)).to(device),  # Convert to torch tensors
                torch.FloatTensor(np.array(actions)).to(device),
                torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
                torch.FloatTensor(np.array(next_states)).to(device),
                torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device))

    def size(self):
        return len(self.buffer)  # Return current size of buffer

# Actor network outputs mean and std dev of Gaussian policy
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),  # First hidden layer
            nn.Linear(256, 256), nn.ReLU(),        # Second hidden layer
        )
        self.mu = nn.Linear(256, action_dim)  # Output mean of Gaussian
        self.log_std = nn.Linear(256, action_dim)  # Output log std dev
        self.max_action = max_action  # For scaling actions after tanh

    def forward(self, state):
        x = self.net(state)  # Pass through hidden layers
        mu = self.mu(x)  # Get mean
        log_std = self.log_std(x)  # Get log std
        log_std = torch.clamp(log_std, -20, 2)  # Clamp for stability
        std = log_std.exp()  # Convert log std to std
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)  # Get distribution parameters
        normal = torch.distributions.Normal(mu, std)  # Create Gaussian
        x_t = normal.rsample()  # Reparameterized sample
        y_t = torch.tanh(x_t)  # Apply tanh to bound actions
        action = y_t * self.max_action  # Scale to action range
        log_prob = normal.log_prob(x_t).sum(1, keepdim=True)  # Log prob of action
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(1, keepdim=True)  # Correction for tanh
        return action, log_prob

# Critic network returns two Q-value estimates
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),  # Q1 net
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),  # Q2 net
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)  # Concatenate state and action
        return self.q1(sa), self.q2(sa)  # Return both Q-values

# Create environment (TimeLimit prevents infinite episodes if robot never falls)
env = gym.wrappers.TimeLimit(AlphaEnv(), max_episode_steps=1000)
state_dim = env.observation_space.shape[0]  # 43
action_dim = env.action_space.shape[0]       # 16
max_action = float(env.action_space.high[0]) # 1.0 (actions are normalised)

# Initialize networks and optimizers
actor = Actor(state_dim, action_dim, max_action).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

critic = Critic(state_dim, action_dim).to(device)
critic_target = Critic(state_dim, action_dim).to(device)
critic_target.load_state_dict(critic.state_dict())  # Copy weights to target
critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

# Alpha tuning for entropy (automatic temperature adjustment)
target_entropy = -action_dim  # Target entropy
log_alpha = torch.zeros(1, requires_grad=True, device=device)  # Learnable log alpha
alpha_optimizer = optim.Adam([log_alpha], lr=LEARNING_RATE)

replay_buffer = ReplayBuffer()  # Initialize replay buffer

global_step = 0
episode = 0

# ---- Resume from latest checkpoint if one exists ----
_ckpts = sorted(
    [f for f in os.listdir(_CKPT_DIR) if f.endswith(".pt")],
    key=lambda f: int(f.split("_")[-1].replace(".pt", ""))
)
if _ckpts:
    _resume_path = os.path.join(_CKPT_DIR, _ckpts[-1])
    print(f"Resuming from {_resume_path} ...")
    _ckpt = torch.load(_resume_path, map_location=device, weights_only=False)
    actor.load_state_dict(_ckpt["actor"])
    critic.load_state_dict(_ckpt["critic"])
    critic_target.load_state_dict(_ckpt["critic_target"])
    log_alpha = _ckpt["log_alpha"].to(device).requires_grad_(True)
    alpha_optimizer = optim.Adam([log_alpha], lr=LEARNING_RATE)
    alpha_optimizer.load_state_dict(_ckpt["alpha_optimizer"])
    actor_optimizer.load_state_dict(_ckpt["actor_optimizer"])
    critic_optimizer.load_state_dict(_ckpt["critic_optimizer"])
    global_step = int(_ckpts[-1].split("_")[-1].replace(".pt", ""))
    print(f"Resumed at step {global_step}, continuing to {TOTAL_TIMESTEPS}")
# -----------------------------------------------------

# --------------- Main training loop ---------------
while global_step < TOTAL_TIMESTEPS:
    state, _ = env.reset()  # Reset env at episode start
    episode_reward = 0
    done = False

    while not done:
        global_step += 1
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # Convert to torch
        with torch.no_grad():
            action, _ = actor.sample(state_tensor)  # Sample action from actor
        action = action.cpu().numpy()[0]  # Convert to numpy

        next_state, reward, terminated, truncated, info = env.step(action)  # Step env
        done = terminated or truncated  # Episode done?

        # AlphaEnv reward already weighted correctly — no extra shaping needed

        episode_reward += reward  # Track reward

        replay_buffer.put((state, action, reward, next_state, float(done)))  # Store in buffer
        state = next_state  # Move to next state

        # Train if enough samples are collected
        if replay_buffer.size() > LEARNING_STARTS:
            for _ in range(1):  # One gradient step
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                new_actions, log_pi = actor.sample(states)
                q1_new, q2_new = critic(states, new_actions)
                q_min = torch.min(q1_new, q2_new)  # Take min Q for stability

                # ----- Alpha tuning -----
                alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                alpha = log_alpha.exp()  # Convert log_alpha to alpha
                # ------------------------

                # ----- Actor update -----
                actor_loss = (alpha * log_pi - q_min).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                # ------------------------

                # ----- Critic update -----
                with torch.no_grad():
                    next_actions, next_log_pi = actor.sample(next_states)
                    q1_next, q2_next = critic_target(next_states, next_actions)
                    q_target = torch.min(q1_next, q2_next) - alpha * next_log_pi
                    target_q = rewards + (1 - dones) * GAMMA * q_target  # Bellman backup

                q1, q2 = critic(states, actions)
                critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                # -------------------------

                # ----- Soft update target critic -----
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                # -------------------------------------

            # Log to TensorBoard
            writer.add_scalar("Loss/actor", actor_loss.item(), global_step)
            writer.add_scalar("Loss/critic", critic_loss.item(), global_step)
            writer.add_scalar("Loss/alpha", alpha_loss.item(), global_step)
            writer.add_scalar("Alpha", alpha.item(), global_step)

        # Save model checkpoint
        if global_step % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(_CKPT_DIR, f"sac2_checkpoint_{global_step}.pt")
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "critic_target": critic_target.state_dict(),
                "log_alpha": log_alpha,
                "alpha_optimizer": alpha_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
            }, ckpt_path)
            print(f"Checkpoint saved at step {global_step}")
            # Spawn GIF generation in background — does not block training
            gif_script = os.path.join(_HERE, "eval", "make_checkpoint_gif.py")
            subprocess.Popen(
                [sys.executable, gif_script,
                 "--ckpt", ckpt_path, "--step", str(global_step)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    # Log episode reward
    writer.add_scalar("Reward/episode", episode_reward, global_step)
    print(f"Episode {episode}, Reward: {episode_reward:.2f}, Step: {global_step}")

    with open(_CSV_PATH, mode="a", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([global_step, episode, episode_reward])
        
    episode += 1

# Clean up
env.close()
writer.close()
