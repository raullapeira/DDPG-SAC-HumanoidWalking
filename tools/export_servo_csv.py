"""
Runs 10 policy steps and exports servo values (0-255) to CSV.

Neutral pose servo values (action=0) are hardcoded from the real robot.
Joint angle (rad) is converted to degrees and added to neutral.

Usage:
    python tools/export_servo_csv.py --ckpt checkpoints/sac_alpha/sac2_checkpoint_1000000.pt
"""
import sys, os, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import numpy as np
import torch
import torch.nn as nn
import csv
import gymnasium as gym
from envs.alpha_env import AlphaEnv, _ACTION_REPEAT, _FRAME_SKIP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_STEPS = 10
STEP_MS = _ACTION_REPEAT * _FRAME_SKIP * 5  # ms per policy step

# Real robot servo names in order 1-16
SERVO_NAMES = [
    "rot_hombro_der",    "ext_hombro_der",    "rot_codo_der",
    "rot_hombro_izq",    "ext_hombro_izq",    "rot_codo_izq",
    "cadera_lat_der",    "cadera_front_der",  "rodilla_der",
    "tobillo_front_der", "tobillo_lat_der",
    "cadera_lat_izq",    "cadera_front_izq",  "rodilla_izq",
    "tobillo_front_izq", "tobillo_lat_izq",
]

# Neutral servo values for real robot (index = servo number - 1)
NEUTRAL_REAL = [90, 90, 90, 90, 90, 90, 90, 60,
                76, 110, 90, 90, 120, 104, 70, 90]

# Mapping: real servo index (0-15) → MuJoCo qpos/actuator index (0-15)
# MuJoCo order: L_Lat_Thigh(0) L_Thigh(1) L_Leg(2) L_Ankle(3) L_Feet(4)
#               L_Shoulder(5) L_Arm(6) L_Hand(7)
#               R_Lat_Thigh(8) R_Thigh(9) R_Leg(10) R_Ankle(11) R_Feet(12)
#               R_Shoulder(13) R_Arm(14) R_Hand(15)
REAL_TO_MUJOCO = [15, 13, 14, 7, 5, 6, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4]

# Neutral values indexed by MuJoCo order (for angle_to_servo)
NEUTRAL_MUJOCO = [NEUTRAL_REAL[REAL_TO_MUJOCO.index(i)] for i in range(16)]


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


def angle_to_servo(angle_rad, neutral):
    """Convert joint angle (rad) to servo value (0-255) using neutral as center."""
    angle_deg = np.degrees(angle_rad)
    value = neutral + angle_deg
    return int(np.clip(round(value), 0, 255))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    out_csv = os.path.join(_HERE, "..", "media", "servo_export.csv")

    env = gym.wrappers.TimeLimit(AlphaEnv(), max_episode_steps=10_000)
    s = env.observation_space.shape[0]
    a = env.action_space.shape[0]
    m = float(env.action_space.high[0])

    actor = Actor(s, a, m).to(DEVICE)
    ckpt  = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    raw_env = env.unwrapped
    obs, _ = env.reset(seed=0)

    rows = []
    for step in range(N_STEPS):
        st = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        action = actor.act(st).cpu().numpy()[0]

        obs, _, term, trunc, _ = env.step(action)

        # Joint angles in radians (qpos[7:] = 16 joints, MuJoCo order)
        joint_angles_mujoco = raw_env.data.qpos[7:].copy()

        # Convert each MuJoCo joint angle to servo value using its neutral
        servo_mujoco = [angle_to_servo(joint_angles_mujoco[i], NEUTRAL_MUJOCO[i]) for i in range(16)]

        # Reorder to real robot servo order (1-16)
        servo_real = [servo_mujoco[REAL_TO_MUJOCO[i]] for i in range(16)]

        row = {"duration_ms": STEP_MS}
        for i, name in enumerate(SERVO_NAMES):
            row[name] = servo_real[i]
        rows.append(row)

        print(f"Step {step+1:2d} ({STEP_MS}ms): {servo_real}")

        if term or trunc:
            print("Episode ended early.")
            break

    env.close()

    fieldnames = ["duration_ms"] + SERVO_NAMES
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved -> {out_csv}")


if __name__ == "__main__":
    main()
