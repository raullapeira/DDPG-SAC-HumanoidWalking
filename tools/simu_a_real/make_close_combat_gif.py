"""
Genera un GIF de evaluacion de un asalto de combate puro (sin fase de
acercamiento) desde los checkpoints de brazos r1/r2.

Usa el propio CloseCombatEnv para simular (antes duplicaba la fisica y la
observacion aqui, y se desincronizaba cada vez que cambiaba el entorno).

Uso:
    python tools/simu_a_real/make_close_combat_gif.py \
        --ckpt_r1 checkpoints/<run>/arm_r1/ckpt_050000.pt \
        --ckpt_r2 checkpoints/<run>/arm_r2/ckpt_050000.pt \
        --step 50000 --out_dir media/
"""
import sys, os, argparse
_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn as nn
import mujoco
import imageio

from close_combat_env import CloseCombatEnv, ARM_ACT_DIM, FIGHT_OBS_DIM

N_STEPS       = 150      # ~15s de asalto (cada step del entorno son 0.1s)
WIDTH, HEIGHT = 800, 480
FPS           = 10       # un frame por step del entorno = tiempo real


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

    def act(self, state):
        with torch.no_grad():
            return torch.tanh(self.mu(self.net(state))) * self.max_action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_r1", required=True)
    parser.add_argument("--ckpt_r2", required=True)
    parser.add_argument("--step",    required=True, type=int)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    actors = []
    for ckpt in (args.ckpt_r1, args.ckpt_r2):
        a = Actor(FIGHT_OBS_DIM, ARM_ACT_DIM, 1.0).to(device)
        a.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["actor"])
        a.eval()
        actors.append(a)

    env = CloseCombatEnv()
    env.model.vis.global_.offwidth  = 1600
    env.model.vis.global_.offheight = 960
    (obs_r1, obs_r2), _ = env.reset(seed=0)

    renderer = mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.25]
    cam.distance  = 1.0
    cam.azimuth   = 90
    cam.elevation = -10

    frames = []
    result = "sin_ko"
    strikes_r1 = strikes_r2 = 0

    for _ in range(N_STEPS):
        a1 = actors[0].act(torch.FloatTensor(obs_r1).unsqueeze(0)).numpy()[0]
        a2 = actors[1].act(torch.FloatTensor(obs_r2).unsqueeze(0)).numpy()[0]
        (obs_r1, obs_r2), _, terminated, truncated, info = env.step(
            np.concatenate([a1, a2]))
        strikes_r1 += info["r1_strikes"]
        strikes_r2 += info["r2_strikes"]

        d = env.data
        cam.lookat[0] = (float(d.qpos[0]) + float(d.qpos[23])) / 2.0
        cam.lookat[1] = (float(d.qpos[1]) + float(d.qpos[24])) / 2.0
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render().copy())

        if terminated:
            if info["r1_fell"] and info["r2_fell"]:
                result = "ambos_caidos"
            elif info["r2_fell"]:
                result = "ko_r1" if info["r2_ko_by_hit"] else "r2_cae_solo"
            else:
                result = "ko_r2" if info["r1_ko_by_hit"] else "r1_cae_solo"
            break
        if truncated:
            break

    renderer.close()
    env.close()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"close_combat_step_{args.step:07d}_{result}_golpes{strikes_r1}-{strikes_r2}.gif")
    imageio.mimsave(out_path, frames, fps=FPS, loop=0)
    print(f"GIF guardado: {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
