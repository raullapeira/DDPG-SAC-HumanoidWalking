"""
Genera un GIF de evaluacion de los dos robots andando desde sus checkpoints.
Uso:
    python tools/simu_a_real/make_fighting_gif.py \
        --ckpt_r1 checkpoints/fighting/r1/ckpt_050000.pt \
        --ckpt_r2 checkpoints/fighting/r2/ckpt_050000.pt \
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
import pathlib

_XML_FIGHT = os.path.join(_HERE, "robot", "configs", "fighting", "alpha_fight.xml")

FRAME_SKIP = 5
ACT_REPEAT = 4
N_STEPS    = 60          # igual que make_walking_gif.py
WIDTH, HEIGHT = 800, 480
FPS = round(1000 / (FRAME_SKIP * 5))   # 40

_LEG_IDX     = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
_ARM_IDX     = np.array([5, 6, 7, 13, 14, 15], dtype=int)
_R1_LEG_CTRL = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
_R2_LEG_CTRL = np.array([16, 17, 18, 19, 20, 24, 25, 26, 27, 28], dtype=int)
_R1_ARM_QPOS = 7  + _ARM_IDX
_R1_ARM_QVEL = 6  + _ARM_IDX
_R2_ARM_QPOS = 30 + _ARM_IDX
_R2_ARM_QVEL = 28 + _ARM_IDX


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


def get_obs(data):
    qpos = data.qpos.flat.copy()
    qvel = data.qvel.flat.copy()
    obs_r1 = np.concatenate([
        qpos[2:7], qvel[0:6],
        qpos[7:23][_LEG_IDX], qvel[6:22][_LEG_IDX],
    ]).astype(np.float32)
    obs_r2 = np.concatenate([
        qpos[25:30], qvel[22:28],
        qpos[30:46][_LEG_IDX], qvel[28:44][_LEG_IDX],
    ]).astype(np.float32)
    return obs_r1, obs_r2


def denorm(action, ctrl_low, ctrl_high):
    half = (ctrl_high - ctrl_low) / 2.0
    return np.clip(action * half, ctrl_low, ctrl_high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_r1",  required=True)
    parser.add_argument("--ckpt_r2",  required=True)
    parser.add_argument("--step",     required=True, type=int)
    parser.add_argument("--out_dir",  required=True)
    args = parser.parse_args()

    device = torch.device("cpu")

    base    = pathlib.Path(_XML_FIGHT).read_text()
    patched = base.replace(
        '<mujoco model="alpha_fight">',
        '<mujoco model="alpha_fight">\n  <visual>\n    <global offwidth="1600" offheight="960"/>\n  </visual>'
    )
    model = mujoco.MjModel.from_xml_string(patched)
    data  = mujoco.MjData(model)

    ctrl_low  = model.actuator_ctrlrange[_R1_LEG_CTRL, 0].copy()
    ctrl_high = model.actuator_ctrlrange[_R1_LEG_CTRL, 1].copy()

    mujoco.mj_resetData(model, data)
    rng = np.random.default_rng(42)
    data.qpos[7:23]  += rng.uniform(-0.03, 0.03, 16)
    data.qpos[30:46] += rng.uniform(-0.03, 0.03, 16)
    mujoco.mj_forward(model, data)

    obs_r1, obs_r2 = get_obs(data)

    # obs_dim = 31, misma arquitectura que walking.py
    actor_r1 = Actor(31, 10, 1.0).to(device)
    actor_r2 = Actor(31, 10, 1.0).to(device)
    ckpt1 = torch.load(args.ckpt_r1, map_location=device, weights_only=False)
    ckpt2 = torch.load(args.ckpt_r2, map_location=device, weights_only=False)
    actor_r1.load_state_dict(ckpt1["actor"]); actor_r1.eval()
    actor_r2.load_state_dict(ckpt2["actor"]); actor_r2.eval()

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.25]
    cam.distance  = 2.5
    cam.azimuth   = 0      # vista frontal (miramos en dirección +X junto a los robots)
    cam.elevation = -15

    frames = []
    result = "ok"

    for step in range(N_STEPS):
        t1 = torch.FloatTensor(obs_r1).unsqueeze(0)
        t2 = torch.FloatTensor(obs_r2).unsqueeze(0)
        a1 = actor_r1.act(t1).numpy()[0]
        a2 = actor_r2.act(t2).numpy()[0]

        ctrl = np.zeros(model.nu, dtype=np.float64)
        ctrl[_R1_LEG_CTRL] = denorm(a1, ctrl_low, ctrl_high)
        ctrl[_R2_LEG_CTRL] = denorm(a2, ctrl_low, ctrl_high)

        for _ in range(ACT_REPEAT):
            data.ctrl[:] = ctrl
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(model, data)
            data.qpos[_R1_ARM_QPOS] = 0.0
            data.qvel[_R1_ARM_QVEL] = 0.0
            data.qpos[_R2_ARM_QPOS] = 0.0
            data.qvel[_R2_ARM_QVEL] = 0.0
            mujoco.mj_forward(model, data)

            # cámara sigue el punto medio entre los dos robots
            mid_x = (float(data.qpos[0]) + float(data.qpos[23])) / 2.0
            mid_y = (float(data.qpos[1]) + float(data.qpos[24])) / 2.0
            cam.lookat[0] = mid_x
            cam.lookat[1] = mid_y
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

        obs_r1, obs_r2 = get_obs(data)

        r1_upz = 1.0 - 2.0 * (float(data.qpos[4])**2 + float(data.qpos[5])**2)
        r2_upz = 1.0 - 2.0 * (float(data.qpos[27])**2 + float(data.qpos[28])**2)
        r1_fell = float(data.qpos[2])  < 0.12 or r1_upz < 0.7
        r2_fell = float(data.qpos[25]) < 0.12 or r2_upz < 0.7
        if r1_fell or r2_fell:
            result = "r2_caido" if r1_fell and not r2_fell else (
                     "r1_caido" if r2_fell and not r1_fell else "ambos_caidos")
            break

    renderer.close()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"fight_step_{args.step:07d}_{result}.gif")
    imageio.mimsave(out_path, frames, fps=FPS, loop=0)
    print(f"GIF guardado: {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
