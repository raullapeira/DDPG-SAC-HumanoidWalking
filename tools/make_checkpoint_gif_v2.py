"""
Genera un GIF lateral de evaluacion del agente v2 en un checkpoint dado.
Uso: python tools/make_checkpoint_gif_v2.py --ckpt <path.pt> --step <N> --out_dir <dir>
"""
import sys, os, argparse
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn as nn
import mujoco
import imageio
import pathlib

_XML_V2 = os.path.join(_HERE, "robot", "reverse_eng_v2", "alpha_single.xml")

FRAME_SKIP = 5
ACT_REPEAT = 4
N_STEPS    = 60
WIDTH, HEIGHT = 640, 640
FPS = round(1000 / (FRAME_SKIP * 5))


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
    parser.add_argument("--ckpt",    required=True)
    parser.add_argument("--step",    required=True, type=int)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    device = torch.device("cpu")

    # Parchear XML con framebuffer grande
    base    = pathlib.Path(_XML_V2).read_text()
    patched = base.replace(
        '<mujoco model="alpha_single">',
        '<mujoco model="alpha_single">\n  <visual>\n    <global offwidth="1280" offheight="1280"/>\n  </visual>'
    )
    model = mujoco.MjModel.from_xml_string(patched)
    data  = mujoco.MjData(model)

    ctrl_low  = model.actuator_ctrlrange[:, 0].copy()
    ctrl_high = model.actuator_ctrlrange[:, 1].copy()

    # Mismos índices que AlphaEnv
    leg_ctrl_idx = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
    leg_obs_idx  = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
    arm_qpos_idx = np.array([7+5, 7+6, 7+7, 7+13, 7+14, 7+15], dtype=int)
    arm_qvel_idx = np.array([6+5, 6+6, 6+7, 6+13, 6+14, 6+15], dtype=int)

    def get_obs():
        qpos = data.qpos.flat.copy()
        qvel = data.qvel.flat.copy()
        leg_qpos = qpos[7:][leg_obs_idx]
        leg_qvel = qvel[6:][leg_obs_idx]
        return np.concatenate([qpos[2:7], qvel[0:6], leg_qpos, leg_qvel]).astype(np.float32)

    def denorm(action):
        low  = ctrl_low[leg_ctrl_idx]
        high = ctrl_high[leg_ctrl_idx]
        half = (high - low) / 2.0
        full = np.zeros(model.nu, dtype=np.float64)
        full[leg_ctrl_idx] = np.clip(action * half, low, high)
        return full

    # Reset
    mujoco.mj_resetData(model, data)
    rng = np.random.default_rng(42)
    data.qpos[7:] += rng.uniform(-0.03, 0.03, model.nq - 7)
    mujoco.mj_forward(model, data)
    obs = get_obs()

    # Cargar actor
    n_obs = obs.shape[0]
    n_act = len(leg_ctrl_idx)   # 10
    actor = Actor(n_obs, n_act, 1.0).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    # Renderer lateral
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.20]
    cam.distance  = 1.0
    cam.azimuth   = 90
    cam.elevation = -10

    frames = []
    fell = False
    for step in range(N_STEPS):
        st = torch.FloatTensor(obs).unsqueeze(0).to(device)
        action = actor.act(st).cpu().numpy()[0]

        for _ in range(ACT_REPEAT):
            data.ctrl[:] = denorm(action)
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(model, data)
            data.qpos[arm_qpos_idx] = 0.0
            data.qvel[arm_qvel_idx] = 0.0
            mujoco.mj_forward(model, data)

            cam.lookat[0] = float(data.qpos[0])
            cam.lookat[1] = float(data.qpos[1])
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

        obs = get_obs()
        if data.qpos[2] < 0.12:
            fell = True
            break

    renderer.close()

    status = "caido" if fell else "ok"
    out_path = os.path.join(args.out_dir, f"step_{args.step:07d}_{status}.gif")
    imageio.mimsave(out_path, frames, fps=FPS, loop=0)
    print(f"GIF saved: {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
