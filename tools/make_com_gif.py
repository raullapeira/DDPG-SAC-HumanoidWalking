"""
Genera un GIF lateral con el centro de masa (COM) del robot pintado en cada frame.

Se muestran dos marcadores:
  - Esfera roja 3D en la posicion real del COM
  - Disco amarillo proyectado en el suelo (z=0) bajo el COM

Uso:
  python tools/make_com_gif.py --ckpt <path.pt> --out <output.gif>
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
N_STEPS    = 80
WIDTH, HEIGHT = 640, 640
FPS = round(1000 / (FRAME_SKIP * 5))

# Índices idénticos a AlphaEnv
_LEG_CTRL_IDX = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
_LEG_OBS_IDX  = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
_ARM_QPOS_IDX = np.array([7+5, 7+6, 7+7, 7+13, 7+14, 7+15], dtype=int)
_ARM_QVEL_IDX = np.array([6+5, 6+6, 6+7, 6+13, 6+14, 6+15], dtype=int)


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
    return np.concatenate([
        qpos[2:7], qvel[0:6],
        qpos[7:][_LEG_OBS_IDX],
        qvel[6:][_LEG_OBS_IDX],
    ]).astype(np.float32)


def denorm(action, ctrl_low, ctrl_high, n_act_total):
    low  = ctrl_low[_LEG_CTRL_IDX]
    high = ctrl_high[_LEG_CTRL_IDX]
    half = (high - low) / 2.0
    full = np.zeros(n_act_total, dtype=np.float64)
    full[_LEG_CTRL_IDX] = np.clip(action * half, low, high)
    return full


def add_com_markers(scene, com_pos):
    """Añade esfera roja en el COM y disco amarillo proyectado en el suelo."""
    identity = np.eye(3).flatten().astype(np.float32)

    # Esfera roja en el COM real
    if scene.ngeom < scene.maxgeom:
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.025, 0.025, 0.025]),
            pos=com_pos.astype(np.float32),
            mat=identity,
            rgba=np.array([1.0, 0.1, 0.1, 0.9], dtype=np.float32),
        )
        scene.ngeom += 1

    # Disco amarillo en el suelo (proyeccion vertical del COM)
    if scene.ngeom < scene.maxgeom:
        g = scene.geoms[scene.ngeom]
        ground_pos = np.array([com_pos[0], com_pos[1], 0.002], dtype=np.float32)
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=np.array([0.03, 0.002, 0.002]),
            pos=ground_pos,
            mat=identity,
            rgba=np.array([1.0, 0.9, 0.0, 0.8], dtype=np.float32),
        )
        scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Ruta al checkpoint .pt")
    parser.add_argument("--out",  required=True, help="Ruta de salida del GIF")
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

    # ID del cuerpo raíz del robot (body 1 = primer cuerpo real, subtree = robot completo)
    root_body_id = 1

    ctrl_low  = model.actuator_ctrlrange[:, 0].copy()
    ctrl_high = model.actuator_ctrlrange[:, 1].copy()

    # Reset
    mujoco.mj_resetData(model, data)
    rng = np.random.default_rng(42)
    data.qpos[7:] += rng.uniform(-0.03, 0.03, model.nq - 7)
    mujoco.mj_forward(model, data)
    obs = get_obs(data)

    # Cargar actor
    n_obs = obs.shape[0]
    n_act = len(_LEG_CTRL_IDX)
    actor = Actor(n_obs, n_act, 1.0).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    print(f"Actor cargado: obs={n_obs}D  act={n_act}D")

    # Renderer
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.20]
    cam.distance  = 1.2
    cam.azimuth   = 90
    cam.elevation = -10

    frames = []
    fell   = False

    for step in range(N_STEPS):
        st     = torch.FloatTensor(obs).unsqueeze(0).to(device)
        action = actor.act(st).cpu().numpy()[0]

        for _ in range(ACT_REPEAT):
            data.ctrl[:] = denorm(action, ctrl_low, ctrl_high, model.nu)
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(model, data)
            data.qpos[_ARM_QPOS_IDX] = 0.0
            data.qvel[_ARM_QVEL_IDX] = 0.0
            mujoco.mj_forward(model, data)

            # COM del robot completo (subtree del body raíz)
            com_pos = data.subtree_com[root_body_id].copy()

            # Seguir al robot con la cámara
            cam.lookat[0] = float(data.qpos[0])
            cam.lookat[1] = float(data.qpos[1])

            renderer.update_scene(data, camera=cam)
            add_com_markers(renderer.scene, com_pos)
            frames.append(renderer.render().copy())

        obs = get_obs(data)
        if data.qpos[2] < 0.12:
            fell = True
            break

    renderer.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimsave(args.out, frames, fps=FPS, loop=0)
    status = "CAIDO" if fell else "OK"
    print(f"GIF guardado: {args.out}  ({len(frames)} frames)  [{status}]")


if __name__ == "__main__":
    main()
