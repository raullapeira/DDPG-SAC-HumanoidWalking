"""
Genera un GIF de evaluacion del pipeline completo (acercamiento -> freno ->
combate) desde los 4 checkpoints (piernas r1/r2, brazos r1/r2).
Uso:
    python tools/simu_a_real/make_full_fight_gif.py \
        --ckpt_leg_r1 checkpoints/2026_09_20_full_fight_acercar_parar_pegar/leg_r1/ckpt_050000.pt \
        --ckpt_leg_r2 checkpoints/2026_09_20_full_fight_acercar_parar_pegar/leg_r2/ckpt_050000.pt \
        --ckpt_arm_r1 checkpoints/2026_09_20_full_fight_acercar_parar_pegar/arm_r1/ckpt_050000.pt \
        --ckpt_arm_r2 checkpoints/2026_09_20_full_fight_acercar_parar_pegar/arm_r2/ckpt_050000.pt \
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

_XML_VERSUS = os.path.join(_HERE, "robot", "configs", "fighting", "alpha_versus.xml")

FRAME_SKIP    = 5
ACT_REPEAT    = 4
N_STEPS       = 150     # cubre acercamiento + combate; se corta antes si hay KO
WIDTH, HEIGHT = 800, 480
FPS = round(1000 / (FRAME_SKIP * 5))   # 40

_FALL_Z    = 0.12
_TILT_Z    = 0.7
_STOP_DIST = 0.20
_TRANSITION_MIN_UPZ = 0.9   # no pasar a combate si alguno ya viene cayendose
_SENSOR_CLIP = 20.0

_LEG_IDX     = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
_ARM_IDX     = np.array([5, 6, 7, 13, 14, 15], dtype=int)
_R1_LEG_CTRL = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=int)
_R1_ARM_CTRL = np.array([5, 6, 7, 13, 14, 15], dtype=int)
_R2_LEG_CTRL = np.array([16, 17, 18, 19, 20, 24, 25, 26, 27, 28], dtype=int)
_R2_ARM_CTRL = np.array([21, 22, 23, 29, 30, 31], dtype=int)

_R1_LEG_QPOS = 7  + _LEG_IDX
_R1_LEG_QVEL = 6  + _LEG_IDX
_R2_LEG_QPOS = 30 + _LEG_IDX
_R2_LEG_QVEL = 28 + _LEG_IDX
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


def get_walk_obs(data):
    qpos = data.qpos.flat.copy()
    qvel = data.qvel.flat.copy()
    r1_x, r1_y, r1_z = qpos[0],  qpos[1],  qpos[2]
    r2_x, r2_y, r2_z = qpos[23], qpos[24], qpos[25]
    obs_r1 = np.concatenate([
        qpos[2:7],  qvel[0:6],
        qpos[7:23][_LEG_IDX],  qvel[6:22][_LEG_IDX],
        [r2_x - r1_x, r2_y - r1_y, r2_z - r1_z],
    ]).astype(np.float32)
    obs_r2 = np.concatenate([
        qpos[25:30], qvel[22:28],
        qpos[30:46][_LEG_IDX], qvel[28:44][_LEG_IDX],
        [r1_x - r2_x, r1_y - r2_y, r1_z - r2_z],
    ]).astype(np.float32)
    return obs_r1, obs_r2


def get_fight_obs(data):
    qpos    = data.qpos.flat.copy()
    qvel    = data.qvel.flat.copy()
    sensors = np.clip(data.sensordata[:4].copy(), 0, _SENSOR_CLIP) / _SENSOR_CLIP
    r1_pos = qpos[0:3];  r2_pos = qpos[23:26]
    r1_vel = qvel[0:3];  r2_vel = qvel[22:25]
    obs_r1 = np.concatenate([
        qpos[7:23][_ARM_IDX],  qvel[6:22][_ARM_IDX],
        qpos[2:7],             qvel[3:6],
        r2_pos - r1_pos,       r2_vel - r1_vel,
        sensors[0:2],
    ]).astype(np.float32)
    obs_r2 = np.concatenate([
        qpos[30:46][_ARM_IDX], qvel[28:44][_ARM_IDX],
        qpos[25:30],           qvel[25:28],
        r1_pos - r2_pos,       r1_vel - r2_vel,
        sensors[2:4],
    ]).astype(np.float32)
    return obs_r1, obs_r2


def denorm_leg(a, low, high):
    half = (high - low) / 2.0
    return np.clip(a * half, low, high)


def denorm_arm(a, low, high):
    center = (high + low) / 2.0
    half   = (high - low) / 2.0
    return np.clip(center + a * half, low, high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_leg_r1", required=True)
    parser.add_argument("--ckpt_leg_r2", required=True)
    parser.add_argument("--ckpt_arm_r1", required=True)
    parser.add_argument("--ckpt_arm_r2", required=True)
    parser.add_argument("--step",        required=True, type=int)
    parser.add_argument("--out_dir",     required=True)
    args = parser.parse_args()

    device = torch.device("cpu")

    base    = pathlib.Path(_XML_VERSUS).read_text()
    patched = base.replace(
        '<mujoco model="alpha_versus">',
        '<mujoco model="alpha_versus">\n  <visual>\n    <global offwidth="1600" offheight="960"/>\n  </visual>'
    )
    model = mujoco.MjModel.from_xml_string(patched)
    data  = mujoco.MjData(model)

    leg_low  = model.actuator_ctrlrange[_R1_LEG_CTRL, 0].copy()
    leg_high = model.actuator_ctrlrange[_R1_LEG_CTRL, 1].copy()
    r1_arm_low  = model.actuator_ctrlrange[_R1_ARM_CTRL, 0].copy()
    r1_arm_high = model.actuator_ctrlrange[_R1_ARM_CTRL, 1].copy()
    r2_arm_low  = model.actuator_ctrlrange[_R2_ARM_CTRL, 0].copy()
    r2_arm_high = model.actuator_ctrlrange[_R2_ARM_CTRL, 1].copy()

    _bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    r1_id = _bid("r1_root")
    r2_id = _bid("r2_root")

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    r1_neutral_leg_qpos = data.qpos[7:23][_LEG_IDX].copy()
    r2_neutral_leg_qpos = data.qpos[30:46][_LEG_IDX].copy()

    rng = np.random.default_rng(42)
    data.qpos[7:23]  += rng.uniform(-0.03, 0.03, 16)
    data.qpos[30:46] += rng.uniform(-0.03, 0.03, 16)
    mujoco.mj_forward(model, data)

    walk_r1, walk_r2   = get_walk_obs(data)
    fight_r1, fight_r2 = get_fight_obs(data)

    leg_actor_r1 = Actor(34, 10, 1.0).to(device)
    leg_actor_r2 = Actor(34, 10, 1.0).to(device)
    arm_actor_r1 = Actor(28, 6,  1.0).to(device)
    arm_actor_r2 = Actor(28, 6,  1.0).to(device)
    leg_actor_r1.load_state_dict(torch.load(args.ckpt_leg_r1, map_location=device, weights_only=False)["actor"]); leg_actor_r1.eval()
    leg_actor_r2.load_state_dict(torch.load(args.ckpt_leg_r2, map_location=device, weights_only=False)["actor"]); leg_actor_r2.eval()
    arm_actor_r1.load_state_dict(torch.load(args.ckpt_arm_r1, map_location=device, weights_only=False)["actor"]); arm_actor_r1.eval()
    arm_actor_r2.load_state_dict(torch.load(args.ckpt_arm_r2, map_location=device, weights_only=False)["actor"]); arm_actor_r2.eval()

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.25]
    cam.distance  = 4.0
    cam.azimuth   = 90
    cam.elevation = -15

    frames = []
    result = "sin_combate"
    fighting = False
    r1_leg_freeze_qpos = r2_leg_freeze_qpos = None
    r1_leg_ctrl_hold = r2_leg_ctrl_hold = None

    for step in range(N_STEPS):
        if fighting:
            t1f = torch.FloatTensor(fight_r1).unsqueeze(0)
            t2f = torch.FloatTensor(fight_r2).unsqueeze(0)
            arm_a1 = arm_actor_r1.act(t1f).numpy()[0]
            arm_a2 = arm_actor_r2.act(t2f).numpy()[0]
        else:
            t1w = torch.FloatTensor(walk_r1).unsqueeze(0)
            t2w = torch.FloatTensor(walk_r2).unsqueeze(0)
            leg_a1 = leg_actor_r1.act(t1w).numpy()[0]
            leg_a2 = leg_actor_r2.act(t2w).numpy()[0]

        for _ in range(ACT_REPEAT):
            ctrl = np.zeros(model.nu, dtype=np.float64)
            if fighting:
                ctrl[_R1_LEG_CTRL] = r1_leg_ctrl_hold
                ctrl[_R2_LEG_CTRL] = r2_leg_ctrl_hold
                ctrl[_R1_ARM_CTRL] = denorm_arm(arm_a1, r1_arm_low, r1_arm_high)
                ctrl[_R2_ARM_CTRL] = denorm_arm(arm_a2, r2_arm_low, r2_arm_high)
            else:
                ctrl[_R1_LEG_CTRL] = denorm_leg(leg_a1, leg_low, leg_high)
                ctrl[_R2_LEG_CTRL] = denorm_leg(leg_a2, leg_low, leg_high)
            data.ctrl[:] = ctrl

            for _ in range(FRAME_SKIP):
                mujoco.mj_step(model, data)
                if fighting:
                    data.qpos[_R1_LEG_QPOS] = r1_leg_freeze_qpos
                    data.qvel[_R1_LEG_QVEL] = 0.0
                    data.qpos[_R2_LEG_QPOS] = r2_leg_freeze_qpos
                    data.qvel[_R2_LEG_QVEL] = 0.0
                else:
                    data.qpos[_R1_ARM_QPOS] = 0.0
                    data.qvel[_R1_ARM_QVEL] = 0.0
                    data.qpos[_R2_ARM_QPOS] = 0.0
                    data.qvel[_R2_ARM_QVEL] = 0.0
                    r1_xy = data.xpos[r1_id][:2]
                    r2_xy = data.xpos[r2_id][:2]
                    if float(np.linalg.norm(r1_xy - r2_xy)) <= _STOP_DIST:
                        data.qvel[0:6]   = 0.0
                        data.qvel[22:28] = 0.0
            mujoco.mj_forward(model, data)

            mid_x = (float(data.qpos[0]) + float(data.qpos[23])) / 2.0
            mid_y = (float(data.qpos[1]) + float(data.qpos[24])) / 2.0
            cam.lookat[0] = mid_x
            cam.lookat[1] = mid_y
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

        walk_r1, walk_r2   = get_walk_obs(data)
        fight_r1, fight_r2 = get_fight_obs(data)

        r1_pos = data.xpos[r1_id]; r2_pos = data.xpos[r2_id]
        dist_2d = float(np.linalg.norm(r1_pos[:2] - r2_pos[:2]))
        r1_upz = 1.0 - 2.0 * (float(data.qpos[4])**2 + float(data.qpos[5])**2)
        r2_upz = 1.0 - 2.0 * (float(data.qpos[27])**2 + float(data.qpos[28])**2)
        r1_fell = float(data.qpos[2])  < _FALL_Z or r1_upz < _TILT_Z
        r2_fell = float(data.qpos[25]) < _FALL_Z or r2_upz < _TILT_Z

        if (not fighting) and dist_2d <= _STOP_DIST \
                and not r1_fell and not r2_fell \
                and r1_upz >= _TRANSITION_MIN_UPZ and r2_upz >= _TRANSITION_MIN_UPZ:
            fighting = True
            result = "combate_sin_ko"
            # Postura neutra fija (no la de media zancada) — mucho mas estable.
            r1_leg_freeze_qpos = r1_neutral_leg_qpos.copy()
            r2_leg_freeze_qpos = r2_neutral_leg_qpos.copy()
            r1_leg_ctrl_hold   = np.clip(r1_leg_freeze_qpos, leg_low, leg_high)
            r2_leg_ctrl_hold   = np.clip(r2_leg_freeze_qpos, leg_low, leg_high)

        if fighting and (r1_fell or r2_fell):
            result = "gana_r2" if r1_fell and not r2_fell else (
                     "gana_r1" if r2_fell and not r1_fell else "ambos_caidos")
            break

    renderer.close()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"full_fight_step_{args.step:07d}_{result}.gif")
    imageio.mimsave(out_path, frames, fps=FPS, loop=0)
    print(f"GIF guardado: {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
