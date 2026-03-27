"""
Runs N policy steps, exports servo values to .aesx (injected into template)
and a slow MP4 video. Both files share the same base name and folder.

Usage:
    python tools/export_aesx_mp4.py \\
        --ckpt checkpoints/sac_alpha/sac2_checkpoint_950000.pt \\
        --n_steps 5 \\
        --template robot/simu_a_real/exportado_por_sw_ubtech.aesx \\
        --out robot/simu_a_real/5_movs_25_03_26
"""
import sys, os, argparse, csv, struct
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import numpy as np
import torch
import torch.nn as nn
import cv2
import mujoco
import gymnasium as gym
from envs.alpha_env import AlphaEnv, _ACTION_REPEAT, _FRAME_SKIP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WIDTH, HEIGHT = 640, 480
PANEL_H = 160                            # altura del panel de servos debajo del render
OUT_H   = HEIGHT + PANEL_H              # altura total del vídeo
CTRL_MS      = _FRAME_SKIP * 5          # 25ms per physics+control step
POLICY_MS    = _ACTION_REPEAT * CTRL_MS  # 100ms per policy step
FPS_RENDER   = round(1000 / CTRL_MS)    # 40 fps (physics rate)
FPS_PLAYBACK = 5                         # 8x slower than real-time

# --- servo conversion constants (from export_servo_csv.py) ---
SERVO_NAMES = [
    "rot_hombro_der",    "ext_hombro_der",    "rot_codo_der",
    "rot_hombro_izq",    "ext_hombro_izq",    "rot_codo_izq",
    "cadera_lat_der",    "cadera_front_der",  "rodilla_der",
    "tobillo_front_der", "tobillo_lat_der",
    "cadera_lat_izq",    "cadera_front_izq",  "rodilla_izq",
    "tobillo_front_izq", "tobillo_lat_izq",
]
NEUTRAL_REAL = [90, 90, 90, 90, 90, 90, 90, 120,
                145, 95, 90, 90, 60, 30, 95, 90]
# Los neutrales v2 están intercambiados entre piernas (der 120, izq 60) respecto a v1
# (der 60, izq 120). Para que la formula neutral-degrees funcione igual que en v1,
# se intercambia qué pierna física recibe los valores de qué pierna MuJoCo:
#   pierna física DER (pos 7-11, índices 6-10) → joints MuJoCo LEFT (0-4)
#   pierna física IZQ (pos 12-16, índices 11-15) → joints MuJoCo RIGHT (8-12)
REAL_TO_MUJOCO = [15, 13, 14, 7, 5, 6, 0, 1, 2, 3, 4, 8, 9, 10, 11, 12]
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
    return int(np.clip(round(neutral - np.degrees(angle_rad)), 0, 255))


# --- aesx injection (from genera.py) ---
def _find_servo_blocks(data):
    offsets = []
    n = len(data)
    for i in range(n - 16 * 8):
        ok = True
        for j in range(16):
            sid = struct.unpack_from("<I", data, i + j*8)[0]
            val = struct.unpack_from("<I", data, i + j*8 + 4)[0]
            if not (1 <= sid <= 16 and 0 <= val <= 300):
                ok = False
                break
        if ok:
            offsets.append(i)
    return offsets


def inject_aesx(template_path, frames, out_path, duration_ms=None):
    """frames: list of (duration_ms, [16 servo values])"""
    with open(template_path, "rb") as f:
        data = bytearray(f.read())
    offsets = _find_servo_blocks(data)
    if not offsets:
        print("❌ No se encontraron bloques de servos en la plantilla")
        return False
    for idx, (frame_dur, servos, *_) in enumerate(frames):
        if idx >= len(offsets):
            break
        base = offsets[idx]
        for i, value in enumerate(servos):
            struct.pack_into("<I", data, base + i*8 + 4, value)
        # Actual playback duration: float32 at base+140 and base+144 (= duration_ms / 10.0)
        # The uint32 at base+128/132 is a fixed field (always 212) — do NOT touch it
        if duration_ms is not None:
            dur_float = float(duration_ms) / 10.0
            flt_offset = base + 16 * 8 + 12  # base + 140
            if flt_offset + 8 <= len(data):
                struct.pack_into("<f", data, flt_offset,     dur_float)
                struct.pack_into("<f", data, flt_offset + 4, dur_float)
    with open(out_path, "wb") as f:
        f.write(data)
    return True


# --- overlay ---
# Short labels for the 16 servos (same order as SERVO_NAMES)
_LABELS = [
    "hom_rot_D", "hom_ext_D", "cod_rot_D",
    "hom_rot_I", "hom_ext_I", "cod_rot_I",
    "cad_lat_D", "cad_frt_D", "rod_D", "tob_frt_D", "tob_lat_D",
    "cad_lat_I", "cad_frt_I", "rod_I", "tob_frt_I", "tob_lat_I",
]

def draw_overlay(frame_bgr, step, n_steps, x_vel, up_z, servo_vals=None):
    h, w = frame_bgr.shape[:2]

    # Top bar
    cv2.rectangle(frame_bgr, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(frame_bgr,
        f"Step {step}/{n_steps}  |  x_vel {x_vel:+.2f} m/s  |  up {up_z:.2f}",
        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)

    # Servo panel: drawn below the render frame, appended as extra rows
    panel = np.zeros((PANEL_H, w, 3), dtype=np.uint8)
    panel[:] = (15, 15, 15)

    if servo_vals is not None:
        col_w = w // 2
        row_h = PANEL_H // 8
        for i, (label, val) in enumerate(zip(_LABELS, servo_vals)):
            col = i // 8
            row = i %  8
            x   = col * col_w + 6
            y   = row * row_h + row_h - 5
            neutral = NEUTRAL_REAL[i]
            color = (0, 200, 100) if abs(val - neutral) <= 20 else (0, 120, 255)
            cv2.putText(panel, f"{label}: {val:3d}",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    return np.vstack([frame_bgr, panel])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     required=True)
    parser.add_argument("--n_steps",  type=int, default=10)
    parser.add_argument("--template", default=os.path.join(
        _HERE, "..", "robot", "simu_a_real", "exportado_por_sw_ubtech.aesx"))
    parser.add_argument("--out",      required=True,
                        help="Base path sin extensión, p.ej. robot/simu_a_real/5_movs_25_03_26")
    parser.add_argument("--xml",      default=None,
                        help="Ruta al XML del modelo (por defecto usa el v1)")
    parser.add_argument("--gif",      action="store_true",
                        help="Genera también un .gif además del .mp4")
    args = parser.parse_args()

    out_aesx = args.out + ".aesx"
    out_mp4  = args.out + ".mp4"
    out_gif  = args.out + ".gif"
    os.makedirs(os.path.dirname(os.path.abspath(out_aesx)), exist_ok=True)

    env = gym.wrappers.TimeLimit(AlphaEnv(xml_path=args.xml), max_episode_steps=10_000)
    s = env.observation_space.shape[0]
    a = env.action_space.shape[0]
    m = float(env.action_space.high[0])

    actor = Actor(s, a, m).to(DEVICE)
    ckpt  = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    raw_env = env.unwrapped
    obs, _ = env.reset(seed=0)

    renderer = mujoco.Renderer(raw_env.model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.20]
    cam.distance  = 1.1
    cam.azimuth   = 150
    cam.elevation = -18

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, FPS_PLAYBACK, (WIDTH, OUT_H))

    servo_frames = [(POLICY_MS, list(NEUTRAL_REAL), [0.0]*16)]  # frame 0: posición de reposo

    # Renderizar frame de reposo (posición inicial)
    renderer.update_scene(raw_env.data, camera=cam)
    frame_rgb = renderer.render().copy()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    total_steps = args.n_steps + 1
    rest_frame = draw_overlay(frame_bgr, 1, total_steps, 0.0, raw_env._torso_up_z(), list(NEUTRAL_REAL))
    for _ in range(FPS_RENDER // _ACTION_REPEAT):  # misma duración que un policy step
        writer.write(rest_frame)

    print(f"Simulando {total_steps} pasos  |  playback {FPS_PLAYBACK}fps ({FPS_RENDER//FPS_PLAYBACK}x lento)...")

    for step in range(args.n_steps):
        st = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        action = actor.act(st).cpu().numpy()[0]

        # Capture servo values for this step (what gets sent to the robot)
        joint_angles = raw_env.data.qpos[7:].copy()
        servo_mujoco = [angle_to_servo(joint_angles[i], NEUTRAL_MUJOCO[i]) for i in range(16)]
        servo_real   = [servo_mujoco[REAL_TO_MUJOCO[i]] for i in range(16)]
        # ángulos en grados reordenados al orden real (mismo reordenamiento que servos)
        angles_mujoco_deg = [np.degrees(joint_angles[i]) for i in range(16)]
        angles_real_deg   = [angles_mujoco_deg[REAL_TO_MUJOCO[i]] for i in range(16)]
        servo_frames.append((POLICY_MS, servo_real, angles_real_deg))

        # Step through each control sub-step, rendering with the sent servo values
        for ctrl_i in range(_ACTION_REPEAT):
            raw_env.data.ctrl[:] = raw_env._denorm_action(action)
            for _ in range(_FRAME_SKIP):
                mujoco.mj_step(raw_env.model, raw_env.data)

            cam.lookat[0] = float(raw_env.data.qpos[0])
            cam.lookat[1] = float(raw_env.data.qpos[1])
            renderer.update_scene(raw_env.data, camera=cam)
            frame_rgb = renderer.render().copy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            x_vel = float(raw_env.data.qvel[0])
            up_z  = raw_env._torso_up_z()
            out_frame = draw_overlay(frame_bgr, step + 2, total_steps, x_vel, up_z, servo_real)
            writer.write(out_frame)

        obs = raw_env._get_obs()
        print(f"  Step {step+1:2d}: {servo_real}")

    writer.release()
    env.close()
    renderer.close()

    # --- export readable table ---
    out_txt = args.out + "_servos.txt"
    col_w = 6
    header = f"{'Paso':<8}" + "".join(f"S{i+1:02d}".rjust(col_w) for i in range(16))
    sep = "-" * len(header)

    lines = ["=== ANGULOS MUJOCO (grados) — inclinacion real de cada joint ===",
             header, sep]
    for idx, (dur, servos, angles_deg) in enumerate(servo_frames):
        label = f"paso {idx+1:2d}"
        row = f"{label:<8}" + "".join(f"{v:+.1f}".rjust(col_w) for v in angles_deg)
        lines.append(row)
    lines += [sep, ""]

    lines += ["=== VALORES SERVO exportados al robot ===",
              header, sep]
    for idx, (dur, servos, angles_deg) in enumerate(servo_frames):
        label = f"paso {idx+1:2d}"
        row = f"{label:<8}" + "".join(str(v).rjust(col_w) for v in servos)
        lines.append(row)
    lines.append(sep)
    neutro_row = f"{'neutro':<8}" + "".join(str(n).rjust(col_w) for n in NEUTRAL_REAL)
    lines.append(neutro_row)

    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ TXT   -> {out_txt}")

    # --- generate .aesx ---
    template = os.path.abspath(args.template)
    if inject_aesx(template, servo_frames, out_aesx, duration_ms=800):
        print(f"✅ AESX  -> {out_aesx}")
    print(f"✅ MP4   -> {out_mp4}")

    # --- generate .gif (optional) ---
    if args.gif:
        import imageio
        cap = cv2.VideoCapture(out_mp4)
        gif_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gif_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        imageio.mimsave(out_gif, gif_frames, fps=FPS_PLAYBACK, loop=0)
        print(f"✅ GIF   -> {out_gif}")


if __name__ == "__main__":
    main()
