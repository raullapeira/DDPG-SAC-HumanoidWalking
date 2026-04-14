"""
Runs N policy steps, exports servo values to .aesx (injected into template)
and a slow MP4 video with:
  - Angulos en GRADOS de cada joint de pierna superpuestos en laterales
  - Pausa de 3 segundos tras cada step (con valores de servo en el panel)
  - Animacion de sub-frames durante cada step

Usage:
    python tools/export_aesx_mp4.py \\
        --ckpt checkpoints/sac_alpha_v13_cog_x2/sac2_checkpoint_700000.pt \\
        --n_steps 10 \\
        --template robot/simu_a_real/exportado_por_sw_ubtech.aesx \\
        --out robot/simu_a_real/v2_depuracion/700k
"""
import sys, os, csv, struct
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
PAUSE_SEGUNDOS = 3
PAUSE_FRAMES   = FPS_PLAYBACK * PAUSE_SEGUNDOS  # frames de pausa tras cada step

# Joints MuJoCo de cada pierna para overlay de grados
JOINTS_DER = [8, 9, 10, 11, 12]  # pierna derecha fisica → MuJoCo RIGHT
JOINTS_IZQ = [0, 1, 2,  3,  4]   # pierna izquierda fisica → MuJoCo LEFT
LEG_LABELS = ["Cad Lat", "Cad Frt", "Rodilla", "Tob Frt", "Tob Lat"]

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
# Signo por servo: -1 → neutral - degrees  |  +1 → neutral + degrees
# Los joints con neutro < 90 (espejo físico) necesitan signo opuesto
SERVO_SIGN = [-1, -1, -1, -1, -1, -1,   # S1-S6  brazos
              -1, -1, -1, -1, -1,         # S7-S11 pierna der
              -1, +1, +1, +1, +1]         # S12-S16 pierna izq
# Los neutrales v2 están intercambiados entre piernas (der 120, izq 60) respecto a v1
# (der 60, izq 120). Para que la formula neutral-degrees funcione igual que en v1,
# se intercambia qué pierna física recibe los valores de qué pierna MuJoCo:
#   pierna física DER (pos 7-11, índices 6-10) → joints MuJoCo LEFT (0-4)
#   pierna física IZQ (pos 12-16, índices 11-15) → joints MuJoCo RIGHT (8-12)
REAL_TO_MUJOCO = [15, 13, 14, 7, 5, 6, 0, 1, 2, 3, 4, 8, 9, 10, 11, 12]
NEUTRAL_MUJOCO = [NEUTRAL_REAL[REAL_TO_MUJOCO.index(i)] for i in range(16)]
# Límite de aducción de cadera lateral (evita cruce físico de piernas).
# L_hip_lat (mj_idx=0): aducción = ángulo negativo → clamp a -_MAX_ADDUCTION_RAD
# R_hip_lat (mj_idx=8): aducción = ángulo positivo → clamp a +_MAX_ADDUCTION_RAD
_MAX_ADDUCTION_DEG = 5.0
_MAX_ADDUCTION_RAD = np.radians(_MAX_ADDUCTION_DEG)


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


def angle_to_servo(angle_rad, neutral, sign=-1):
    return int(np.clip(round(neutral + sign * np.degrees(angle_rad)), 0, 255))


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


# --- overlay de angulos de piernas ---
def draw_leg_angles(frame, deg_der, deg_izq):
    """Superpone angulos en grados en los laterales del frame (modifica in-place)."""
    h, w = frame.shape[:2]
    font, fsize, thick, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1, 25

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 38), (158, 38 + lh*5 + 35), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    overlay = frame.copy()
    cv2.rectangle(overlay, (w-158, 38), (w, 38 + lh*5 + 35), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "PIERNA DER", (5, 56), font, 0.45, (100, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "PIERNA IZQ", (w-155, 56), font, 0.45, (100, 255, 180), 1, cv2.LINE_AA)

    for i, label in enumerate(LEG_LABELS):
        y  = 56 + (i + 1) * lh
        vd = deg_der[i]
        vi = deg_izq[i]
        cd = (0, 200, 80) if abs(vd) < 10 else (0, 140, 255)
        ci = (0, 200, 80) if abs(vi) < 10 else (0, 140, 255)
        cv2.putText(frame, f"{label}:{vd:+.1f}", (5, y), font, fsize, cd, thick, cv2.LINE_AA)
        cv2.putText(frame, f"{label}:{vi:+.1f}", (w-155, y), font, fsize, ci, thick, cv2.LINE_AA)


# --- overlay ---
# Short labels para los 16 servos en orden numérico S1-S16
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

    # Servo panel: 2 columnas de 8, en orden numérico S1-S16
    panel = np.zeros((PANEL_H, w, 3), dtype=np.uint8)
    panel[:] = (15, 15, 15)

    if servo_vals is not None:
        col_w = w // 2
        row_h = PANEL_H // 8
        for i, (label, val) in enumerate(zip(_LABELS, servo_vals)):
            col = i // 8          # columna 0 = S1-S8, columna 1 = S9-S16
            row = i %  8
            x   = col * col_w + 6
            y   = row * row_h + row_h - 5
            neutral = NEUTRAL_REAL[i]
            color = (0, 200, 100) if abs(val - neutral) <= 20 else (0, 120, 255)
            cv2.putText(panel, f"S{i+1:02d} {label}: {val:3d}",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    return np.vstack([frame_bgr, panel])


_CKPT_DIR  = os.path.join(_HERE, "..", "checkpoints", "sac_alpha_v13_cog_x2")
_TEMPLATE  = os.path.join(_HERE, "..", "robot", "simu_a_real", "exportado_por_sw_ubtech.aesx")
_XML       = os.path.join(_HERE, "..", "robot", "reverse_eng_v2", "alpha_single.xml")
_OUT_DIR   = os.path.join(_HERE, "..", "robot", "simu_a_real", "v2_depuracion")
_N_STEPS   = 10


def main():
    # Coge el checkpoint mas reciente de _CKPT_DIR
    ckpts = sorted(
        [f for f in os.listdir(_CKPT_DIR) if f.endswith(".pt")],
        key=lambda f: int(f.split("_")[-1].replace(".pt", ""))
    )
    if not ckpts:
        raise FileNotFoundError(f"No hay checkpoints en {_CKPT_DIR}")
    ckpt_path = os.path.join(_CKPT_DIR, ckpts[-1])
    step_num  = ckpts[-1].split("_")[-1].replace(".pt", "")
    print(f"Usando checkpoint: {ckpts[-1]}")

    os.makedirs(_OUT_DIR, exist_ok=True)
    out_base = os.path.join(_OUT_DIR, f"step_{step_num}")
    out_aesx = out_base + ".aesx"
    out_mp4  = out_base + ".mp4"

    env = gym.wrappers.TimeLimit(AlphaEnv(xml_path=_XML), max_episode_steps=10_000)
    s = env.observation_space.shape[0]
    a = env.action_space.shape[0]
    m = float(env.action_space.high[0])

    actor = Actor(s, a, m).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
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

    renderer_lat = mujoco.Renderer(raw_env.model, height=HEIGHT, width=WIDTH)
    cam_lat = mujoco.MjvCamera()
    cam_lat.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam_lat.lookat[:] = [0.0, 0.0, 0.20]
    cam_lat.distance  = 1.1
    cam_lat.azimuth   = 90
    cam_lat.elevation = -10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer     = cv2.VideoWriter(out_mp4,                    fourcc, FPS_PLAYBACK, (WIDTH, OUT_H))
    writer_lat = cv2.VideoWriter(out_mp4.replace(".mp4", "_lateral.mp4"), fourcc, FPS_PLAYBACK, (WIDTH, HEIGHT))

    servo_frames = [(POLICY_MS, list(NEUTRAL_REAL), [0.0]*16)]  # frame 0: posición de reposo

    # Renderizar frame de reposo (posición inicial)
    renderer.update_scene(raw_env.data, camera=cam)
    frame_rgb = renderer.render().copy()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    total_steps = _N_STEPS + 1
    rest_frame = draw_overlay(frame_bgr, 1, total_steps, 0.0, raw_env._torso_up_z(), list(NEUTRAL_REAL))
    renderer_lat.update_scene(raw_env.data, camera=cam_lat)
    rest_frame_lat = cv2.cvtColor(renderer_lat.render().copy(), cv2.COLOR_RGB2BGR)
    for _ in range(FPS_RENDER // _ACTION_REPEAT):  # misma duración que un policy step
        writer.write(rest_frame)
        writer_lat.write(rest_frame_lat)

    print(f"Simulando {total_steps} pasos  |  playback {FPS_PLAYBACK}fps ({FPS_RENDER//FPS_PLAYBACK}x lento)...")

    for step in range(_N_STEPS):
        st = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        action = actor.act(st).cpu().numpy()[0]

        last_raw_bgr = None
        last_deg_der = last_deg_izq = None
        last_x_vel = last_up_z = 0.0

        # Step through each control sub-step
        for ctrl_i in range(_ACTION_REPEAT):
            full_ctrl = np.zeros(raw_env.model.nu, dtype=np.float64)
            full_ctrl[raw_env._leg_ctrl_idx] = raw_env._denorm_action(action)
            raw_env.data.ctrl[:] = full_ctrl
            for _ in range(_FRAME_SKIP):
                mujoco.mj_step(raw_env.model, raw_env.data)
            raw_env.data.qpos[raw_env._arm_qpos_idx] = 0.0
            raw_env.data.qvel[raw_env._arm_qvel_idx] = 0.0
            mujoco.mj_forward(raw_env.model, raw_env.data)

            qpos_now = raw_env.data.qpos[7:].copy()
            deg_der  = [np.degrees(qpos_now[j]) for j in JOINTS_DER]
            deg_izq  = [np.degrees(qpos_now[j]) for j in JOINTS_IZQ]
            x_vel    = float(raw_env.data.qvel[0])
            up_z     = raw_env._torso_up_z()

            cam.lookat[0] = float(raw_env.data.qpos[0])
            cam.lookat[1] = float(raw_env.data.qpos[1])
            cam_lat.lookat[0] = float(raw_env.data.qpos[0])
            cam_lat.lookat[1] = float(raw_env.data.qpos[1])

            renderer.update_scene(raw_env.data, camera=cam)
            frame_rgb = renderer.render().copy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            last_raw_bgr = frame_bgr.copy()
            last_deg_der, last_deg_izq = deg_der, deg_izq
            last_x_vel, last_up_z = x_vel, up_z

            # Overlay angulos en grados en los laterales
            draw_leg_angles(frame_bgr, deg_der, deg_izq)
            # Panel inferior sin servos durante la animacion
            writer.write(draw_overlay(frame_bgr, step + 2, total_steps, x_vel, up_z))

            renderer_lat.update_scene(raw_env.data, camera=cam_lat)
            frame_lat = cv2.cvtColor(renderer_lat.render().copy(), cv2.COLOR_RGB2BGR)
            writer_lat.write(frame_lat)

        # Capturar angulos DESPUES del step — esto es el estado que vera el robot
        joint_angles = raw_env.data.qpos[7:].copy()
        angles_mujoco_deg = [np.degrees(joint_angles[i]) for i in range(16)]

        servo_real = []
        for i in range(16):
            mj_idx   = REAL_TO_MUJOCO[i]
            ang_rad  = joint_angles[mj_idx]
            # Clampear aducción de caderas laterales para evitar cruce físico de piernas
            if mj_idx == 0:   # L_hip_lat: aducción = negativo
                ang_rad = max(ang_rad, -_MAX_ADDUCTION_RAD)
            elif mj_idx == 8: # R_hip_lat: aducción = positivo
                ang_rad = min(ang_rad, +_MAX_ADDUCTION_RAD)
            ang_deg  = np.degrees(ang_rad)
            neutral  = NEUTRAL_REAL[i]
            sign     = SERVO_SIGN[i]
            raw_val  = neutral + sign * ang_deg
            clipped  = int(np.clip(round(raw_val), 0, 255))
            servo_real.append(clipped)

        angles_real_deg = [angles_mujoco_deg[REAL_TO_MUJOCO[i]] for i in range(16)]
        servo_frames.append((POLICY_MS, servo_real, angles_real_deg))

        obs = raw_env._get_obs()

        # ── DEBUG completo de conversion ─────────────────────────────────────
        print(f"\n{'='*90}")
        print(f"  STEP {step+1:2d}   x_vel={float(raw_env.data.qvel[0]):+.3f} m/s   "
              f"torso_z={float(raw_env.data.qpos[2]):.3f}m")
        print(f"{'='*90}")
        print(f"  {'S#':<5} {'Nombre':<16} {'MjIdx':>5} {'MjJoint':<14} "
              f"{'deg':>7} {'neutro':>7} {'sign':>5} {'raw':>7} {'servo':>6}  {'delta':>6}")
        print(f"  {'-'*90}")
        mj_joint_names = [
            "L_hip_lat","L_hip_fwd","L_knee","L_ank_fwd","L_ank_lat",
            "arm_L1","arm_L2","arm_L3",
            "R_hip_lat","R_hip_fwd","R_knee","R_ank_fwd","R_ank_lat",
            "arm_R1","arm_R2","arm_R3",
        ]
        for i in range(16):
            mj_idx  = REAL_TO_MUJOCO[i]
            ang_deg = angles_mujoco_deg[mj_idx]
            neutral = NEUTRAL_REAL[i]
            sign    = SERVO_SIGN[i]
            raw_val = neutral + sign * ang_deg
            servo   = servo_real[i]
            delta   = servo - neutral
            marker  = "  ←" if abs(delta) > 15 else ""
            print(f"  S{i+1:02d}  {_LABELS[i]:<16} {mj_idx:>5}  {mj_joint_names[mj_idx]:<14} "
                  f"{ang_deg:>+7.1f}  {neutral:>6}  {sign:>+5}  {raw_val:>7.1f}  {servo:>5} "
                  f" {delta:>+5d}{marker}")
        print(f"  {'-'*90}")
        print(f"  Posicion torso: x={float(raw_env.data.qpos[0]):+.3f}  "
              f"y={float(raw_env.data.qpos[1]):+.3f}  z={float(raw_env.data.qpos[2]):.3f}")
        qw,qx,qy,qz = raw_env.data.qpos[3:7]
        roll  = np.degrees(np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy)))
        pitch = np.degrees(np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1)))
        yaw   = np.degrees(np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))
        print(f"  Orientacion torso: roll={roll:+.1f}°  pitch={pitch:+.1f}°  yaw={yaw:+.1f}°")

        # Pausa de 3 segundos con el ultimo frame del step + panel de servos relleno
        pause_bgr = last_raw_bgr.copy()
        draw_leg_angles(pause_bgr, last_deg_der, last_deg_izq)
        pause_frame = draw_overlay(pause_bgr, step + 2, total_steps, last_x_vel, last_up_z, servo_real)
        for _ in range(PAUSE_FRAMES):
            writer.write(pause_frame)

    writer.release()
    writer_lat.release()
    env.close()
    renderer.close()
    renderer_lat.close()

    # --- export readable table ---
    out_txt = out_base + "_servos.txt"
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
    template = os.path.abspath(_TEMPLATE)
    if inject_aesx(template, servo_frames, out_aesx, duration_ms=800):
        print(f"✅ AESX  -> {out_aesx}")
    print(f"✅ MP4   -> {out_mp4}")
    print(f"✅ MP4   -> {out_mp4.replace('.mp4', '_lateral.mp4')}")



if __name__ == "__main__":
    main()
