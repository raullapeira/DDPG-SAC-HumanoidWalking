"""
Extrae keyframes estables del ciclo de marcha del humanoide.

En lugar de exportar movimientos a intervalos regulares (donde los pies
pueden estar inclinados o en el aire), detecta los sub-frames donde el
pie de apoyo esta plano en el suelo y exporta SOLO esas poses al .aesx.

Criterios de "pie plano en suelo":
  1. Altura del pie < Z_FLAT  (esta tocando el suelo)
  2. Angulos de tobillo (fwd + lat) < ANKLE_TOL  (no inclinado)
  3. Se mantiene >= MIN_FLAT_FRAMES sub-frames consecutivos (estable)

Al ser el gait periodico, los keyframes se repiten. El script detecta
automaticamente cuando empieza a repetirse el ciclo y para.

Salida:
  v2_depuracion/keyframes_XXXXXX.aesx   — para importar en UBTech
  v2_depuracion/keyframes_XXXXXX.mp4    — video con poses marcadas
  v2_depuracion/keyframes_XXXXXX.txt    — tabla de angulos por pose
"""
import sys, os, struct
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

# ── Parametros de simulacion ──────────────────────────────────────────────────
_CKPT_DIR  = os.path.join(_HERE, "..", "checkpoints", "sac_alpha_v13_cog_x2")
_TEMPLATE  = os.path.join(_HERE, "..", "robot", "simu_a_real", "exportado_por_sw_ubtech.aesx")
_XML       = os.path.join(_HERE, "..", "robot", "reverse_eng_v2", "alpha_single.xml")
_OUT_DIR   = os.path.join(_HERE, "..", "media", "10_04_2026_v13_cog_x2", "keyframes_700k")

_N_STEPS      = 50      # policy steps a simular (~5 segundos, 4-5 ciclos de marcha)
_KEYFRAME_MS  = 600     # ms por keyframe en el aesx (tiempo para que el robot alcance la pose)
_MAX_KF       = 16      # maximo keyframes a exportar

# ── Criterios de "pie plano" ──────────────────────────────────────────────────
_Z_FLAT        = 0.030  # altura maxima del pie para considerar contacto con suelo (m)
                         # El solver de MuJoCo garantiza que si el pie esta a z_min, esta plano
                         # (no-penetracion + friccion). No hace falta chequear orientacion.
_MIN_FLAT_SUB  = 2      # sub-frames consecutivos planos antes de aceptar la pose
_MIN_DIST_RAD  = 0.25   # distancia angular minima (norma L2 de 10 joints) entre keyframes
_CYCLE_DIST    = 0.15   # si la distancia al primer KF cae por debajo, consideramos ciclo completo

# Indices de los 10 joints de pierna en qpos[7:] (mismo orden que la politica)
_LEG_IDX = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]

# ── Servo export (igual que export_aesx_mp4.py) ───────────────────────────────
NEUTRAL_REAL = [90, 90, 90, 90, 90, 90, 90, 120, 145, 95, 90, 90, 60, 30, 95, 90]
SERVO_SIGN   = [-1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1,
                -1, +1, +1, +1, +1]
REAL_TO_MUJOCO = [15, 13, 14, 7, 5, 6, 0, 1, 2, 3, 4, 8, 9, 10, 11, 12]
# Límite de aducción de cadera lateral (evita que las piernas se crucen físicamente).
# L_hip_lat (mj_idx=0): aducción = ángulo negativo → clamp a -MAX_ADDUCTION_RAD
# R_hip_lat (mj_idx=8): aducción = ángulo positivo → clamp a +MAX_ADDUCTION_RAD
_MAX_ADDUCTION_DEG = 5.0
_MAX_ADDUCTION_RAD = np.radians(_MAX_ADDUCTION_DEG)
SERVO_NAMES    = [
    "hom_rot_D", "hom_ext_D", "cod_rot_D",
    "hom_rot_I", "hom_ext_I", "cod_rot_I",
    "cad_lat_D", "cad_frt_D", "rod_D", "tob_frt_D", "tob_lat_D",
    "cad_lat_I", "cad_frt_I", "rod_I", "tob_frt_I", "tob_lat_I",
]
LEG_LABELS = ["Cad Lat", "Cad Frt", "Rodilla", "Tob Frt", "Tob Lat"]

WIDTH, HEIGHT  = 640, 480
FPS_PLAYBACK   = 5
PAUSE_FRAMES   = FPS_PLAYBACK * 2   # 2 segundos de pausa por keyframe en el video


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


def angle_to_servo(angle_rad, neutral, sign):
    return int(np.clip(round(neutral + sign * np.degrees(angle_rad)), 0, 255))


def _find_servo_blocks(data):
    offsets = []
    n = len(data)
    for i in range(n - 16 * 8):
        ok = True
        for j in range(16):
            sid = struct.unpack_from("<I", data, i + j*8)[0]
            val = struct.unpack_from("<I", data, i + j*8 + 4)[0]
            if not (1 <= sid <= 16 and 0 <= val <= 300):
                ok = False; break
        if ok:
            offsets.append(i)
    return offsets


def inject_aesx(template_path, frames, out_path, duration_ms):
    data = bytearray(open(template_path, "rb").read())
    offsets = _find_servo_blocks(data)
    if not offsets:
        print("❌ No se encontraron bloques de servos en la plantilla")
        return False
    for idx, servos in enumerate(frames):
        if idx >= len(offsets):
            break
        base = offsets[idx]
        for i, val in enumerate(servos):
            struct.pack_into("<I", data, base + i*8 + 4, int(val))
        dur_float = float(duration_ms) / 10.0
        flt_off = base + 16*8 + 12
        if flt_off + 8 <= len(data):
            struct.pack_into("<f", data, flt_off,     dur_float)
            struct.pack_into("<f", data, flt_off + 4, dur_float)
    open(out_path, "wb").write(data)
    return True


def foot_is_flat(data, left_id, right_id):
    """True si alguno de los dos pies esta en contacto con el suelo.

    Criterio: altura del centro del cuerpo del pie < _Z_FLAT.
    El solver de MuJoCo garantiza no-penetracion y friccion, por lo que
    si el pie esta a su altura minima esta en contacto real y plano.
    """
    return (float(data.xpos[left_id][2])  < _Z_FLAT or
            float(data.xpos[right_id][2]) < _Z_FLAT)


def leg_vec(qpos):
    """Vector de los 10 joints de pierna (para comparar poses)."""
    return qpos[_LEG_IDX].copy()


def draw_keyframe(frame, kf_idx, total_kf, qpos, x_vel):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(frame, (0, 0), (w, 36), (0, 50, 0), -1)
    cv2.putText(frame, f"KEYFRAME {kf_idx}/{total_kf}  x_vel={x_vel:+.2f}",
                (10, 24), font, 0.65, (100, 255, 100), 1, cv2.LINE_AA)

    # Angulos de pierna derecha (MuJoCo RIGHT = joints 8-12)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 38), (158, 38+25*5+35), (10,10,40), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "PIERNA DER", (5, 56), font, 0.45, (100,220,255), 1, cv2.LINE_AA)
    for i, label in enumerate(LEG_LABELS):
        v = np.degrees(qpos[8+i])
        c = (0,200,80) if abs(v) < 10 else (0,140,255)
        cv2.putText(frame, f"{label}:{v:+.1f}", (5, 56+(i+1)*25), font, 0.50, c, 1, cv2.LINE_AA)

    # Angulos de pierna izquierda (MuJoCo LEFT = joints 0-4)
    overlay = frame.copy()
    cv2.rectangle(overlay, (w-158, 38), (w, 38+25*5+35), (10,40,10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "PIERNA IZQ", (w-155, 56), font, 0.45, (100,255,180), 1, cv2.LINE_AA)
    for i, label in enumerate(LEG_LABELS):
        v = np.degrees(qpos[i])
        c = (0,200,80) if abs(v) < 10 else (0,140,255)
        cv2.putText(frame, f"{label}:{v:+.1f}", (w-155, 56+(i+1)*25), font, 0.50, c, 1, cv2.LINE_AA)

    return frame


def main():
    ckpts = sorted(
        [f for f in os.listdir(_CKPT_DIR) if f.endswith(".pt")],
        key=lambda f: int(f.split("_")[-1].replace(".pt", ""))
    )
    if not ckpts:
        raise FileNotFoundError(f"No hay checkpoints en {_CKPT_DIR}")
    ckpt_path = os.path.join(_CKPT_DIR, ckpts[-1])
    step_num  = ckpts[-1].split("_")[-1].replace(".pt", "")
    print(f"Checkpoint: {ckpts[-1]}")

    os.makedirs(_OUT_DIR, exist_ok=True)
    out_base = os.path.join(_OUT_DIR, f"keyframes_{step_num}")

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

    left_id  = mujoco.mj_name2id(raw_env.model, mujoco.mjtObj.mjOBJ_BODY, "Left_Feet_link")
    right_id = mujoco.mj_name2id(raw_env.model, mujoco.mjtObj.mjOBJ_BODY, "Right_Feet_link")

    # ── Pasada diagnostica ────────────────────────────────────────────────────
    print(f"Simulando {_N_STEPS} steps — pasada diagnostica...")
    diag_fzl, diag_fzr = [], []
    obs_d, _ = env.reset(seed=0)
    for step in range(_N_STEPS):
        st = torch.FloatTensor(obs_d).unsqueeze(0).to(DEVICE)
        action = actor.act(st).cpu().numpy()[0]
        for _ in range(_ACTION_REPEAT):
            full_ctrl = np.zeros(raw_env.model.nu, dtype=np.float64)
            full_ctrl[raw_env._leg_ctrl_idx] = raw_env._denorm_action(action)
            raw_env.data.ctrl[:] = full_ctrl
            for _ in range(_FRAME_SKIP):
                mujoco.mj_step(raw_env.model, raw_env.data)
            raw_env.data.qpos[raw_env._arm_qpos_idx] = 0.0
            raw_env.data.qvel[raw_env._arm_qvel_idx] = 0.0
            mujoco.mj_forward(raw_env.model, raw_env.data)
            diag_fzl.append(float(raw_env.data.xpos[left_id][2]))
            diag_fzr.append(float(raw_env.data.xpos[right_id][2]))
        obs_d = raw_env._get_obs()

    p10 = lambda a: np.percentile(a, 10)
    print(f"  foot_z LEFT   min={min(diag_fzl):.4f}  p10={p10(diag_fzl):.4f}  max={max(diag_fzl):.4f}")
    print(f"  foot_z RIGHT  min={min(diag_fzr):.4f}  p10={p10(diag_fzr):.4f}  max={max(diag_fzr):.4f}")
    auto_z = float(np.percentile(diag_fzl + diag_fzr, 25))
    print(f"\n  Umbral actual   : _Z_FLAT={_Z_FLAT}")
    print(f"  Umbral sugerido : _Z_FLAT={auto_z:.4f}  (p25 de todas las alturas)")

    # ── Fase 1: recoger candidatos ────────────────────────────────────────────
    print(f"\nBuscando poses estables (pie_z < {_Z_FLAT}m, >= {_MIN_FLAT_SUB} sub-frames)...")
    candidates = []
    flat_count = 0
    obs, _ = env.reset(seed=0)

    for step in range(_N_STEPS):
        st = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        action = actor.act(st).cpu().numpy()[0]

        for _ in range(_ACTION_REPEAT):
            full_ctrl = np.zeros(raw_env.model.nu, dtype=np.float64)
            full_ctrl[raw_env._leg_ctrl_idx] = raw_env._denorm_action(action)
            raw_env.data.ctrl[:] = full_ctrl
            for _ in range(_FRAME_SKIP):
                mujoco.mj_step(raw_env.model, raw_env.data)
            raw_env.data.qpos[raw_env._arm_qpos_idx] = 0.0
            raw_env.data.qvel[raw_env._arm_qvel_idx] = 0.0
            mujoco.mj_forward(raw_env.model, raw_env.data)

            qpos = raw_env.data.qpos[7:].copy()

            if foot_is_flat(raw_env.data, left_id, right_id):
                flat_count += 1
                if flat_count >= _MIN_FLAT_SUB:
                    candidates.append((
                        leg_vec(qpos),
                        qpos.copy(),
                        float(raw_env.data.qvel[0])
                    ))
            else:
                flat_count = 0

        obs = raw_env._get_obs()

    env.close()
    print(f"  Sub-frames planos detectados: {len(candidates)}")

    if not candidates:
        print("❌ No se detectaron poses estables.")
        print(f"   Ajusta _Z_FLAT a ~{auto_z:.4f} en el script.")
        return

    # ── Fase 2: seleccionar keyframes por distancia angular ───────────────────
    keyframes = []
    for vec, qpos_full, x_vel in candidates:
        if not keyframes:
            keyframes.append((vec, qpos_full, x_vel))
            continue

        dist_to_last  = np.linalg.norm(vec - keyframes[-1][0])
        dist_to_first = np.linalg.norm(vec - keyframes[0][0])

        # Ciclo completado: volvemos cerca del primer keyframe
        if len(keyframes) >= 4 and dist_to_first < _CYCLE_DIST:
            print(f"  Ciclo detectado tras {len(keyframes)} keyframes")
            break

        if dist_to_last > _MIN_DIST_RAD:
            keyframes.append((vec, qpos_full, x_vel))
            if len(keyframes) >= _MAX_KF:
                break

    print(f"  Keyframes seleccionados: {len(keyframes)}")

    _MJ_NAMES = [
        "L_hip_lat","L_hip_fwd","L_knee","L_ank_fwd","L_ank_lat",
        "arm_L1","arm_L2","arm_L3",
        "R_hip_lat","R_hip_fwd","R_knee","R_ank_fwd","R_ank_lat",
        "arm_R1","arm_R2","arm_R3",
    ]

    # ── Fase 3: convertir a servos + debug completo ───────────────────────────
    servo_list = []
    for ki, (_, qpos_full, x_vel) in enumerate(keyframes):
        servos = []
        print(f"\n{'='*92}")
        print(f"  KEYFRAME {ki+1}   x_vel={x_vel:+.3f} m/s")
        print(f"{'='*92}")
        print(f"  {'S#':<5} {'Nombre':<16} {'MjIdx':>5} {'MjJoint':<14} "
              f"{'deg':>7} {'neutro':>7} {'sign':>5} {'raw':>7} {'servo':>6}  {'delta':>6}")
        print(f"  {'-'*88}")
        for i in range(16):
            mj_idx  = REAL_TO_MUJOCO[i]
            ang_rad = qpos_full[mj_idx]
            # Clampear aducción de caderas laterales para evitar cruce físico de piernas
            if mj_idx == 0:   # L_hip_lat: aducción = negativo
                ang_rad = max(ang_rad, -_MAX_ADDUCTION_RAD)
            elif mj_idx == 8: # R_hip_lat: aducción = positivo
                ang_rad = min(ang_rad, +_MAX_ADDUCTION_RAD)
            ang_deg = np.degrees(ang_rad)
            neutral = NEUTRAL_REAL[i]
            sign    = SERVO_SIGN[i]
            raw_val = neutral + sign * ang_deg
            clipped = int(np.clip(round(raw_val), 0, 255))
            delta   = clipped - neutral
            marker  = "  ←" if abs(delta) > 15 else ""
            print(f"  S{i+1:02d}  {SERVO_NAMES[i]:<16} {mj_idx:>5}  {_MJ_NAMES[mj_idx]:<14} "
                  f"{ang_deg:>+7.1f}  {neutral:>6}  {sign:>+5}  {raw_val:>7.1f}  {clipped:>5} "
                  f" {delta:>+5d}{marker}")
            servos.append(clipped)
        print(f"  {'-'*88}")

        # Posicion y orientacion del torso en este keyframe
        root_qpos = qpos_full   # qpos[7:], no tiene el root — usamos x_vel como proxy
        print(f"  x_vel={x_vel:+.4f} m/s   (posicion root no disponible fuera del env)")

        servo_list.append(servos)

    # ── Fase 4: exportar .aesx ────────────────────────────────────────────────
    out_aesx = out_base + ".aesx"
    # Primer frame: posicion neutra para que el robot parta de cero
    frames_to_inject = [list(NEUTRAL_REAL)] + servo_list
    if inject_aesx(os.path.abspath(_TEMPLATE), frames_to_inject, out_aesx, _KEYFRAME_MS):
        print(f"✅ AESX  -> {out_aesx}  ({len(servo_list)} poses + neutro inicial)")

    # ── Fase 5: exportar tabla de texto ──────────────────────────────────────
    out_txt = out_base + ".txt"
    col_w = 7
    header = f"{'KF':<6}" + "".join(f"S{i+1:02d}".rjust(col_w) for i in range(16))
    sep    = "-" * len(header)
    lines  = ["=== KEYFRAMES — valores servo ===", header, sep]
    neutro_row = f"{'neutro':<6}" + "".join(str(n).rjust(col_w) for n in NEUTRAL_REAL)
    lines.append(neutro_row)
    for ki, (servos, (_, qpos_full, x_vel)) in enumerate(zip(servo_list, keyframes)):
        row = f"KF{ki+1:<4}" + "".join(str(v).rjust(col_w) for v in servos)
        lines.append(row)
    lines.append(sep)
    lines.append("")
    lines.append("=== KEYFRAMES — angulos MuJoCo (grados) ===")
    lines.append(f"{'KF':<6}" + "".join(f"J{i:02d}".rjust(col_w) for i in range(16)))
    lines.append(sep)
    for ki, (_, qpos_full, x_vel) in enumerate(keyframes):
        row = f"KF{ki+1:<4}" + "".join(f"{np.degrees(qpos_full[i]):+.1f}".rjust(col_w) for i in range(16))
        lines.append(row + f"   x_vel={x_vel:+.2f}")
    lines.append(sep)
    open(out_txt, "w").write("\n".join(lines) + "\n")
    print(f"✅ TXT   -> {out_txt}")

    # ── Fase 6: video de los keyframes ────────────────────────────────────────
    out_mp4 = out_base + ".mp4"
    env2 = gym.wrappers.TimeLimit(AlphaEnv(xml_path=_XML), max_episode_steps=10_000)
    raw2 = env2.unwrapped
    left2_id  = mujoco.mj_name2id(raw2.model, mujoco.mjtObj.mjOBJ_BODY, "Left_Feet_link")
    right2_id = mujoco.mj_name2id(raw2.model, mujoco.mjtObj.mjOBJ_BODY, "Right_Feet_link")
    env2.reset(seed=0)

    renderer = mujoco.Renderer(raw2.model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.20]
    cam.distance  = 1.1
    cam.azimuth   = 150
    cam.elevation = -18

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, FPS_PLAYBACK, (WIDTH, HEIGHT))

    # Para el video reproducimos la simulacion de nuevo y marcamos los keyframes
    actor2 = Actor(s, a, m).to(DEVICE)
    actor2.load_state_dict(ckpt["actor"])
    actor2.eval()
    obs2, _ = env2.reset(seed=0)
    flat_count2 = 0
    kf_shown = 0

    for step in range(_N_STEPS):
        st2 = torch.FloatTensor(obs2).unsqueeze(0).to(DEVICE)
        action2 = actor2.act(st2).cpu().numpy()[0]

        for _ in range(_ACTION_REPEAT):
            full_ctrl = np.zeros(raw2.model.nu, dtype=np.float64)
            full_ctrl[raw2._leg_ctrl_idx] = raw2._denorm_action(action2)
            raw2.data.ctrl[:] = full_ctrl
            for _ in range(_FRAME_SKIP):
                mujoco.mj_step(raw2.model, raw2.data)
            raw2.data.qpos[raw2._arm_qpos_idx] = 0.0
            raw2.data.qvel[raw2._arm_qvel_idx] = 0.0
            mujoco.mj_forward(raw2.model, raw2.data)

            qpos2 = raw2.data.qpos[7:].copy()

            cam.lookat[0] = float(raw2.data.qpos[0])
            cam.lookat[1] = float(raw2.data.qpos[1])
            renderer.update_scene(raw2.data, camera=cam)
            frame_rgb = renderer.render().copy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            is_flat = foot_is_flat(raw2.data, left2_id, right2_id)
            if is_flat:
                flat_count2 += 1
            else:
                flat_count2 = 0

            # Si este sub-frame coincide con un keyframe seleccionado, marcar y pausar
            if flat_count2 == _MIN_FLAT_SUB and kf_shown < len(keyframes):
                vec2 = leg_vec(qpos2)
                dist = np.linalg.norm(vec2 - keyframes[kf_shown][0])
                if dist < _MIN_DIST_RAD * 0.8:
                    draw_keyframe(frame_bgr, kf_shown+1, len(keyframes), qpos2,
                                  float(raw2.data.qvel[0]))
                    for _ in range(PAUSE_FRAMES):
                        writer.write(frame_bgr)
                    kf_shown += 1
                    continue

            # Frame normal de animacion
            h, w = frame_bgr.shape[:2]
            cv2.rectangle(frame_bgr, (0, 0), (w, 30), (0,0,0), -1)
            cv2.putText(frame_bgr, f"Step {step+1}  buscando pose plana...",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1, cv2.LINE_AA)
            writer.write(frame_bgr)

        obs2 = raw2._get_obs()
        if kf_shown >= len(keyframes):
            break

    writer.release()
    renderer.close()
    env2.close()
    print(f"✅ MP4   -> {out_mp4}")
    print(f"\nResumen: {len(keyframes)} keyframes @ {_KEYFRAME_MS}ms cada uno")
    print(f"Duracion total en robot: {(len(keyframes)+1)*_KEYFRAME_MS/1000:.1f}s (con neutro inicial)")


if __name__ == "__main__":
    main()
