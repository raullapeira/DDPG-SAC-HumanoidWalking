"""
Test3: pone el robot en MuJoCo con ambas caderas frontales a 90 grados hacia adelante,
lee los angulos del simulador, los convierte a servo con el mismo codigo que
export_aesx_mp4.py y exporta .aesx + MP4 + TXT para comparar con el software UBTech.

Uso:
    python tools/test3_mujoco_lift_legs.py \
        --out robot/simu_a_real/test3_mujoco_lift_legs
"""
import sys, os, argparse, struct
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import numpy as np
import cv2
import mujoco

_XML_V2 = os.path.join(_HERE, "..", "robot", "reverse_eng_v2", "alpha_single.xml")

# --- mismas constantes que export_aesx_mp4.py ---
NEUTRAL_REAL = [90, 90, 90, 90, 90, 90, 90, 120,
                145, 95, 90, 90, 60, 30, 95, 90]
SERVO_SIGN   = [-1, -1, -1, -1, -1, -1,
                -1, -1, -1, +1, +1,
                +1, +1, +1, -1, -1]
REAL_TO_MUJOCO = [15, 13, 14, 7, 5, 6, 0, 1, 2, 3, 4, 8, 9, 10, 11, 12]

POLICY_MS    = 800
WIDTH, HEIGHT = 640, 480
FPS_PLAYBACK = 5


def angle_to_servo(angle_rad, neutral, sign=-1):
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
                ok = False
                break
        if ok:
            offsets.append(i)
    return offsets


def inject_aesx(template_path, frames, out_path, duration_ms=None):
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
        if duration_ms is not None:
            dur_float = float(duration_ms) / 10.0
            flt_offset = base + 16 * 8 + 12
            if flt_offset + 8 <= len(data):
                struct.pack_into("<f", data, flt_offset,     dur_float)
                struct.pack_into("<f", data, flt_offset + 4, dur_float)
    with open(out_path, "wb") as f:
        f.write(data)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(
        _HERE, "..", "robot", "simu_a_real", "test3_mujoco_lift_legs"))
    parser.add_argument("--template", default=os.path.join(
        _HERE, "..", "robot", "simu_a_real", "exportado_por_sw_ubtech.aesx"))
    args = parser.parse_args()

    out_aesx = args.out + ".aesx"
    out_mp4  = args.out + ".mp4"
    out_txt  = args.out + "_servos.txt"
    os.makedirs(os.path.dirname(os.path.abspath(out_aesx)), exist_ok=True)

    model = mujoco.MjModel.from_xml_path(_XML_V2)
    data  = mujoco.MjData(model)

    # --- Frame 1: posicion neutra ---
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    joint_angles_neutro = data.qpos[7:].copy()
    servo_neutro = [angle_to_servo(joint_angles_neutro[REAL_TO_MUJOCO[i]],
                                   NEUTRAL_REAL[i], SERVO_SIGN[i]) for i in range(16)]
    angles_deg_neutro = [np.degrees(joint_angles_neutro[REAL_TO_MUJOCO[i]]) for i in range(16)]

    # --- Frame 2: ambas caderas frontales a 90 grados ---
    mujoco.mj_resetData(model, data)
    # S08 cadera_front_der -> REAL_TO_MUJOCO[7] = joint MuJoCo 1
    # S13 cadera_front_izq -> REAL_TO_MUJOCO[12] = joint MuJoCo 9
    # sign S08=-1: servo = 120 - deg => para servo=30, deg=+90 => angle=+pi/2
    # sign S13=+1: servo = 60 + deg => para servo=150, deg=+90 => angle=+pi/2
    data.qpos[7 + 1] = np.pi / 2   # cadera frontal MuJoCo joint 1
    data.qpos[7 + 9] = np.pi / 2   # cadera frontal MuJoCo joint 9
    mujoco.mj_forward(model, data)
    joint_angles_lift = data.qpos[7:].copy()
    servo_lift = [angle_to_servo(joint_angles_lift[REAL_TO_MUJOCO[i]],
                                 NEUTRAL_REAL[i], SERVO_SIGN[i]) for i in range(16)]
    angles_deg_lift = [np.degrees(joint_angles_lift[REAL_TO_MUJOCO[i]]) for i in range(16)]

    frames = [
        (POLICY_MS, servo_neutro, angles_deg_neutro),
        (POLICY_MS, servo_lift,   angles_deg_lift),
    ]

    # --- MP4 ---
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.20]
    cam.distance  = 1.4
    cam.azimuth   = 150
    cam.elevation = -18

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, FPS_PLAYBACK, (WIDTH, HEIGHT))

    # neutro: varios frames para que se vea
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    frame = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
    for _ in range(FPS_PLAYBACK * 2):
        writer.write(frame)

    # lift: varios frames
    mujoco.mj_resetData(model, data)
    data.qpos[7 + 1] = np.pi / 2
    data.qpos[7 + 9] = np.pi / 2
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    frame = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
    for _ in range(FPS_PLAYBACK * 2):
        writer.write(frame)

    writer.release()
    renderer.close()
    print(f"✅ MP4  -> {out_mp4}")

    # --- TXT ---
    col_w  = 6
    header = f"{'Paso':<8}" + "".join(f"S{i+1:02d}".rjust(col_w) for i in range(16))
    sep    = "-" * len(header)

    lines = ["=== ANGULOS MUJOCO (grados) ===", header, sep,
             f"{'neutro':<8}" + "".join(f"{v:+.1f}".rjust(col_w) for v in angles_deg_neutro),
             f"{'lift':<8}"   + "".join(f"{v:+.1f}".rjust(col_w) for v in angles_deg_lift),
             sep, "",
             "=== VALORES SERVO ===", header, sep,
             f"{'neutro':<8}" + "".join(str(v).rjust(col_w) for v in servo_neutro),
             f"{'lift':<8}"   + "".join(str(v).rjust(col_w) for v in servo_lift),
             sep,
             f"{'neutro_ref':<8}" + "".join(str(n).rjust(col_w) for n in NEUTRAL_REAL)]
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ TXT  -> {out_txt}")

    # --- AESX ---
    if inject_aesx(os.path.abspath(args.template), frames, out_aesx, duration_ms=POLICY_MS):
        print(f"✅ AESX -> {out_aesx}")

    print(f"\nEsperado:")
    print(f"  S08 cadera_front_der: {servo_neutro[7]} -> {servo_lift[7]}  (deberia ser 120 -> 30)")
    print(f"  S13 cadera_front_izq: {servo_neutro[12]} -> {servo_lift[12]}  (deberia ser 60 -> 150)")


if __name__ == "__main__":
    main()
