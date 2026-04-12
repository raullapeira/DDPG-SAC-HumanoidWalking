"""
Test de mapeo joint-a-servo: mueve UN solo joint a un angulo fijo,
el resto en neutro. Genera .aesx y muestra el servo resultante.

Uso:
  python tools/test_joint_mapping.py --joint 7 --deg 20 --template robot/simu_a_real/exportado_por_sw_ubtech.aesx

  --joint  : numero de joint MuJoCo (0-15, en el orden de qpos[7:])
  --deg    : angulo en grados a aplicar (puede ser negativo)
  --out    : prefijo de salida (por defecto robot/simu_a_real/test_joint)
"""
import sys, os, argparse, struct
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import numpy as np
import mujoco
import pathlib

_XML_V2 = os.path.join(_HERE, "..", "robot", "reverse_eng_v2", "alpha_single.xml")

REAL_TO_MUJOCO = [15, 13, 14, 7, 5, 6, 0, 1, 2, 3, 4, 8, 9, 10, 11, 12]
NEUTRAL_REAL   = [90, 90, 90, 90, 90, 90, 90, 120, 145, 95, 90, 90, 60, 30, 95, 90]
SERVO_SIGN     = [-1, -1, -1, -1, -1, -1,
                  +1, +1, -1, -1, -1,
                  -1, -1, +1, +1, +1]
SERVO_NAMES    = [
    "hom_rot_D","hom_ext_D","cod_rot_D",
    "hom_rot_I","hom_ext_I","cod_rot_I",
    "cad_lat_D","cad_frt_D","rod_D","tob_frt_D","tob_lat_D",
    "cad_lat_I","cad_frt_I","rod_I","tob_frt_I","tob_lat_I",
]
MUJOCO_JOINT_NAMES = [
    "L_hip_lat","L_hip_fwd","L_knee","L_ank_fwd","L_ank_lat",
    "arm_L1","arm_L2","arm_L3",
    "R_hip_lat","R_hip_fwd","R_knee","R_ank_fwd","R_ank_lat",
    "arm_R1","arm_R2","arm_R3",
]


def angle_to_servo(angle_rad, neutral, sign=-1):
    return int(np.clip(round(neutral + sign * np.degrees(angle_rad)), 0, 255))


def _find_servo_blocks(data):
    """Mismo algoritmo que export_aesx_mp4.py — busca bloques uint32 sid+val."""
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


def inject_aesx(template_path, out_path, frames):
    """Mismo formato que export_aesx_mp4.py — escribe uint32 en offset+4."""
    data = bytearray(pathlib.Path(template_path).read_bytes())
    offsets = _find_servo_blocks(data)
    if not offsets:
        print("❌ No se encontraron bloques de servos en la plantilla")
        return
    print(f"   Bloques en plantilla: {len(offsets)}  |  frames a escribir: {len(frames)}")
    for idx, (frame_dur, servos, *_) in enumerate(frames):
        if idx >= len(offsets):
            print(f"   ⚠️  Frame {idx+1} ignorado — la plantilla solo tiene {len(offsets)} bloques")
            break
        base = offsets[idx]
        for i, value in enumerate(servos):
            struct.pack_into("<I", data, base + i*8 + 4, int(value))
    pathlib.Path(out_path).write_bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint",    type=int,   required=True, help="Índice MuJoCo joint (0-15)")
    parser.add_argument("--deg",      type=float, required=True, help="Ángulo en grados")
    parser.add_argument("--template", required=True)
    parser.add_argument("--out",      default="robot/simu_a_real/test_joint")
    args = parser.parse_args()

    assert 0 <= args.joint <= 15, "joint debe estar entre 0 y 15"

    # Construir joint_angles: todos a 0 salvo el elegido
    joint_angles = np.zeros(16)
    joint_angles[args.joint] = np.radians(args.deg)

    NEUTRAL_MUJOCO = [NEUTRAL_REAL[REAL_TO_MUJOCO.index(i)] for i in range(16)]

    servo_real = [angle_to_servo(joint_angles[REAL_TO_MUJOCO[i]],
                                 NEUTRAL_REAL[i],
                                 SERVO_SIGN[i]) for i in range(16)]

    print(f"\nJoint MuJoCo {args.joint} ({MUJOCO_JOINT_NAMES[args.joint]}) = {args.deg:+.1f}°")
    print(f"{'S#':<4} {'Nombre':<14} {'MuJoCo_idx':>10} {'MuJoCo_deg':>11} {'Neutro':>7} {'Sign':>5} {'Servo':>6}  Cambio")
    print("-" * 80)
    for i in range(16):
        mj_idx  = REAL_TO_MUJOCO[i]
        angle_d = np.degrees(joint_angles[mj_idx])
        neutral = NEUTRAL_REAL[i]
        sign    = SERVO_SIGN[i]
        servo   = servo_real[i]
        delta   = servo - neutral
        marker  = f"  ← MOVIDO {delta:+d}" if abs(delta) > 0 else ""
        print(f"S{i+1:02d}  {SERVO_NAMES[i]:<14} {mj_idx:>10}  {angle_d:>+10.2f}°  {neutral:>6}  {sign:>+5}  {servo:>6}{marker}")

    # Generar aesx con 3 frames: neutro → posicion → neutro
    POLICY_MS = 500
    neutral_frame  = (POLICY_MS, list(NEUTRAL_REAL), [])
    test_frame     = (POLICY_MS, servo_real, [])

    out_aesx = args.out + f"_j{args.joint}_{args.deg:+.0f}deg.aesx"
    os.makedirs(os.path.dirname(os.path.abspath(out_aesx)), exist_ok=True)
    inject_aesx(args.template, out_aesx, [neutral_frame, test_frame, neutral_frame])
    print(f"\n✅ AESX -> {out_aesx}")
    print(f"   Secuencia: NEUTRO → J{args.joint}={args.deg:+.0f}° → NEUTRO")


if __name__ == "__main__":
    main()
