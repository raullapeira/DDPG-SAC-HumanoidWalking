"""
Test de calibracion de signos de servo.
Genera un movimiento desde neutro hasta neutro+50 en todos los servos.
No requiere checkpoint ni simulacion MuJoCo.

Uso:
    python tools/test_servo_50deg.py \
        --out robot/simu_a_real/test_50deg_27_03_26
"""
import sys, os, argparse, struct
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import numpy as np

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

DELTA = 50   # grados a sumar a cada servo


# --- aesx injection ---
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
    parser.add_argument("--out",      required=True,
                        help="Base path sin extension, p.ej. robot/simu_a_real/test_50deg")
    parser.add_argument("--template", default=os.path.join(
        _HERE, "..", "robot", "simu_a_real", "exportado_por_sw_ubtech.aesx"))
    parser.add_argument("--duration_ms", type=int, default=2000,
                        help="Duracion del movimiento en ms (default 2000)")
    args = parser.parse_args()

    out_aesx = args.out + ".aesx"
    out_txt  = args.out + "_servos.txt"
    os.makedirs(os.path.dirname(os.path.abspath(out_aesx)), exist_ok=True)

    neutro   = list(NEUTRAL_REAL)
    desplazado = [int(np.clip(n + DELTA, 0, 255)) for n in NEUTRAL_REAL]

    frames = [
        (args.duration_ms, neutro,     [0.0] * 16),
        (args.duration_ms, desplazado, [float(DELTA)] * 16),
    ]

    # --- TXT ---
    col_w = 6
    header = f"{'Paso':<8}" + "".join(f"S{i+1:02d}".rjust(col_w) for i in range(16))
    sep    = "-" * len(header)

    lines = ["=== VALORES SERVO exportados al robot ===", header, sep]
    labels = ["neutro (paso 1)", f"+{DELTA} (paso 2)"]
    for label, (_, servos, _) in zip(labels, frames):
        row = f"{label:<8}" + "".join(str(v).rjust(col_w) for v in servos)
        lines.append(row)
    lines.append(sep)

    lines += ["", "=== DELTA aplicado por servo ===", header, sep]
    delta_row = f"{'delta':<8}" + "".join(str(desplazado[i] - neutro[i]).rjust(col_w) for i in range(16))
    lines.append(delta_row)
    lines.append(sep)

    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ TXT  -> {out_txt}")

    # --- AESX ---
    template = os.path.abspath(args.template)
    if inject_aesx(template, frames, out_aesx, duration_ms=args.duration_ms):
        print(f"✅ AESX -> {out_aesx}")

    # --- Resumen en consola ---
    print(f"\nDelta aplicado: +{DELTA} a todos los servos\n")
    print(f"{'Servo':<6} {'Nombre':<20} {'Neutro':>7} {'Paso2':>6} {'Direccion esperada'}")
    print("-" * 75)
    direcciones = [
        "brazos (sin calibrar)",
        "brazos (sin calibrar)",
        "brazos (sin calibrar)",
        "brazos (sin calibrar)",
        "brazos (sin calibrar)",
        "brazos (sin calibrar)",
        "cadera lat der se CIERRA (aduce)",
        "cadera frt der va hacia ATRAS (extension)",
        "rodilla der se EXTIENDE mas",
        "tobillo frt der punta hacia ARRIBA",
        "tobillo lat der gira hacia ADENTRO",
        "cadera lat izq se ABRE (abduce)",
        "cadera frt izq va hacia ADELANTE",
        "rodilla izq se DOBLA",
        "tobillo frt izq punta hacia ABAJO",
        "tobillo lat izq gira hacia AFUERA",
    ]
    for i in range(16):
        print(f"S{i+1:02d}    {SERVO_NAMES[i]:<20} {neutro[i]:>7} {desplazado[i]:>6}  {direcciones[i]}")


if __name__ == "__main__":
    main()
