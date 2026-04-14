"""
Test2: robot levanta las dos piernas 90 grados hacia adelante (flexion de cadera).
No requiere checkpoint ni simulacion MuJoCo.

S08 (cadera_front_der): neutral=120, BAJA=adelante -> 120 - 90 = 30
S13 (cadera_front_izq): neutral=60,  SUBE=adelante -> 60  + 90 = 150

Uso:
    python tools/test2_lift_legs_90deg.py \
        --out robot/simu_a_real/test2_lift_legs
"""
import sys, os, argparse, struct
_HERE = os.path.dirname(os.path.abspath(__file__))

NEUTRAL_REAL = [90, 90, 90, 90, 90, 90, 90, 120,
                145, 95, 90, 90, 60, 30, 95, 90]


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
        _HERE, "..", "robot", "simu_a_real", "test2_lift_legs"))
    parser.add_argument("--template", default=os.path.join(
        _HERE, "..", "robot", "simu_a_real", "exportado_por_sw_ubtech.aesx"))
    parser.add_argument("--duration_ms", type=int, default=2000)
    args = parser.parse_args()

    out_aesx = args.out + ".aesx"
    out_txt  = args.out + "_servos.txt"
    os.makedirs(os.path.dirname(os.path.abspath(out_aesx)), exist_ok=True)

    target = list(NEUTRAL_REAL)
    target[7]  = 30   # S08 cadera_front_der: 120 - 90 = 30
    target[12] = 150  # S13 cadera_front_izq:  60 + 90 = 150

    frames = [
        (args.duration_ms, list(NEUTRAL_REAL), []),
        (args.duration_ms, target, []),
    ]

    # TXT
    col_w = 6
    header = f"{'Paso':<8}" + "".join(f"S{i+1:02d}".rjust(col_w) for i in range(16))
    sep    = "-" * len(header)
    lines  = ["=== VALORES SERVO ===", header, sep,
              f"{'neutro':<8}" + "".join(str(v).rjust(col_w) for v in NEUTRAL_REAL),
              f"{'lift':<8}"   + "".join(str(v).rjust(col_w) for v in target),
              sep]
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ TXT  -> {out_txt}")

    if inject_aesx(os.path.abspath(args.template), frames, out_aesx, duration_ms=args.duration_ms):
        print(f"✅ AESX -> {out_aesx}")

    print(f"\nMovimiento: neutro -> ambas caderas 90° hacia adelante")
    print(f"  S08 cadera_front_der: {NEUTRAL_REAL[7]} -> {target[7]}  (-90)")
    print(f"  S13 cadera_front_izq: {NEUTRAL_REAL[12]} -> {target[12]}  (+90)")
    print(f"  Resto de servos: en neutro")


if __name__ == "__main__":
    main()
