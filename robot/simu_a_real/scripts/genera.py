import struct
import csv
import sys


def find_servo_blocks(data):
    """
    Encuentra offsets donde hay bloques de 16 servos (id, valor)
    """
    offsets = []
    n = len(data)

    for i in range(n - 16 * 8):
        ok = True

        for j in range(16):
            servo_id = struct.unpack_from("<I", data, i + j*8)[0]
            value = struct.unpack_from("<I", data, i + j*8 + 4)[0]

            if not (1 <= servo_id <= 16 and 0 <= value <= 300):
                ok = False
                break

        if ok:
            offsets.append(i)

    return offsets


def load_csv(csv_path):
    frames = []

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            duration = int(row[0])
            servos = list(map(int, row[1:17]))
            frames.append((duration, servos))

    return frames


def inject_frames(template_path, csv_path, output_path):
    with open(template_path, "rb") as f:
        data = bytearray(f.read())

    frames = load_csv(csv_path)
    offsets = find_servo_blocks(data)

    if not offsets:
        print("❌ No se encontraron bloques de servos")
        return

    print(f"Bloques encontrados: {len(offsets)}")

    for idx, (duration, servos) in enumerate(frames):
        if idx >= len(offsets):
            break

        base = offsets[idx]

        for i, value in enumerate(servos):
            struct.pack_into("<I", data, base + i*8 + 4, value)

    with open(output_path, "wb") as f:
        f.write(data)

    print(f"✅ Fichero generado: {output_path}")


def main():
    if len(sys.argv) < 4:
        print("Uso: python inject_aesx.py template.aesx input.csv output.aesx")
        return

    inject_frames(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()