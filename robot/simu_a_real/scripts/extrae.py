import struct
import sys


def extract_frames(data):
    frames = []
    n = len(data)

    i = 0
    while i < n - (16 * 8):
        start = i
        servos = [0] * 16
        valid = True

        # leer 16 pares (id, valor)
        for _ in range(16):
            if i + 8 > n:
                valid = False
                break

            servo_id = struct.unpack_from("<I", data, i)[0]
            value = struct.unpack_from("<I", data, i + 4)[0]
            i += 8

            if not (1 <= servo_id <= 16 and 0 <= value <= 300):
                valid = False
                break

            servos[servo_id - 1] = value

        if valid:
            # intentar leer duración si existe
            duration = None

            if i + 8 <= n:
                d1 = struct.unpack_from("<I", data, i)[0]
                d2 = struct.unpack_from("<I", data, i + 4)[0]

                if d1 == d2 and 1 <= d1 <= 5000:
                    duration = d1

            frames.append({
                "time": duration,
                "servos": servos
            })

        # 🔥 SIEMPRE avanzar solo 1 byte
        i = start + 1

    return frames


def main():
    if len(sys.argv) < 2:
        print("Uso: python extract_aesx.py <fichero.aesx>")
        return

    with open(sys.argv[1], "rb") as f:
        data = f.read()

    frames = extract_frames(data)

    print(f"\nFrames encontrados: {len(frames)}\n")

    for i, frame in enumerate(frames):
        print(f"Frame {i}: time={frame['time']} servos={frame['servos']}")


if __name__ == "__main__":
    main()