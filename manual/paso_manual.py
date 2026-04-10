"""
Secuencia manual de un paso completo (pierna izquierda).
6 fases que representan la biomecanica correcta del paso:

  1. Reposo (neutro)
  2. Desplazamiento lateral: peso sobre pierna derecha
  3. Levanta pierna izquierda: cadera avanza, rodilla flexionada
  4. Avance pierna izquierda: pierna extendida hacia delante, cuerpo ligeramente atras
  5. Apoyo pie izquierdo: pie toca suelo, cuerpo empieza a pasar
  6. Transferencia de peso a izquierda: listo para el siguiente paso

Brazos pegados al cuerpo (neutro durante toda la secuencia).

Uso:
    python manual/paso_manual.py
    python manual/paso_manual.py --out manual/paso_manual --duration_ms 1500
"""
import sys, os, argparse, struct
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import numpy as np

# ---------------------------------------------------------------
# Constantes de conversion (mismas que export_aesx_mp4.py)
# ---------------------------------------------------------------
SERVO_NAMES = [
    "rot_hombro_der",    "ext_hombro_der",    "rot_codo_der",   # S01-S03
    "rot_hombro_izq",    "ext_hombro_izq",    "rot_codo_izq",   # S04-S06
    "cadera_lat_der",    "cadera_front_der",  "rodilla_der",    # S07-S09
    "tobillo_front_der", "tobillo_lat_der",                     # S10-S11
    "cadera_lat_izq",    "cadera_front_izq",  "rodilla_izq",    # S12-S14
    "tobillo_front_izq", "tobillo_lat_izq",                     # S15-S16
]

# Neutrales fisicos del robot
N = [90, 90, 90, 90, 90, 90,   # brazos (S01-S06)
     90, 120, 145, 95, 90,      # pierna der (S07-S11)
     90, 60,  30,  95, 90]      # pierna izq (S12-S16)

# Referencia del comportamiento fisico (calibrado):
#   S07 cad_lat_der : BAJA=abre/abduce  SUBE=cierra/aduce
#   S08 cad_frt_der : BAJA=adelante     SUBE=atras
#   S09 rodilla_der : BAJA=dobla        SUBE=extiende
#   S10 tob_frt_der : SUBE=punta_arriba BAJA=punta_abajo
#   S11 tob_lat_der : SUBE=adentro      BAJA=afuera
#   S12 cad_lat_izq : SUBE=abre/abduce  BAJA=cierra/aduce
#   S13 cad_frt_izq : SUBE=adelante     BAJA=atras
#   S14 rodilla_izq : SUBE=dobla        BAJA=extiende
#   S15 tob_frt_izq : BAJA=punta_arriba SUBE=punta_abajo
#   S16 tob_lat_izq : BAJA=adentro      SUBE=afuera

def pose(
    # pierna derecha (stance en fases 2-5)
    cad_lat_der=0, cad_frt_der=0, rod_der=0,
    tob_frt_der=0, tob_lat_der=0,
    # pierna izquierda (swing en fases 3-5)
    cad_lat_izq=0, cad_frt_izq=0, rod_izq=0,
    tob_frt_izq=0, tob_lat_izq=0,
):
    """Devuelve lista de 16 valores servo a partir de deltas sobre neutro.
    Signos: positivo = direccion indicada en el comentario de arriba.
    """
    s = list(N)
    # der: cad_lat SUBE=cierra (+), cad_frt BAJA=adelante (-)
    s[6]  = int(np.clip(N[6]  + cad_lat_der,  0, 255))  # S07: +cierra/aduce
    s[7]  = int(np.clip(N[7]  - cad_frt_der,  0, 255))  # S08: -=adelante
    s[8]  = int(np.clip(N[8]  - rod_der,       0, 255))  # S09: -=dobla
    s[9]  = int(np.clip(N[9]  + tob_frt_der,  0, 255))  # S10: +=punta_arriba
    s[10] = int(np.clip(N[10] + tob_lat_der,  0, 255))  # S11: +=adentro
    # izq: cad_lat SUBE=abre (+), cad_frt SUBE=adelante (+)
    s[11] = int(np.clip(N[11] + cad_lat_izq,  0, 255))  # S12: +=abre/abduce
    s[12] = int(np.clip(N[12] + cad_frt_izq,  0, 255))  # S13: +=adelante
    s[13] = int(np.clip(N[13] + rod_izq,       0, 255))  # S14: +=dobla
    s[14] = int(np.clip(N[14] - tob_frt_izq,  0, 255))  # S15: -=punta_arriba
    s[15] = int(np.clip(N[15] - tob_lat_izq,  0, 255))  # S16: -=adentro
    return s


# ---------------------------------------------------------------
# 6 fases del paso
# ---------------------------------------------------------------
FASES = [
    {
        "nombre": "reposo",
        "desc":   "Posicion neutra de pie",
        "servos": pose(),
    },
    {
        "nombre": "peso_derecha",
        "desc":   "Desplaza peso a pierna derecha (cuerpo sobre pie der)",
        "servos": pose(
            cad_lat_der=+15,   # aduce der -> cuerpo va a la derecha
            cad_lat_izq=+15,   # abduce izq -> cuerpo va a la derecha
            rod_der=+10,        # leve flexion de stance para bajar COM
        ),
    },
    {
        "nombre": "levanta_izq",
        "desc":   "Levanta pierna izquierda con cuerpo sobre pie derecho",
        "servos": pose(
            cad_lat_der=+15,
            cad_lat_izq=+15,
            rod_der=+12,
            cad_frt_izq=+20,   # cadera izq ligeramente adelante
            rod_izq=+70,        # rodilla muy doblada para maxima altura del pie
            tob_frt_izq=+25,   # punta pie izq bien arriba
        ),
    },
    {
        "nombre": "avance_izq",
        "desc":   "Pierna izq avanza, rodilla aun doblada para no rozar suelo",
        "servos": pose(
            cad_lat_der=+15,
            cad_lat_izq=+15,
            rod_der=+12,
            cad_frt_der=-10,   # cuerpo/cadera der ligeramente atras
            cad_frt_izq=+45,   # cadera izq adelante
            rod_izq=+50,        # rodilla sigue bastante doblada durante el vuelo
            tob_frt_izq=+20,   # punta arriba aun
        ),
    },
    {
        "nombre": "apoyo_izq",
        "desc":   "Pie izquierdo toca suelo, cuerpo pasa hacia delante",
        "servos": pose(
            cad_lat_der=+5,    # empieza a soltar el peso lateral
            cad_lat_izq=+5,
            rod_der=+8,
            cad_frt_der=+15,   # cuerpo avanza sobre pie izq
            cad_frt_izq=+35,   # pierna izq plantada adelante
            rod_izq=+5,         # casi extendida
            tob_frt_izq=0,     # pie plano al suelo
        ),
    },
    {
        "nombre": "peso_izquierda",
        "desc":   "Peso transferido a pierna izquierda, listo para paso derecho",
        "servos": pose(
            cad_lat_der=-15,   # abduce der -> cuerpo va a la izquierda
            cad_lat_izq=-15,   # aduce izq -> cuerpo va a la izquierda
            cad_frt_der=-20,   # pierna der empieza a subir
            rod_der=+25,        # rodilla der doblada
            tob_frt_der=+15,   # punta pie der arriba
        ),
    },
]


# ---------------------------------------------------------------
# AESX injection
# ---------------------------------------------------------------
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


def inject_aesx(template_path, frames, out_path, duration_ms):
    with open(template_path, "rb") as f:
        data = bytearray(f.read())
    offsets = _find_servo_blocks(data)
    if not offsets:
        print("❌ No se encontraron bloques de servos en la plantilla")
        return False
    for idx, (servos,) in enumerate(frames):
        if idx >= len(offsets):
            break
        base = offsets[idx]
        for i, value in enumerate(servos):
            struct.pack_into("<I", data, base + i*8 + 4, value)
        dur_float = float(duration_ms) / 10.0
        flt_offset = base + 16 * 8 + 12
        if flt_offset + 8 <= len(data):
            struct.pack_into("<f", data, flt_offset,     dur_float)
            struct.pack_into("<f", data, flt_offset + 4, dur_float)
    with open(out_path, "wb") as f:
        f.write(data)
    return True


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(_HERE, "paso_manual"))
    parser.add_argument("--template", default=os.path.join(
        _HERE, "..", "robot", "simu_a_real", "exportado_por_sw_ubtech.aesx"))
    parser.add_argument("--duration_ms", type=int, default=1500,
                        help="Duracion de cada fase en ms")
    args = parser.parse_args()

    out_aesx = args.out + ".aesx"
    out_txt  = args.out + "_servos.txt"
    os.makedirs(os.path.dirname(os.path.abspath(out_aesx)), exist_ok=True)

    frames = [([f["servos"]],) for f in FASES]
    # flatten for inject
    flat_frames = [(f["servos"],) for f in FASES]

    # TXT
    col_w  = 6
    header = f"{'Fase':<22}" + "".join(f"S{i+1:02d}".rjust(col_w) for i in range(16))
    sep    = "-" * len(header)
    lines  = ["=== PASO MANUAL - VALORES SERVO ===", "", header, sep]
    for i, fase in enumerate(FASES):
        row = f"{fase['nombre']:<22}" + "".join(str(v).rjust(col_w) for v in fase["servos"])
        lines.append(row)
    lines += [sep, "", "=== DESCRIPCION DE FASES ==="]
    for i, fase in enumerate(FASES):
        lines.append(f"  {i+1}. {fase['nombre']}: {fase['desc']}")
    lines.append("")
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ TXT  -> {out_txt}")

    # AESX
    if inject_aesx(os.path.abspath(args.template), flat_frames, out_aesx, args.duration_ms):
        print(f"✅ AESX -> {out_aesx}")

    # Resumen en consola
    print(f"\n{'Fase':<4} {'Nombre':<22} {'S07':>4} {'S08':>4} {'S09':>4} {'S10':>4} "
          f"{'S11':>4} {'S12':>4} {'S13':>4} {'S14':>4} {'S15':>4} {'S16':>4}")
    print("-" * 76)
    for i, fase in enumerate(FASES):
        s = fase["servos"]
        print(f"{i+1:<4} {fase['nombre']:<22} "
              f"{s[6]:>4} {s[7]:>4} {s[8]:>4} {s[9]:>4} {s[10]:>4} "
              f"{s[11]:>4} {s[12]:>4} {s[13]:>4} {s[14]:>4} {s[15]:>4}")


if __name__ == "__main__":
    main()
