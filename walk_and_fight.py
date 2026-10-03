"""
Asalto completo encadenando las dos politicas YA entrenadas (no entrena nada):

  1. ANDAR   — piernas con la politica de acercamiento de versus_env.py
               (checkpoints/*_versus_stop_dist/{r1,r2}), en las mismas
               condiciones con las que se entreno: brazos en 0.
  2. CUADRAR — en cuanto los troncos llegan a la distancia de combate
               (close_combat_env._SPAWN_DIST, comprobado en cada sub-paso
               fisico para no pasarse), 0.5s de transicion que lleva a cada
               robot a la postura de arranque del combate: piernas neutras,
               tronco recto mirando al rival, brazos en guardia. Se mantiene
               la posicion donde han llegado.
  3. PEGAR   — brazos con la politica de close_combat.py
               (checkpoints/*_close_combat_*/{arm_r1,arm_r2}) usando el propio
               CloseCombatEnv, hasta que uno cae (KO) o se acaba el asalto.

Por que el paso 2 no es fisico: el andar entrenado llega con el tronco
inclinado hacia delante (zancada casi en caida) y las piernas a media
zancada. Medido pasando directamente a combate (con o sin freno de la raiz,
con o sin esperar un momento "bueno"): ~50% de los robots se caian SOLOS en
0.2-5s aunque los brazos estuvieran quietos. Con la transicion: 0 caidas
pasivas en 12 seeds, y 11/12 asaltos acaban en KO por golpe.

Genera VARIAS simulaciones, cada una con una parametrizacion distinta
(lista _SIMULACIONES, editable). Parametros de cada simulacion:
  seed        ruido inicial de las articulaciones (+-0.05 rad, como VersusEnv)
  start_dist  separacion inicial raiz-raiz en metros (entrenado a 2.0)
  lateral     desfase lateral en metros (r1 en +lateral/2, r2 en -lateral/2)
  yaw_r1/r2   giro inicial de cada robot en grados respecto a mirar al rival
  fight_ckpt  step del checkpoint de combate (None = el mas reciente)

Uso:
    py -3 walk_and_fight.py               # todas las simulaciones de la lista
    py -3 walk_and_fight.py --sim 3       # solo la simulacion 3
    py -3 walk_and_fight.py --walk_dir checkpoints/2026_09_19_versus_stop_dist                             --fight_dir checkpoints/2026_10_02_close_combat_ciclo_golpe
Salida en media/YYYY_MM_DD_walk_and_fight/:
    simNN_<nombre>_<resultado>_golpesA-B.gif   un GIF por simulacion
    resumen.csv                                parametros y resultado de cada una
"""
import os, re, csv, glob, argparse, datetime
import numpy as np
import torch
import torch.nn as nn
import mujoco
import imageio

import versus_env as V
import close_combat_env as C

_HERE  = os.path.dirname(os.path.abspath(__file__))
_TODAY = datetime.date.today().strftime("%Y_%m_%d")
_MEDIA_DIR = os.path.join(_HERE, "media", f"{_TODAY}_walk_and_fight")

WALK_OBS_DIM   = 34
WALK_ACT_DIM   = 10
_MAX_WALK_STEPS = 300      # 30s para llegar; si no, se da por perdido
_BLEND_STEPS    = 5        # 0.5s de transicion andar -> guardia
_POST_KO_STEPS  = 10       # frames extra tras el KO para ver la caida entera

WIDTH, HEIGHT = 800, 480
FPS           = 10         # un frame por step (0.1s) = tiempo real

# Cada simulacion cambia UNA cosa respecto a la base (sim 1) para poder
# comparar que efecto tiene cada parametro.
_BASE = dict(seed=4, start_dist=1.2, lateral=0.0, yaw_r1=0.0, yaw_r2=0.0, fight_ckpt=None)
_SIMULACIONES = [
    dict(_BASE, nombre="base"),
    dict(_BASE, nombre="otra_semilla",       seed=6),
    dict(_BASE, nombre="muy_cerca",          start_dist=0.8),
    dict(_BASE, nombre="mas_lejos",          start_dist=1.6),
    dict(_BASE, nombre="dist_entrenamiento", start_dist=2.0),
    dict(_BASE, nombre="desfase_lateral",    lateral=0.10),
    dict(_BASE, nombre="r1_girado",          yaw_r1=15.0),
    dict(_BASE, nombre="ambos_girados",      yaw_r1=-10.0, yaw_r2=10.0),
    dict(_BASE, nombre="combate_250k",       fight_ckpt=250000),
]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mu      = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def act(self, obs):
        with torch.no_grad():
            return torch.tanh(self.mu(self.net(torch.FloatTensor(obs).unsqueeze(0))))[0].numpy()


def _latest_ckpt(folder, step=None):
    if step is not None:
        return os.path.join(folder, f"ckpt_{step}.pt")
    ckpts = glob.glob(os.path.join(folder, "ckpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No hay checkpoints en {folder}")
    return max(ckpts, key=lambda p: int(re.search(r"ckpt_(\d+)\.pt", p).group(1)))


def _latest_run(pattern):
    runs = sorted(glob.glob(os.path.join(_HERE, "checkpoints", pattern)))
    if not runs:
        raise FileNotFoundError(f"No hay carpetas checkpoints/{pattern}")
    return runs[-1]   # empiezan por YYYY_MM_DD -> la ultima es la mas reciente


def _load_actor(path, obs_dim, act_dim):
    a = Actor(obs_dim, act_dim)
    a.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["actor"])
    a.eval()
    print(f"  {os.path.relpath(path, _HERE)}")
    return a


def _slerp(p, q, t):
    dot = float(np.dot(p, q))
    if dot < 0.0:
        q, dot = -q, -dot
    if dot > 0.9995:
        r = p + t * (q - p)
        return r / np.linalg.norm(r)
    th = np.arccos(dot)
    return (np.sin((1 - t) * th) * p + np.sin(t * th) * q) / np.sin(th)


def _upright(qpos, off):
    return 1.0 - 2.0 * (qpos[off + 4] ** 2 + qpos[off + 5] ** 2)


def run_sim(sim, legs, arms, fight, walk, spawn_qpos, max_fight_steps):
    """Ejecuta una simulacion completa (andar -> cuadrar -> pegar).
    Devuelve (frames, resumen)."""
    m, d = fight.model, fight.data

    # Arranque del andar: igual que VersusEnv.reset() pero con la
    # parametrizacion de esta simulacion.
    mujoco.mj_resetData(m, d)
    rng = np.random.default_rng(sim["seed"])
    d.qpos[7:23]  += rng.uniform(-0.05, 0.05, 16)
    d.qpos[30:46] += rng.uniform(-0.05, 0.05, 16)
    d.qpos[0]  =  sim["start_dist"] / 2.0
    d.qpos[23] = -sim["start_dist"] / 2.0
    d.qpos[1]  =  sim["lateral"] / 2.0
    d.qpos[24] = -sim["lateral"] / 2.0
    # r1 mira a -X (yaw 180), r2 a +X (yaw 0); se suma el giro de cada uno.
    for off, base_yaw, extra in ((0, np.pi, sim["yaw_r1"]), (23, 0.0, sim["yaw_r2"])):
        yaw = base_yaw + np.radians(extra)
        d.qpos[off + 3:off + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth   = 90
    cam.elevation = -10
    frames = []

    def dist_xy():
        return float(np.hypot(d.qpos[0] - d.qpos[23], d.qpos[1] - d.qpos[24]))

    def snap():
        cam.lookat[:] = [(d.qpos[0] + d.qpos[23]) / 2.0, (d.qpos[1] + d.qpos[24]) / 2.0, 0.25]
        cam.distance  = max(1.0, 0.6 + dist_xy())
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render().copy())

    snap()

    # ── 1. ANDAR ─────────────────────────────────────────────────────────────
    result = None
    arrived = False
    for t in range(_MAX_WALK_STEPS):
        o1, o2 = walk._get_obs()
        a1, a2 = legs[0].act(o1), legs[1].act(o2)
        for _ in range(V._ACTION_REPEAT):
            ctrl = np.zeros(m.nu)
            ctrl[V._R1_LEG_CTRL] = walk._denorm(a1)
            ctrl[V._R2_LEG_CTRL] = walk._denorm(a2)
            d.ctrl[:] = ctrl
            for _ in range(V._FRAME_SKIP):
                mujoco.mj_step(m, d)
                if dist_xy() <= C._SPAWN_DIST:
                    arrived = True
                    break
            # Brazos en 0, como en el entrenamiento de versus_env.
            d.qpos[walk._r1_arm_qpos] = 0.0; d.qvel[walk._r1_arm_qvel] = 0.0
            d.qpos[walk._r2_arm_qpos] = 0.0; d.qvel[walk._r2_arm_qvel] = 0.0
            mujoco.mj_forward(m, d)
            if arrived:
                break
        snap()
        r1_fell = d.qpos[2]  < C._FALL_Z or _upright(d.qpos, 0)  < C._TILT_Z
        r2_fell = d.qpos[25] < C._FALL_Z or _upright(d.qpos, 23) < C._TILT_Z
        if r1_fell or r2_fell:
            result = "ambos_caen_andando" if (r1_fell and r2_fell) else \
                     ("r1_cae_andando" if r1_fell else "r2_cae_andando")
            break
        if arrived:
            break
    if result is None and not arrived:
        result = "no_llegan"
    walk_steps = t + 1

    strikes = [0, 0]
    fight_steps = 0
    if result is None:
        # ── 2. CUADRAR: transicion a la postura de spawn del combate ─────────
        q0  = d.qpos.copy()
        tgt = q0.copy()
        for off, opp in ((0, 23), (23, 0)):
            tgt[off + 2:off + 23] = spawn_qpos[off + 2:off + 23]   # altura, piernas, brazos
            yaw = np.arctan2(q0[opp + 1] - q0[off + 1], q0[opp] - q0[off])
            tgt[off + 3:off + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        for k in range(_BLEND_STEPS):
            al = (k + 1) / _BLEND_STEPS
            q = q0 + al * (tgt - q0)
            for off in (0, 23):
                q[off + 3:off + 7] = _slerp(q0[off + 3:off + 7], tgt[off + 3:off + 7], al)
            d.qpos[:] = q
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            snap()

        # ── 3. PEGAR ─────────────────────────────────────────────────────────
        o1, o2 = fight.begin_fight()
        result = "sin_ko"
        for k in range(max_fight_steps):
            (o1, o2), _, terminated, _, info = fight.step(
                np.concatenate([arms[0].act(o1), arms[1].act(o2)]))
            strikes[0] += info["r1_strikes"]
            strikes[1] += info["r2_strikes"]
            snap()
            if terminated:
                if info["r1_fell"] and info["r2_fell"]:
                    result = "ambos_caidos"
                elif info["r2_fell"]:
                    result = "ko_r1" if info["r2_ko_by_hit"] else "r2_cae_solo"
                else:
                    result = "ko_r2" if info["r1_ko_by_hit"] else "r1_cae_solo"
                for _ in range(_POST_KO_STEPS):
                    fight.step(np.concatenate([arms[0].act(o1), arms[1].act(o2)]))
                    snap()
                break
        fight_steps = k + 1

    renderer.close()
    return frames, {
        "andar_s":   round(walk_steps * 0.1, 1),
        "combate_s": round(fight_steps * 0.1, 1),
        "resultado": result,
        "golpes_r1": strikes[0],
        "golpes_r2": strikes[1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim",        type=int, default=None,
                        help="ejecutar solo esta simulacion (1..N) de _SIMULACIONES")
    parser.add_argument("--walk_dir",   default=None)
    parser.add_argument("--fight_dir",  default=None)
    parser.add_argument("--max_fight_steps", type=int, default=300)
    parser.add_argument("--out_dir",    default=_MEDIA_DIR)
    args = parser.parse_args()

    walk_dir  = args.walk_dir  or _latest_run("*_versus_stop_dist")
    fight_dir = args.fight_dir or _latest_run("*_close_combat_*")
    print("Politica de andar:")
    legs = [_load_actor(_latest_ckpt(os.path.join(walk_dir, r)), WALK_OBS_DIM, WALK_ACT_DIM)
            for r in ("r1", "r2")]
    arms_cache = {}

    # Un unico modelo/estado fisico: el del CloseCombatEnv. El VersusEnv solo
    # aporta su observacion y su desnormalizacion de piernas, apuntando al
    # mismo MjData (los dos cargan alpha_versus.xml).
    fight = C.CloseCombatEnv()
    walk  = V.VersusEnv()
    walk.model, walk.data = fight.model, fight.data
    fight.model.vis.global_.offwidth  = 1600
    fight.model.vis.global_.offheight = 960

    # Postura de arranque del combate (la del entrenamiento de close_combat).
    fight.reset()
    spawn_qpos = fight.data.qpos.copy()

    sims = list(enumerate(_SIMULACIONES, start=1))
    if args.sim is not None:
        sims = [sims[args.sim - 1]]

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for n, sim in sims:
        if sim["fight_ckpt"] not in arms_cache:
            print(f"Politica de combate ({sim['fight_ckpt'] or 'ultima'}):")
            arms_cache[sim["fight_ckpt"]] = [
                _load_actor(_latest_ckpt(os.path.join(fight_dir, r), sim["fight_ckpt"]),
                            C.FIGHT_OBS_DIM, C.ARM_ACT_DIM)
                for r in ("arm_r1", "arm_r2")]
        frames, res = run_sim(sim, legs, arms_cache[sim["fight_ckpt"]],
                              fight, walk, spawn_qpos, args.max_fight_steps)
        gif = f"sim{n:02d}_{sim['nombre']}_{res['resultado']}_golpes{res['golpes_r1']}-{res['golpes_r2']}.gif"
        imageio.mimsave(os.path.join(args.out_dir, gif), frames, fps=FPS, loop=0)
        row = {"sim": n, **sim, "fight_ckpt": sim["fight_ckpt"] or "ultima", **res, "gif": gif}
        rows.append(row)
        print(f"[sim {n:02d}] {sim['nombre']:<18} seed={sim['seed']} dist={sim['start_dist']} "
              f"lat={sim['lateral']} yaw=({sim['yaw_r1']},{sim['yaw_r2']}) "
              f"ckpt={row['fight_ckpt']} -> andar {res['andar_s']}s, combate {res['combate_s']}s, "
              f"{res['resultado']}, golpes {res['golpes_r1']}-{res['golpes_r2']}")

    fight.close()
    csv_path = os.path.join(args.out_dir, "resumen.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} GIFs y resumen en {args.out_dir}")


if __name__ == "__main__":
    main()
