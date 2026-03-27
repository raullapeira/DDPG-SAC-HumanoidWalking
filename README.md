# Alpha Humanoid Walking — SAC

Bipedal locomotion for the **Alpha humanoid robot** (16 DOF, ~1.58 kg) using Soft Actor-Critic in MuJoCo.

![Walking Demo](media/23_03_2026.1_buena_pinta_sin_rozar_suelo/alpha_step_1000000.gif)

## What it does

Trains a policy that walks stably forward at ~0.3 m/s with no falls over 1M+ steps. Each policy step = 100ms, directly mappable to real servo commands (position + transition time).

## Key design choices

- **Action space**: normalised joint positions [-1, 1] centered on neutral pose — `action=0` maps to standing pose
- **Policy step = 100ms**: matches real servo API timing
- **Reward shaping**: forward velocity + foot height (rear leg only) + push-off + yaw/lateral penalties
- **Foot height reward**: forces weight transfer between legs to achieve real walking gait

## Train

```
# Geometry v1 (original)
python -u main_sac_alpha1.py

# Geometry v2 (corrected knee pivot) — auto-generates GIF per checkpoint
python -u main_sac_alpha_v2.py
```

Checkpoints saved every 50k steps to `checkpoints/sac_alpha/` (v1) or `checkpoints/sac_alpha_v2/` (v2).
GIFs auto-generated in `media/<date>_<description>/` at each checkpoint.

## Evaluate

```
# GIF from a v1 checkpoint
python tools/make_checkpoint_gif.py --ckpt checkpoints/sac_alpha/sac2_checkpoint_1000000.pt --step 1000000

# GIF from a v2 checkpoint
python tools/make_checkpoint_gif_v2.py --ckpt checkpoints/sac_alpha_v2/sac2_checkpoint_50000.pt --step 50000 --out_dir media/

# Sweep all checkpoints in a folder → comparison table
python tools/sweep_eval.py --ckpt_dir checkpoints/sac_alpha/

# Detailed metrics for a single checkpoint
python tools/eval_check.py --ckpt checkpoints/sac_alpha/sac2_checkpoint_1000000.pt
```

## Export to real robot (UBTech Alpha)

Generates a `.aesx` action file (importable in UBTech software) and a slow MP4 with servo value overlay:

```
python tools/export_aesx_mp4.py \
    --ckpt checkpoints/sac_alpha/sac2_checkpoint_950000.pt \
    --n_steps 10 \
    --out robot/simu_a_real/10_movs_25_03_26
```

- `--n_steps`: number of policy steps to export (template supports up to 19)
- `--out`: base output path (no extension) — produces `<out>.aesx` and `<out>.mp4`
- MP4 plays at 5 fps (8x slower than real-time), overlays servo values per joint per step
- Step 1 = rest position, Step 2+ = simulation steps (matches UBTech software numbering)
- Servo values inverted from simulation angles to match real robot direction
- Template: `robot/simu_a_real/exportado_por_sw_ubtech.aesx`

### Known issues (as of 25/03/2026)

- Robot moves but gait is not yet reliable: rear foot doesn't push off properly
- Servo direction mapping needs further validation against physical robot

## Scripts reference

### Training
| Script | Description |
|---|---|
| `main_sac_alpha1.py` | SAC training loop — geometry v1 |
| `main_sac_alpha_v2.py` | SAC training loop — geometry v2 (corrected knee pivot) |

### Environment
| Script | Description |
|---|---|
| `envs/alpha_env.py` | MuJoCo environment: obs, reward shaping, reset, step. Accepts optional `xml_path` for v1 or v2 |

### Evaluation / visualisation
| Script | Description |
|---|---|
| `tools/make_checkpoint_gif.py` | Lateral-view evaluation GIF for v1 checkpoints |
| `tools/make_checkpoint_gif_v2.py` | Lateral-view evaluation GIF for v2 checkpoints — called automatically every 50k steps |
| `tools/eval_check.py` | Evaluates a checkpoint and prints metrics (reward, distance, falls) |
| `tools/eval_gif.py` | Generic evaluation GIF (checkpoint + XML as arguments) |
| `tools/make_long_gif.py` | Multi-episode GIF for comparing checkpoints side by side |
| `tools/sweep_eval.py` | Evaluates all checkpoints in a directory and generates a comparison table |

### Real robot export
| Script | Description |
|---|---|
| `tools/export_aesx_mp4.py` | **Main export script**: runs simulation, writes `.aesx` (UBTech format) + `.mp4` with servo value overlay |
| `tools/export_mp4.py` | MP4 only (no `.aesx`) |
| `tools/export_servo_csv.py` | Export servo values to `.csv` |
| `tools/export_servo_intervals.py` | Export servo values with timing intervals between steps |

### Utilities
| Script | Description |
|---|---|
| `robot/simu_a_real/extrae.py` | Reads a `.aesx` file and decodes its binary structure (frames, durations, servo values) |
| `robot/simu_a_real/genera.py` | Injects servo values from a CSV into a `.aesx` template (legacy, superseded by `export_aesx_mp4.py`) |

## Structure

```
DDPG-SAC-HumanoidWalking/
├── main_sac_alpha1.py          # Training loop v1
├── main_sac_alpha_v2.py        # Training loop v2
├── envs/
│   └── alpha_env.py            # MuJoCo environment + reward shaping
├── robot/
│   ├── alpha_single.xml        # Robot model v1 (16 actuators, kp=20)
│   ├── reverse_eng_v1/         # Geometry v1 (original URDF + MuJoCo XML)
│   ├── reverse_eng_v2/         # Geometry v2 (corrected knee pivot)
│   │   ├── alpha_back_engineer_v2.urdf
│   │   ├── alpha_single.xml
│   │   └── scene.xml
│   └── simu_a_real/            # Simulation → real robot export files
│       ├── exportado_por_sw_ubtech.aesx  # 19-frame template
│       ├── extrae.py
│       ├── genera.py
│       └── *.aesx / *.mp4
├── tools/                      # Evaluation and export scripts
├── checkpoints/
│   ├── sac_alpha/              # v1 checkpoints
│   └── sac_alpha_v2/           # v2 checkpoints
└── media/                      # GIFs/videos per training run (DD_MM_YYYY.N_description)
```
