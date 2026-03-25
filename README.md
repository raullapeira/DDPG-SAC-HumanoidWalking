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
cd DDPG-SAC-HumanoidWalking
python -u main_sac_alpha1.py
```

Checkpoints saved every 50k steps to `checkpoints/sac_alpha/`. GIFs auto-generated in `media/`.

## Evaluate

```
# Generate MP4 from a checkpoint
python tools/export_mp4.py --ckpt checkpoints/sac_alpha/sac2_checkpoint_1000000.pt --step 1000000

# Generate GIF
python tools/make_checkpoint_gif.py --ckpt checkpoints/sac_alpha/sac2_checkpoint_1000000.pt --step 1000000
```

## Export to real robot (UBTech Alpha)

Generates a `.aesx` action file (importable in UBTech software) and a slow MP4 for comparison, both with the same base name:

```
python tools/export_aesx_mp4.py \
    --ckpt checkpoints/sac_alpha/sac2_checkpoint_950000.pt \
    --n_steps 10 \
    --out robot/simu_a_real/10_movs_25_03_26
```

- `--n_steps`: number of policy steps to export (template supports up to 19)
- `--out`: base output path (no extension) — produces `<out>.aesx` and `<out>.mp4`
- MP4 plays at 5 fps (8x slower than real-time) for easy comparison with the physical robot
- Servo values are inverted from simulation angles to match real robot direction
- Template: `robot/simu_a_real/exportado_por_sw_ubtech.aesx`

### Known issues (as of 25/03/2026)

- Robot moves but gait is not yet reliable: rear foot doesn't push off properly
- Servo direction mapping needs further validation against physical robot

## Structure

```
DDPG-SAC-HumanoidWalking/
├── main_sac_alpha1.py          # Training loop
├── envs/alpha_env.py           # MuJoCo environment + reward shaping
├── robot/
│   ├── alpha_single.xml        # Robot model (16 actuators, kp=20)
│   └── simu_a_real/            # Simulation → real robot export files
│       ├── exportado_por_sw_ubtech.aesx  # Template (19 frames)
│       ├── extrae.py           # Parse/inspect .aesx files
│       ├── genera.py           # Inject CSV servo values into .aesx template
│       └── *.aesx / *.mp4      # Generated exports
├── tools/
│   ├── export_aesx_mp4.py      # Export .aesx + slow MP4 from checkpoint
│   ├── export_mp4.py           # Smooth MP4 with joint angle bars
│   ├── export_servo_csv.py     # Export servo values to CSV
│   └── make_checkpoint_gif.py
└── media/                      # GIFs and videos per training run
```
