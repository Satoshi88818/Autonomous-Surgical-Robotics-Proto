
# Autonomous Surgical Robotics AI Training Pipeline  
**Prototype v19.0** – ACT-Inspired Chunked Transformer + Hierarchical Task Conditioning + CVAE + NVIDIA Warp + Digital Twins

**Status:** Research prototype – simulation-heavy, early-stage sim2real bridging  
**Date of this snapshot:** February 2026  
**License:** (not specified – assume research/academic use only)

## Overview

This repository contains an **end-to-end research prototype** for training **vision-language-action** policies for **autonomous da Vinci-style surgical robots**, with strong emphasis on:

- soft-tissue simulation realism  
- stochastic action generation (CVAE)  
- hierarchical & chunked action prediction (inspired by ACT / Diffusion Forcing)  
- offline BC → DAgger loop  
- digital twin integration from real CT/MRI scans  
- sim-to-real gap analysis

Current focus: **suturing, cutting, grasping, tissue manipulation** tasks using replayed JIGSAWS teleoperation data + synthetic Warp-based simulation.

## Key Technical Features (v19 highlights)

- **Physics** → NVIDIA Warp (XPBD solver) – GPU-accelerated, differentiable soft bodies  
- **Digital Twins** → patient-specific meshes from NIfTI (CT/MRI) via marching cubes  
- **Blood flow** → simple particle advection + bleeding triggers  
- **Variable tissue properties** → diseased vs. healthy elasticity randomization  
- **Stochastic policy** → Conditional Variational Autoencoder (CVAE) for action chunks  
- **Observation** → multi-modal: joint states (76-dim) + RGB-D + segmentation (8 ch)  
- **Action space** → position deltas (6DoF × 2 arms) + gripper + tool swap  
- **Temporal modeling** → history of 5 steps + predict next 10-step action chunk  
- **Training** → Behavior Cloning → DAgger → KL-regularized CVAE loss  
- **Evaluation** → multi-seed sim success rate + reward + haptic violation metrics  
- **Deployment stub** → ROS 2 node with haptic force visualization  

## Architecture

```
Data Sources
├─ JIGSAWS teleop kinematics & video (real expert)
└─ Warp simulation (synthetic augmentation + curriculum)

↓ (offline collection + DAgger)

SurgicalDataset
├─ History vector (T=5 × 76)
├─ History images (T=5 × 224×224×8)
├─ Future action chunk (T=10 × action_dim)
└─ Task ID embedding

↓

CVAE-ACT Model
├─ Vision: ConvNeXt backbone → 384-dim emb per frame
├─ Vector state → Linear projection
├─ Concat + Task embedding + PosEnc
├─ TransformerEncoder (6 layers)
├─ CVAE head: μ, logvar → z ~ N(μ,σ²)
└─ Decoder: z + task → 10-step action chunk

↓ (MSE + smoothness + KL)

Optimizer: AdamW · LR 3e-5 · grad clip 1.5

↓ (DAgger loop)

Policy rollouts in Warp env → expert correction → aggregate

↓

Evaluation: success rate, avg reward, force/penetration violation

↓ (optional)

ROS 2 inference node → Warp sim + haptic viz
```

## Requirements

- Python 3.10–3.12  
- PyTorch 2.1+ (CUDA 12.x recommended)  
- NVIDIA Warp (`pip install warp-lang`)  
- `nibabel`, `scikit-image`, `torchvision`, `pandas`, `opencv-python`  
- ROS 2 Humble / Iron (for deployment node)  
- (optional) JIGSAWS dataset (~20–30 GB)

```bash
# Minimal working set (Ubuntu 22.04/24.04 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install warp-lang nibabel scikit-image pandas opencv-python matplotlib rclpy
```

## Installation & Quick Start

1. Clone repo
```bash
git clone https://github.com/your-org/autonomous-surgical-robotics.git
cd autonomous-surgical-robotics
```

2. Install dependencies (see above)

3. Prepare data
   - Place JIGSAWS kinematics & video files under `./jigsaws_data/`
   - (optional) Add sample NIfTI file as `patient_ct.nii`

4. Train from real JIGSAWS data
```bash
python main.py   # runs acquire → train → DAgger → eval → save
```

Outputs:
- `surgical_model_v18.pth` (or v19)
- `norms_v18.pth`
- `losses_v18.png`
- `logs_v18.json`

5. Visualize simulation (GUI mode)
```python
env = WarpSurgicalEnv(gui=True)
```

6. Run ROS 2 inference loop
```bash
ros2 run <your_package> surgical_node_v18
```

## Current Weaknesses & Known Limitations

| Area                     | Issue                                                                 | Severity |
|--------------------------|-----------------------------------------------------------------------|----------|
| Action application       | Very simplistic joint control (no proper IK / OSC)                    | ★★★★     |
| Cutting & suturing       | Extremely crude geometric primitives – no real topology change       | ★★★★★    |
| Thread / needle physics  | Basic spring chain – no real needle threading or knotting             | ★★★★     |
| Digital twin             | Marching cubes surface mesh only – no good tet mesh yet               | ★★★★     |
| Sim2real gap             | No domain randomization for camera / lighting / latency               | ★★★      |
| Tool changing            | Stub – no geometry or collision change                                 | ★★★      |
| Haptic rendering         | Very approximate – no proper tool-tissue force model                   | ★★★      |
| Scalability              | Full Warp sim inside training loop is slow even on high-end GPU       | ★★★      |
| Task diversity           | Only 5 coarse tasks – no fine-grained gesture parsing                  | ★★       |

## Suggested Improvements (Short–Medium Term)

1. Replace manual joint stepping with proper **Operational Space Control** (OSC) or differential IK  
2. Integrate **ARCSim / SOFA / MuJoCo-FEM** hybrid for more realistic cutting & suturing  
3. Use **tetgen / PyTetWild** to generate proper tetrahedral meshes from digital twins  
4. Implement **domain randomization** pipeline (camera intrinsics, lighting, specular, blood amount, breathing motion)  
5. Replace chunk prediction with **diffusion policy** or **autoregressive tokenization + LLM-style decoding**  
6. Add **force/torque prediction head** and train with haptic augmentation loss  
7. Port evaluation to **real da Vinci Research Kit (dVRK)** or **da Vinci Si/Xi** via ROS bridge  
8. Record **multi-view RGB-D** streams and fuse in model (stereo + overhead)  
9. Implement **online RL fine-tuning** loop (PPO / DrQ-v2 style) using Warp gradients  

## Longer-term Future Directions

- Full topology-aware tissue cutting & needle threading  
- Multi-task hierarchical policy with LLM-style task decomposition  
- Foundation model pre-training on large-scale surgical video datasets  
- Zero-shot generalization to unseen anatomies via digital twin + language  
- Closed-loop autonomous suturing on physical phantom / cadaver  
- Regulatory-grade sim2real validation pipeline (FDA / ISO 13485 track)

## Citation

If you use ideas or code from this prototype in your research, please consider citing:

```bibtex
@misc{surgical-robotics-prototype-2026,
  author       = {Your Name / Team},
  title        = {Autonomous Surgical Robotics – Warp + CVAE + Digital Twins Prototype},
  year         = {2026},
  note         = {Research prototype – v19.0}
}
```

## Contributing

This is currently a solo / small-team research prototype.

Welcomed: bug reports, sim realism suggestions, better cutting/suturing ideas, digital-twin meshing help.

Pull requests welcome – especially in the following areas:

- realistic soft-body primitives  
- better action representation / control  
- domain randomization suite  
- real hardware integration stubs

---

**Happy hacking – and stay precise!** ✂️🧵
```

Feel free to modify author names, license, repository link, version numbering, or add badges (Python version, CUDA, arXiv link, etc.).

