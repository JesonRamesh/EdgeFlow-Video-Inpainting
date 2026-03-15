# EdgeFlow — Localized Clean Plate Pipeline

A computer vision pipeline that automatically removes a subject from video and reconstructs the background behind them — producing what VFX artists call a **clean plate**. Built entirely on Apple Silicon (MPS) without any cloud compute or CUDA GPU.

---

## What It Does

Given a raw video with a person in front of a background, the pipeline outputs a new video where the person has been seamlessly erased and the background reconstructed frame-by-frame.

```
inputs/sample.mp4  →  [Pipeline]  →  outputs/clean_plate.mp4
```

The subject is not cropped or blurred — the background behind them is *synthesised* from surrounding frames using optical flow-guided video inpainting.

---

## Pipeline Architecture

The pipeline runs four sequential steps, each building on the last:

```
Step 1 — MPS Setup
         Verify Apple GPU (Metal Performance Shaders) is available.
         Route all model weights and tensors onto the unified memory GPU.

Step 2 — SAM 2 Object Tracking
         User draws a bounding box around the subject on frame 0.
         SAM 2's memory-bank architecture propagates the mask forward
         through every frame. Output: one binary PNG mask per frame.

Step 3 — RAFT Optical Flow
         RAFT Large computes dense per-pixel motion vectors between
         every consecutive frame pair. These vectors describe exactly
         how every pixel in the scene moved between frames.
         Output: one .npy flow field per frame pair.

Step 4 — ProPainter Video Inpainting
         Three-stage neural inpainting:
           (a) Bidirectional flow estimation in masked regions
           (b) Flow completion — predict background motion behind subject
           (c) Sparse transformer — warp real pixels + synthesise unseen texture
         Output: outputs/clean_plate.mp4
```

### Data Flow

```
inputs/sample.mp4
       │
       ▼
intermediate/frames/        ← 05-digit JPEGs (00000.jpg … N.jpg)
       │
       ├──► SAM 2 ──────────► intermediate/masks/    ← binary PNGs
       │
       ├──► RAFT ───────────► intermediate/flow/     ← .npy float32 [2,H,W]
       │                      intermediate/flow/viz/ ← HSV colour previews
       │
       └──► ProPainter ─────► outputs/clean_plate.mp4
```

---

## Models Used

| Model | Purpose | Source |
|---|---|---|
| **SAM 2.1 Hiera Small** | Per-frame segmentation masks via video propagation | Meta AI |
| **RAFT Large** (C_T_SKHT_V2) | Dense optical flow estimation | torchvision built-in |
| **ProPainter** | Flow-guided video inpainting | S-Lab, NTU Singapore |

All weights are stored locally in `weights/` — no internet connection required at inference time.

---

## Requirements

- **Hardware:** Apple Silicon Mac (M1 or later), 16 GB+ unified memory recommended, 24 GB for best results
- **OS:** macOS 12.3+
- **Python:** 3.10+

### Setup

```bash
# 1. Clone the repo (with submodules)
git clone https://github.com/JesonRamesh/EdgeFlow-Video-Inpainting.git
cd EdgeFlow-Video-Inpainting

# 2. Create and activate a conda environment
conda create -n computer-vision python=3.12
conda activate computer-vision

# 3. Install PyTorch with MPS support
pip install torch==2.5.1 torchvision==0.20.1

# 4. Install other dependencies
pip install opencv-python numpy scipy imageio pillow tqdm

# 5. Install SAM 2 from source
pip install -e "third_party/sam2"

# 6. Download model weights
mkdir -p weights

curl -L -o weights/sam2.1_hiera_small.pt \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"

curl -L -o weights/ProPainter.pth \
  "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth"

curl -L -o weights/raft-things.pth \
  "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth"

curl -L -o weights/recurrent_flow_completion.pth \
  "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth"
```

---

## Usage

```bash
# Place your video at:
cp your_video.mp4 inputs/sample.mp4

# Run the full pipeline:
python main.py
```

On first run, an OpenCV window opens on the first frame. **Click and drag** a bounding box around the subject, then press **Enter** to confirm. The pipeline then runs fully automatically — no further input needed.

Steps are **idempotent** — if masks or flow fields already exist they are skipped. Re-run `main.py` at any time to resume from where it left off.

### Output

```
outputs/clean_plate.mp4   ← Final inpainted video
intermediate/masks/       ← Per-frame binary masks (for inspection)
intermediate/flow/viz/    ← Colour-coded flow visualisations (every 50 frames)
```

---

## Key Technical Decisions

**Why bounding box instead of a point click for SAM 2?**
A single click is spatially ambiguous — clicking a shirt gives SAM 2 no information about whether you want the shirt, the torso, or the full body. A bounding box encodes the intended spatial extent explicitly, reliably producing a full-body mask every time.

**Why run RAFT at 1024×576 instead of 1920×1080?**
At full resolution, RAFT's 4D all-pairs correlation volume requires ~4.2 GB — too large for MPS. 1024×576 was chosen because both dimensions are divisible by 8 (RAFT's encoder stride requirement) AND the scale factor back to 1080p is exactly 1.875× in both axes — one scalar corrects both u and v displacement channels uniformly.

**Why run ProPainter as a subprocess?**
ProPainter uses package-relative imports that only resolve when Python's working directory is inside `third_party/propainter/`. Rather than patching sys.path, we launch it as a child process with the correct `cwd`. The child inherits `PYTORCH_ENABLE_MPS_FALLBACK=1` from the parent, so unsupported MPS ops fall back to CPU automatically.

**Why `PYTORCH_ENABLE_MPS_FALLBACK=1` before `import torch`?**
PyTorch builds its MPS dispatch tables at import time. The environment variable must be set *before* `import torch` — setting it after has no effect.

---

## Project Structure

```
EdgeFlow-Video-Inpainting/
├── main.py                 # Full pipeline — single entry point
├── PIPELINE_GUIDE.md       # Deep-dive learning guide (CV concepts + code walkthrough)
├── inputs/                 # Place your .mp4 here
├── weights/                # Model checkpoints (gitignored)
├── third_party/
│   ├── sam2/               # Meta's SAM 2 (pip installed)
│   └── propainter/         # ProPainter inference code
├── intermediate/
│   ├── frames/             # Extracted JPEGs (gitignored)
│   ├── masks/              # SAM 2 binary PNG masks
│   └── flow/               # RAFT .npy flow fields + viz/
└── outputs/                # clean_plate.mp4 lives here
```

---

## Learning Resource

See [`PIPELINE_GUIDE.md`](PIPELINE_GUIDE.md) for a full breakdown of every CV concept in the pipeline — written for beginners. Covers: Apple MPS unified memory, SAM 2's memory bank architecture, bounding box vs point prompts, optical flow math, RAFT's all-pairs correlation volume, and ProPainter's three-stage inpainting approach.

---

## Acknowledgements

- [Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/segment-anything-2) — Meta AI Research
- [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://github.com/princeton-vl/RAFT) — Teed & Deng, Princeton (via torchvision)
- [ProPainter: Improving Propagation and Transformer for Video Inpainting](https://github.com/sczhou/ProPainter) — Zhou et al., S-Lab NTU Singapore
