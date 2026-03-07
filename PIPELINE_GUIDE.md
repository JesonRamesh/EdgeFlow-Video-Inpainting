# Localized Clean Plate Pipeline — Complete Guide

> **How to read this document:** This is both a technical plan and a learning guide.
> Every concept is explained before the code that implements it, so you understand
> *why* each decision was made, not just *what* the code does.

---

## Table of Contents

1. [What Are We Actually Building?](#1-what-are-we-actually-building)
2. [Hardware Foundation — Apple MPS & Unified Memory](#2-hardware-foundation--apple-mps--unified-memory)
3. [The Full Pipeline at a Glance](#3-the-full-pipeline-at-a-glance)
4. [Step 1 — Environment Setup ✅](#4-step-1--environment-setup-)
5. [Step 2 — SAM 2 Object Tracking ✅](#5-step-2--sam-2-object-tracking-)
6. [Step 3 — RAFT Optical Flow ⬜](#6-step-3--raft-optical-flow-)
7. [Step 4 — ProPainter Video Inpainting ⬜](#7-step-4--propainter-video-inpainting-)
8. [Data Flow: How Files Move Between Steps](#8-data-flow-how-files-move-between-steps)
9. [Directory Structure Reference](#9-directory-structure-reference)

---

## 1. What Are We Actually Building?

### The Problem

Imagine you filmed a scene with a person standing in front of a brick wall. Now you want to
remove that person in post-production — but what goes *behind* them? The camera never recorded
that part of the wall. You need to reconstruct it.

In professional VFX, this reconstructed background is called a **clean plate** — a version of
the shot with the foreground subject removed, leaving only the background as if the subject
was never there.

We are building a **localized** clean plate pipeline. "Localized" means we only reconstruct
the background in the specific region covered by the subject — we don't touch the rest of the
frame. This is far more tractable than full-frame reconstruction and produces better results.

### The Output

- **Input:** A raw `.mp4` video with a person (or object) in front of a background.
- **Output:** A new `.mp4` video of the same shot, where the subject has been seamlessly
  removed and the background has been filled in convincingly.

### Why Is This Hard?

Three things make this a genuine computer vision problem rather than a simple photo edit:

1. **Temporal consistency** — The background fill must be stable across hundreds of frames.
   A single-frame approach leaves flickering, which looks artificial.
2. **Accurate silhouettes** — To fill behind something, you first need to know exactly where
   it is in every frame. A rough mask produces halos and dirty edges.
3. **Background motion** — The background itself may be moving (camera pan, parallax, wind
   in trees). The fill must understand and continue that motion, not freeze it.

Our four-step pipeline addresses each of these challenges in sequence.

---

## 2. Hardware Foundation — Apple MPS & Unified Memory

### Why This Matters

Neural networks operate on **tensors** — large multi-dimensional arrays of numbers.
A ResNet processes an image as a tensor of shape `[batch, channels, height, width]`.
All the matrix multiplications and convolutions that constitute "inference" need to happen
on a processor optimised for this kind of parallel floating-point math.

On a standard PC, you have a CPU for general computation and a discrete GPU (CUDA) with its
own separate VRAM for neural network math. Data must be **copied** across the PCIe bus to move
between them — this is slow and a frequent bottleneck.

### The M4 Pro Difference: Unified Memory

Your MacBook Pro M4 Pro has no separate GPU VRAM. Instead, the CPU and GPU share the same
physical DRAM pool. Apple calls this **Unified Memory Architecture (UMA)**.

```
STANDARD PC (CUDA)                        APPLE M4 PRO (MPS)
─────────────────────────────────         ──────────────────────────────
  CPU RAM  ←──PCIe bus──→  GPU VRAM         CPU + GPU share one DRAM pool
  [slow copy between them]                  [no copy needed — same memory]
```

The practical implication for our pipeline: when PyTorch moves a tensor to `mps`, it is not
physically copying data. It is registering a pointer to the same memory region with the Metal
GPU command queue. This means:

- `tensor.to('mps')` is nearly free (no bus transfer).
- `tensor.cpu()` when pulling a mask back to NumPy/OpenCV is also fast.
- We can hold large feature tensors (like SAM 2's memory bank) in GPU-addressable memory
  without worrying about VRAM limits.

### Why We Set `PYTORCH_ENABLE_MPS_FALLBACK=1`

Apple's MPS backend is a partial implementation. Not every PyTorch operator has a Metal
kernel yet. When SAM 2's mask decoder needs to run `upsample_bicubic2d` (a bilinear resize
of the small predicted mask back up to full image resolution), no Metal kernel exists for it.

Without the fallback flag, PyTorch throws a `NotImplementedError`. With it, PyTorch silently
routes just that one op to the CPU and continues. The performance cost is negligible because
the tensor being resized is tiny (a single low-resolution mask, not the full feature volume).

**Critical:** This flag must be set before `import torch`, or the MPS dispatch table is
already locked in and the flag is ignored. That is why it sits at line 9 of `main.py`,
before any other imports:

```python
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch   # dispatch tables built HERE, so the flag is already active
```

`setdefault` is used (not a plain assignment) so that if you manually set the variable in
your shell before running the script, your shell value takes precedence.

### Float16 vs Float32 — When We Use Each

| Situation | Precision | Reason |
|---|---|---|
| Pixel tensors we compute on ourselves | `float16` | We control the ops; all are MPS-native |
| SAM 2 model weights and inference | `float32` | SAM 2's attention layers hit MPS ops that don't support fp16 |
| RAFT flow fields (Step 3) | `float32` | Flow vectors need range beyond fp16 can represent |
| ProPainter inference (Step 4) | `float16` | ProPainter's ops are MPS-safe; fp16 halves memory and improves throughput |

---

## 3. The Full Pipeline at a Glance

```
inputs/sample.mp4 (1920×1080 @ 50fps, 505 frames)
         │
         ▼
┌─────────────────────────────┐
│  STEP 2 — SAM 2 Tracking    │  "Where is the subject in every frame?"
│  Model: sam2.1_hiera_small  │
│  Prompt: bounding box       │
│  Device: MPS (float32)      │
└──────────────┬──────────────┘
               │  505 binary PNG masks (white=subject, black=background)
               ▼
     intermediate/masks/
         00000.png … 00504.png
               │
               │ (also feeds back the original video)
               ▼
┌─────────────────────────────┐
│  STEP 3 — RAFT Optical Flow │  "How is the background moving between frames?"
│  Model: raft-things.pth     │
│  Input: original frames     │
│         + inverted masks    │  ← we only compute flow in background regions
│  Device: MPS (float32)      │
└──────────────┬──────────────┘
               │  505 flow fields (.npy tensors, shape [2, H, W])
               ▼
     intermediate/flow/
         00000.npy … 00504.npy
               │
               ▼
┌─────────────────────────────┐
│  STEP 4 — ProPainter        │  "Fill the masked regions with plausible background"
│  Model: ProPainter + E2FGVI │
│  Input: frames + masks      │
│         + flow fields       │
│  Device: MPS (float16)      │
└──────────────┬──────────────┘
               │  Single inpainted MP4
               ▼
     outputs/clean_plate.mp4
```

---

## 4. Step 1 — Environment Setup ✅

### Concept: Why Check the Device at Runtime?

Deep learning code must be hardware-agnostic at the source level but hardware-explicit at
runtime. Writing `torch.device('mps')` hardcoded everywhere would crash on any machine
without Apple Silicon. The pattern is always: detect the best available device once at
startup, store it in a `device` variable, and pass that variable everywhere else.

### Code Walkthrough: `check_mps_availability()`

```python
def check_mps_availability() -> torch.device:
    if not torch.backends.mps.is_available():
```
`torch.backends.mps.is_available()` checks two things internally:
(1) Is the PyTorch binary built with MPS support compiled in?
(2) Is the OS version macOS 12.3 or higher (the minimum for Metal compute shaders)?
If either fails, we can't use MPS.

```python
        if not torch.backends.mps.is_built():
```
This distinguishes *why* MPS is unavailable — a PyTorch build issue vs a hardware/OS issue.
Useful diagnostic information when setting up new machines.

```python
    return torch.device("mps")
```
`torch.device` is a lightweight object that acts as a label. When you later call
`tensor.to(device)`, PyTorch reads this label and routes accordingly. It costs nothing
to create and holding it as a module-level variable is the standard practice.

---

## 5. Step 2 — SAM 2 Object Tracking ✅

### Concept: What SAM 2 Is

**SAM 2 (Segment Anything Model 2)** is Meta's video object segmentation model, released
in 2024. Given a single prompt (a click, a box, or a drawn mask) on the *first* frame of a
video, it tracks and re-segments that object in every subsequent frame automatically.

It's important to understand what SAM 2 is *not*: it is not an optical flow model. It does
not track where pixels move. Instead, it tracks where an *object* is, using a combination
of appearance understanding and temporal memory. This makes it robust to occlusion, fast
motion, and significant appearance changes (lighting, clothing folds, etc.).

### Concept: The Memory Bank Architecture

SAM 2's most important innovation is its **memory bank**. Here is how it works:

```
Frame 0 (prompted)
    │
    ▼
[Hiera ViT Encoder] → Image Features (spatial map of "what's here")
    │
    ▼
[Mask Decoder] ← Memory Attention ← Memory Bank (starts empty)
    │                                      │
    ▼                                      │
Binary Mask for Frame 0                    │
    │                                      │
    └──── Features + Mask written ─────────┘
          into Memory Bank

Frame 1 (no prompt needed)
    │
    ▼
[Hiera ViT Encoder] → Image Features
    │
    ▼
[Mask Decoder] ← Memory Attention ← Memory Bank (contains Frame 0)
    │
    ▼
Binary Mask for Frame 1
    │
    └──── Written into Memory Bank (Frame 0 still there, up to 6 frames stored)

... and so on for every frame
```

The memory bank is a **ring buffer** of 6 frames. For each new frame, the mask decoder
runs **cross-attention** between the current frame's features and all memories in the bank.
Cross-attention lets the model ask: *"Does this pixel look like the thing I've been tracking?"*
The answer comes from the accumulated visual history — not just the previous frame, but
the last 6 frames, giving it resilience to single-frame occlusions.

The prompted frame (Frame 0) is kept in the bank permanently and never evicted. This acts
as an "anchor" — no matter how many frames have passed, the model always has the original
clean reference of what the object looked like when first shown to it.

### Concept: SAM 2 vs Optical Flow (Why We Need Both)

| Property | SAM 2 (Step 2) | RAFT Optical Flow (Step 3) |
|---|---|---|
| Tracks | Objects (semantic) | Pixel motion (geometric) |
| Output | Binary mask per frame | 2D displacement field per frame |
| Handles occlusion | Yes (memory bank fills the gap) | No (breaks down) |
| Handles fast motion | Reasonably (appearance-based) | No (large displacements break RAFT) |
| Tells you *where* pixels came from | No | Yes |
| Needed for inpainting | As mask input | As warp guidance |

We need SAM 2 to know *where* to fill. We need RAFT to know *how* to fill it coherently.

### Code Walkthrough: `extract_frames()`

```python
def extract_frames(video_path: str, frames_dir: str) -> int:
```

SAM 2's `VideoPredictor.init_state()` does not accept a raw `.mp4` file. It requires a
flat directory of image files, sorted lexicographically. This is because:
1. It needs random access to any frame at any time during propagation.
2. It supports lazy loading — it doesn't decode the entire video upfront.

```python
    existing = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    if existing:
        return len(existing)
```
**Fast-path restart:** If you re-run the script after a crash or interrupted run, we skip
re-extraction. Decoding a 505-frame 1080p video takes several seconds and produces ~500 MB
of JPEGs — no need to repeat it.

```python
    out_path = os.path.join(frames_dir, f"{frame_idx:05d}.jpg")
    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
```
`{frame_idx:05d}` produces zero-padded filenames: `00000.jpg`, `00001.jpg`, etc. This is
essential for correct lexicographic sorting (without zero-padding, `frame10.jpg` would sort
before `frame2.jpg`). Quality 95 is high enough that the JPEG compression artifacts don't
confuse SAM 2's image encoder.

### Concept: Coordinate Spaces — The Bug We Fixed

Every image has two coordinate spaces in our pipeline:

```
ORIGINAL SPACE  (1920 × 1080)            DISPLAY SPACE  (960 × 540 at scale=0.5)
─────────────────────────────            ────────────────────────────────────────
What SAM 2 expects                       What OpenCV's window shows
What we save to disk                     What the mouse callback reports
The ground truth                         A scaled-down view for the user
```

When the user drags a box in the display window, the mouse callback gives coordinates in
display space. Before passing them to SAM 2, we must divide by the scale factor to convert
back to original space:

```python
scale = min(1.0, 1280 / w, 720 / h)   # e.g. 0.667 for a 1920-wide frame

# Mouse callback gives (x, y) in display space.
# Divide by scale to get original-image coordinates for SAM 2:
x1 = int(min(_box_start[0], _box_end[0]) / scale)
```

The original bug was multiplying display-space coordinates *by* scale before drawing them
on the display image — which moved the green dot toward the origin (upper-left), producing
the "offset to the left" effect you observed.

### Concept: Bounding Box vs Point Prompt

SAM 2 accepts three prompt types. Understanding the tradeoff:

```
Point prompt:   "The object I want is near pixel (960, 400)"
                → Ambiguous. SAM 2 must guess: shirt? torso? full person? table?
                → Picks the smallest, most confident object at that location.

Bounding box:   "The object I want fits within this rectangle"
                → Explicit spatial extent.
                → SAM 2 segments the largest coherent thing whose boundary
                  is consistent with the box.  Reliably captures full body.

Mask prompt:    "Here is a rough silhouette of the object"
                → Most explicit, requires prior segmentation.
                → Used for refinement, not initial prompting.
```

### Code Walkthrough: `collect_box_prompt()`

```python
scale = min(1.0, 1280 / w, 720 / h)
display = cv2.resize(frame, (dw, dh))
```
We compute a single scale factor that fits the frame within a 1280×720 display window while
preserving aspect ratio. `min(1.0, ...)` ensures we never upscale a small video — only
shrink large ones.

```python
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, dw, dh)
cv2.setMouseCallback(window, _mouse_callback)
```
`WINDOW_NORMAL` allows the window to be resized by the user. `setMouseCallback` registers
our function to receive mouse events. OpenCV's event loop is single-threaded — the callback
runs *synchronously* inside the `waitKey` call in our loop below.

```python
# ENTER / SPACE to confirm — but only if a real box was drawn, not just a click
box_drawn = (_box_start and _box_end and
             abs(_box_end[0] - _box_start[0]) > 5 and
             abs(_box_end[1] - _box_start[1]) > 5)
```
The `> 5` guard prevents accidental confirmation if the user clicks without dragging. A
5-pixel threshold in display space is ~10 pixels in original space at our scale — too small
to be a meaningful prompt but large enough to distinguish from a stray click.

### Code Walkthrough: `load_sam2_predictor()`

```python
predictor = build_sam2_video_predictor(
    config_file="configs/sam2.1/sam2.1_hiera_s.yaml",
    ckpt_path=ckpt,
    device=device,
)
```
`build_sam2_video_predictor` uses **Hydra**, a configuration framework by Meta. Hydra reads
the YAML config file (which defines the model architecture: number of layers, attention
heads, feature dimensions) and uses it to construct the Python object graph. The checkpoint
`.pt` file then fills in the learned weights. This separation of architecture (YAML) from
weights (`.pt`) is a common pattern in research code.

The `_hiera_s` in the config name stands for **Hiera Small** — the backbone image encoder
variant. Hiera is a hierarchical Vision Transformer that processes images at multiple
resolutions simultaneously, giving it both fine spatial detail and broad semantic context.

### Code Walkthrough: `run_sam2_tracking()`

```python
with torch.inference_mode():
```
`inference_mode` is stronger than `no_grad()`. It disables the autograd engine entirely
and tells PyTorch's memory allocator that no gradients will ever be needed — enabling
more aggressive memory reuse and preventing the accidental retention of computation graphs.
Always use this for production inference, not `no_grad()`.

```python
state = predictor.init_state(video_path=frames_dir)
```
`init_state` scans the frames directory, builds a sorted list of filenames, and pre-computes
some internal lookup structures. On unified memory, the frame tensors are allocated in a
region accessible to both CPU (for disk I/O) and GPU (for the encoder) — no explicit
host-to-device copies occur.

```python
predictor.add_new_points_or_box(
    inference_state=state,
    frame_idx=0,
    obj_id=1,
    box=box,
)
```
`obj_id=1` is an arbitrary integer label for this tracked object. SAM 2 supports tracking
multiple objects simultaneously in the same video — each gets a different `obj_id`. For our
single-subject use case, `1` is conventional. This call runs the encoder + decoder on
frame 0 and writes the result into the memory bank as the permanent anchor.

```python
for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
```
`propagate_in_video` is a Python **generator** — it yields one frame's result at a time
and pauses until the `for` loop requests the next. This is critical for memory efficiency:
at any given moment, only one frame's set of logits lives in memory alongside the memory
bank. The full video is never loaded into RAM at once.

```python
    binary_mask = (mask_logits[0, 0] > 0.0)          # bool tensor, still on MPS
    mask_np     = (binary_mask.cpu().numpy() * 255).astype(np.uint8)
```
`mask_logits[0, 0]` extracts the logit map for object 0 (first object), channel 0
(there's only one channel — the foreground probability). The shape is `[H, W]`.

`> 0.0` thresholds the **raw logit** (unbounded float) into a boolean mask. A logit of 0.0
corresponds exactly to 50% probability in sigmoid space — anything above this is "probably
foreground." We apply this threshold on the MPS device before calling `.cpu()`, so we
transfer a compact boolean tensor rather than a full float32 tensor. Less data crossing
the memory bus = faster.

```python
    predictor.reset_state(state)
```
The memory bank accumulates tensors throughout propagation — 6 frames × encoded feature
maps × multiple resolution levels. For a 1080p video this can be several GB. `reset_state`
releases all reference counts, making those tensors eligible for garbage collection. Without
this line, the memory remains occupied until the Python process exits.

---

## 6. Step 3 — RAFT Optical Flow ⬜

### Concept: What Is Optical Flow?

**Optical flow** is a 2D vector field that describes, for each pixel in frame N, where that
pixel came from in frame N-1 (or equivalently, where it went to in frame N+1).

```
Frame 0                          Frame 1
┌────────────────────┐           ┌────────────────────┐
│      🧱 wall       │           │     🧱 wall         │
│   🚶 person        │    →      │       🚶 person      │   (person moved right)
└────────────────────┘           └────────────────────┘

Optical Flow Field (u, v) at each pixel:
  Background pixels:  (u≈0, v≈0)   — wall didn't move
  Person pixels:      (u≈+20, v≈0) — person moved 20px right
```

The flow field has shape `[2, H, W]`:
- Channel 0: `u` — horizontal displacement in pixels
- Channel 1: `v` — vertical displacement in pixels

Why do we need this for inpainting? Because ProPainter (Step 4) uses flow fields to **warp**
background pixels from nearby frames into the masked (missing) region of the current frame.
Without flow, it has no geometric guidance and must rely purely on texture synthesis, which
produces blurry or inconsistent results.

### Concept: Why RAFT Over Classic Methods

Classic optical flow (Lucas-Kanade, Farnebäck) works by assuming brightness constancy —
pixels keep their intensity as they move — and solving a local least-squares problem.
This fails on large displacements, low-texture regions, and lighting changes.

**RAFT (Recurrent All-Pairs Field Transforms)** works differently:
1. It builds a 4D correlation volume: every pixel in frame 0 compared to every pixel in
   frame 1 (the "all-pairs" part). This is expensive but thorough.
2. A recurrent GRU network iteratively refines the flow field over multiple passes —
   starting from zero and progressively correcting errors. This handles large displacements.
3. The result is dense, sub-pixel-accurate flow that generalises across motion types.

### What We Will Do in Step 3

- Clone RAFT into `third_party/raft` and download `raft-things.pth` (trained on synthetic
  data with large motions, best for general video).
- **Invert our SAM 2 masks** to get background-only masks. We only want flow in the
  background region — the subject region will be filled by ProPainter, so computing flow
  there is wasted work and would produce noisy vectors.
- Run RAFT on consecutive frame pairs: (0→1), (1→2), …, (503→504).
- Save each flow field as a `.npy` file in `intermediate/flow/`.
- Tensors stay on MPS in `float32` throughout; we only call `.cpu()` to hand off to NumPy
  for disk serialisation.

### What the Output Looks Like

Flow fields are typically visualised using **HSV colour coding**: hue encodes direction,
saturation/value encodes magnitude. A still background is grey; rightward motion is red;
leftward is cyan; upward is blue; downward is yellow. We will write a visualisation helper
so you can verify the flow looks sensible before feeding it to ProPainter.

---

## 7. Step 4 — ProPainter Video Inpainting ⬜

### Concept: What Is Video Inpainting?

**Inpainting** is the task of filling in a missing or masked region of an image (or video)
with plausible content. For a single photo, this can be done with texture synthesis or
generative models. For *video*, the challenge is temporal consistency — every filled frame
must agree with every other filled frame, or the result will flicker.

### Concept: How ProPainter Uses Flow

ProPainter's core idea is **flow-guided pixel propagation**:

```
Frame 5 (has a clean background pixel at position P)
    │
    │  Flow field says: pixel at P in frame 5
    │  corresponds to pixel at Q in frame 8
    ▼
Frame 8 (subject is covering position Q — it's masked)
    │
    │  ProPainter: "I know what's at Q — warp it from frame 5 using the flow"
    ▼
Frame 8 (Q filled with the warped pixel from frame 5)
```

This is called **flow-guided temporal propagation**. Pixels from unmasked regions of nearby
frames are warped into the masked region of the current frame. This gives ProPainter real
background content to work with, rather than hallucinating texture.

For regions that cannot be filled by propagation (e.g., a region masked in *every* frame
of the video — something that never moves), ProPainter falls back to its learned
**temporal attention** mechanism — essentially a spatial-temporal transformer that invents
plausible content conditioned on surrounding context.

### What We Will Do in Step 4

- Clone ProPainter into `third_party/propainter`.
- Feed it the original frames, the SAM 2 masks, and the RAFT flow fields.
- Run inference with MPS float16.
- Stitch the output frames back into a `.mp4` using OpenCV's `VideoWriter`.
- Save to `outputs/clean_plate.mp4`.

---

## 8. Data Flow: How Files Move Between Steps

```
inputs/sample.mp4
    │
    ├──[Step 2: extract_frames()]─────────────────────────────────────────────┐
    │                                                                          │
    │                                                                          ▼
    │                                                             intermediate/frames/
    │                                                             00000.jpg … 00504.jpg
    │                                                                          │
    │                                                            [Step 2: run_sam2_tracking()]
    │                                                                          │
    │                                                                          ▼
    │                                                             intermediate/masks/
    │                                                             00000.png … 00504.png
    │                                                             (binary, lossless PNG)
    │                                                                          │
    ├──[Step 3: RAFT on original frames + inverted masks]──────────────────────┤
    │                                                                          │
    │                                                                          ▼
    │                                                             intermediate/flow/
    │                                                             00000.npy … 00503.npy
    │                                                             (shape: [2, H, W], float32)
    │                                                                          │
    └──[Step 4: ProPainter]────────────────────────────────────────────────────┤
           ↑                    ↑                    ↑                         │
      original frames       masks/             flow fields                     │
                                                                               ▼
                                                                    outputs/clean_plate.mp4
```

Note that `intermediate/frames/` is gitignored — these extracted JPEGs (~500 MB) are
re-derived from the input video and don't need to be version-controlled.

### Why PNG for Masks, Not JPEG

JPEG uses **lossy compression**. Near a sharp black-white edge (like a mask boundary),
JPEG introduces "ringing" artifacts — semi-transparent grey fringe pixels. If ProPainter
ingests a mask with grey fringe pixels instead of a clean hard edge, it will partially
inpaint those border pixels, leaving a halo around where the subject was. PNG is lossless:
the white pixels are exactly 255 and the black pixels are exactly 0, always.

### Why `.npy` for Flow Fields, Not Images

Flow vectors are `float32` values with a range of roughly ±200 pixels. You cannot store
them in a regular 8-bit image format without quantisation loss. `.npy` is NumPy's native
binary format — it preserves full float32 precision, loads instantly, and has no
external dependencies.

---

## 9. Directory Structure Reference

```
Video Rotoscoping/
│
├── main.py                    ← Single execution script for all 4 steps
├── PIPELINE_GUIDE.md          ← This document
│
├── inputs/
│   └── sample.mp4             ← Your raw source video (1920×1080 @ 50fps, 505 frames)
│
├── weights/
│   ├── sam2.1_hiera_small.pt  ← SAM 2 Small backbone (~183 MB)
│   ├── raft-things.pth        ← RAFT flow model (~21 MB)  [Step 3]
│   └── ProPainter.pth         ← ProPainter inpainting model (~200 MB)  [Step 4]
│
├── third_party/
│   ├── sam2/                  ← git clone facebookresearch/sam2
│   ├── raft/                  ← git clone princeton-vl/RAFT  [Step 3]
│   └── propainter/            ← git clone sczhou/ProPainter  [Step 4]
│
├── intermediate/
│   ├── frames/                ← Extracted JPEGs for SAM 2 (gitignored, ~500 MB)
│   ├── masks/                 ← Binary PNG masks from SAM 2 (one per frame)
│   └── flow/                  ← RAFT flow fields (one .npy per consecutive pair)
│
└── outputs/
    └── clean_plate.mp4        ← Final inpainted video
```

---

*Last updated: Step 2 complete. SAM 2 bounding-box tracking verified on 505-frame 1920×1080 @ 50fps source.*
