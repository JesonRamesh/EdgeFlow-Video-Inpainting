import os
import sys

# Must be set before `import torch` so MPS dispatch tables are built with
# the fallback enabled.  This allows PyTorch to silently route any MPS-
# unsupported op (e.g. upsample_bicubic2d) to CPU rather than crashing.
# The performance hit is negligible — only the final small mask upsample
# in SAM 2's decoder falls back; all heavy ops stay on the Metal GPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — SAM 2 import
#
#  We add third_party/sam2 to sys.path defensively.  If you installed with
#  `pip install -e "third_party/sam2"`, Python already knows about it, so
#  this insert is a no-op.  Either way, the import below will succeed.
# ─────────────────────────────────────────────────────────────────────────────
SAM2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "sam2")
if SAM2_ROOT not in sys.path:
    sys.path.insert(0, SAM2_ROOT)

from sam2.build_sam import build_sam2_video_predictor


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — MPS device check  (unchanged from last session)
# ─────────────────────────────────────────────────────────────────────────────

def check_mps_availability() -> torch.device:
    """
    Verify Apple MPS is reachable and return the correct torch.device.
    Falls back to CPU so the script still runs on any machine.
    """
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("[!] PyTorch was not built with MPS support.")
        else:
            print("[!] MPS unavailable — macOS 12.3+ and an Apple Silicon chip required.")
        print("    Falling back to CPU.")
        return torch.device("cpu")

    print("[ok] MPS available — using Apple GPU (Metal Performance Shaders).")
    return torch.device("mps")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2a — Frame extraction
#
#  SAM 2's VideoPredictor.init_state() requires a flat directory of image
#  files sorted lexicographically by filename — it does NOT accept a raw
#  .mp4 file.  We decode every frame here and write zero-padded JPEGs.
#
#  CV concept: OpenCV reads frames in BGR order.  SAM 2 internally converts
#  to RGB (it uses torchvision transforms).  We store BGR on disk — this is
#  fine because SAM 2's frame loader does its own PIL-based RGB conversion.
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, frames_dir: str) -> int:
    """
    Decode all frames from video_path into frames_dir as JPEG images.
    Returns the total frame count.
    Skips extraction if frames are already present (fast restart).
    """
    os.makedirs(frames_dir, exist_ok=True)

    # Fast-path: if frames already exist, count and return them.
    existing = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    if existing:
        print(f"[extract_frames] Found {len(existing)} existing frames in {frames_dir} — skipping re-extraction.")
        return len(existing)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"[extract_frames] Video: {total_frames} frames @ {fps:.2f} fps")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(frames_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_idx += 1

    cap.release()
    print(f"[extract_frames] Wrote {frame_idx} frames → {frames_dir}")
    return frame_idx


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2b — Bounding-box prompt via OpenCV drag
#
#  CV concept: SAM 2 accepts three prompt types — points, boxes, or masks.
#  A single foreground point is ambiguous: clicking a shirt gives SAM 2 no
#  information about whether you want the shirt, the torso, or the full body.
#  A bounding box eliminates that ambiguity by explicitly encoding the spatial
#  extent of the object.  The mask decoder then segments everything inside the
#  box that belongs together, reliably producing a full-body mask.
#
#  Coordinate system note:
#    OpenCV's mouse callback always delivers coordinates in the DISPLAY window's
#    pixel space (the scaled-down version).  SAM 2 expects coordinates in the
#    ORIGINAL image's pixel space.  We must divide by the display scale factor
#    before handing coordinates to the model.  All drawing happens in display
#    space, so no scaling is applied there.
# ─────────────────────────────────────────────────────────────────────────────

# Module-level state written by the mouse callback and read by the event loop.
_box_start   = None   # (x, y) in display space — set on mouse-down
_box_end     = None   # (x, y) in display space — updated on mouse-move / mouse-up
_is_drawing  = False  # True while the left button is held


def _mouse_callback(event, x, y, flags, param):
    global _box_start, _box_end, _is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        _box_start  = (x, y)
        _box_end    = (x, y)
        _is_drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and _is_drawing:
        _box_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        _box_end    = (x, y)
        _is_drawing = False


def collect_box_prompt(first_frame_path: str) -> np.ndarray:
    """
    Display the first frame.  User click-drags to draw a bounding box around
    the subject.  Press ENTER or SPACE to confirm.  ESC to abort.

    Returns a float32 array [x1, y1, x2, y2] in ORIGINAL image pixel space,
    ready to pass directly to SAM 2's add_new_points_or_box().
    """
    global _box_start, _box_end, _is_drawing
    _box_start = _box_end = None
    _is_drawing = False

    frame = cv2.imread(first_frame_path)
    h, w  = frame.shape[:2]

    # Compute a scale so the window fits comfortably on screen.
    # scale < 1.0 means we shrink the image for display only.
    scale   = min(1.0, 1280 / w, 720 / h)
    dw, dh  = int(w * scale), int(h * scale)
    display = cv2.resize(frame, (dw, dh))

    window = "SAM 2 Prompt  |  Drag a box around the subject → ENTER to confirm"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, dw, dh)
    cv2.setMouseCallback(window, _mouse_callback)

    print("\n[prompt] Click and drag a box around the FULL body of the subject.")
    print("         Press ENTER or SPACE to confirm.  ESC to abort.\n")

    while True:
        vis = display.copy()

        if _box_start and _box_end:
            # Both corners are in display space — draw directly, no re-scaling.
            cv2.rectangle(vis, _box_start, _box_end, (0, 255, 0), 2)
            # Show the box dimensions in original-image pixels for reference.
            orig_w = int(abs(_box_end[0] - _box_start[0]) / scale)
            orig_h = int(abs(_box_end[1] - _box_start[1]) / scale)
            label  = f"{orig_w} x {orig_h} px (original)"
            cv2.putText(vis, label,
                        (min(_box_start[0], _box_end[0]),
                         min(_box_start[1], _box_end[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF

        # Only allow confirm if the user has actually drawn a box (not just a click).
        box_drawn = (_box_start and _box_end and
                     abs(_box_end[0] - _box_start[0]) > 5 and
                     abs(_box_end[1] - _box_start[1]) > 5)

        if key in (13, 32) and box_drawn and not _is_drawing:  # ENTER / SPACE
            break
        if key == 27:                                           # ESC
            cv2.destroyAllWindows()
            raise RuntimeError("User aborted prompt selection.")

    cv2.destroyAllWindows()

    # ── Convert display-space corners → original-image-space box ─────────────
    # Divide by scale to invert the resize we applied for display.
    # np.clip ensures we never go outside valid image bounds.
    x1 = int(np.clip(min(_box_start[0], _box_end[0]) / scale, 0, w - 1))
    y1 = int(np.clip(min(_box_start[1], _box_end[1]) / scale, 0, h - 1))
    x2 = int(np.clip(max(_box_start[0], _box_end[0]) / scale, 0, w - 1))
    y2 = int(np.clip(max(_box_start[1], _box_end[1]) / scale, 0, h - 1))

    box = np.array([x1, y1, x2, y2], dtype=np.float32)
    print(f"[prompt] Box confirmed (original image space): {box}")
    return box


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2c — Load SAM 2 VideoPredictor onto MPS
#
#  MPS memory note: build_sam2_video_predictor() allocates all model weights
#  into unified memory and registers them with the Metal command queue.
#  On a 24 GB M4 Pro, the 'small' model uses ~700 MB — well within budget.
#
#  We use float32 here.  SAM 2's Hiera ViT uses scaled dot-product attention
#  (SDPA) internally; PyTorch 2.5 routes MPS-SDPA through an optimised Metal
#  kernel automatically.  Forcing float16 on SAM 2 can hit unsupported MPS
#  ops in the attention layers — we leave precision management to SAM 2.
# ─────────────────────────────────────────────────────────────────────────────

def load_sam2_predictor(weights_dir: str, device: torch.device):
    """
    Build and return the SAM 2.1 Small VideoPredictor on `device`.
    The config file path is resolved via Hydra relative to the installed
    sam2 package's config directory.
    """
    ckpt = os.path.join(weights_dir, "sam2.1_hiera_small.pt")

    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"\n[!] Weights not found at: {ckpt}\n"
            "    Download with:\n"
            "    curl -L -o weights/sam2.1_hiera_small.pt \\\n"
            '      "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"'
        )

    print(f"[sam2] Loading predictor onto {device} ...")
    predictor = build_sam2_video_predictor(
        config_file="configs/sam2.1/sam2.1_hiera_s.yaml",
        ckpt_path=ckpt,
        device=device,
    )
    print("[sam2] Predictor ready.")
    return predictor


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2d — Propagate masks through the video and save to disk
#
#  CV concept: After the initial prompt on frame 0, SAM 2 runs forward-only
#  propagation.  For each new frame it:
#    1. Encodes the frame with the Hiera ViT.
#    2. Runs cross-attention against the memory bank (6 past frames + the
#       prompted frame).  The memory bank stores BOTH the spatial feature maps
#       AND the predicted mask from each past frame.
#    3. The mask decoder outputs a logit map: positive = foreground.
#    4. The current frame's encoding + mask is written back into the bank,
#       evicting the oldest entry (FIFO ring buffer).
#
#  Output: mask_logits shape is [N_objects, 1, H, W] on MPS device.
#  We threshold at 0.0 (the zero-crossing of the logit) and write uint8 PNGs.
#  PNG is lossless — critical for masks since any compression artifact would
#  corrupt the binary edge, which matters for ProPainter in Step 4.
#
#  MPS memory management:
#    - We binarise the logit tensor on the MPS device *before* calling .cpu()
#      to minimise the data transferred across the unified memory bus.
#    - After all frames are processed, reset_state() releases the memory bank
#      tensors so they can be reclaimed.  Without this, a long video will
#      quietly consume several GB of unified memory.
# ─────────────────────────────────────────────────────────────────────────────

def run_sam2_tracking(
    predictor,
    frames_dir: str,
    box: np.ndarray,
    masks_dir: str,
    device: torch.device,
) -> int:
    """
    Run SAM 2 video propagation using a bounding-box prompt on frame 0.
    `box` is [x1, y1, x2, y2] in original image pixel space (float32).
    Saves one binary PNG mask per frame into `masks_dir`.
    Returns the number of masks saved.
    """
    os.makedirs(masks_dir, exist_ok=True)
    mask_count = 0

    with torch.inference_mode():
        # ── 1. Initialise memory-bank state from the frames directory ─────────
        #
        # init_state() reads all frame filenames, builds internal lookup tables,
        # and lazily loads frames on demand.  On unified memory this is fast —
        # frame tensors will be placed directly into MPS-accessible memory.
        print("[sam2] Initialising video state ...")
        state = predictor.init_state(video_path=frames_dir)

        # ── 2. Inject the bounding box as the prompt on frame 0 ──────────────
        #
        # Using box= instead of points=/labels= gives SAM 2 explicit spatial
        # extent information.  The mask decoder treats every pixel inside the
        # box as a candidate and produces one unified segment — the largest
        # coherent object whose boundary fits within the box.  This is why a
        # box reliably returns a full-body mask while a single point on the
        # torso returns only the shirt.
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            box=box,
        )

        # ── 3. Propagate forward through the entire video ─────────────────────
        #
        # propagate_in_video() is a Python generator.  Each call yields one
        # frame's results.  This is memory-efficient: only one frame's worth
        # of logits lives in memory at a time (plus the memory bank).
        print("[sam2] Propagating masks ...")
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):

            # mask_logits: [N_objects, 1, H, W] — still on MPS device.
            # Threshold at 0.0 on-device (avoids transferring float data).
            binary_mask = (mask_logits[0, 0] > 0.0)          # bool, on MPS
            mask_np     = (binary_mask.cpu().numpy() * 255).astype(np.uint8)

            out_path = os.path.join(masks_dir, f"{frame_idx:05d}.png")
            cv2.imwrite(out_path, mask_np)
            mask_count += 1

            if frame_idx % 30 == 0:
                coverage = binary_mask.float().mean().item() * 100
                print(f"  [{frame_idx:05d}] mask saved  |  subject coverage: {coverage:.1f}%")

        # ── 4. Release the memory bank from unified memory ────────────────────
        predictor.reset_state(state)

    print(f"\n[sam2] Done. {mask_count} masks saved → {masks_dir}")
    return mask_count


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — RAFT Optical Flow
#
#  CV concept: We need a per-pixel motion map (optical flow) between every
#  consecutive frame pair so ProPainter can warp real background pixels into
#  the masked region instead of hallucinating them.
#
#  We use RAFT (Recurrent All-Pairs Field Transforms) from torchvision — no
#  extra repo clone needed.  RAFT is trained on synthetic + real datasets and
#  produces dense sub-pixel-accurate flow even for large displacements.
#
#  Resolution strategy:
#    Full 1920×1080 would create a 4D correlation volume of ~4.2 GB — too
#    large for MPS.  We downscale to 1024×576 (both divisible by 8, which
#    RAFT's feature encoder requires).  The scale factor back to 1920×1080 is
#    exactly 1.875 in both axes, so one scalar corrects both u and v channels.
#
#  Output: one .npy file per consecutive frame pair, shape [2, 1080, 1920],
#  float32, in units of original-image pixels.
# ─────────────────────────────────────────────────────────────────────────────

# Resolution we feed into RAFT (must be divisible by 8).
# 1024/1920 = 576/1080 = 0.5333... → scale back by 1920/1024 = 1.875 exactly.
_RAFT_W     = 1024
_RAFT_H     = 576
_FLOW_SCALE = 1920 / _RAFT_W   # = 1.875 — applied to both u and v after upsampling


def load_raft_model(device: torch.device):
    """
    Load RAFT Large from torchvision onto `device` and return both the model
    and its paired preprocessing transforms.

    WHY we return transforms too:
      torchvision's RAFT was trained with a specific normalisation pipeline.
      `weights.transforms()` converts uint8 [0, 255] → float32 [-1, 1].
      Skipping this and passing float32 [0, 255] directly gives the model
      inputs ~127× too large, producing completely nonsensical flow values
      (347 px mean per frame instead of the correct ~0.01–20 px range).

    We use C_T_SKHT_V2 — trained on Chairs, Things, Sintel, KITTI, HD1K, and
    Tartanair — the most generalised weight set available in torchvision 0.20.
    """
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    print("[raft] Loading RAFT Large (C_T_SKHT_V2 weights) ...")
    weights    = Raft_Large_Weights.C_T_SKHT_V2
    model      = raft_large(weights=weights, progress=True).to(device).eval()
    transforms = weights.transforms()   # OpticalFlow(): uint8 [0,255] → float32 [-1,1]
    print("[raft] Model ready.")
    return model, transforms


def _bgr_to_raft_tensor(bgr_img: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert an OpenCV BGR uint8 frame (H, W, 3) to a uint8 RGB tensor
    (1, 3, H, W) on `device`.

    We return uint8 here — NOT float32 — because torchvision's OpticalFlow
    transform (weights.transforms()) expects uint8 input and handles the
    float conversion + normalisation to [-1, 1] internally.

    We do BGR→RGB by reversing channel axis (::-1).  The .copy() makes the
    array contiguous; torch.from_numpy() requires a contiguous buffer.
    """
    rgb = bgr_img[:, :, ::-1].copy()
    t   = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)   # uint8 [1,3,H,W]
    return t.to(device)


def flow_to_color_image(flow_np: np.ndarray) -> np.ndarray:
    """
    Convert a flow field [2, H, W] (u, v in pixels) to a BGR colour image
    for human inspection.

    Encoding convention (standard optical flow visualisation):
      Hue    → direction of motion (colour wheel)
      Value  → magnitude of motion (bright = fast, dark = still)
      Saturation = 255 always

    We clip magnitude at the 95th percentile so a few large outlier
    vectors don't compress the rest of the visualisation into near-black.
    """
    u, v = flow_np[0], flow_np[1]

    angle     = np.arctan2(v, u)                        # radians in [-π, π]
    magnitude = np.sqrt(u ** 2 + v ** 2)

    # Normalise magnitude; 95th percentile avoids outlier domination
    mag_cap  = np.percentile(magnitude, 95) + 1e-5
    mag_norm = np.clip(magnitude / mag_cap, 0.0, 1.0)

    # OpenCV HSV: hue in [0, 180] (wraps at 180, not 360)
    hsv        = np.zeros((*u.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (mag_norm * 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def compute_optical_flow(
    model,
    raft_transforms,
    frames_dir: str,
    flow_dir: str,
    device: torch.device,
    visualize_every: int = 50,
) -> int:
    """
    Run RAFT on every consecutive frame pair (0→1, 1→2, …, N-2→N-1).
    Saves each flow field as a .npy file: shape [2, orig_H, orig_W], float32.
    Also saves HSV colour visualisations every `visualize_every` frames
    into flow_dir/viz/ so you can spot-check motion quality.
    Returns the number of flow fields written.
    """
    os.makedirs(flow_dir, exist_ok=True)
    viz_dir = os.path.join(flow_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    frame_paths = sorted(
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.endswith(".jpg")
    )
    N          = len(frame_paths)
    flow_count = 0

    print(f"[raft] Computing flow for {N - 1} consecutive frame pairs ...")
    print(f"       RAFT resolution: {_RAFT_W}×{_RAFT_H}  →  upscaled ×{_FLOW_SCALE} to 1920×1080")

    with torch.inference_mode():
        for i in range(N - 1):
            out_path = os.path.join(flow_dir, f"{i:05d}.npy")

            # ── Fast-restart: skip pairs already computed ─────────────────────
            if os.path.exists(out_path):
                flow_count += 1
                if i % visualize_every == 0:
                    print(f"  [{i:05d}→{i+1:05d}] already exists — skipping")
                continue

            # ── Load and resize the two frames ────────────────────────────────
            #
            # We resize HERE (not pre-baked) so the frames/ directory stays at
            # original resolution for ProPainter in Step 4.
            f1_bgr = cv2.resize(cv2.imread(frame_paths[i]),     (_RAFT_W, _RAFT_H))
            f2_bgr = cv2.resize(cv2.imread(frame_paths[i + 1]), (_RAFT_W, _RAFT_H))

            # Build uint8 tensors then apply the model's normalisation transform.
            # transforms() converts uint8 [0, 255] → float32 [-1, 1].
            # Passing the raw uint8 (or raw float [0,255]) to the model skips
            # this normalisation and produces ~100× inflated flow magnitudes.
            t1_raw = _bgr_to_raft_tensor(f1_bgr, device)   # uint8 [1,3,576,1024]
            t2_raw = _bgr_to_raft_tensor(f2_bgr, device)
            t1, t2 = raft_transforms(t1_raw, t2_raw)        # float32 [-1, 1]

            # ── Run RAFT ──────────────────────────────────────────────────────
            #
            # RAFT returns a Python list of flow tensors, one per GRU iteration
            # (default: 12 iterations).  Each entry is [1, 2, H, W] float32.
            # The list is ordered from coarsest (index 0) to finest (index -1).
            # We only need the final, most-refined prediction.
            flow_preds = model(t1, t2)
            flow_small = flow_preds[-1]          # [1, 2, 576, 1024]

            # ── Upsample flow to original 1920×1080 resolution ────────────────
            #
            # F.interpolate resizes the spatial dimensions (H, W) using bilinear
            # interpolation.  But it does NOT know these are displacement vectors,
            # so we must multiply by the scale factor manually.
            #
            # Why multiply?  If a pixel moved 10px right at 1024 wide, it moved
            # 10 × (1920/1024) = 18.75px right at 1920 wide.  The displacement
            # magnitude scales linearly with image resolution.
            flow_full = torch.nn.functional.interpolate(
                flow_small,
                size=(1080, 1920),
                mode="bilinear",
                align_corners=False,
            ) * _FLOW_SCALE                      # [1, 2, 1080, 1920]

            # ── Save to disk ──────────────────────────────────────────────────
            #
            # .cpu() moves from MPS-accessible unified memory to the NumPy
            # bridge.  On M4 Pro this is not a physical copy — just a pointer
            # handoff — but we still call it explicitly for correctness.
            flow_np = flow_full[0].cpu().numpy()  # [2, 1080, 1920]
            np.save(out_path, flow_np)
            flow_count += 1

            # ── Periodically save a colour visualisation ──────────────────────
            if i % visualize_every == 0:
                viz = flow_to_color_image(flow_np)
                cv2.imwrite(os.path.join(viz_dir, f"{i:05d}_flow.png"), viz)
                mean_mag = np.sqrt(flow_np[0] ** 2 + flow_np[1] ** 2).mean()
                print(f"  [{i:05d}→{i+1:05d}] saved  |  mean motion: {mean_mag:.2f} px")

    print(f"\n[raft] Done. {flow_count} flow fields → {flow_dir}")
    print(f"       Colour visualisations → {viz_dir}/")
    return flow_count


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — Orchestrates all steps
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Paths ─────────────────────────────────────────────────────────────────
    INPUT_VIDEO  = os.path.join("inputs",       "sample.mp4")
    WEIGHTS_DIR  = "weights"
    FRAMES_DIR   = os.path.join("intermediate", "frames")
    MASKS_DIR    = os.path.join("intermediate", "masks")
    FLOW_DIR     = os.path.join("intermediate", "flow")
    os.makedirs("intermediate", exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1 — MPS setup
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 1 — Environment Check")
    print("=" * 60)
    device = check_mps_availability()

    if not os.path.exists(INPUT_VIDEO):
        print(f"\n[!] No video found at '{INPUT_VIDEO}'.")
        sys.exit(0)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2 — SAM 2 Tracking
    #
    #  Fast-path: if masks already exist we skip re-prompting and re-tracking.
    #  This means you can re-run the script to work on Step 3 or 4 without
    #  sitting through the SAM 2 propagation again.
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 2 — SAM 2 Object Tracking")
    print("=" * 60)

    total_frames = extract_frames(INPUT_VIDEO, FRAMES_DIR)

    existing_masks = sorted(
        f for f in os.listdir(MASKS_DIR) if f.endswith(".png")
    ) if os.path.isdir(MASKS_DIR) else []

    if len(existing_masks) == total_frames:
        print(f"[sam2] {len(existing_masks)} masks already exist — skipping tracking.")
        mask_count = len(existing_masks)
    else:
        first_frame_path = os.path.join(FRAMES_DIR, "00000.jpg")
        box              = collect_box_prompt(first_frame_path)
        predictor        = load_sam2_predictor(WEIGHTS_DIR, device)
        mask_count       = run_sam2_tracking(predictor, FRAMES_DIR, box, MASKS_DIR, device)

        # Free SAM 2 weights from unified memory before loading RAFT.
        del predictor
        if device.type == "mps":
            torch.mps.empty_cache()

    print(f"[step 2] {mask_count} masks in {MASKS_DIR}/")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3 — RAFT Optical Flow
    #
    #  Fast-path: if all flow fields already exist, skip re-computation.
    #  504 frame pairs at 1024×576 takes roughly 5-8 minutes on MPS.
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 3 — RAFT Optical Flow")
    print("=" * 60)

    existing_flow = sorted(
        f for f in os.listdir(FLOW_DIR) if f.endswith(".npy")
    ) if os.path.isdir(FLOW_DIR) else []

    # We expect N-1 flow fields for N frames (one per consecutive pair).
    if len(existing_flow) == total_frames - 1:
        print(f"[raft] {len(existing_flow)} flow fields already exist — skipping.")
        flow_count = len(existing_flow)
    else:
        raft_model, raft_transforms = load_raft_model(device)
        flow_count = compute_optical_flow(raft_model, raft_transforms, FRAMES_DIR, FLOW_DIR, device)

        # Free RAFT weights before ProPainter loads in Step 4.
        del raft_model, raft_transforms
        if device.type == "mps":
            torch.mps.empty_cache()

    print(f"[step 3] {flow_count} flow fields in {FLOW_DIR}/")
    print(f"\nInspect flow visualisations:")
    print(f"  open {os.path.join(FLOW_DIR, 'viz')}/")
    print("\nConfirm the flow looks correct, then we will begin Step 4 (ProPainter).")


if __name__ == "__main__":
    main()
