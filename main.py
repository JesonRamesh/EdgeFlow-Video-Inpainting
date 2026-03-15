import os
import sys

# Must be set before `import torch` — MPS dispatch tables are built at import
# time and the fallback routes for unsupported ops (e.g. upsample_bicubic2d,
# deform_conv2d) must be registered before any model is loaded.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch
import numpy as np

# Add third_party/sam2 to sys.path defensively; a no-op if already pip-installed.
SAM2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "sam2")
if SAM2_ROOT not in sys.path:
    sys.path.insert(0, SAM2_ROOT)

from sam2.build_sam import build_sam2_video_predictor


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1 — Device setup
# ─────────────────────────────────────────────────────────────────────────────

def check_mps_availability() -> torch.device:
    """Return MPS device if available, otherwise fall back to CPU."""
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("[!] PyTorch was not built with MPS support.")
        else:
            print("[!] MPS unavailable — macOS 12.3+ and Apple Silicon required.")
        print("    Falling back to CPU.")
        return torch.device("cpu")

    print("[ok] MPS available — using Apple GPU (Metal Performance Shaders).")
    return torch.device("mps")


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2 — SAM 2 object tracking
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, frames_dir: str) -> int:
    """
    Decode all frames from video_path into frames_dir as zero-padded JPEGs.
    SAM 2's init_state() requires a flat directory of lexicographically sorted
    image files rather than a raw video. Returns the total frame count.
    """
    os.makedirs(frames_dir, exist_ok=True)

    existing = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    if existing:
        print(f"[extract_frames] Found {len(existing)} existing frames — skipping.")
        return len(existing)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"[extract_frames] {total_frames} frames @ {fps:.2f} fps")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(
            os.path.join(frames_dir, f"{frame_idx:05d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        frame_idx += 1

    cap.release()
    print(f"[extract_frames] Wrote {frame_idx} frames → {frames_dir}")
    return frame_idx


# Mouse callback state for bounding-box collection.
_box_start  = None
_box_end    = None
_is_drawing = False


def _mouse_callback(event, x, y, flags, param):
    global _box_start, _box_end, _is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        _box_start, _box_end, _is_drawing = (x, y), (x, y), True
    elif event == cv2.EVENT_MOUSEMOVE and _is_drawing:
        _box_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        _box_end, _is_drawing = (x, y), False


def collect_box_prompt(first_frame_path: str) -> np.ndarray:
    """
    Display frame 0 and let the user drag a bounding box around the subject.
    Returns [x1, y1, x2, y2] float32 in original image pixel space.

    A bounding box is used instead of a point prompt because a single point
    is spatially ambiguous — SAM 2 defaults to the smallest confident object
    at the clicked location (e.g. a shirt rather than a full body). The box
    encodes the intended spatial extent explicitly.

    Mouse coordinates are in display space (scaled-down window); we divide by
    the scale factor to convert back to original image space before returning.
    """
    global _box_start, _box_end, _is_drawing
    _box_start = _box_end = None
    _is_drawing = False

    frame   = cv2.imread(first_frame_path)
    h, w    = frame.shape[:2]
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
            cv2.rectangle(vis, _box_start, _box_end, (0, 255, 0), 2)
            orig_w = int(abs(_box_end[0] - _box_start[0]) / scale)
            orig_h = int(abs(_box_end[1] - _box_start[1]) / scale)
            cv2.putText(
                vis, f"{orig_w} x {orig_h} px (original)",
                (min(_box_start[0], _box_end[0]), min(_box_start[1], _box_end[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )

        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF

        box_drawn = (
            _box_start and _box_end
            and abs(_box_end[0] - _box_start[0]) > 5
            and abs(_box_end[1] - _box_start[1]) > 5
        )
        if key in (13, 32) and box_drawn and not _is_drawing:
            break
        if key == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("User aborted prompt selection.")

    cv2.destroyAllWindows()

    x1 = int(np.clip(min(_box_start[0], _box_end[0]) / scale, 0, w - 1))
    y1 = int(np.clip(min(_box_start[1], _box_end[1]) / scale, 0, h - 1))
    x2 = int(np.clip(max(_box_start[0], _box_end[0]) / scale, 0, w - 1))
    y2 = int(np.clip(max(_box_start[1], _box_end[1]) / scale, 0, h - 1))

    box = np.array([x1, y1, x2, y2], dtype=np.float32)
    print(f"[prompt] Box confirmed (original image space): {box}")
    return box


def load_sam2_predictor(weights_dir: str, device: torch.device):
    """Load SAM 2.1 Small VideoPredictor onto device."""
    ckpt = os.path.join(weights_dir, "sam2.1_hiera_small.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"\n[!] Weights not found: {ckpt}\n"
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


def run_sam2_tracking(
    predictor,
    frames_dir: str,
    box: np.ndarray,
    masks_dir: str,
    device: torch.device,
) -> int:
    """
    Run SAM 2 video propagation from a bounding-box prompt on frame 0.
    Saves one binary PNG mask per frame into masks_dir.
    Returns the number of masks saved.
    """
    os.makedirs(masks_dir, exist_ok=True)
    mask_count = 0

    with torch.inference_mode():
        state = predictor.init_state(video_path=frames_dir)

        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            box=box,
        )

        print("[sam2] Propagating masks ...")
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            # Threshold logits at 0.0 on-device before transferring to CPU.
            binary_mask = (mask_logits[0, 0] > 0.0)
            mask_np     = (binary_mask.cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(masks_dir, f"{frame_idx:05d}.png"), mask_np)
            mask_count += 1

            if frame_idx % 30 == 0:
                coverage = binary_mask.float().mean().item() * 100
                print(f"  [{frame_idx:05d}] mask saved  |  coverage: {coverage:.1f}%")

        # Release memory bank tensors before next model loads.
        predictor.reset_state(state)

    print(f"\n[sam2] Done. {mask_count} masks → {masks_dir}")
    return mask_count


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3 — RAFT optical flow
# ─────────────────────────────────────────────────────────────────────────────

# Run RAFT at 1024×576 (both divisible by 8, RAFT's encoder stride requirement).
# Scale factor 1920/1024 = 1080/576 = 1.875 is uniform across both axes, so
# one scalar corrects both u and v displacement channels after upsampling.
_RAFT_W     = 1024
_RAFT_H     = 576
_FLOW_SCALE = 1920 / _RAFT_W  # = 1.875


def load_raft_model(device: torch.device):
    """
    Load RAFT Large (C_T_SKHT_V2) onto device.
    Returns (model, transforms) — transforms must be applied to uint8 tensors
    before inference; skipping normalisation produces ~127× inflated flow magnitudes.
    """
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    print("[raft] Loading RAFT Large ...")
    weights    = Raft_Large_Weights.C_T_SKHT_V2
    model      = raft_large(weights=weights, progress=True).to(device).eval()
    transforms = weights.transforms()
    print("[raft] Model ready.")
    return model, transforms


def _bgr_to_raft_tensor(bgr_img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a BGR uint8 frame to a uint8 RGB tensor (1, 3, H, W) on device."""
    rgb = bgr_img[:, :, ::-1].copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)


def flow_to_color_image(flow_np: np.ndarray) -> np.ndarray:
    """
    Encode a flow field [2, H, W] as a BGR colour image for inspection.
    Hue → direction, Value → magnitude (95th-percentile normalised).
    """
    u, v      = flow_np[0], flow_np[1]
    angle     = np.arctan2(v, u)
    magnitude = np.sqrt(u ** 2 + v ** 2)
    mag_cap   = np.percentile(magnitude, 95) + 1e-5
    mag_norm  = np.clip(magnitude / mag_cap, 0.0, 1.0)

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
    Compute RAFT flow for every consecutive frame pair (0→1, 1→2, …, N-2→N-1).
    Saves each field as [2, orig_H, orig_W] float32 .npy in original-image pixels.
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
    N = len(frame_paths)
    flow_count = 0

    print(f"[raft] Computing flow for {N - 1} frame pairs at {_RAFT_W}×{_RAFT_H} ...")

    with torch.inference_mode():
        for i in range(N - 1):
            out_path = os.path.join(flow_dir, f"{i:05d}.npy")

            if os.path.exists(out_path):
                flow_count += 1
                continue

            f1_bgr = cv2.resize(cv2.imread(frame_paths[i]),     (_RAFT_W, _RAFT_H))
            f2_bgr = cv2.resize(cv2.imread(frame_paths[i + 1]), (_RAFT_W, _RAFT_H))

            t1_raw = _bgr_to_raft_tensor(f1_bgr, device)
            t2_raw = _bgr_to_raft_tensor(f2_bgr, device)
            t1, t2 = raft_transforms(t1_raw, t2_raw)

            flow_preds = model(t1, t2)
            flow_small = flow_preds[-1]  # finest refinement iteration

            # Upsample spatial dims then scale displacement magnitudes to match.
            flow_full = torch.nn.functional.interpolate(
                flow_small, size=(1080, 1920), mode="bilinear", align_corners=False
            ) * _FLOW_SCALE

            flow_np = flow_full[0].cpu().numpy()
            np.save(out_path, flow_np)
            flow_count += 1

            if i % visualize_every == 0:
                cv2.imwrite(os.path.join(viz_dir, f"{i:05d}_flow.png"), flow_to_color_image(flow_np))
                mean_mag = np.sqrt(flow_np[0] ** 2 + flow_np[1] ** 2).mean()
                print(f"  [{i:05d}→{i+1:05d}]  mean motion: {mean_mag:.2f} px")

    print(f"\n[raft] Done. {flow_count} flow fields → {flow_dir}")
    return flow_count


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4 — ProPainter video inpainting
# ─────────────────────────────────────────────────────────────────────────────

def run_propainter(
    input_video: str,
    frames_dir:  str,
    masks_dir:   str,
    output_dir:  str,
    fps:         float,
    device:      torch.device,
) -> str:
    """
    Invoke ProPainter's inference script as a subprocess and return the path
    of the finished clean-plate video.

    Subprocess design: ProPainter uses package-relative imports that only
    resolve when cwd is third_party/propainter/. The child inherits
    PYTORCH_ENABLE_MPS_FALLBACK=1, allowing deform_conv2d to fall back to CPU.

    frames_dir is passed as -i (not the raw video) to bypass a PyAV/macOS bug
    where torchvision.io.read_video() raises BlockingIOError on BT.709 content.
    ProPainter's directory branch reads frames with OpenCV instead.
    """
    import subprocess
    import shutil
    import tempfile
    import shutil as _shutil

    propainter_dir = os.path.abspath(os.path.join("third_party", "propainter"))

    # Symlink project-level weights/ into ProPainter's directory so its
    # load_file_from_url() finds checkpoints without re-downloading.
    weights_link = os.path.join(propainter_dir, "weights")
    if not os.path.lexists(weights_link):
        os.symlink(os.path.abspath("weights"), weights_link)
        print(f"[propainter] Weights symlink created: {weights_link}")

    # ProPainter iterates every file in the input directories without filtering
    # by extension. Non-image files (.gitkeep, .DS_Store) cause cv2.imread to
    # return None, crashing cvtColor. Move them out for the subprocess duration.
    _tmp_dir   = tempfile.mkdtemp(prefix="propainter_hidden_")
    _evacuated = []
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
    for _dir in (frames_dir, masks_dir):
        for _fname in os.listdir(_dir):
            if os.path.splitext(_fname)[1].lower() not in _IMAGE_EXTS:
                _orig = os.path.join(_dir, _fname)
                _tmp  = os.path.join(_tmp_dir, f"{os.path.basename(_dir)}_{_fname}")
                _shutil.move(_orig, _tmp)
                _evacuated.append((_orig, _tmp))

    try:
        cmd = [
            sys.executable, "inference_propainter.py",
            "-i",  os.path.abspath(frames_dir),
            "-m",  os.path.abspath(masks_dir),
            "-o",  os.path.abspath(output_dir),
            "--width",           "640",
            "--height",          "360",
            "--save_fps",        str(int(fps)),
            "--subvideo_length", "40",
            "--neighbor_length", "10",
            "--ref_stride",      "10",
            "--mask_dilation",   "4",
        ]
        print(f"\n[propainter] Launching inference at 640×360 ...")
        print(f"  Masks:  {masks_dir}/")
        print(f"  Output: {output_dir}/\n")
        subprocess.run(cmd, cwd=propainter_dir, check=True)

    finally:
        for _orig, _tmp in _evacuated:
            if os.path.exists(_tmp):
                _shutil.move(_tmp, _orig)
        _shutil.rmtree(_tmp_dir, ignore_errors=True)

    # ProPainter writes to {output_dir}/{basename(frames_dir)}/inpaint_out.mp4.
    video_name = os.path.basename(frames_dir.rstrip("/"))
    src = os.path.join(os.path.abspath(output_dir), video_name, "inpaint_out.mp4")
    dst = os.path.join(os.path.abspath(output_dir), "clean_plate.mp4")

    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"\n[propainter] Clean plate saved → {dst}")
    else:
        print(f"[!] Expected output not found at: {src}")
        print(f"    Check {output_dir}/{video_name}/ manually.")
        dst = src

    return dst


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    INPUT_VIDEO = os.path.join("inputs",       "sample.mp4")
    WEIGHTS_DIR = "weights"
    FRAMES_DIR  = os.path.join("intermediate", "frames")
    MASKS_DIR   = os.path.join("intermediate", "masks")
    FLOW_DIR    = os.path.join("intermediate", "flow")
    OUTPUT_DIR  = "outputs"
    CLEAN_PLATE = os.path.join(OUTPUT_DIR, "clean_plate.mp4")
    os.makedirs("intermediate", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1 ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Environment Check")
    print("=" * 60)
    device = check_mps_availability()

    if not os.path.exists(INPUT_VIDEO):
        print(f"\n[!] No video found at '{INPUT_VIDEO}'.")
        sys.exit(0)

    # ── Step 2 — SAM 2 tracking ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — SAM 2 Object Tracking")
    print("=" * 60)

    total_frames = extract_frames(INPUT_VIDEO, FRAMES_DIR)

    existing_masks = (
        sorted(f for f in os.listdir(MASKS_DIR) if f.endswith(".png"))
        if os.path.isdir(MASKS_DIR) else []
    )

    if len(existing_masks) == total_frames:
        print(f"[sam2] {len(existing_masks)} masks already exist — skipping.")
        mask_count = len(existing_masks)
    else:
        box       = collect_box_prompt(os.path.join(FRAMES_DIR, "00000.jpg"))
        predictor = load_sam2_predictor(WEIGHTS_DIR, device)
        mask_count = run_sam2_tracking(predictor, FRAMES_DIR, box, MASKS_DIR, device)
        del predictor
        if device.type == "mps":
            torch.mps.empty_cache()

    print(f"[step 2] {mask_count} masks in {MASKS_DIR}/")

    # ── Step 3 — RAFT optical flow ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — RAFT Optical Flow")
    print("=" * 60)

    existing_flow = (
        sorted(f for f in os.listdir(FLOW_DIR) if f.endswith(".npy"))
        if os.path.isdir(FLOW_DIR) else []
    )

    if len(existing_flow) == total_frames - 1:
        print(f"[raft] {len(existing_flow)} flow fields already exist — skipping.")
        flow_count = len(existing_flow)
    else:
        raft_model, raft_transforms = load_raft_model(device)
        flow_count = compute_optical_flow(raft_model, raft_transforms, FRAMES_DIR, FLOW_DIR, device)
        del raft_model, raft_transforms
        if device.type == "mps":
            torch.mps.empty_cache()

    print(f"[step 3] {flow_count} flow fields in {FLOW_DIR}/")

    # ── Step 4 — ProPainter inpainting ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — ProPainter Inpainting")
    print("=" * 60)

    if os.path.exists(CLEAN_PLATE):
        print(f"[propainter] {CLEAN_PLATE} already exists — skipping.")
    else:
        cap = cv2.VideoCapture(INPUT_VIDEO)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        result_path = run_propainter(INPUT_VIDEO, FRAMES_DIR, MASKS_DIR, OUTPUT_DIR, fps, device)
        print(f"[propainter] inpainting complete → {result_path}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"  Masks  : {MASKS_DIR}/")
    print(f"  Flow   : {FLOW_DIR}/")
    print(f"  Output : {CLEAN_PLATE}")
    print(f"\n  open {CLEAN_PLATE}")


if __name__ == "__main__":
    main()
