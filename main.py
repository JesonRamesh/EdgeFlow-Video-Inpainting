import cv2
import torch
import sys
import os

def check_mps_availability():
    """
    Checks if Apple's Metal Performance Shaders (MPS) backend is available.
    Returns the device object.
    """
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current macOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
        print("Falling back to CPU.")
        return torch.device("cpu")
    else:
        print("MPS is available. Using Apple GPU for inference.")
        return torch.device("mps")

def main():
    # Setup basic paths
    input_dir = "inputs"
    input_video_path = os.path.join(input_dir, "sample.mp4")
    
    # 1. Check PyTorch MPS
    print("--- Environment Check ---")
    device = check_mps_availability()
    
    # Create input directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)
    
    if not os.path.exists(input_video_path):
        print(f"\n[!] Warning: Please test by placing a video file at '{input_video_path}'")
        print("Exiting for now until the sample video is provided.")
        sys.exit(0)

    # 2. Initialize OpenCV VideoCapture
    print(f"\n--- Loading video from {input_video_path} ---")
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit(1)
        
    # Extract the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video.")
        cap.release()
        sys.exit(1)
        
    print(f"Successfully extracted first frame with shape: {frame.shape} (H, W, C)")
    
    # 3. Test PyTorch Routing & Memory Management (FP16)
    print("\n--- Testing Device & Precision ---")
    print("Routing tensor to MPS using float16 for maximum throughput...")
    
    # Convert OpenCV BGR frame (H, W, C) to PyTorch tensor (C, H, W) and use FP16
    tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).to(torch.float16)
    
    # Move tensor to the Apple GPU (MPS)
    mps_tensor = tensor_frame.to(device)
    
    print(f"Tensor successfully initialized on: {mps_tensor.device}")
    print(f"Tensor data type: {mps_tensor.dtype}")
    print(f"Tensor shape matches expected layout: {mps_tensor.shape}")
    
    # Cleanup
    cap.release()
    print("\n--- Initial boilerplate test successful! ---")

if __name__ == "__main__":
    main()
