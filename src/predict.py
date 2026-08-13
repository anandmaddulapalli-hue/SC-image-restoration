import os
import time
import numpy as np
import torch

from model import RestorationNet


# ============================================================
# CONFIGURATION
# ============================================================

# Test input directory
TEST_DIR = r"D:\SEMICON\dataset\test\NoisyLR"

# Trained model
MODEL_PATH = "checkpoints/edge_aware/best_model.pth"

# Output directory
OUTPUT_DIR = "results/test_restored"


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("TEST SET INFERENCE")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD MODEL
# ============================================================

model = RestorationNet()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("\nModel loaded successfully.")
print(f"Model: {MODEL_PATH}")


# ============================================================
# FIND TEST FILES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

test_files = sorted(
    [
        f for f in os.listdir(TEST_DIR)
        if f.endswith(".npy")
    ]
)

print(f"\nTest images found: {len(test_files)}")


# ============================================================
# INFERENCE
# ============================================================

total_time = 0.0

with torch.no_grad():

    for index, filename in enumerate(test_files):

        input_path = os.path.join(
            TEST_DIR,
            filename
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        # ----------------------------------------------------
        # Load NumPy image
        # ----------------------------------------------------

        noisy = np.load(input_path).astype(np.float32)

        # ----------------------------------------------------
        # NumPy → PyTorch
        # Shape:
        # (128,128)
        #      ↓
        # (1,1,128,128)
        # ----------------------------------------------------

        input_tensor = torch.from_numpy(noisy)

        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

        input_tensor = input_tensor.to(device)

        # ----------------------------------------------------
        # GPU synchronization for accurate timing
        # ----------------------------------------------------

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        restored = model(input_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time

        total_time += elapsed

        # ----------------------------------------------------
        # Tensor → NumPy
        # ----------------------------------------------------

        restored = restored.squeeze().cpu().numpy()

        # ----------------------------------------------------
        # Save restored image
        # ----------------------------------------------------

        np.save(
            output_path,
            restored.astype(np.float32)
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (index + 1) % 50 == 0 or index == 0:

            print(
                f"Processed {index + 1}/{len(test_files)}"
            )


# ============================================================
# SUMMARY
# ============================================================

average_time = total_time / len(test_files)

print("\n" + "=" * 70)
print("INFERENCE COMPLETE")
print("=" * 70)

print(f"Images processed       : {len(test_files)}")
print(f"Average inference time : {average_time * 1000:.2f} ms/image")
print(
    f"Inference speed        : "
    f"{1 / average_time:.2f} images/second"
)

print(f"\nRestored images saved to:")
print(OUTPUT_DIR)