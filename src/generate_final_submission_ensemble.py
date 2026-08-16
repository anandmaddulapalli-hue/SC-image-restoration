import os
import sys
import glob

import numpy as np
import torch
from model_hf_residual import HFResidualSRNet


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "checkpoints/hf_residual/best_model.pth"


# ============================================================
# CHECK ARGUMENTS
# ============================================================

if len(sys.argv) != 3:

    print("\nUsage:")
    print(
        "python src/generate_final_submission_ensemble.py "
        "<test_input_directory> <output_directory>"
    )

    print("\nExample:")
    print(
        r"python src/generate_final_submission_ensemble.py "
        r"D:\SEMICON\dataset\test\NoisyLR "
        r"results\final_submission"
    )

    sys.exit(1)


TEST_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("FINAL SUBMISSION - HF RESIDUAL + SELF-ENSEMBLE")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.isdir(TEST_DIR):

    print("\nERROR: Test directory not found:")
    print(TEST_DIR)

    sys.exit(1)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.isfile(MODEL_PATH):

    print("\nERROR: Model checkpoint not found:")
    print(MODEL_PATH)

    sys.exit(1)


# ============================================================
# FIND TEST FILES
# ============================================================

test_files = sorted(
    glob.glob(
        os.path.join(
            TEST_DIR,
            "*.npy"
        )
    )
)

print(
    f"\nTest images found: {len(test_files)}"
)

if len(test_files) == 0:

    print("\nERROR: No .npy files found.")

    sys.exit(1)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading HF Residual model...")

model = HFResidualSRNet(
    channels=96,
    num_blocks=8
).to(device)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# SELF-ENSEMBLE
# ============================================================

def self_ensemble_predict(
    model,
    image
):

    # --------------------------------------------------------
    # 1. Original
    # --------------------------------------------------------

    pred_original = model(image)

    # --------------------------------------------------------
    # 2. Horizontal flip
    # --------------------------------------------------------

    image_h = torch.flip(
        image,
        dims=[3]
    )

    pred_h = model(image_h)

    pred_h = torch.flip(
        pred_h,
        dims=[3]
    )

    # --------------------------------------------------------
    # 3. Vertical flip
    # --------------------------------------------------------

    image_v = torch.flip(
        image,
        dims=[2]
    )

    pred_v = model(image_v)

    pred_v = torch.flip(
        pred_v,
        dims=[2]
    )

    # --------------------------------------------------------
    # 4. Horizontal + vertical flip
    # --------------------------------------------------------

    image_hv = torch.flip(
        image,
        dims=[2, 3]
    )

    pred_hv = model(image_hv)

    pred_hv = torch.flip(
        pred_hv,
        dims=[2, 3]
    )

    # --------------------------------------------------------
    # Average
    # --------------------------------------------------------

    prediction = (
        pred_original
        + pred_h
        + pred_v
        + pred_hv
    ) / 4.0

    return prediction


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating final predictions...\n")


with torch.no_grad():

    for index, file_path in enumerate(test_files):

        # ----------------------------------------------------
        # Load LR image
        # ----------------------------------------------------

        image = np.load(
            file_path
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Convert to tensor
        # ----------------------------------------------------

        tensor = torch.from_numpy(
            image
        ).unsqueeze(0).unsqueeze(0)

        tensor = tensor.to(device)

        # ----------------------------------------------------
        # Self-ensemble inference
        # ----------------------------------------------------

        prediction = self_ensemble_predict(
            model,
            tensor
        )

        # ----------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------

        prediction = (
            prediction
            .squeeze()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Valid image range
        # ----------------------------------------------------

        prediction = np.clip(
            prediction,
            0.0,
            1.0
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Preserve original filename
        # ----------------------------------------------------

        filename = os.path.basename(
            file_path
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        np.save(
            output_path,
            prediction
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        current = index + 1

        if (
            current == 1
            or current % 25 == 0
            or current == len(test_files)
        ):

            print(
                f"Processed {current}/{len(test_files)}"
            )


# ============================================================
# FINAL VALIDATION
# ============================================================

output_files = sorted(
    glob.glob(
        os.path.join(
            OUTPUT_DIR,
            "*.npy"
        )
    )
)


print("\n" + "=" * 70)
print("FINAL SUBMISSION READY")
print("=" * 70)

print(
    f"Input images : {len(test_files)}"
)

print(
    f"Output images: {len(output_files)}"
)


if len(test_files) != len(output_files):

    print(
        "\nERROR: Input/output count mismatch!"
    )

    sys.exit(1)


# ------------------------------------------------------------
# Check all output shapes
# ------------------------------------------------------------

print("\nChecking output shapes...")

for file_path in output_files:

    output = np.load(
        file_path
    )

    if output.shape != (256, 256):

        print(
            "\nERROR: Invalid output shape:"
        )

        print(
            file_path,
            output.shape
        )

        sys.exit(1)


# ------------------------------------------------------------
# Check sample
# ------------------------------------------------------------

sample_path = output_files[0]

sample = np.load(
    sample_path
)

print("\nSample output:")
print(sample_path)

print(
    "Shape:",
    sample.shape
)

print(
    "Minimum:",
    sample.min()
)

print(
    "Maximum:",
    sample.max()
)

print("\nALL CHECKS PASSED!")

print("\nSubmission folder:")
print(
    os.path.abspath(
        OUTPUT_DIR
    )
)