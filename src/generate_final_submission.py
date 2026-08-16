import os
import glob
import numpy as np
import torch

from model_hf_residual import HFResidualSRNet


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = r"D:\SEMICON\dataset\test\NoisyLR"

CHECKPOINT = (
    r"checkpoints\hf_residual\best_model.pth"
)

OUTPUT_DIR = (
    r"results\final_submission"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("FINAL SUBMISSION GENERATION")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = HFResidualSRNet(
    channels=96,
    num_blocks=8
).to(device)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# TRANSFORMATIONS
# ============================================================

def transform(x, mode):

    if mode == 0:
        return x

    if mode == 1:
        return torch.flip(
            x,
            dims=[-1]
        )

    if mode == 2:
        return torch.flip(
            x,
            dims=[-2]
        )

    if mode == 3:
        return torch.flip(
            x,
            dims=[-2, -1]
        )

    if mode == 4:
        return torch.rot90(
            x,
            1,
            dims=[-2, -1]
        )

    if mode == 5:
        return torch.rot90(
            x,
            2,
            dims=[-2, -1]
        )

    if mode == 6:
        return torch.rot90(
            x,
            3,
            dims=[-2, -1]
        )

    if mode == 7:

        x = torch.rot90(
            x,
            1,
            dims=[-2, -1]
        )

        return torch.flip(
            x,
            dims=[-1]
        )

    return x


def inverse_transform(x, mode):

    if mode == 0:
        return x

    if mode == 1:
        return torch.flip(
            x,
            dims=[-1]
        )

    if mode == 2:
        return torch.flip(
            x,
            dims=[-2]
        )

    if mode == 3:
        return torch.flip(
            x,
            dims=[-2, -1]
        )

    if mode == 4:
        return torch.rot90(
            x,
            3,
            dims=[-2, -1]
        )

    if mode == 5:
        return torch.rot90(
            x,
            2,
            dims=[-2, -1]
        )

    if mode == 6:
        return torch.rot90(
            x,
            1,
            dims=[-2, -1]
        )

    if mode == 7:

        x = torch.flip(
            x,
            dims=[-1]
        )

        return torch.rot90(
            x,
            3,
            dims=[-2, -1]
        )

    return x


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

print()
print(
    "Test images found:",
    len(test_files)
)

if len(test_files) != 400:

    raise RuntimeError(
        f"Expected 400 test images, "
        f"found {len(test_files)}"
    )


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print()
print("Generating final predictions...")
print()

with torch.no_grad():

    for index, file_path in enumerate(
        test_files
    ):

        filename = os.path.basename(
            file_path
        )

        # ----------------------------------------------------
        # Load input
        # ----------------------------------------------------

        image = np.load(
            file_path
        )

        tensor = torch.from_numpy(
            image
        ).float()

        tensor = tensor.unsqueeze(
            0
        ).unsqueeze(
            0
        )

        tensor = tensor.to(
            device
        )

        # ----------------------------------------------------
        # 8-way self ensemble
        # ----------------------------------------------------

        predictions = []

        for mode in range(8):

            transformed = transform(
                tensor,
                mode
            )

            prediction = model(
                transformed
            )

            prediction = inverse_transform(
                prediction,
                mode
            )

            predictions.append(
                prediction
            )

        # ----------------------------------------------------
        # Average predictions
        # ----------------------------------------------------

        prediction = torch.stack(
            predictions,
            dim=0
        ).mean(
            dim=0
        )

        # ----------------------------------------------------
        # Valid image range
        # ----------------------------------------------------

        prediction = torch.clamp(
            prediction,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------

        output = (
            prediction
            .squeeze()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        # ----------------------------------------------------
        # Save with SAME filename
        # ----------------------------------------------------

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        np.save(
            output_path,
            output
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (index + 1) % 25 == 0
            or index == 0
        ):

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(test_files)}"
            )


# ============================================================
# FINAL CHECK
# ============================================================

output_files = sorted(
    glob.glob(
        os.path.join(
            OUTPUT_DIR,
            "*.npy"
        )
    )
)

print()
print("=" * 70)
print("FINAL SUBMISSION READY")
print("=" * 70)

print(
    "Input images :",
    len(test_files)
)

print(
    "Output images:",
    len(output_files)
)

# Check one output

sample = np.load(
    output_files[0]
)

print()
print(
    "Sample output:",
    output_files[0]
)

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

if len(output_files) != 400:

    raise RuntimeError(
        "Output count is not 400!"
    )

if sample.shape != (
    256,
    256
):

    raise RuntimeError(
        "Output shape is not 256x256!"
    )

print()
print("ALL CHECKS PASSED!")
print()
print(
    "Submission folder:"
)
print(
    os.path.abspath(
        OUTPUT_DIR
    )
)