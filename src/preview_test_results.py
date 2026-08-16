import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from model_hf_residual import HFResidualSRNet


# ============================================================
# PATHS
# ============================================================

TEST_DIR = r"D:\SEMICON\dataset\test\NoisyLR"

CHECKPOINT = (
    r"checkpoints\hf_residual\best_model.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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
# TRANSFORMS
# ============================================================

def transform(x, mode):

    if mode == 0:
        return x

    if mode == 1:
        return torch.flip(x, dims=[-1])

    if mode == 2:
        return torch.flip(x, dims=[-2])

    if mode == 3:
        return torch.flip(x, dims=[-2, -1])

    if mode == 4:
        return torch.rot90(
            x, 1,
            dims=[-2, -1]
        )

    if mode == 5:
        return torch.rot90(
            x, 2,
            dims=[-2, -1]
        )

    if mode == 6:
        return torch.rot90(
            x, 3,
            dims=[-2, -1]
        )

    if mode == 7:

        x = torch.rot90(
            x, 1,
            dims=[-2, -1]
        )

        return torch.flip(
            x,
            dims=[-1]
        )


def inverse_transform(x, mode):

    if mode == 0:
        return x

    if mode == 1:
        return torch.flip(x, dims=[-1])

    if mode == 2:
        return torch.flip(x, dims=[-2])

    if mode == 3:
        return torch.flip(x, dims=[-2, -1])

    if mode == 4:
        return torch.rot90(
            x, 3,
            dims=[-2, -1]
        )

    if mode == 5:
        return torch.rot90(
            x, 2,
            dims=[-2, -1]
        )

    if mode == 6:
        return torch.rot90(
            x, 1,
            dims=[-2, -1]
        )

    if mode == 7:

        x = torch.flip(
            x,
            dims=[-1]
        )

        return torch.rot90(
            x, 3,
            dims=[-2, -1]
        )


# ============================================================
# GET FIRST 3 TEST IMAGES
# ============================================================

test_files = sorted(
    glob.glob(
        os.path.join(
            TEST_DIR,
            "*.npy"
        )
    )
)

test_files = test_files[:3]

print()
print("Previewing:")

for f in test_files:
    print(
        os.path.basename(f)
    )


# ============================================================
# GENERATE + DISPLAY
# ============================================================

with torch.no_grad():

    for file_path in test_files:

        filename = os.path.basename(
            file_path
        )

        # ----------------------------------------------------
        # Load LR test image
        # ----------------------------------------------------

        image = np.load(
            file_path
        ).astype(
            np.float32
        )

        tensor = torch.from_numpy(
            image
        )

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
        # Average
        # ----------------------------------------------------

        restored = torch.stack(
            predictions,
            dim=0
        ).mean(
            dim=0
        )

        restored = torch.clamp(
            restored,
            0.0,
            1.0
        )

        restored = (
            restored
            .squeeze()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        plt.figure(
            figsize=(10, 5)
        )

        plt.subplot(
            1, 2, 1
        )

        plt.imshow(
            image,
            cmap="gray"
        )

        plt.title(
            f"Test LR\n{filename}"
        )

        plt.axis("off")

        plt.subplot(
            1, 2, 2
        )

        plt.imshow(
            restored,
            cmap="gray"
        )

        plt.title(
            "HF Residual + 8-way Ensemble"
        )

        plt.axis("off")

        plt.tight_layout()

        plt.show()

        print()
        print(
            filename,
            "->",
            "Input:",
            image.shape,
            "| Restored:",
            restored.shape
        )