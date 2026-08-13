import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# DATASET PATHS
# ============================================================

GT_DIR = r"D:\SEMICON\dataset\train\GT"
NOISY_DIR = r"D:\SEMICON\dataset\train\NoisyLR"


# ============================================================
# SAMPLE FILES
# ============================================================

samples = [
    "000000.npy",
    "000001.npy",
    "000002.npy",
    "000003.npy"
]


# ============================================================
# VISUALIZE
# ============================================================

fig, axes = plt.subplots(
    len(samples),
    2,
    figsize=(10, 16)
)


for row, filename in enumerate(samples):

    gt = np.load(
        os.path.join(GT_DIR, filename)
    )

    noisy = np.load(
        os.path.join(NOISY_DIR, filename)
    )

    # --------------------------------------------------------
    # Noisy LR
    # --------------------------------------------------------

    axes[row, 0].imshow(
        noisy,
        cmap="gray"
    )

    axes[row, 0].set_title(
        f"NoisyLR - {filename}\n"
        f"128x128 | min={noisy.min():.3f}, "
        f"max={noisy.max():.3f}"
    )

    axes[row, 0].axis("off")

    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    axes[row, 1].imshow(
        gt,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[row, 1].set_title(
        f"Ground Truth - {filename}\n"
        f"256x256"
    )

    axes[row, 1].axis("off")


plt.tight_layout()

plt.show()