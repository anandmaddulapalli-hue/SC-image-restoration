import os
import numpy as np


# ============================================================
# CHANGE THESE PATHS TO YOUR ACTUAL DATASET PATH
# ============================================================

TRAIN_GT_DIR = r"D:\SEMICON\dataset\train\GT"
TRAIN_NOISY_DIR = r"D:\SEMICON\dataset\train\NoisyLR"


# ============================================================
# FUNCTION TO INSPECT A DIRECTORY
# ============================================================

def inspect_directory(directory, name):

    print("\n" + "=" * 70)
    print(f"INSPECTING: {name}")
    print("=" * 70)

    files = sorted(
        f for f in os.listdir(directory)
        if f.endswith(".npy")
    )

    print(f"Number of .npy files: {len(files)}")

    if len(files) == 0:
        print("ERROR: No .npy files found!")
        return

    # --------------------------------------------------------
    # Inspect first file
    # --------------------------------------------------------

    first_file = os.path.join(directory, files[0])

    image = np.load(first_file)

    print("\nFirst file:")
    print(f"  Filename       : {files[0]}")
    print(f"  Shape          : {image.shape}")
    print(f"  Data type      : {image.dtype}")
    print(f"  Minimum value  : {image.min()}")
    print(f"  Maximum value  : {image.max()}")
    print(f"  Mean           : {image.mean():.6f}")
    print(f"  Std deviation  : {image.std():.6f}")

    # --------------------------------------------------------
    # Inspect first 10 files
    # --------------------------------------------------------

    print("\nFirst 10 files:")

    for filename in files[:10]:

        path = os.path.join(directory, filename)

        arr = np.load(path)

        print(
            f"  {filename:20s}"
            f" shape={str(arr.shape):15s}"
            f" dtype={str(arr.dtype):10s}"
            f" min={arr.min():10.3f}"
            f" max={arr.max():10.3f}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    inspect_directory(
        TRAIN_GT_DIR,
        "TRAIN - GROUND TRUTH"
    )

    inspect_directory(
        TRAIN_NOISY_DIR,
        "TRAIN - NOISY LR"
    )