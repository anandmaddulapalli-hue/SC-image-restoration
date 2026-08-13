import os
import numpy as np

OUTPUT_DIR = "results/test_restored"

files = sorted(
    f for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".npy")
)

print("=" * 70)
print("CHECKING RESTORED TEST OUTPUTS")
print("=" * 70)

print(f"Number of output files: {len(files)}")

shapes = {}
global_min = float("inf")
global_max = float("-inf")

for filename in files:

    path = os.path.join(OUTPUT_DIR, filename)

    image = np.load(path)

    shapes[image.shape] = shapes.get(image.shape, 0) + 1

    global_min = min(global_min, float(image.min()))
    global_max = max(global_max, float(image.max()))

print("\nOutput shapes:")

for shape, count in shapes.items():
    print(f"{shape}: {count} images")

print("\nGlobal output statistics:")
print(f"Minimum : {global_min:.6f}")
print(f"Maximum : {global_max:.6f}")

print("\nFirst 5 files:")

for filename in files[:5]:

    image = np.load(
        os.path.join(OUTPUT_DIR, filename)
    )

    print(
        f"{filename:15s}"
        f" shape={str(image.shape):15s}"
        f" min={image.min():.4f}"
        f" max={image.max():.4f}"
    )

print("\nVerification complete.")