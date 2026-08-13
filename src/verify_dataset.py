import os
import numpy as np


# ============================================================
# DATASET PATHS
# ============================================================

GT_DIR = r"D:\SEMICON\dataset\train\GT"
NOISY_DIR = r"D:\SEMICON\dataset\train\NoisyLR"


# ============================================================
# GET FILES
# ============================================================

gt_files = sorted(
    f for f in os.listdir(GT_DIR)
    if f.endswith(".npy")
)

noisy_files = sorted(
    f for f in os.listdir(NOISY_DIR)
    if f.endswith(".npy")
)


print("=" * 70)
print("DATASET VERIFICATION")
print("=" * 70)

print(f"GT files      : {len(gt_files)}")
print(f"NoisyLR files : {len(noisy_files)}")


# ============================================================
# CHECK FILENAMES
# ============================================================

print("\nChecking filenames...")

if gt_files == noisy_files:
    print("[OK] All GT and NoisyLR filenames match.")
else:
    print("[ERROR] Filenames do not match.")

    gt_set = set(gt_files)
    noisy_set = set(noisy_files)

    print("\nMissing from NoisyLR:")
    print(sorted(gt_set - noisy_set)[:20])

    print("\nMissing from GT:")
    print(sorted(noisy_set - gt_set)[:20])


# ============================================================
# CHECK SHAPES
# ============================================================

print("\nChecking image shapes...")

gt_shapes = {}
noisy_shapes = {}

for filename in gt_files:

    gt = np.load(os.path.join(GT_DIR, filename))
    noisy = np.load(os.path.join(NOISY_DIR, filename))

    gt_shapes[gt.shape] = gt_shapes.get(gt.shape, 0) + 1
    noisy_shapes[noisy.shape] = noisy_shapes.get(noisy.shape, 0) + 1


print("\nGT shapes:")
for shape, count in gt_shapes.items():
    print(f"  {shape}: {count} images")


print("\nNoisyLR shapes:")
for shape, count in noisy_shapes.items():
    print(f"  {shape}: {count} images")


# ============================================================
# GLOBAL STATISTICS
# ============================================================

print("\nCalculating global statistics...")

gt_global_min = float("inf")
gt_global_max = float("-inf")

noisy_global_min = float("inf")
noisy_global_max = float("-inf")

gt_sum = 0.0
noisy_sum = 0.0

gt_sum_sq = 0.0
noisy_sum_sq = 0.0

gt_count = 0
noisy_count = 0


for filename in gt_files:

    gt = np.load(os.path.join(GT_DIR, filename)).astype(np.float64)
    noisy = np.load(os.path.join(NOISY_DIR, filename)).astype(np.float64)

    # GT
    gt_global_min = min(gt_global_min, gt.min())
    gt_global_max = max(gt_global_max, gt.max())

    gt_sum += gt.sum()
    gt_sum_sq += np.square(gt).sum()
    gt_count += gt.size

    # Noisy
    noisy_global_min = min(noisy_global_min, noisy.min())
    noisy_global_max = max(noisy_global_max, noisy.max())

    noisy_sum += noisy.sum()
    noisy_sum_sq += np.square(noisy).sum()
    noisy_count += noisy.size


gt_mean = gt_sum / gt_count
noisy_mean = noisy_sum / noisy_count

gt_std = np.sqrt(gt_sum_sq / gt_count - gt_mean ** 2)
noisy_std = np.sqrt(noisy_sum_sq / noisy_count - noisy_mean ** 2)


print("\nGT GLOBAL STATISTICS")
print("-" * 40)
print(f"Minimum : {gt_global_min:.6f}")
print(f"Maximum : {gt_global_max:.6f}")
print(f"Mean    : {gt_mean:.6f}")
print(f"Std     : {gt_std:.6f}")


print("\nNOISYLR GLOBAL STATISTICS")
print("-" * 40)
print(f"Minimum : {noisy_global_min:.6f}")
print(f"Maximum : {noisy_global_max:.6f}")
print(f"Mean    : {noisy_mean:.6f}")
print(f"Std     : {noisy_std:.6f}")


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)