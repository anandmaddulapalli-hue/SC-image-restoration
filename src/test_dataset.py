import os
import torch

from dataset import SemiconductorDataset


# ============================================================
# CHANGE THESE PATHS
# ============================================================

NOISY_DIR = r"D:\SEMICON\dataset\train\NoisyLR"
GT_DIR = r"D:\SEMICON\dataset\train\GT"


# ============================================================
# CREATE DATASET
# ============================================================

dataset = SemiconductorDataset(
    noisy_dir=NOISY_DIR,
    gt_dir=GT_DIR
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nDataset length:")
print(len(dataset))


# ============================================================
# GET FIRST SAMPLE
# ============================================================

noisy, gt = dataset[0]


print("\nFirst sample:")
print("Noisy shape :", noisy.shape)
print("GT shape    :", gt.shape)

print("\nNoisy:")
print("dtype :", noisy.dtype)
print("min   :", noisy.min().item())
print("max   :", noisy.max().item())

print("\nGT:")
print("dtype :", gt.dtype)
print("min   :", gt.min().item())
print("max   :", gt.max().item())


# ============================================================
# CHECK THAT VALUES ARE FINITE
# ============================================================

print("\nChecking for NaN / Inf:")

print(
    "Noisy contains NaN:",
    torch.isnan(noisy).any().item()
)

print(
    "Noisy contains Inf:",
    torch.isinf(noisy).any().item()
)

print(
    "GT contains NaN:",
    torch.isnan(gt).any().item()
)

print(
    "GT contains Inf:",
    torch.isinf(gt).any().item()
)