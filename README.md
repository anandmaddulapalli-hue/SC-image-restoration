# AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Overview

Semiconductor inspection images can suffer from noise and reduced spatial resolution. These degradations can hide important structural details and make defect inspection difficult.

This project uses a deep learning model to restore degraded grayscale semiconductor inspection images.

The model takes a noisy, low-resolution image as input and produces a clean, high-resolution image.

---

## Problem Statement

The provided degraded images contain two major problems:

1. Pixel-level noise
2. Spatial resolution reduction

The model must learn to remove noise while reconstructing the fine details lost during downsampling.

### Input

- Grayscale image
- Resolution: 128 × 128
- Noisy and low-resolution

### Ground Truth

- Grayscale image
- Resolution: 256 × 256
- Clean and high-resolution

### Output

- Restored grayscale image
- Resolution: 256 × 256

---

## Dataset

The training dataset contains 3200 paired samples.

Each sample consists of:

- `GT` — clean high-resolution ground truth image
- `NoisyLR` — noisy low-resolution input image

### Dataset split

| Set | Images |
|---|---:|
| Training | 2560 |
| Validation | 640 |
| Total | 3200 |

The test dataset contains 400 degraded images without ground-truth images.

All images are stored as NumPy `.npy` files.

---

## Model Architecture

The restoration model is a lightweight convolutional neural network consisting of:

- Initial convolution layer
- Residual blocks
- Feature reconstruction convolution
- PixelShuffle-based upsampling
- Final reconstruction convolution

### Architecture

```text
128×128 Noisy LR Image
          │
          ▼
     Input Conv
          │
          ▼
    Residual Blocks
          │
          ▼
    Feature Reconstruction
          │
          ▼
   PixelShuffle ×2
          │
          ▼
     Output Conv
          │
          ▼
256×256 Restored Image