import torch
import torch.nn as nn
import torch.nn.functional as F


class DetailAwareLoss(nn.Module):

    def __init__(
        self,
        alpha=1.0,
        beta=0.15,
        gamma=0.05
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Sobel X
        sobel_x = torch.tensor(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        # Sobel Y
        sobel_y = torch.tensor(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer(
            "sobel_x",
            sobel_x
        )

        self.register_buffer(
            "sobel_y",
            sobel_y
        )

    # ========================================================
    # GRADIENT EXTRACTION
    # ========================================================

    def gradients(self, image):

        sobel_x = self.sobel_x.to(
            device=image.device,
            dtype=image.dtype
        )

        sobel_y = self.sobel_y.to(
            device=image.device,
            dtype=image.dtype
        )

        gx = F.conv2d(
            image,
            sobel_x,
            padding=1
        )

        gy = F.conv2d(
            image,
            sobel_y,
            padding=1
        )

        return gx, gy

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, prediction, target):

        # ----------------------------------------------------
        # 1. Pixel reconstruction
        # ----------------------------------------------------

        pixel_loss = F.l1_loss(
            prediction,
            target
        )

        # ----------------------------------------------------
        # 2. Edge/gradient preservation
        # ----------------------------------------------------

        pred_gx, pred_gy = self.gradients(
            prediction
        )

        target_gx, target_gy = self.gradients(
            target
        )

        edge_loss = (
            F.l1_loss(pred_gx, target_gx)
            +
            F.l1_loss(pred_gy, target_gy)
        )

        # ----------------------------------------------------
        # 3. High-frequency detail
        #
        # Laplacian-like operation
        # ----------------------------------------------------

        laplacian_kernel = torch.tensor(
            [
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ],
            dtype=prediction.dtype,
            device=prediction.device
        ).view(1, 1, 3, 3)

        pred_detail = F.conv2d(
            prediction,
            laplacian_kernel,
            padding=1
        )

        target_detail = F.conv2d(
            target,
            laplacian_kernel,
            padding=1
        )

        detail_loss = F.l1_loss(
            pred_detail,
            target_detail
        )

        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        total_loss = (
            self.alpha * pixel_loss
            +
            self.beta * edge_loss
            +
            self.gamma * detail_loss
        )

        return total_loss