import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareLoss(nn.Module):

    def __init__(
        self,
        alpha=1.0,
        beta=0.2,
        gamma=0.1
    ):
        super().__init__()

        # Weight for pixel-level reconstruction
        self.alpha = alpha

        # Weight for structural similarity
        self.beta = beta

        # Weight for edge preservation
        self.gamma = gamma

        # ----------------------------------------------------
        # Sobel X filter
        # ----------------------------------------------------

        sobel_x = torch.tensor(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        # ----------------------------------------------------
        # Sobel Y filter
        # ----------------------------------------------------

        sobel_y = torch.tensor(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        # Store Sobel filters as part of the model
        self.register_buffer(
            "sobel_x",
            sobel_x
        )

        self.register_buffer(
            "sobel_y",
            sobel_y
        )

    # ========================================================
    # EDGE EXTRACTION
    # ========================================================

    def get_edges(self, image):

        sobel_x = self.sobel_x.to(image.device)
        sobel_y = self.sobel_y.to(image.device)

        edge_x = F.conv2d(
            image,
            sobel_x,
            padding=1
        )

        edge_y = F.conv2d(
            image,
            sobel_y,
            padding=1
        )

        edges = torch.sqrt(
            edge_x ** 2 +
            edge_y ** 2 +
            1e-6
        )

        return edges

    # ========================================================
    # SSIM LOSS
    # ========================================================

    def ssim_loss(self, prediction, target):

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Local mean
        mu_x = F.avg_pool2d(
            prediction,
            kernel_size=7,
            stride=1,
            padding=3
        )

        mu_y = F.avg_pool2d(
            target,
            kernel_size=7,
            stride=1,
            padding=3
        )

        # Mean squares
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2

        # Mean product
        mu_xy = mu_x * mu_y

        # Variance of prediction
        sigma_x_sq = F.avg_pool2d(
            prediction ** 2,
            kernel_size=7,
            stride=1,
            padding=3
        ) - mu_x_sq

        # Variance of target
        sigma_y_sq = F.avg_pool2d(
            target ** 2,
            kernel_size=7,
            stride=1,
            padding=3
        ) - mu_y_sq

        # Covariance
        sigma_xy = F.avg_pool2d(
            prediction * target,
            kernel_size=7,
            stride=1,
            padding=3
        ) - mu_xy

        # SSIM map
        ssim_map = (
            (2 * mu_xy + C1) *
            (2 * sigma_xy + C2)
        ) / (
            (mu_x_sq + mu_y_sq + C1) *
            (sigma_x_sq + sigma_y_sq + C2)
        )

        return 1.0 - ssim_map.mean()

    # ========================================================
    # COMBINED LOSS
    # ========================================================

    def forward(self, prediction, target):

        # ----------------------------------------------------
        # 1. Pixel-level L1 loss
        # ----------------------------------------------------

        l1 = F.l1_loss(
            prediction,
            target
        )

        # ----------------------------------------------------
        # 2. Structural SSIM loss
        # ----------------------------------------------------

        ssim = self.ssim_loss(
            prediction,
            target
        )

        # ----------------------------------------------------
        # 3. Edge loss
        # ----------------------------------------------------

        prediction_edges = self.get_edges(
            prediction
        )

        target_edges = self.get_edges(
            target
        )

        edge = F.l1_loss(
            prediction_edges,
            target_edges
        )

        # ----------------------------------------------------
        # 4. Combine everything
        # ----------------------------------------------------

        total_loss = (
            self.alpha * l1
            + self.beta * ssim
            + self.gamma * edge
        )

        return total_loss