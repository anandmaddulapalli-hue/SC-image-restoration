import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAwareLoss(nn.Module):

    def __init__(
        self,
        alpha=1.0,
        beta=0.15,
        gamma=0.05,
        delta=0.05
    ):

        super().__init__()

        # ----------------------------------------------------
        # Loss weights
        # ----------------------------------------------------

        self.alpha = alpha      # L1 reconstruction
        self.beta = beta        # SSIM
        self.gamma = gamma      # Edge preservation
        self.delta = delta      # High-frequency preservation

        # ----------------------------------------------------
        # Sobel filters
        # ----------------------------------------------------

        sobel_x = torch.tensor(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

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
    # EDGE EXTRACTION
    # ========================================================

    def get_edges(self, image):

        edge_x = F.conv2d(
            image,
            self.sobel_x,
            padding=1
        )

        edge_y = F.conv2d(
            image,
            self.sobel_y,
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

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            F.avg_pool2d(
                prediction ** 2,
                kernel_size=7,
                stride=1,
                padding=3
            )
            - mu_x_sq
        )

        sigma_y_sq = (
            F.avg_pool2d(
                target ** 2,
                kernel_size=7,
                stride=1,
                padding=3
            )
            - mu_y_sq
        )

        sigma_xy = (
            F.avg_pool2d(
                prediction * target,
                kernel_size=7,
                stride=1,
                padding=3
            )
            - mu_xy
        )

        ssim_map = (
            (2 * mu_xy + C1) *
            (2 * sigma_xy + C2)
        ) / (
            (mu_x_sq + mu_y_sq + C1) *
            (sigma_x_sq + sigma_y_sq + C2)
        )

        return 1.0 - ssim_map.mean()

    # ========================================================
    # HIGH-FREQUENCY EXTRACTION
    # ========================================================

    def get_high_frequency(self, image):

        # Smooth image = low-frequency information
        low_frequency = F.avg_pool2d(
            image,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # Original - smooth = high-frequency information
        high_frequency = (
            image - low_frequency
        )

        return high_frequency

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, prediction, target):

        # ----------------------------------------------------
        # 1. Pixel reconstruction
        # ----------------------------------------------------

        l1 = F.l1_loss(
            prediction,
            target
        )

        # ----------------------------------------------------
        # 2. Structural similarity
        # ----------------------------------------------------

        ssim = self.ssim_loss(
            prediction,
            target
        )

        # ----------------------------------------------------
        # 3. Edge preservation
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
        # 4. High-frequency preservation
        # ----------------------------------------------------

        prediction_hf = self.get_high_frequency(
            prediction
        )

        target_hf = self.get_high_frequency(
            target
        )

        frequency = F.l1_loss(
            prediction_hf,
            target_hf
        )

        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        total_loss = (

            self.alpha * l1

            + self.beta * ssim

            + self.gamma * edge

            + self.delta * frequency
        )

        return total_loss