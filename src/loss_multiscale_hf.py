import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleHFLoss(nn.Module):

    def __init__(
        self,
        alpha=1.0,
        beta=0.15,
        gamma=0.10,
        delta=0.05
    ):

        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

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

    def edges(self, image):

        gx = F.conv2d(
            image,
            self.sobel_x,
            padding=1
        )

        gy = F.conv2d(
            image,
            self.sobel_y,
            padding=1
        )

        return torch.sqrt(
            gx ** 2
            +
            gy ** 2
            +
            1e-6
        )

    # ========================================================
    # SSIM
    # ========================================================

    def ssim_loss(
        self,
        prediction,
        target
    ):

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(
            prediction,
            7,
            1,
            3
        )

        mu_y = F.avg_pool2d(
            target,
            7,
            1,
            3
        )

        sigma_x = (
            F.avg_pool2d(
                prediction ** 2,
                7,
                1,
                3
            )
            -
            mu_x ** 2
        )

        sigma_y = (
            F.avg_pool2d(
                target ** 2,
                7,
                1,
                3
            )
            -
            mu_y ** 2
        )

        sigma_xy = (
            F.avg_pool2d(
                prediction * target,
                7,
                1,
                3
            )
            -
            mu_x * mu_y
        )

        ssim = (
            (2 * mu_x * mu_y + C1)
            *
            (2 * sigma_xy + C2)
        ) / (
            (mu_x ** 2 + mu_y ** 2 + C1)
            *
            (sigma_x + sigma_y + C2)
            +
            1e-8
        )

        return 1.0 - ssim.mean()

    # ========================================================
    # HIGH FREQUENCY EXTRACTION
    # ========================================================

    def high_frequency(
        self,
        image
    ):

        blurred = F.avg_pool2d(
            image,
            kernel_size=5,
            stride=1,
            padding=2
        )

        return image - blurred

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        prediction,
        target
    ):

        # ----------------------------------------------------
        # Full-resolution pixel loss
        # ----------------------------------------------------

        pixel_loss = F.l1_loss(
            prediction,
            target
        )

        # ----------------------------------------------------
        # SSIM
        # ----------------------------------------------------

        structural_loss = self.ssim_loss(
            prediction,
            target
        )

        # ----------------------------------------------------
        # Edge loss
        # ----------------------------------------------------

        pred_edges = self.edges(
            prediction
        )

        target_edges = self.edges(
            target
        )

        edge_loss = F.l1_loss(
            pred_edges,
            target_edges
        )

        # ----------------------------------------------------
        # High-frequency loss
        # ----------------------------------------------------

        pred_hf = self.high_frequency(
            prediction
        )

        target_hf = self.high_frequency(
            target
        )

        hf_loss = F.l1_loss(
            pred_hf,
            target_hf
        )

        # ----------------------------------------------------
        # Multi-scale loss
        # ----------------------------------------------------

        pred_128 = F.interpolate(
            prediction,
            size=(128, 128),
            mode="area"
        )

        target_128 = F.interpolate(
            target,
            size=(128, 128),
            mode="area"
        )

        pred_64 = F.interpolate(
            prediction,
            size=(64, 64),
            mode="area"
        )

        target_64 = F.interpolate(
            target,
            size=(64, 64),
            mode="area"
        )

        multi_scale_loss = (
            0.7 * F.l1_loss(
                pred_128,
                target_128
            )
            +
            0.3 * F.l1_loss(
                pred_64,
                target_64
            )
        )

        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        total_loss = (

            self.alpha * pixel_loss

            +

            self.beta * structural_loss

            +

            self.gamma * edge_loss

            +

            self.delta * (
                hf_loss
                +
                multi_scale_loss
            )
        )

        return total_loss