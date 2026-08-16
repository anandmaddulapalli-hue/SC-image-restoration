import torch
import torch.nn as nn
import torch.nn.functional as F


class HFRestorationLoss(nn.Module):

    def __init__(
        self,
        alpha=1.0,
        beta=0.10,
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
    # CHARBONNIER LOSS
    # ========================================================

    def charbonnier(
        self,
        prediction,
        target
    ):

        epsilon = 1e-3

        diff = prediction - target

        loss = torch.sqrt(
            diff * diff +
            epsilon * epsilon
        )

        return loss.mean()


    # ========================================================
    # GRADIENTS
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
    # GRADIENT LOSS
    # ========================================================

    def gradient_loss(
        self,
        prediction,
        target
    ):

        pred_gx, pred_gy = self.gradients(
            prediction
        )

        target_gx, target_gy = self.gradients(
            target
        )

        loss_x = F.l1_loss(
            pred_gx,
            target_gx
        )

        loss_y = F.l1_loss(
            pred_gy,
            target_gy
        )

        return loss_x + loss_y


    # ========================================================
    # SSIM LOSS
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
            (2 * mu_xy + C1)
            * (2 * sigma_xy + C2)
        ) / (
            (mu_x_sq + mu_y_sq + C1)
            * (sigma_x_sq + sigma_y_sq + C2)
        )

        return 1.0 - ssim_map.mean()


    # ========================================================
    # COMBINED LOSS
    # ========================================================

    def forward(
        self,
        prediction,
        target
    ):

        reconstruction = self.charbonnier(
            prediction,
            target
        )

        gradient = self.gradient_loss(
            prediction,
            target
        )

        ssim = self.ssim_loss(
            prediction,
            target
        )

        total = (
            self.alpha * reconstruction
            +
            self.beta * gradient
            +
            self.gamma * ssim
        )

        return total