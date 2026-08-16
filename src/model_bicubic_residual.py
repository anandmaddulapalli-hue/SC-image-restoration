import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x + residual


# ============================================================
# BICUBIC RESIDUAL SUPER-RESOLUTION NETWORK
# ============================================================

class BicubicResidualSRNet(nn.Module):

    def __init__(
        self,
        channels=64,
        num_blocks=8
    ):

        super().__init__()

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        self.input_conv = nn.Conv2d(
            1,
            channels,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Deep residual processing
        # ----------------------------------------------------

        self.residual_blocks = nn.Sequential(

            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]

        )

        # ----------------------------------------------------
        # Feature refinement
        # ----------------------------------------------------

        self.middle_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------

        self.upsample = nn.Sequential(

            nn.Conv2d(
                channels,
                channels * 4,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # Residual image prediction
        # ----------------------------------------------------

        self.residual_output = nn.Conv2d(
            channels,
            1,
            kernel_size=3,
            padding=1
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Bicubic baseline
        # ----------------------------------------------------

        bicubic = F.interpolate(
            x,
            scale_factor=2,
            mode="bicubic",
            align_corners=False
        )

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        features = self.input_conv(x)

        residual = features

        # ----------------------------------------------------
        # Deep processing
        # ----------------------------------------------------

        features = self.residual_blocks(
            features
        )

        features = self.middle_conv(
            features
        )

        features = features + residual

        # ----------------------------------------------------
        # 2× upsampling
        # ----------------------------------------------------

        features = self.upsample(
            features
        )

        # ----------------------------------------------------
        # Predict HIGH-FREQUENCY residual
        # ----------------------------------------------------

        detail = self.residual_output(
            features
        )

        # ----------------------------------------------------
        # Add learned detail to bicubic image
        # ----------------------------------------------------

        output = bicubic + detail

        # ----------------------------------------------------
        # Valid image range
        # ----------------------------------------------------

        output = torch.clamp(
            output,
            0.0,
            1.0
        )

        return output


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    model = BicubicResidualSRNet(
        channels=64,
        num_blocks=8
    )

    test_input = torch.randn(
        2,
        1,
        128,
        128
    )

    test_output = model(
        test_input
    )

    print(model)

    print()
    print("Input shape:")
    print(test_input.shape)

    print()
    print("Output shape:")
    print(test_output.shape)

    print()
    print("Output range:")

    print(
        "Minimum:",
        test_output.min().item()
    )

    print(
        "Maximum:",
        test_output.max().item()
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print("Total parameters:")
    print(f"{total_params:,}")