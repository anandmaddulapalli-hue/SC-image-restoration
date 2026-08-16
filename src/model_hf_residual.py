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

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        return out + residual


# ============================================================
# HIGH-FREQUENCY RESIDUAL SUPER-RESOLUTION NETWORK
# ============================================================

class HFResidualSRNet(nn.Module):

    def __init__(
        self,
        channels=96,
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
        # Deep residual feature processing
        # ----------------------------------------------------

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        # ----------------------------------------------------
        # Feature fusion
        # ----------------------------------------------------

        self.middle_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Upsampling
        # 128x128 -> 256x256
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
        # High-frequency residual prediction
        # ----------------------------------------------------

        self.detail_head = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                1,
                kernel_size=3,
                padding=1
            )
        )


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
        # Extract LR features
        # ----------------------------------------------------

        features = self.input_conv(x)

        residual = features

        # ----------------------------------------------------
        # Deep residual processing
        # ----------------------------------------------------

        features = self.residual_blocks(features)

        features = self.middle_conv(features)

        # Global residual connection
        features = features + residual

        # ----------------------------------------------------
        # Upsample features
        # ----------------------------------------------------

        features = self.upsample(features)

        # ----------------------------------------------------
        # Predict high-frequency correction
        # ----------------------------------------------------

        detail = self.detail_head(features)

        # ----------------------------------------------------
        # Add learned detail to bicubic reconstruction
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

    model = HFResidualSRNet()

    test_input = torch.randn(
        2,
        1,
        128,
        128
    )

    test_output = model(test_input)

    print(model)

    print("\nInput shape:")
    print(test_input.shape)

    print("\nOutput shape:")
    print(test_output.shape)

    print("\nOutput range:")

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

    print("\nTotal parameters:")
    print(f"{total_params:,}")