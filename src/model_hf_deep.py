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

        self.relu = nn.ReLU(
            inplace=True
        )

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
# DEEP HF RESIDUAL SUPER-RESOLUTION NETWORK
# ============================================================

class DeepHFResidualSRNet(nn.Module):

    def __init__(
        self,
        channels=64,
        num_blocks=12
    ):

        super().__init__()

        # ----------------------------------------------------
        # Initial feature extraction
        # ----------------------------------------------------

        self.input_conv = nn.Conv2d(
            1,
            channels,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Deep residual feature extraction
        # ----------------------------------------------------

        blocks = []

        for _ in range(num_blocks):

            blocks.append(
                ResidualBlock(
                    channels
                )
            )

        self.residual_blocks = nn.Sequential(
            *blocks
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
        # 2× upsampling
        # ----------------------------------------------------

        self.upsample = nn.Sequential(

            nn.Conv2d(
                channels,
                channels * 4,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.ReLU(
                inplace=True
            )
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        self.output_conv = nn.Conv2d(
            channels,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Initial features
        # ----------------------------------------------------

        features = self.input_conv(x)

        # ----------------------------------------------------
        # Global residual
        # ----------------------------------------------------

        residual = features

        # ----------------------------------------------------
        # Deep residual processing
        # ----------------------------------------------------

        features = self.residual_blocks(
            features
        )

        features = self.middle_conv(
            features
        )

        # Global skip connection
        features = features + residual

        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------

        features = self.upsample(
            features
        )

        # ----------------------------------------------------
        # Final reconstruction
        # ----------------------------------------------------

        output = self.output_conv(
            features
        )

        # ----------------------------------------------------
        # Valid image range
        # ----------------------------------------------------

        output = torch.sigmoid(
            output
        )

        return output


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    model = DeepHFResidualSRNet(
        channels=64,
        num_blocks=12
    )

    print(model)

    test_input = torch.randn(
        2,
        1,
        128,
        128
    )

    test_output = model(
        test_input
    )

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

    print(
        f"{total_params:,}"
    )