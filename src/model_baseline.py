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

        out = out + residual

        return out


# ============================================================
# RESTORATION MODEL
# ============================================================

class RestorationNet(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        self.input_conv = nn.Conv2d(
            1,
            64,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Residual feature processing
        # ----------------------------------------------------

        self.residual_blocks = nn.Sequential(

            ResidualBlock(64),

            ResidualBlock(64),

            ResidualBlock(64),

            ResidualBlock(64)
        )

        # ----------------------------------------------------
        # Additional feature processing
        # ----------------------------------------------------

        self.middle_conv = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Upsampling
        #
        # 128 x 128 -> 256 x 256
        # ----------------------------------------------------

        self.upsample = nn.Sequential(

            nn.Conv2d(
                64,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

        self.output_conv = nn.Conv2d(
            64,
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
        # Residual processing
        # ----------------------------------------------------

        residual = features

        features = self.residual_blocks(features)

        features = self.middle_conv(features)

        features = features + residual

        # ----------------------------------------------------
        # 2x super-resolution
        # ----------------------------------------------------

        features = self.upsample(features)

        # ----------------------------------------------------
        # Final restored image
        # ----------------------------------------------------

        output = self.output_conv(features)

        # Keep output in valid image range
        output = torch.sigmoid(output)

        return output


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    model = RestorationNet()

    print(model)

    # Fake input
    test_input = torch.randn(
        2,
        1,
        128,
        128
    )

    test_output = model(test_input)

    print("\nInput shape :")
    print(test_input.shape)

    print("\nOutput shape:")
    print(test_output.shape)

    print("\nOutput range:")
    print("Minimum:", test_output.min().item())
    print("Maximum:", test_output.max().item())

    # Number of parameters
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nTotal parameters:")
    print(f"{total_params:,}")

# Restoration network:
# Input  : 128x128 noisy low-resolution image
# Output : 256x256 restored image