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
# RESIDUAL SUPER-RESOLUTION MODEL
# ============================================================

class ResidualSRNet(nn.Module):

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
        # Middle feature processing
        # ----------------------------------------------------

        self.middle_conv = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Upsampling
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
        # Predict HIGH-FREQUENCY RESIDUAL
        #
        # Important:
        # No sigmoid here.
        #
        # Residuals can be positive OR negative.
        # ----------------------------------------------------

        self.residual_output = nn.Conv2d(
            64,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Bicubic baseline
        # ----------------------------------------------------

        bicubic = F.interpolate(
            x,
            size=(256, 256),
            mode="bicubic",
            align_corners=False
        )

        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features = self.input_conv(x)

        # ----------------------------------------------------
        # Residual feature processing
        # ----------------------------------------------------

        residual = features

        features = self.residual_blocks(features)

        features = self.middle_conv(features)

        features = features + residual

        # ----------------------------------------------------
        # Upsample features
        # ----------------------------------------------------

        features = self.upsample(features)

        # ----------------------------------------------------
        # Predict correction/detail
        # ----------------------------------------------------

        learned_residual = self.residual_output(
            features
        )

        # ----------------------------------------------------
        # Add learned correction to bicubic image
        # ----------------------------------------------------

        output = bicubic + learned_residual

        # ----------------------------------------------------
        # Keep image in valid range
        # ----------------------------------------------------

        output = torch.clamp(
            output,
            0.0,
            1.0
        )

        return output


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    model = ResidualSRNet()

    print(model)

    # --------------------------------------------------------
    # Fake input
    # --------------------------------------------------------

    test_input = torch.randn(
        2,
        1,
        128,
        128
    )

    test_output = model(test_input)

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

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nTotal parameters:")

    print(
        f"{total_params:,}"
    )