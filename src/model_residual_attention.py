import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CHANNEL ATTENTION
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        reduced_channels = max(
            channels // reduction,
            4
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(
                channels,
                reduced_channels,
                kernel_size=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                reduced_channels,
                channels,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        attention = self.pool(x)

        attention = self.fc(attention)

        return x * attention


# ============================================================
# ATTENTION RESIDUAL BLOCK
# ============================================================

class AttentionResidualBlock(nn.Module):

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

        self.attention = ChannelAttention(
            channels
        )

    def forward(self, x):

        residual = x

        out = self.conv1(x)

        out = self.relu(out)

        out = self.conv2(out)

        # ----------------------------------------------------
        # Channel attention
        # ----------------------------------------------------

        out = self.attention(out)

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        out = out + residual

        return out


# ============================================================
# RESIDUAL ATTENTION SUPER-RESOLUTION NETWORK
# ============================================================

class ResidualAttentionSRNet(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Input feature extraction
        # ----------------------------------------------------

        self.input_conv = nn.Conv2d(
            1,
            64,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Attention residual blocks
        # ----------------------------------------------------

        self.residual_blocks = nn.Sequential(

            AttentionResidualBlock(64),

            AttentionResidualBlock(64),

            AttentionResidualBlock(64),

            AttentionResidualBlock(64)
        )

        # ----------------------------------------------------
        # Middle convolution
        # ----------------------------------------------------

        self.middle_conv = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Global feature attention
        # ----------------------------------------------------

        self.global_attention = ChannelAttention(
            64
        )

        # ----------------------------------------------------
        # Upsampling
        #
        # 128 × 128 → 256 × 256
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
        # Learned residual/detail
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
        # Feature extraction
        # ----------------------------------------------------

        features = self.input_conv(x)

        # ----------------------------------------------------
        # Long residual connection
        # ----------------------------------------------------

        residual = features

        # ----------------------------------------------------
        # Attention residual processing
        # ----------------------------------------------------

        features = self.residual_blocks(
            features
        )

        # ----------------------------------------------------
        # Middle processing
        # ----------------------------------------------------

        features = self.middle_conv(
            features
        )

        # ----------------------------------------------------
        # Global channel attention
        # ----------------------------------------------------

        features = self.global_attention(
            features
        )

        # ----------------------------------------------------
        # Long skip
        # ----------------------------------------------------

        features = features + residual

        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------

        features = self.upsample(
            features
        )

        # ----------------------------------------------------
        # Predict learned detail
        # ----------------------------------------------------

        learned_residual = self.residual_output(
            features
        )

        # ----------------------------------------------------
        # Add learned detail to bicubic
        # ----------------------------------------------------

        output = (
            bicubic +
            learned_residual
        )

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
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    model = ResidualAttentionSRNet()

    print(model)

    # --------------------------------------------------------
    # Test input
    # --------------------------------------------------------

    test_input = torch.randn(
        2,
        1,
        128,
        128
    )

    test_output = model(
        test_input
    )

    print("\nInput shape:")
    print(
        test_input.shape
    )

    print("\nOutput shape:")
    print(
        test_output.shape
    )

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