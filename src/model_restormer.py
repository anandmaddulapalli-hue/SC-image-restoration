import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# GATED DEPTHWISE FEED-FORWARD NETWORK
# Restormer-inspired
# ============================================================

class GDFN(nn.Module):

    def __init__(self, dim, expansion_factor=2.0):

        super().__init__()

        hidden_dim = int(dim * expansion_factor)

        self.project_in = nn.Conv2d(
            dim,
            hidden_dim * 2,
            kernel_size=1
        )

        self.dwconv = nn.Conv2d(
            hidden_dim * 2,
            hidden_dim * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_dim * 2
        )

        self.project_out = nn.Conv2d(
            hidden_dim,
            dim,
            kernel_size=1
        )

    def forward(self, x):

        x = self.project_in(x)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2

        x = self.project_out(x)

        return x


# ============================================================
# LIGHTWEIGHT CHANNEL ATTENTION
# Restormer-inspired
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(self, dim, num_heads=4):

        super().__init__()

        self.num_heads = num_heads

        self.temperature = nn.Parameter(
            torch.ones(num_heads, 1, 1)
        )

        self.qkv = nn.Conv2d(
            dim,
            dim * 3,
            kernel_size=1,
            bias=False
        )

        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            padding=1,
            groups=dim * 3,
            bias=False
        )

        self.project_out = nn.Conv2d(
            dim,
            dim,
            kernel_size=1
        )

    def forward(self, x):

        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(
            self.qkv(x)
        )

        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(
            b,
            self.num_heads,
            c // self.num_heads,
            h * w
        )

        k = k.reshape(
            b,
            self.num_heads,
            c // self.num_heads,
            h * w
        )

        v = v.reshape(
            b,
            self.num_heads,
            c // self.num_heads,
            h * w
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attention = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        attention = attention * self.temperature

        attention = attention.softmax(
            dim=-1
        )

        out = torch.matmul(
            attention,
            v
        )

        out = out.reshape(
            b,
            c,
            h,
            w
        )

        out = self.project_out(out)

        return out


# ============================================================
# RESTORMER-INSPIRED BLOCK
# ============================================================

class RestormerBlock(nn.Module):

    def __init__(self, dim=64, num_heads=4):

        super().__init__()

        self.norm1 = nn.GroupNorm(
            1,
            dim
        )

        self.attention = ChannelAttention(
            dim,
            num_heads
        )

        self.norm2 = nn.GroupNorm(
            1,
            dim
        )

        self.ffn = GDFN(
            dim
        )

    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x


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

        x = self.conv1(x)

        x = self.relu(x)

        x = self.conv2(x)

        return x + residual


# ============================================================
# HYBRID RESTORATION NETWORK
# ============================================================

class HybridRestormerNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_conv = nn.Conv2d(
            1,
            64,
            kernel_size=3,
            padding=1
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.transformer_blocks = nn.Sequential(
            RestormerBlock(
                dim=64,
                num_heads=4
            ),
            RestormerBlock(
                dim=64,
                num_heads=4
            )
        )

        self.middle_conv = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1
        )

        self.upsample = nn.Sequential(

            nn.Conv2d(
                64,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.ReLU(
                inplace=True
            )
        )

        self.output_conv = nn.Conv2d(
            64,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        x = self.input_conv(x)

        residual = x

        x = self.residual_blocks(x)

        x = self.transformer_blocks(x)

        x = self.middle_conv(x)

        x = x + residual

        x = self.upsample(x)

        x = self.output_conv(x)

        return torch.sigmoid(x)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = HybridRestormerNet().to(device)

    x = torch.randn(
        2,
        1,
        128,
        128
    ).to(device)

    with torch.no_grad():

        y = model(x)

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        y.shape
    )

    print(
        "Output range:"
    )

    print(
        "Minimum:",
        y.min().item()
    )

    print(
        "Maximum:",
        y.max().item()
    )

    print(
        "Total parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )