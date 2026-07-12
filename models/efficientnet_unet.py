"""
EfficientNet-B4 U-Net with a multi-scale attention decoder, for binary
brain tumor segmentation on Dataset001_BrainTumorProcessed (1-channel,
256x256 MRI slices).

Encoder
    timm's efficientnet_b4 (MBConv blocks w/ built-in Squeeze-and-Excitation,
    ImageNet-pretrained), used as a feature pyramid via features_only=True.
    Grayscale input is repeated to 3 channels so the pretrained stem is used
    unmodified.

Decoder
    At each upsampling stage: upsample -> concat with the matching encoder
    skip connection -> MultiScaleAttentionBlock (parallel 1x1/3x3/5x5 conv
    branches, fused, channel-attention (SE) recalibrated, then added back
    to a residual shortcut).

Output
    1x1 conv -> sigmoid -> single-channel tumor probability map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SEAttention(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class MultiScaleAttentionBlock(nn.Module):
    """
    Parallel 1x1 / 3x3 / 5x5 conv branches capturing fine detail and wider
    context, fused, SE-recalibrated, and added to a residual shortcut.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 2

        def branch(kernel_size):
            return nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=kernel_size,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )

        self.branch1 = branch(1)
        self.branch3 = branch(3)
        self.branch5 = branch(5)

        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.attention = SEAttention(out_channels)

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        feats = torch.cat([self.branch1(x), self.branch3(x), self.branch5(x)], dim=1)
        feats = self.attention(self.fuse(feats))
        return self.act(feats + self.shortcut(x))


class DecoderStage(nn.Module):
    """Upsample -> concat skip (if any) -> MultiScaleAttentionBlock."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.block = MultiScaleAttentionBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class EfficientNetB4UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, pretrained=True, decoder_channels=(256, 128, 64, 32, 16)):
        """
        n_channels: input channels (1 for grayscale MRI; internally repeated to 3
                    so the ImageNet-pretrained EfficientNet-B4 stem applies unchanged)
        n_classes: output channels (1 for a sigmoid tumor-probability map)
        pretrained: load ImageNet weights for the EfficientNet-B4 encoder
        decoder_channels: output width of each of the 5 decoder stages
        """
        super().__init__()
        assert len(decoder_channels) == 5
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.encoder = timm.create_model(
            "efficientnet_b4", features_only=True, pretrained=pretrained, in_chans=3
        )
        enc_channels = self.encoder.feature_info.channels()  # e.g. [24, 32, 56, 160, 448]

        c0, c1, c2, c3, c4 = enc_channels
        d0, d1, d2, d3, d4 = decoder_channels

        self.decoder4 = DecoderStage(c4, c3, d0)   # 8x8   -> 16x16
        self.decoder3 = DecoderStage(d0, c2, d1)   # 16x16 -> 32x32
        self.decoder2 = DecoderStage(d1, c1, d2)   # 32x32 -> 64x64
        self.decoder1 = DecoderStage(d2, c0, d3)   # 64x64 -> 128x128
        self.decoder0 = DecoderStage(d3, 0, d4)    # 128x128 -> 256x256 (no skip)

        self.out_conv = nn.Conv2d(d4, n_classes, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feat0, feat1, feat2, feat3, feat4 = self.encoder(x)

        x = self.decoder4(feat4, feat3)
        x = self.decoder3(x, feat2)
        x = self.decoder2(x, feat1)
        x = self.decoder1(x, feat0)
        x = self.decoder0(x, None)

        return torch.sigmoid(self.out_conv(x))


if __name__ == "__main__":
    model = EfficientNetB4UNet(n_channels=1, n_classes=1, pretrained=False)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}  range=[{y.min().item():.3f}, {y.max().item():.3f}]")
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params:,} (trainable: {n_trainable:,})")
