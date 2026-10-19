""" InfiltrNet architecture."""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(out_ch)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=4, base_filters=32):
        super().__init__()
        f = base_filters
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, f, 3, padding=1, bias=False),
            nn.InstanceNorm3d(f),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.level1 = ResBlock3D(f, f)
        self.level2 = ResBlock3D(f, f, stride=2)
        self.level3 = ResBlock3D(f, f * 2, stride=2)
        self.level4 = ResBlock3D(f * 2, f * 4, stride=2)
        self.level5 = ResBlock3D(f * 4, f * 8, stride=2)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.level1(x0)
        x2 = self.level2(x1)
        x3 = self.level3(x2)
        x4 = self.level4(x3)
        x5 = self.level5(x4)
        return [x1, x2, x3, x4, x5]


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between CNN and Swin feature maps."""

    def __init__(self, cnn_ch, swin_ch, out_ch):
        super().__init__()
        self.cnn_proj = nn.Conv3d(cnn_ch, out_ch, 1, bias=False)
        self.swin_proj = nn.Conv3d(swin_ch, out_ch, 1, bias=False)
        self.q_cnn = nn.Conv3d(out_ch, out_ch, 1, bias=False)
        self.k_swin = nn.Conv3d(out_ch, out_ch, 1, bias=False)
        self.v_swin = nn.Conv3d(out_ch, out_ch, 1, bias=False)
        self.q_swin = nn.Conv3d(out_ch, out_ch, 1, bias=False)
        self.k_cnn = nn.Conv3d(out_ch, out_ch, 1, bias=False)
        self.v_cnn = nn.Conv3d(out_ch, out_ch, 1, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv3d(out_ch * 2, out_ch, 1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.scale = out_ch ** -0.5

    def spatial_attention(self, q, k, v):
        B, C, D, H, W = q.shape
        N = D * H * W
        q = q.view(B, C, N).permute(0, 2, 1)
        k = k.view(B, C, N)
        v = v.view(B, C, N).permute(0, 2, 1)

        # Memory-bounded key/value pooling used in the original implementation.
        if N > 512:
            k_pool = k.view(B, C, D, H, W)
            v_pool = v.permute(0, 2, 1).view(B, C, D, H, W)
            ps = max(1, int(round((max(1, N // 512)) ** (1 / 3))))
            if ps > 1:
                k_pool = F.avg_pool3d(k_pool, ps)
                v_pool = F.avg_pool3d(v_pool, ps)
            N_k = k_pool.shape[2] * k_pool.shape[3] * k_pool.shape[4]
            k = k_pool.view(B, C, N_k)
            v = v_pool.view(B, C, N_k).permute(0, 2, 1)

        attn = F.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        return torch.bmm(attn, v).permute(0, 2, 1).view(B, -1, D, H, W)

    def forward(self, cnn_feat, swin_feat):
        if cnn_feat.shape[2:] != swin_feat.shape[2:]:
            swin_feat = F.interpolate(
                swin_feat,
                size=cnn_feat.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        c = self.cnn_proj(cnn_feat)
        s = self.swin_proj(swin_feat)
        cnn_enh = self.spatial_attention(self.q_cnn(c), self.k_swin(s), self.v_swin(s))
        swin_enh = self.spatial_attention(self.q_swin(s), self.k_cnn(c), self.v_cnn(c))
        return self.out_proj(torch.cat([cnn_enh, swin_enh], dim=1))


class FusionDecoder(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class InfiltrNet(nn.Module):
    """Dual-branch CNN + Swin Transformer with multi-scale cross-attention fusion."""

    def __init__(self, in_channels=4, out_channels=4, sf=24, cf=32):
        super().__init__()
        self.cnn_encoder = CNNEncoder(in_channels, cf)
        self.swin = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=sf,
            use_checkpoint=True,
            spatial_dims=3,
        )

        self.fusion_bottom = CrossAttentionFusion(cf * 8, sf * 8, sf * 4)
        self.fusion_3 = CrossAttentionFusion(cf * 4, sf * 4, sf * 2)
        self.fusion_2 = CrossAttentionFusion(cf * 2, sf * 2, sf)
        self.fusion_1 = CrossAttentionFusion(cf, sf, sf)

        self.dec4 = FusionDecoder(sf * 4, sf * 2, sf * 2)
        self.dec3 = FusionDecoder(sf * 2, sf, sf)
        self.dec2 = FusionDecoder(sf, sf, sf)
        self.dec1 = nn.Sequential(
            nn.Conv3d(sf, sf, 3, padding=1, bias=False),
            nn.InstanceNorm3d(sf),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.out_conv = nn.Conv3d(sf, out_channels, 1)
        self.ds_head_2x = nn.Conv3d(sf, out_channels, 1)
        self.ds_head_4x = nn.Conv3d(sf, out_channels, 1)
        self.ds_head_8x = nn.Conv3d(sf * 2, out_channels, 1)

    def forward(self, x):
        cnn_feats = self.cnn_encoder(x)

        swin_hidden = self.swin.swinViT(x, self.swin.normalize)
        swin_enc1 = self.swin.encoder2(swin_hidden[0])
        swin_enc2 = self.swin.encoder3(swin_hidden[1])
        swin_enc3 = self.swin.encoder4(swin_hidden[2])

        fused_bottom = self.fusion_bottom(cnn_feats[4], swin_hidden[3])
        skip3 = self.fusion_3(cnn_feats[3], swin_enc3)
        skip2 = self.fusion_2(cnn_feats[2], swin_enc2)
        skip1 = self.fusion_1(cnn_feats[1], swin_enc1)

        dec3 = self.dec4(fused_bottom, skip3)
        dec2 = self.dec3(dec3, skip2)
        dec1 = self.dec2(dec2, skip1)

        out = self.dec1(dec1)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="trilinear", align_corners=False)
        main_out = self.out_conv(out)

        if self.training:
            return (
                main_out,
                self.ds_head_2x(dec1),
                self.ds_head_4x(dec2),
                self.ds_head_8x(dec3),
            )
        return main_out


def load_swin_pretrained(model, checkpoint_path):
    """Load compatible Swin encoder tensors and return the number loaded."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    pt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(pt, dict) and "state_dict" in pt:
        pt = pt["state_dict"]

    md = model.state_dict()
    loaded = 0
    for k, v in pt.items():
        candidates = [k]
        if k.startswith("module."):
            candidates.append(k.replace("module.", "", 1))
        matched = False
        for clean in candidates:
            for pfx in (f"swin.swinViT.{clean}", f"swin.{clean}"):
                if pfx in md and md[pfx].shape == v.shape:
                    md[pfx] = v
                    loaded += 1
                    matched = True
                    break
            if matched:
                break
    model.load_state_dict(md)
    return loaded
