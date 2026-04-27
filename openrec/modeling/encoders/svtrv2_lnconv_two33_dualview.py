"""
SVTRv2LNConvTwo33DualView — Dual-view encoder for Local-Global IDS verification.

Identical architecture to SVTRv2LNConvTwo33, but returns TWO outputs:
  1. feat_1d: [B, W, C_out] — height-pooled 1D sequence (for text branch, same as before)
  2. feat_2d: [B, H, W, C_last_stage] — pre-collapse 2D feature map (for IDS branch)

The 2D features preserve vertical spatial structure needed for stroke/radical
verification, while the 1D features remain optimal for text recognition.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_

from .svtrv2_lnconv_two33 import (
    POPatchEmbed,
    SVTRStage,
    LastStage,
)


class SVTRv2LNConvTwo33DualView(nn.Module):

    def __init__(self,
                 max_sz=[32, 128],
                 in_channels=3,
                 out_channels=192,
                 depths=[3, 6, 3],
                 dims=[64, 128, 256],
                 mixer=[['Conv'] * 3, ['Conv'] * 3 + ['Global'] * 3,
                        ['Global'] * 3],
                 use_pos_embed=True,
                 sub_k=[[1, 1], [2, 1], [1, 1]],
                 num_heads=[2, 4, 8],
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 last_drop=0.1,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 act=nn.GELU,
                 last_stage=True,
                 feat2d=False,
                 eps=1e-6,
                 num_convs=[[2] * 3, [2] * 3 + [3] * 3, [3] * 3],
                 kernel_sizes=[[3] * 3, [3] * 3 + [3] * 3, [3] * 3],
                 pope_bias=False,
                 **kwargs):
        super().__init__()
        num_stages = len(depths)
        self.num_features = dims[-1]

        feat_max_size = [max_sz[0] // 4, max_sz[1] // 4]
        self.pope = POPatchEmbed(in_channels=in_channels,
                                 feat_max_size=feat_max_size,
                                 embed_dim=dims[0],
                                 use_pos_embed=use_pos_embed,
                                 flatten=mixer[0][0] != 'Conv',
                                 bias=pope_bias)

        dpr = np.linspace(0, drop_path_rate, sum(depths))

        self.stages = nn.ModuleList()
        for i_stage in range(num_stages):
            stage = SVTRStage(
                dim=dims[i_stage],
                out_dim=dims[i_stage + 1] if i_stage < num_stages - 1 else 0,
                depth=depths[i_stage],
                mixer=mixer[i_stage],
                kernel_sizes=kernel_sizes[i_stage]
                if len(kernel_sizes[i_stage]) == len(mixer[i_stage]) else [3] *
                len(mixer[i_stage]),
                sub_k=sub_k[i_stage],
                num_heads=num_heads[i_stage],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                act=act,
                downsample=False if i_stage == num_stages - 1 else True,
                eps=eps,
                num_conv=num_convs[i_stage] if len(num_convs[i_stage]) == len(
                    mixer[i_stage]) else [2] * len(mixer[i_stage]),
            )
            self.stages.append(stage)

        self.out_channels = out_channels
        self.feat_2d_channels = dims[-1]

        self.apply(self._init_weights)

        # LastStage for 1D output (height pooling + projection)
        assert last_stage, "DualView encoder requires last_stage=True"
        self.last_stage = LastStage(self.num_features, out_channels, last_drop)

        # LayerNorm for 2D features (applied before returning)
        self.feat_2d_norm = nn.LayerNorm(dims[-1])

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'patch_embed', 'downsample', 'pos_embed'}

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.flatten(0, 1)
        x, sz = self.pope(x)

        for stage in self.stages:
            x, sz = stage(x, sz)

        # After all SVTRStages, x is [B, H*W, C] (Global mixer), sz = [H, W]
        # For config with all-Global last stage: x shape is [B, H*W, C]
        H, W = sz[0], sz[1]
        C = self.num_features

        # Ensure x is [B, H*W, C] for both LastStage and 2D reshape
        if x.dim() == 4:
            # Conv mixer output: [B, C, H, W] -> [B, H*W, C]
            x = x.permute(0, 2, 3, 1).reshape(-1, H * W, C)

        # Save 2D features BEFORE height collapse
        feat_2d = x.reshape(-1, H, W, C)
        feat_2d = self.feat_2d_norm(feat_2d)  # [B, H, W, C]

        # Run LastStage: height-pool + project to out_channels
        # LastStage expects [B, H*W, C] and sz=[H, W]
        feat_1d, _ = self.last_stage(x, sz)  # [B, W, C_out]

        return {
            'feat_1d': feat_1d,       # [B, W, C_out] for text branch
            'feat_2d': feat_2d,       # [B, H, W, C_2d] for IDS branch
            'sz_2d': (H, W),          # tuple (H, W)
        }
