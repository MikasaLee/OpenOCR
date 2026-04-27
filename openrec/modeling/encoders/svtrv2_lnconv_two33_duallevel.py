"""
SVTRv2LNConvTwo33DualLevel — Dual-level encoder for structure-aware IDS verification.

Returns TWO feature levels from the same SVTR backbone:
    1. feat_local:  [B, H_local*W_local, C_local] — configurable intermediate stage
                                    output selected by local_stage_idx, preserving local
                                    stroke/component structure information.
  2. feat_global: [B, W, C_out] — after LastStage height-pooling + projection,
                  captures whole-line context for text recognition.

Design motivation:
  - SVTR front stages (Conv mixers) capture local patterns: stroke, radical.
  - SVTR deep stages (Global attention) capture inter-character dependencies.
  - IDS branch needs local structure → use feat_local.
  - Text branch needs global context → use feat_global.
  - Both share the same backbone; no extra feature extraction.
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


class SVTRv2LNConvTwo33DualLevel(nn.Module):

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
                 eps=1e-6,
                 num_convs=[[2] * 3, [2] * 3 + [3] * 3, [3] * 3],
                 kernel_sizes=[[3] * 3, [3] * 3 + [3] * 3, [3] * 3],
                 pope_bias=False,
                 local_stage_idx=0,
                 return_local = True,
                 **kwargs):
        """
        Args:
            local_stage_idx: which stage's output to use as feat_local.
                0 = stage0 output (dims[1] channels after downsample, highest spatial resolution)
                1 = stage1 output (dims[2] channels after downsample)
                Default 0 — highest resolution, best for stroke/component capture.
        """
        super().__init__()
        num_stages = len(depths)
        self.num_features = dims[-1]
        self.local_stage_idx = local_stage_idx
        self.return_local = return_local

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
        self.feat_local_channels = dims[min(local_stage_idx + 1, len(dims) - 1)]

        self.apply(self._init_weights)

        # LastStage for 1D global output (height pooling + projection)
        assert last_stage, "DualLevel encoder requires last_stage=True"
        self.last_stage = LastStage(self.num_features, out_channels, last_drop)

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

        feat_local = None
        feat_local_sz = None

        for i, stage in enumerate(self.stages):
            x, sz = stage(x, sz)
            if i == self.local_stage_idx:
                # Capture intermediate feature as local representation.
                # After downsample in this stage, x may be 4D (Conv) or 3D (Global).
                H_l, W_l = sz[0], sz[1]
                if x.dim() == 4:
                    # Conv output: [B, C, H, W] -> [B, H*W, C]
                    feat_local = x.permute(0, 2, 3, 1).reshape(-1, H_l * W_l, x.size(1))
                else:
                    # Global/1D output: [B, H*W, C]
                    feat_local = x.clone()
                feat_local_sz = (H_l, W_l)

        # After all stages, run LastStage for global 1D output
        # x at this point is the last stage output: [B, H*W, C] or [B, C, H, W]
        H, W = sz[0], sz[1]
        C = self.num_features
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).reshape(-1, H * W, C)

        feat_global, _ = self.last_stage(x, sz)  # [B, W, C_out]

        if self.return_local:
            return {
                'feat_local': feat_local,         # [B, H_l*W_l, C_local]
                'feat_global': feat_global,        # [B, W, C_out]
                'local_sz': feat_local_sz,         # (H_l, W_l)
            }
        else: return feat_global       # [B, W, C_out]
           
