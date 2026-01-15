import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Bottleneck(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super().__init__()
        inter = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(inter)
        self.conv1 = nn.Conv2d(n_channels, inter, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(inter, growth_rate, kernel_size=3, padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DenseNet 瓶颈：先 1x1 降维再 3x3 卷积，最后与输入拼接形成致密连接
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


class _SingleLayer(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DenseNet 基础层：单次 3x3 卷积后与输入拼接
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


class _Transition(nn.Module):
    def __init__(self, n_channels: int, n_out_channels: int, use_dropout: bool):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(n_out_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 过渡层：1x1 压缩通道后 2x2 平均池化（ceil_mode 保留边界）
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class _DenseNetBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        num_layers: int,
        reduction: float = 0.5,
        bottleneck: bool = True,
        use_dropout: bool = True,
    ):
        super().__init__()
        n_dense_blocks = num_layers
        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(in_channels, n_channels, kernel_size=7, padding=3, stride=2, bias=False)
        self.norm1 = nn.BatchNorm2d(n_channels)
        self.dense1 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout)
        self.out_channels = n_channels + n_dense_blocks * growth_rate
        self.post_norm = nn.BatchNorm2d(self.out_channels)

    @staticmethod
    def _make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout):
        layers = []
        for _ in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(_Bottleneck(n_channels, growth_rate, use_dropout))
            else:
                layers.append(_SingleLayer(n_channels, growth_rate, use_dropout))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Stem：7x7 stride2 卷积 + BN + ReLU + 2x2 最大池化，mask 同步下采样
        out = self.conv1(x)
        out = self.norm1(out)
        out_mask = x_mask[:, 0::2, 0::2]
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out_mask = out_mask[:, 0::2, 0::2]
        # 三个 Dense 块 + 两个 Transition 逐级下采样并提升感受野，同时下采样 mask
        out = self.dense1(out)
        out = self.trans1(out)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense2(out)
        out = self.trans2(out)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense3(out)
        out = self.post_norm(out)
        return out, out_mask


class TAMER_Encoder(nn.Module):
    """基于 DenseNet 的编码器，输出序列特征供 Transformer 解码器使用。

    参数说明：
    - in_channels: 输入图像通道数，RGB=3、灰度=1。
    - d_model: 最终投影后的通道维度（解码器期望的特征维度）。
    - growth_rate: DenseNet 每层的增长率（输出通道增量）。
    - num_layers: 每个 Dense block 内的层数（共有 3 个 block，均使用该层数）。
    返回： (seq, (h, w))，其中 seq 形状 [B, H*W, d_model]，(h,w) 为特征图网格高宽。
    """

    def __init__(self, in_channels: int, d_model: int = 512, growth_rate: int = 32, num_layers: int = 4):
        super().__init__()
        self.backbone = _DenseNetBackbone(
            in_channels=in_channels, growth_rate=growth_rate, num_layers=num_layers
        )
        self.proj = nn.Conv2d(self.backbone.out_channels, d_model, kernel_size=1)
        self.out_channels = d_model
        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, img: torch.Tensor, img_mask: torch.Tensor = None):
        # 若未提供 mask，则默认全有效区域（与不 padding 的 resize 相符）；约定 mask: True=padding
        if img_mask is None:
            img_mask = torch.zeros((img.shape[0], img.shape[2], img.shape[3]), dtype=torch.bool, device=img.device)

        # CNN 主干提取特征，mask 同步下采样
        feat, mask = self.backbone(img, img_mask)  # feat: [B, C, H, W], mask: [B, H, W]

        # 1x1 投影到 d_model，保持二维网格以便加 2D 位置编码
        feat = self.proj(feat)   # [B, d, H, W]
        feat = feat.permute(0, 2, 3, 1).contiguous()  # [B, H, W, d]
        feat = self.pos_enc_2d(feat, mask)            # 加 2D 位置编码（带归一化坐标）
        feat = self.norm(feat)

        # 保持 2D 形状返回，方便后续模块按需 flatten / coverage
        return feat, mask


class ImgPosEnc(nn.Module):
    """与官方 TAMER 一致的 2D 正弦位置编码，支持按 mask 归一化坐标。"""

    def __init__(self, d_model: int = 512, temperature: float = 10000.0, normalize: bool = False, scale: float = None):
        super().__init__()
        assert d_model % 2 == 0
        self.half_d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize 应为 True 才能指定 scale")
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 强制转换为 bool，约定 True=padding；取反后累加生成坐标
        mask = mask.bool()
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(0, self.half_d_model, 2, dtype=torch.float32, device=x.device)
        inv_freq = 1.0 / (self.temperature ** (dim_t / self.half_d_model))

        pos_x = torch.einsum("b h w, d -> b h w d", x_embed, inv_freq)
        pos_y = torch.einsum("b h w, d -> b h w d", y_embed, inv_freq)

        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=4).flatten(3)
        pos = torch.cat((pos_x, pos_y), dim=3)

        return x + pos