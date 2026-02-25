import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class PPHDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels=6625,
                 num_heads=12,
                 perturb_samples=3,
                 perturb_std=0.05,
                 return_feats=False,
                 **kwargs):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels 必须能被 num_heads 整除"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.return_feats = return_feats
        self.perturb_samples = perturb_samples
        self.perturb_std = perturb_std

        self.char_token = nn.Parameter(
            torch.zeros([1, num_heads, self.head_dim]))
        trunc_normal_(self.char_token, std=0.02)

        self.permutation_bias = nn.Parameter(
            torch.zeros([1, num_heads, self.head_dim]))
        trunc_normal_(self.permutation_bias, std=0.02)

        self.fc_kv = nn.Linear(in_channels, 2 * in_channels)
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x, data=None):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)

        kv = self.fc_kv(x_flat).reshape(B, H * W, 2, self.num_heads,
                                        self.head_dim)
        kv = kv.permute(2, 0, 3, 4, 1)
        x_k, x_v = kv[0], kv[1]

        base_token = self.char_token + self.permutation_bias
        base_token = base_token.expand(B, -1, -1)

        query_list = [base_token]
        for _ in range(self.perturb_samples - 1):
            noise = torch.randn_like(base_token) * self.perturb_std
            query_list.append(base_token + noise)

        attn_scores_all = []
        for q in query_list:
            q = q.unsqueeze(2)
            scores = torch.matmul(q, x_k)
            attn_scores_all.append(scores)
        attn_scores = torch.stack(attn_scores_all, dim=0).mean(0)

        attn_2d = attn_scores.view(B * self.num_heads, 1, H, W)
        attn_2d = F.softmax(attn_2d, dim=2)
        attn_2d = attn_2d.permute(0, 3, 1, 2)

        x_v_4d = x_v.reshape(B * self.num_heads, self.head_dim, H, W)
        x_v_for_attn = x_v_4d.permute(0, 3, 2, 1)
        head_feats = (attn_2d @ x_v_for_attn).squeeze(2)

        head_feats = head_feats.view(B, self.num_heads, W, self.head_dim)
        feats = head_feats.permute(0, 2, 1, 3).reshape(B, W, C)

        logits = self.fc(feats)

        if self.return_feats and self.training:
            return feats, logits
        if not self.training:
            return F.softmax(logits, dim=2)
        return logits
