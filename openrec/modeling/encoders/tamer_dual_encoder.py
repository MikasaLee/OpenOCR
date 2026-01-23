import torch
import torch.nn as nn

from .tamer_encoder import TAMER_Encoder

__all__ = ["TAMER_DualEncoder"]


class TAMER_DualEncoder(nn.Module):
    """Two independent TAMER encoders (text/ids) to avoid shared features.

    Returns a dict with 'text' and 'ids' keys, each a tuple (feat2d, mask2d)
    so decoders can consume task-specific memories without interference.
    """

    def __init__(self, in_channels: int, d_model: int = 512, growth_rate: int = 32, num_layers: int = 4):
        super().__init__()
        self.text_encoder = TAMER_Encoder(in_channels=in_channels, d_model=d_model, growth_rate=growth_rate, num_layers=num_layers)
        self.ids_encoder = TAMER_Encoder(in_channels=in_channels, d_model=d_model, growth_rate=growth_rate, num_layers=num_layers)
        self.out_channels = d_model  # decoder expects a scalar; both branches use same d_model

    def forward(self, img: torch.Tensor, img_mask: torch.Tensor = None):
        text_feat, text_mask = self.text_encoder(img, img_mask)
        ids_feat, ids_mask = self.ids_encoder(img, img_mask)
        return {
            'text': (text_feat, text_mask),
            'ids': (ids_feat, ids_mask),
        }
