import torch.nn as nn


class TANLoss(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, predicts, batch):
        return {
            'loss': predicts['loss'],
            'lpred_loss': predicts['lpred_loss'],
            'rrepred_loss': predicts['rrepred_loss'],
            'triplet_loss': predicts['triplet_loss'],
        }
