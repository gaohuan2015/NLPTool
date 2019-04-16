import torch
import copy
import MultiHeadAttention
import torch.nn as nn


class SublayerConnection(nn.Module):
    def __init__(self, size):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, Sublayer):
        norm = self.norm(x)
        return x+self.dropout(Sublayer(norm))
