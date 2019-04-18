import torch
import copy
import MultiHeadAttention
import torch.nn as nn


class SublayerConnection(nn.Module):
    def __init__(self):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, Sublayer):
        self.norm = nn.LayerNorm(x.size())
        norm = self.norm(x)
        return x+self.dropout(Sublayer(norm))
