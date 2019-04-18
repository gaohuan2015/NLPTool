import torch
import Utilities
import torch.nn as nn
import EncoderLayer
import MultiHeadAttention
import PositionwiseFeedForward


class Encoder(nn.Module):
    def __init__(self, encoder_layer, layer_nmuber):
        super(Encoder, self).__init__()
        self.encoder_layers = Utilities.clone(encoder_layer, layer_nmuber)

    def forward(self, x):
        self.norm = nn.LayerNorm(x.size())
        for l in self.encoder_layers:
            x = l(x)
        return self.norm(x)

