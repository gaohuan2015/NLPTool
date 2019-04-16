import torch
import Utilities
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, size, decoder_layer, layer_number):
        self.layers = Utilities.clone(decoder_layer, layer_number)
        self.norm = nn.LayerNorm(size)

    def forward(self, x, e):
        for l in self.layers:
            x = l(x, e)
        return self.norm(x)
