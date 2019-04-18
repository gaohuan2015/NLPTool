import torch
import Utilities
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, decoder_layer, layer_number):
        super(Decoder, self).__init__()
        self.layers = Utilities.clone(decoder_layer, layer_number)

    def forward(self, x, e):
        self.norm = nn.LayerNorm(x.size())
        for l in self.layers:
            x = l(x, e)
        return self.norm(x)
