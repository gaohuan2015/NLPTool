import torch
import Utilities
import torch.nn as nn
import SublayerConnection as sc


class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, forward_layer):
        super(EncoderLayer, self).__init__()
        self.att = attention_layer
        self.fw = forward_layer
        self.subcon = Utilities.clone(sc.SublayerConnection(), 2)

    def forward(self, x):
        x = self.att(x, x, x)
        x = self.subcon[0](x, nn.Dropout(0.3))
        x = self.fw(x)
        x = self.subcon[1](x, nn.Dropout(0.3))
        return x
