import torch
import SublayerConnection as sc
import Utilities
import torch.nn as nn


class DecoderLayer(nn.Module):
    def __init__(self, attention_layer, en_attention_layer, forward_layer):
        super(DecoderLayer, self).__init__()
        self.att = attention_layer
        self.en_att = attention_layer
        self.fw = forward_layer
        self.sc = Utilities.clone(sc.SublayerConnection(), 3)

    def forward(self, x, e):
        x = self.att(x, x, x)
        x = self.sc[0](x, nn.Dropout(0.3))
        x = self.en_att(x, e, e)
        x = self.sc[1](x, nn.Dropout(0.3))
        x = self.fw(x)
        x = self.sc[2](x, nn.Dropout(0.3))
        return x
