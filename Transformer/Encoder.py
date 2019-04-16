import torch
import Utilities
import torch.nn as nn
import EncoderLayer
import MultiHeadAttention
import PositionwiseFeedForward


class Encoder(nn.Module):
    def __init__(self, size, encoder_layer, layer_nmuber):
        super(Encoder, self).__init__()
        self.encoder_layers = Utilities.clone(encoder_layer, layer_nmuber)
        self.norm = nn.LayerNorm(size)

    def forward(self, x):
        for l in self.encoder_layers:
            x = l(x)
        return self.norm(x)


if __name__ == "__main__":
    x = torch.randn(2, 3, 20)
    attention_layer = MultiHeadAttention.MultiHeadAttention(2, 10)
    forward_layer = PositionwiseFeedForward.PositionwiseFeedForward(20, 20)
    el = EncoderLayer.EncoderLayer(x.size(), attention_layer, forward_layer)
    ed = Encoder(x.size(), el, 6)
    x = ed(x)
    print(x.size())
