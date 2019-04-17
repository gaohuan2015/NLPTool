import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, embedding_layer1, embedding_layer2, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.scr_embedding = embedding_layer1
        self.tg_embedding = embedding_layer2
        self.decoder = decoder

    def forward(self, s, t):
        x_embed = self.scr_embedding(s)
        x_encoder = self.encoder(x_embed)
        t_embed = self.tg_embedding(t)
        target = self.decoder(t_embed, x_encoder)
        return target
