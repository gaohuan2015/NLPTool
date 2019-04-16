import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, embedding_layer, encoder_layer, decoder_layer):
        self.encoder = encoder_layer
        self.embedding = embedding_layer
        self.decoder = decoder_layer

    def forward(self, x):
        x_embed = self.embedding(x)
        x_encoder = self.encoder(x)
        x_encoder = self.decoder(x, e)
