import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, word_embed_layer, position_embed_layer):
        super(EmbeddingLayer, self).__init__()
        self.word_embed_layer = word_embed_layer
        self.position_embed_layer = position_embed_layer

    def forward(self, x):
        x = self.word_embed_layer(x)
        x = self.position_embed_layer(x)
        return x
