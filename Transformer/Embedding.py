import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embed(x)
