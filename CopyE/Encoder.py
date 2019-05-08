import torch
import torch.nn as nn


class Encoder(nn.modules):
    def __init__(self, vocab_size, hidden_dim):
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
