import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import Embedding


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, model_dim):
        super(PositionEmbedding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., model_dim, 2)
                             * -(math.log(10000.0)/model_dim))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + \
            Variable(self.pe[:, :x.size(-1)], requires_grad=False)
        return x
