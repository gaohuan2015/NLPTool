import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import Utilities


class MultiHeadAttention(nn.Module):
    def __init__(self, number_head, d_model):
        super(MultiHeadAttention, self).__init__()
        self.h = number_head
        self.dim = d_model
        self.lines = Utilities.clone(
            nn.Linear(d_model*number_head, d_model*number_head), 4)

    def attention(self, query, key, value):
        score = torch.matmul(query, key.transpose(-2, -1))
        score = score / math.sqrt(self.dim)
        score = F.softmax(score, dim=-1)
        return torch.matmul(score, value)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        Q, K, V = [layer(x)
                   for layer, x in zip(self.lines, (query, key, value))]
        Q, K, V = [x.view(batch_size, -1, self.h, self.dim).transpose(1, 2)
                   for x in (Q, K, V)]
        score = self.attention(Q, K, V)
        score = score.transpose(1, 2).contiguous().view(batch_size, -1, self.h*self.dim)
        return self.lines[-1](score)

