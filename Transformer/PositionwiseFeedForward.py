import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, convert_dim):
        super(PositionwiseFeedForward, self).__init__()
        self.line1 = nn.Linear(embedding_dim, convert_dim)
        self.line2 = nn.Linear(convert_dim, embedding_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.line1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.line2(x)
        return x
