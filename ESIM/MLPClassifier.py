import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, class_number):
        super(MLP, self).__init__()
        self.line = nn.Linear(input_dim, input_dim)
        self.classfier = nn.Linear(input_dim, class_number)

    def forward(self, value):
        tmp = self.line(value)
        tmp = torch.tanh(tmp)
        score = self.classfier(tmp)
        return score
