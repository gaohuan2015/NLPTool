import torch
import torch.nn as nn

def clone(layer, number):
    return nn.ModuleList([layer for x in range(number)])