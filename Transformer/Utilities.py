import torch
import torch.nn as nn
import torch.autograd.variable as Variable
import numpy as np


def clone(layer, number):
    return nn.ModuleList([layer for x in range(number)])


def build_mask(sentence, pad=0):
    mask = (sentence != 0).unsqueeze(-2)
    shape = (1, sentence.size(-1), sentence.size(-1))
    sub_maks = np.triu(np.ones(shape), k=1).astype('uint8')
    sub_maks = torch.from_numpy(sub_maks) == 0
    final_mask = mask & Variable(sub_maks.type_as(sub_maks.data))
    return final_mask


if __name__ == "__main__":
    tgt = torch.tensor([[1, 2, 3], [1, 2, 0]])
    print(build_mask(tgt))
