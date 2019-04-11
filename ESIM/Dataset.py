import torch
import Utilities
from torch.utils import data


class ESIMDataSet(data.Dataset):
    def __init__(self, sentence_list):
        self.sentence_list = sentence_list

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, idx):
        return self.sentence_list[idx]

