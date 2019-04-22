import torch
import torch.utils.data as data


class SequenceTag(data.Dataset):
    def __init__(self, sequences, tags):
        super(SequenceTag, self).__init__()
        self.seqs = sequences
        self.t = tags

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.t[idx]
