import torch
import os
import numpy as np
import pandas as pd
import torch.nn.utils.rnn as rnn_utils


def padd_sentence(sentences):
    sentences.sort(key=lambda x: len(x), reverse=True)
    sentences_length = [len(x) for x in sentences]
    sentences = rnn_utils.pad_sequence(
        sentences, batch_first=True, padding_value=0)
    return sentences, sentences_length


def build_mask(batch_size, max_length, hidden_dim):
    mask = torch.zeros([batch_size, max_length, hidden_dim], dtype=torch.float)
    for i in range(batch_size):
        for j in range(hidden_dim):
            mask[i, j, j] = 1
    return mask


def get_mask(batch_size, max_length, length):
    mask = torch.ones([batch_size, max_length, 1])
    for b, i in enumerate(length):
        mask[b, i:] = 0
    return mask


def build_dic_from_csv(data, columnhead, word_to_idx):
    for i in range(data.shape[0]):
        sentence = str(data.loc[[i], columnhead].iat[0])
        for w in sentence.split():
            if w.lower() not in word_to_idx:
                word_to_idx[w.lower()] = len(word_to_idx) + 1


def prepare_sequence(seq, to_ix):
    idxs = [torch.tensor(to_ix[w.lower()], dtype=torch.long)
            for w in seq.split()]
    return torch.tensor(idxs, dtype=torch.long)


def build_tag_from_csv(data, columnhead):
    return np.array(data.loc[:, columnhead])


def read_csv(path):
    path = os.getcwd() + path
    with open(path, encoding='utf-8') as f:
        data = pd.read_csv(path)
    return data
