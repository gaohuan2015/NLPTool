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


def get_mask(batch_size, length):
    max_length = torch.max(torch.tensor(length, dtype=torch.long))
    mask = torch.ones([batch_size, max_length.data, 1])
    for b, i in enumerate(length):
        mask[b, i:] = 0
    return mask


def build_dic_from_csv(data, columnhead, word_to_idx):
    for i in range(data.shape[0]):
        sentence = str(data.loc[[i], columnhead].iat[0])
        for w in sentence.split():
            if w not in word_to_idx:
                word_to_idx[w.lower()] = len(word_to_idx) + 1

def prepare_sequence(seq, to_ix,max_length):
    idxs = [to_ix[w.lower()] for w in seq.split()]
    return torch.tensor(idxs, dtype=torch.long)

def build_tag_from_csv(data, columnhead):
    return np.array(data.loc[:, columnhead])

def read_csv(path):
    path = os.getcwd() + path
    with open(path, encoding='utf-8') as f:
        data = pd.read_csv(path)
    return data


if __name__ == "__main__":
    word_to_idx = {}
    data = read_csv('/NLPTool/ESIM/data/train_test.csv')
    build_dic_from_csv(data, 'question1', word_to_idx)
    build_dic_from_csv(data, 'question2', word_to_idx)
    tag = build_tag_from_csv(data, 'is_duplicate')
    print('end')
