import torch
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
