import json
import torch.tensor
import os
import SentencesAndRelation as sar
import torch.nn.utils.rnn as rnn_utils


def load_relation2id_from_json(path):
    path = os.getcwd() + path
    relation2id = {}
    with open(path) as f:
        relation2id = json.load(f)
    return relation2id


def load_data_from_json(path):
    path = os.getcwd() + path
    data = {}
    with open(path) as f:
        data = json.load(f)
    return data


def load_wordembedding_from_json(paht):
    path = os.getcwd() + path
    word_embedding = {}
    word_embedding_list = []
    with open(path) as f:
        word_embedding = json.load(f)
    word_embedding_list = [v for k, v in word_embedding]
    return word_embedding


def load_dic_from_json(path):
    path = os.getcwd() + path
    dic = {}
    with open(path) as f:
        dic = json.load(f)
    return dic


def buildSentenceFromJson(data):
    sentences = data[1]
    labels = data[2]
    build_data = sar.SentenceAndRelation(sentences, labels)
    return build_data


def pad_sentence(data):
    sentences = [torch.tensor(s) for s, l in data]
    labels = [torch.tensor(l) for s, l in data]
    sentences.sort(key=lambda x: len(x), reverse=True)
    sentences_length = [len(x) for x in sentences]
    sentences = rnn_utils.pad_sequence(
        sentences, batch_first=True, padding_value=0)
    return sentences, labels, sentences_length
