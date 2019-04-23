import torch
from Utilities import *
from SequenceTag import *
from LSTMCRF import *

if __name__ == "__main__":
    batchsize = 1
    word_2_idx = {}
    tag_to_ix = {"b": 0, "i": 1, "o": 2, "<start>": 3, "<stop>": 4, "<pad>": 5}
    # read data
    training_data = [
        "the wall street journal reported today that apple corporation made money",
        "georgia tech is a university in georgia",
        'Jean Pierre lives in New York',
        'The European Union is a political and economic union',
        'A French American actor won an oscar'
    ]
    tag_data = [
        "B I I I O O O B I O O", "B I O O O O B", "B I O O B I",
        "O B I O O O O O O", "O B I O O O O"
    ]
    sentences_to_id = []
    tag_to_id = []
    build_voc_size(training_data, word_2_idx)
    for s, t in zip(training_data, tag_data):
        sentences_to_id.append(prepare_sequence(s, word_2_idx))
        tag_to_id.append(prepare_sequence(t, tag_to_ix))
    training = SequenceTag(sentences_to_id, tag_to_id)
    dataloader = data.DataLoader(
        training,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4,
        collate_fn=padd_sentence_crf)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('CRF')
    correct = 0.0
    number = 0.0
    for s, t, l in dataloader:
        l = torch.tensor(l, dtype=torch.long)
        t = t.to(device)
        score, tag = model(s.to(device), l.to(device))
        for i in range(len(tag)):
            if tag[i] == t.squeeze(0)[i]:
                correct = correct + 1
        number = number + t.size(1)
        print(correct / number)