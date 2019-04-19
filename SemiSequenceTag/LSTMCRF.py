import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from UnlabelData import *
from Utilities import *


class LSTMCRF(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layer_number, tag_size):
        super(LSTMCRF, self).__init__()
        self.layer_num = layer_number
        self.hidden_dim = hidden_size
        self.tag_size = tag_size
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size//2,
                            layer_number, batch_first=True, bidirectional=True)
        self.hid2tag = nn.Linear(hidden_size, tag_size)

    def init_weight(self, data):
        return (torch.randn(self.layer_num*2, data.shape[0], self.hidden_dim // 2),
                torch.randn(self.layer_num*2, data.shape[0], self.hidden_dim // 2))

    def lstm_feature(self, x, length):
        h = self.init_weight(x)
        embed = self.embedding_layer(x)
        packed_embedding = rnn_utils.pack_padded_sequence(
            embed, length, batch_first=True)
        out, _ = self.lstm(packed_embedding, h)
        out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
        tag_score = self.hid2tag(out)
        return tag_score

    def forward_inference(self, lstm_feature):
        init_alphas = torch.full(1, self.tag_size)

    def forward(self, x, len):
        return self.lstm_feature(x, len)


if __name__ == "__main__":
    word_2_idx = {}
    tag_to_ix = {"b": 0, "i": 1, "o": 2, "<start>": 3, "<stop>": 4}
    # read data
    training_data = ["the wall street journal reported today that apple corporation made money",
                     "georgia tech is a university in georgia"]
    tag_data = ["B I I I O O O B I O O", "B I O O O O B"]
    sentences_to_id = []
    tag_to_id = []
    build_voc_size(training_data, word_2_idx)
    for s, t in zip(training_data, tag_data):
        sentences_to_id.append(prepare_sequence(s, word_2_idx))
        tag_to_id.append(prepare_sequence(t, tag_to_ix))
    training = UnlabelData(sentences_to_id)
    dataloader = data.DataLoader(
        training, batch_size=2, shuffle=False, num_workers=4, collate_fn=padd_sentence_crf)
    model = LSTMCRF(len(word_2_idx)+1, 100, 50, 2, len(tag_to_ix)+1)
    for s, l in dataloader:
        print(model(s, l))
