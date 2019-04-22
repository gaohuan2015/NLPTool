import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from UnlabelData import *
from Utilities import *
from SequenceTag import *


class LSTMCRF(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layer_number,
                 tag_to_idx):
        super(LSTMCRF, self).__init__()
        self.layer_num = layer_number
        self.start_tag = '<start>'
        self.end_tag = '<stop>'
        self.hidden_dim = hidden_size
        self.tag_to_idx = tag_to_idx
        self.tag_size = len(tag_to_idx)
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size // 2,
            layer_number,
            batch_first=True,
            bidirectional=True)
        self.hid2tag = nn.Linear(hidden_size, self.tag_size)
        self.transition = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size))
        self.transition[tag_to_idx[self.start_tag], :] = -10000
        self.transition[:, tag_to_idx[self.end_tag]] = -10000

    def init_weight(self, data):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return (torch.randn(self.layer_num * 2, data.shape[0],
                            self.hidden_dim // 2).to(device),
                torch.randn(self.layer_num * 2, data.shape[0],
                            self.hidden_dim // 2).to(device))

    def lstm_feature(self, x, length):
        h = self.init_weight(x)
        embed = self.embedding_layer(x)
        packed_embedding = rnn_utils.pack_padded_sequence(
            embed, length, batch_first=True)
        out, _ = self.lstm(packed_embedding, h)
        out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
        tag_score = self.hid2tag(out)
        return tag_score

    def log_exp_sum(self, vec):
        max_value, indexs = torch.max(vec, -1)
        max_value_broadcast = max_value.view(vec.size(0), -1).expand(
            vec.size(0), vec.size(1))
        return max_value + torch.log(
            torch.sum(torch.exp(vec - max_value_broadcast)))

    def sentence_score(self, feature, tags):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = feature.size(0)
        score = torch.zeros(batch_size, 1).to(device)
        start_tags = torch.tensor([self.tag_to_idx[self.start_tag]],
                                  dtype=torch.long).to(device).view(1).expand(
                                      batch_size, 1)
        tags = torch.cat([start_tags, tags], dim=1)
        for i, feat in enumerate(feature.permute(1, 0, 2)):
            score = score + self.transition[
                tags[:, i + 1], tags[:, i]] + feat[:, tags[:, i + 1]][:, 0]

        score = score + self.transition[tags[:, self.tag_to_idx[self.end_tag]],
                                        tags[:, -1]]
        return score[0]

    def neg_log_likehood(self, sentence, tags, length):
        feature = self.lstm_feature(sentence, length)
        forward_score = self.forward_inference(feature)
        sentence_score = self.sentence_score(feature, tags)
        return forward_score - sentence_score

    def forward_inference(self, lstm_feature):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = lstm_feature.size(0)
        init_alphas = torch.full((batch_size, self.tag_size),
                                 -10000).to(device)
        init_alphas[:, self.tag_to_idx[self.start_tag]] = 0
        forward_var = init_alphas
        for feature in lstm_feature.permute(1, 0, 2):
            alphas = []
            for next_tag in range(self.tag_size):
                emition_scoare = feature[:, next_tag].view(
                    batch_size, -1).expand(batch_size, self.tag_size)
                transition_score = self.transition[next_tag].view(
                    1, -1).expand(batch_size, self.tag_size)
                next_tag_var = forward_var + emition_scoare + transition_score
                alphas.append(
                    self.log_exp_sum(next_tag_var).view(batch_size, -1))
            forward_var = torch.cat(alphas, 1).view(batch_size, -1)
        terminal_var = forward_var + self.transition[self.tag_to_idx[
            self.end_tag]]
        alphas = self.log_exp_sum(terminal_var)
        return alphas

    def forward(self, x, len):
        lstm = self.lstm_feature(x, len)
        forward_score = self.forward_inference(lstm)
        return lstm


if __name__ == "__main__":
    batchsize = 2
    word_2_idx = {}
    tag_to_ix = {"b": 0, "i": 1, "o": 2, "<start>": 3, "<stop>": 4}
    # read data
    training_data = [
        "the wall street journal reported today that apple corporation made money",
        "georgia tech is a university in georgia"
    ]
    tag_data = ["B I I I O O O B I O O", "B I O O O O B"]
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
    model = LSTMCRF(len(word_2_idx) + 1, 100, 50, 2, tag_to_ix).to(device)
    batch_index = 0
    for s, t, l in dataloader:
        l = torch.tensor(l, dtype=torch.long)
        model.neg_log_likehood(s.to(device), t.to(device), l.to(device))
        print(model(s.to(device), l.to(device)))
