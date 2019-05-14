import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from UnlabelData import *
from Utilities import *
from SequenceTag import *
from tqdm import *
import matplotlib.pyplot as plt
import torch.autograd as autograd


class LSTMCRF(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layer_number,
                 tag_to_idx):
        super(LSTMCRF, self).__init__()
        self.layer_num = layer_number
        self.start_tag = '<start>'
        self.end_tag = '<stop>'
        self.pad_tag = '<pad>'
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transition = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size)).to(device)
        self.transition[tag_to_idx[self.start_tag], :] = -10000
        self.transition[:, tag_to_idx[self.end_tag]] = -10000
        self.transition[tag_to_idx[self.pad_tag], :] = -10000
        self.transition[:, tag_to_idx[self.pad_tag]] = -10000
        self.transition[tag_to_idx[self.end_tag], tag_to_idx[self.pad_tag]] = 0
        self.transition[tag_to_idx[self.start_tag], tag_to_idx[self.pad_tag]] = 0

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
            torch.sum(torch.exp(vec - max_value_broadcast), 1))

    def sentence_score(self, feature, tags):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = feature.size(0)
        score = torch.zeros(batch_size, 1).to(device)
        start_tags = torch.tensor([self.tag_to_idx[self.start_tag]],
                                  dtype=torch.long).to(device).view(1).expand(
                                      batch_size, 1)
        tags = torch.cat([start_tags, tags], dim=1)
        for i, feat in enumerate(feature.permute(1, 0, 2)):
            index = tags[:, i + 1].view(batch_size, -1)
            emition_scoare = torch.gather(feat, 1, index)
            transition_score = self.transition[tags[:, i +
                                                    1], tags[:, i]].view(
                                                        batch_size, -1)
            tmp_score = torch.add(emition_scoare, transition_score)
            score = torch.add(score, tmp_score)
        score = score + self.transition[tags[:, self.tag_to_idx[self.end_tag]],
                                        tags[:, -1]].view(batch_size, -1)
        return score

    def neg_log_likehood(self, sentence, tags, length):
        feature = self.lstm_feature(sentence, length)
        forward_score = self.forward_inference(feature)
        sentence_score = self.sentence_score(feature, tags)
        return torch.mean(torch.abs(forward_score - sentence_score))

    def forward_inference(self, lstm_feature):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = lstm_feature.size(0)
        init_alphas = torch.full((batch_size, self.tag_size), -10000)
        init_alphas[:, self.tag_to_idx[self.start_tag]] = 0
        forward_var = autograd.Variable(init_alphas)
        forward_var = forward_var.to(device)
        for feature in lstm_feature.permute(1, 0, 2):
            alphas = []
            for next_tag in range(self.tag_size):
                emtion_score = feature[:, next_tag].view(
                    batch_size, -1).expand(batch_size, self.tag_size)
                transition_score = self.transition[next_tag].view(
                    1, -1).expand(batch_size, self.tag_size)
                score = transition_score + emtion_score
                next_tag_var = torch.add(forward_var, score)
                alphas.append(
                    self.log_exp_sum(next_tag_var).view(batch_size, -1))
            forward_var = torch.cat(alphas, 1).view(batch_size, -1)
        end_var = self.transition[self.tag_to_idx[self.end_tag]]
        terminal_var = torch.add(forward_var, end_var[:self.tag_size])
        alphas = self.log_exp_sum(terminal_var)
        return alphas.view(batch_size, -1)

    def viterbi_decode(self, feats):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = feats.size(0)
        backpointers = []
        init_vvars = torch.full((1, self.tag_size), -10000.)
        init_vvars[0][self.tag_to_idx[self.start_tag]] = 0
        forward_var = init_vvars.to(device)
        for feat in feats.permute(1, 0, 2):
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tag_size):
                transition_score = self.transition[next_tag].view(1, -1)
                next_tag_var = forward_var + transition_score
                _, best_tag_id = torch.max(next_tag_var, -1)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        terminal_var = forward_var + self.transition[self.tag_to_idx[
            self.end_tag]][:self.tag_size].view(1, -1)
        _, best_tag_id = torch.max(terminal_var, -1)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_idx[self.start_tag]
        best_path.reverse()
        return path_score, best_path

    def forward(self, x, len):
        lstm = self.lstm_feature(x, len)
        score, tag = self.viterbi_decode(lstm)
        return score, tag


if __name__ == "__main__":
    batchsize = 3
    word_2_idx = {}
    tag_to_ix = {"b": 0, "i": 1, "o": 2, "<start>": 3, "<stop>": 4, "<pad>": 5}
    # read data
    training_data = [
        "the wall street journal reported is good",
        "georgia tech is a bad university",
        'Jean Pierre lives in a New York city',
        'The European Union is a political union',
        'A French American actor won 1997 cup'
    ]
    tag_data = [
        "B I I I O O O", 
        "B I O O O O", 
        "B I O O O B I O", 
        "O B I O O O O",
        "O B I O O O O"
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
    model = LSTMCRF(len(word_2_idx) + 1, 100, 50, 2, tag_to_ix).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = []
    y = []
    for epoch in tqdm(range(100)):
        total_loss = 0
        for s, t, l in dataloader:
            model.zero_grad()
            l = torch.tensor(l, dtype=torch.long)
            loss = model.neg_log_likehood(
                s.to(device), t.to(device), l.to(device))
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss = total_loss + loss
        x.append(epoch)
        y.append(total_loss)
    torch.save(model, 'CRF')
    plt.plot(x, y, 'ro-')
    plt.title('loss function')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
