import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class LSTMCRF(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layer_number, tag_size):
        super(LSTMCRF, self).__init__()
        self.layer_num = layer_number
        self.hidden_dim = hidden_size
        self.tag_size = tag_size
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_size, hidden_size//2,
                            layer_number, batch_first=True, bidirectional=True)
        self.hid2tag = nn.Linear(hidden_size, tag_size)
        self.transition = nn.Parameter(tag_size, tag_size)

    def init_weight(self, data):
        return (torch.randn(self.layer_num*2, data.shape[0], self.hidden_dim // 2),
                torch.randn(self.layer_num*2, data.shape[0], self.hidden_dim // 2))

    def lstm_feature(self, x, length):
        self.init_weight(x)
        embed = self.embedding_layer(x)
        out, _ = rnn_utils.pack_padded_sequence(x, length)
        out, len = rnn_utils.pad_packed_sequence(out, batch_first=True)
        tag_score = self.hid2tag(out)
        return tag_score

    def forward_inference(self, lstm_feature):
        init_alphas = torch.full(1, self.tag_size)

    def forward(self, x):
        return self.lstm_feature(x)


if __name__ == "__main__":
    
