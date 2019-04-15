import torch
import torch.nn as nn
import Utilities
import Dataset
import MLPClassifier
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils import data


class ESIM(nn.Module):
    # init our model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, number_layers):
        super(ESIM, self).__init__()
        # define parameters
        self.vocab_size = vocab_size
        self.number_layers = number_layers
        self.hidden_dim = hidden_dim
        # define layers
        self.emebeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, number_layers, batch_first=True)
        self.line = nn.Linear(4 * hidden_dim, hidden_dim)
        self.composition = nn.LSTM(
            hidden_dim, hidden_dim, number_layers, batch_first=True)
        self.classifier = MLPClassifier.MLP(4*hidden_dim, 2)
        self.dropout = torch.nn.Dropout(0.5)
    # init hidden layers
    def init_hidden(self, sentence):
        return (torch.randn(self.number_layers, sentence.shape[0],
                            self.hidden_dim),
                torch.randn(self.number_layers, sentence.shape[0],
                            self.hidden_dim))

    # encode sentence
    def sentence_encoder(self, sentence, length):
        h = self.init_hidden(sentence)
        sentence_embed = self.emebeddings(sentence)
        sentence_embedding_packed = rnn_utils.pack_padded_sequence(
            sentence_embed, length, batch_first=True)
        sentence_encode, _ = self.lstm(sentence_embedding_packed, h)
        sentence_encode, out_len = rnn_utils.pad_packed_sequence(
            sentence_encode, batch_first=True)
        sentence_encode = self.dropout(sentence_encode)
        return sentence_encode

    # get similarity matrix
    def get_similarity(self, sentence1, sentence2):
        return sentence1.bmm(sentence2.transpose(2, 1))

    def softmax_attention(self, sentence, similarity_matrix):
        result = nn.functional.softmax(similarity_matrix, dim=-1)
        result_atten = result.bmm(sentence)
        return result_atten

    def forward(self, sentence1, length1, sentence2, length2):
        # encoder sentence
        sentence1_embedding = self.sentence_encoder(sentence1, length1)
        sentence2_embedding = self.sentence_encoder(sentence2, length2)
        # pad again
        if length1[0] >= length2[0]:
            mask = Utilities.build_mask(
                sentence1.shape[0], length1[0], length2[0])
            sentence2_embedding = mask.bmm(sentence2_embedding)
        else:
            mask = Utilities.build_mask(
                sentence2.shape[0], length2[0], length1[0])
            sentence1_embedding = mask.bmm(sentence1_embedding)

        # similarity between two sentences
        similarity_matrix = self.get_similarity(
            sentence1_embedding, sentence2_embedding)
        # attention of each sentence
        attented_sentence1 = self.softmax_attention(
            sentence1_embedding, similarity_matrix)
        attented_sentence2 = self.softmax_attention(
            sentence2_embedding, similarity_matrix.permute(0, 2, 1))
        # enhance sentence representation
        enhance_embedding1 = torch.cat([
            sentence1_embedding, attented_sentence1,
            sentence1_embedding - attented_sentence1,
            sentence1_embedding * attented_sentence1
        ], dim=-1)
        enhance_embedding2 = torch.cat([
            sentence2_embedding, attented_sentence2,
            sentence2_embedding - attented_sentence2,
            sentence2_embedding * attented_sentence2
        ], dim=-1)
        # projection representaion
        projection1 = F.relu(self.line(enhance_embedding1))
        projection2 = F.relu(self.line(enhance_embedding2))
        # composition representaion
        composition1, _ = self.composition(projection1)
        composition2, _ = self.composition(projection2)
        composition1 = self.dropout(composition1)
        composition2 = self.dropout(composition2)
        # classifier
        if length1[0] >= length2[0]:
            mask1 = Utilities.get_mask(sentence1.shape[0], length1[0], length1)
            mask2 = Utilities.get_mask(sentence2.shape[0], length1[0], length2)
        else:
            mask1 = Utilities.get_mask(sentence1.shape[0], length2[0], length1)
            mask2 = Utilities.get_mask(sentence2.shape[0], length2[0], length2)
        v_avg1 = torch.sum((composition1 * mask1) /
                           torch.sum(mask1, 1).unsqueeze(-1), 1)
        v_avg2 = torch.sum((composition2 * mask2) /
                           torch.sum(mask2, 1).unsqueeze(-1), 1)
        v_max1, _ = torch.max(composition1, 1)
        v_max2, _ = torch.max(composition2, 1)
        v = torch.cat([v_avg1, v_max1, v_avg2, v_max2], dim=-1)
        p = self.classifier(v)
        p = nn.functional.softmax(p, dim=-1)
        return p
