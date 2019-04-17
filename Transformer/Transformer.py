import torch
import torch.nn as nn
import EncoderLayer
import DecoderLayer
import MultiHeadAttention
import Encoder
import PositionwiseFeedForward as pf
import Decoder


class Transformer(nn.Module):
    def __init__(self, embedding_layer1, embedding_layer2, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.scr_embedding = embedding_layer1
        self.tg_embedding = embedding_layer2
        self.decoder = decoder

    def forward(self, s, t):
        x_embed = self.scr_embedding(s)
        x_encoder = self.encoder(x_embed)
        t_embed = self.tg_embedding(t)
        target = self.decoder(t_embed, x_encoder)
        return target


if __name__ == "__main__":
    # config
    head_num = 2
    max_length = 3
    model_dim = 10
    batch = 1
    embedding_dim = head_num * model_dim
    size = (batch, max_length, embedding_dim)
    # define model
    sentence1 = torch.tensor([1, 2, 3]).unsqueeze(0)
    sentence2 = torch.tensor([2, 3, 4]).unsqueeze(0)
    embedding_layer = nn.Embedding(5, embedding_dim)
    attention_layer = MultiHeadAttention.MultiHeadAttention(
        head_num, model_dim)
    forward_layer = pf.PositionwiseFeedForward(embedding_dim, 100)
    encoder_layer = EncoderLayer.EncoderLayer(
        size, attention_layer, forward_layer)
    encoder = Encoder.Encoder(size, encoder_layer, 6)
    attention_layer2 = MultiHeadAttention.MultiHeadAttention(
        head_num, model_dim)
    forward_layer2 = pf.PositionwiseFeedForward(embedding_dim, 100)
    decoder_layer = DecoderLayer.DecoderLayer(
        size, attention_layer2, attention_layer, forward_layer2)
    decoder = Decoder.Decoder(size, decoder_layer, 6)
    t = Transformer(embedding_layer, embedding_layer,
                    encoder, decoder)
    x = t(sentence1, sentence2)
    print(x.data.cpu().numpy())
