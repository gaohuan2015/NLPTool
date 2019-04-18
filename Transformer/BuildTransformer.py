import torch
import torch.nn as nn
import MultiHeadAttention as multatt
import PositionwiseFeedForward as pfw
import PositionEmbedding as pe
import EncoderLayer as ec
import DecoderLayer as dc
import copy
import Encoder
import Decoder
import Transformer
import Embedding
import PositionEmbedding
import EmbeddingLayer
import Generator


def BuildTransformer(scr_vocab_size, tg_vocabe_size, number_head, model_dim, embedding_dim, convert_dim, max_len):
    attn = multatt.MultiHeadAttention(number_head, model_dim)
    fw = pfw.PositionwiseFeedForward(embedding_dim, convert_dim)
    src_word_embedding = Embedding.Embedding(scr_vocab_size, embedding_dim)
    tag_word_embedding = Embedding.Embedding(tg_vocabe_size, embedding_dim)
    position_embedding = PositionEmbedding.PositionEmbedding(
        max_len, embedding_dim)
    src_embedding_layer = EmbeddingLayer.EmbeddingLayer(
        src_word_embedding, copy.deepcopy(position_embedding))
    tg_embedding_layer = EmbeddingLayer.EmbeddingLayer(
        tag_word_embedding, copy.deepcopy(position_embedding))
    encoder_layer = ec.EncoderLayer(
        copy.deepcopy(attn), copy.deepcopy(fw))
    decoder_layer = dc.DecoderLayer(
        copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(fw))
    encoder = Encoder.Encoder(encoder_layer, 2)
    decodeer = Decoder.Decoder(decoder_layer, 2)
    generator = Generator.Generator(embedding_dim, tg_vocabe_size)
    encoderdecoder = Transformer.Transformer(
        src_embedding_layer, tg_embedding_layer, encoder, decodeer, generator)

    for p in encoderdecoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return encoderdecoder
