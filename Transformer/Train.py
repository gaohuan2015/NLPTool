import torch
import torch.nn as nn
import BuildTransformer

if __name__ == "__main__":
    # config
    head_num = 2
    max_length = 3
    model_dim = 10
    batch = 1
    embedding_dim = head_num * model_dim
    size = (batch, max_length, embedding_dim)
    # define model
    model = BuildTransformer.BuildTransformer(
        size, 4, 5, head_num, model_dim, embedding_dim, 100)
    sentence1 = torch.tensor([1, 2, 3]).unsqueeze(0)
    sentence2 = torch.tensor([2, 3, 4]).unsqueeze(0)
    x = model(sentence1, sentence2)
    print(x.data.cpu().numpy())
