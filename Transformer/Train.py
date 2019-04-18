import torch
import torch.nn as nn
import numpy as np
import BuildTransformer
import Dataset
import torch.utils.data as data
from tqdm import tqdm

if __name__ == "__main__":
    # config
    head_num = 2
    max_length = 3
    model_dim = 10
    batch = 1
    embedding_dim = head_num * model_dim
    size = (batch, max_length, embedding_dim)
    # define model
    sentence1 = torch.randint(1, 3, (16, max_length))
    sentence2 = torch.randint(1, 4, (16, max_length))
    dataset = Dataset.Data(sentence1, sentence2)
    dataloader = data.DataLoader(dataset, batch, shuffle=False, num_workers=4)
    model = BuildTransformer.BuildTransformer(
        3, 4, head_num, model_dim, embedding_dim, 100, max_length)
    criterion = nn.CrossEntropyLoss()
    optimize = torch.optim.Adam(model.parameters(), 0.0001)
    for i in tqdm(range(200)):
        total_loss = 0
        for i, sentence in enumerate(dataloader):
            x = model(sentence[0])
            optimize.zero_grad()
            loss = criterion(
                x.reshape(x.size(0)*x.size(1), x.size(-1)), sentence[1].reshape(sentence[1].size(-2)*sentence[1].size(-1)))
            loss.backward()
            optimize.step()
            total_loss += loss
        print('loss is %f' % (total_loss))

    for i, sentence in enumerate(dataloader):
        x = model(sentence[0])
        v, index = torch.max(x, -1)
        print(sentence[1])
        print(index)
