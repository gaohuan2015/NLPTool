import torch
import Model
import torch.nn as nn
import Dataset
import Utilities
import torch.utils.data as data
from tqdm import tqdm

if __name__ == "__main__":
    train_x = [
        torch.tensor([1, 1, 1, 1, 1, 1, 1]),
        torch.tensor([2, 2, 2, 2, 2, 2]),
        torch.tensor([3, 3, 3, 3, 3]),
        torch.tensor([4, 4, 4, 4]),
        torch.tensor([5, 5, 5]),
        torch.tensor([6, 6]),
        torch.tensor([7])
    ]
    testdata = Dataset.ESIMDataSet(train_x)
    data_loader = data.DataLoader(
        testdata,
        batch_size=2,
        shuffle=False,
        collate_fn=Utilities.padd_sentence)
    model = Model.ESIM(8, 1, 2, 1)
    criterion = nn.CrossEntropyLoss()
    optimize = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = 0

    for i in tqdm(range(100)):
        iter = data.dataloader._DataLoaderIter(data_loader)
        for i in range(len(data_loader)):
            s,l = iter.next()
            model.zero_grad()
            label = torch.ones([s.shape[0]], dtype=torch.long)
            p = model(s, l, s, l)
            loss = criterion(p, label)
            loss.backward()
            optimize.step()
    print(loss.data)

    for s, l in data_loader:
        p = model(s, l, s, l)
        _, label = torch.max(p, dim=1)
        print(p.data)
