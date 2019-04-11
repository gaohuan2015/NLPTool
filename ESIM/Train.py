import torch
import Model
import torch.nn as nn
import Dataset
import Utilities
import torch.utils.data as data
from tqdm import tqdm

if __name__ == "__main__":
    # read data from CSV
    word_to_idx = {}
    train_x = []
    train_y = []
    csv_data = Utilities.read_csv('/NLPTool/ESIM/data/train_test.csv')
    Utilities.build_dic_from_csv(csv_data, 'question1', word_to_idx)
    Utilities.build_dic_from_csv(csv_data, 'question2', word_to_idx)
    for i in range(csv_data.shape[0]):
        sentence = str(csv_data.loc[[i], 'question1'].iat[0])
        train_x.append(Utilities.prepare_sequence(sentence, word_to_idx))

    for i in range(csv_data.shape[0]):
        sentence = str(csv_data.loc[[i], 'question2'].iat[0])
        train_y.append(Utilities.prepare_sequence(sentence, word_to_idx))

    label = Utilities.build_tag_from_csv(csv_data, 'is_duplicate')
    # build data loader
    traindataX = Dataset.ESIMDataSet(train_x)
    data_loaderX = data.DataLoader(
        traindataX,
        batch_size=64,
        shuffle=False,
        collate_fn=Utilities.padd_sentence)

    traindataY = Dataset.ESIMDataSet(train_y)
    data_loaderY = data.DataLoader(
        train_y,
        batch_size=64,
        shuffle=False,
        collate_fn=Utilities.padd_sentence)
    # build model
    model = Model.ESIM(len(word_to_idx) + 2, 1, 2, 1)
    criterion = nn.CrossEntropyLoss()
    optimize = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in tqdm(range(100)):
        iter1 = data.dataloader._DataLoaderIter(data_loaderX)
        iter2 = data.dataloader._DataLoaderIter(data_loaderY)
        for i in range(len(data_loaderX)):
            s1, l1 = iter1.next()
            s2, l2 = iter2.next()
            target = torch.tensor(
                label[i*s1.shape[0]:i*s1.shape[0]+s1.shape[0]], dtype=torch.long)
            model.zero_grad()
            p = model(s1, l1, s2, l2)
            loss = criterion(p, target)
            loss.backward()
            optimize.step()
    print(loss.data)

    t1 = []
    t2 = []
    test_tag = [0, 1]
    test_sentence1 = 'war III?'
    test_sentence2 = 'World War 3?'
    t1.append(Utilities.prepare_sequence(test_sentence1, word_to_idx))
    t2.append(Utilities.prepare_sequence(test_sentence2, word_to_idx))
    # build data loader
    testdataX = Dataset.ESIMDataSet(t1)
    data_loaderX = data.DataLoader(
        testdataX,
        batch_size=1,
        shuffle=False,
        collate_fn=Utilities.padd_sentence)

    testdataY = Dataset.ESIMDataSet(t2)
    data_loaderY = data.DataLoader(
        testdataY,
        batch_size=1,
        shuffle=False,
        collate_fn=Utilities.padd_sentence)
    iter1 = data.dataloader._DataLoaderIter(data_loaderX)
    iter2 = data.dataloader._DataLoaderIter(data_loaderY)
    for i in range(len(data_loaderX)):
        s1, l1 = iter1.next()
        s2, l2 = iter2.next()
        p = model(s1, l1, s2, l2)
        print(p.data)