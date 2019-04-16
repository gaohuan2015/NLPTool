import torch
import Model
import pickle
import torch.nn as nn
import Dataset
import Utilities
import torch.utils.data as data
import os
from tqdm import tqdm

if __name__ == "__main__":
    # read data from CSV
    word_to_idx = {}
    train_x = []
    train_y = []
    s1, s2, label = Utilities.read_from_json(
        '/NLPTool/ESIM/Data/test.json')
    Utilities.build_dic(s1, word_to_idx)
    Utilities.build_dic(s2, word_to_idx)
    for s in s1:
        train_x.append(Utilities.prepare_sequence(s, word_to_idx))
    for s in s2:
        train_y.append(Utilities.prepare_sequence(s, word_to_idx))
    # csv_data = Utilities.read_csv('/NLPTool/ESIM/Data/train_test.csv')
    # Utilities.build_dic_from_csv(csv_data, 'question1', word_to_idx)
    # Utilities.build_dic_from_csv(csv_data, 'question2', word_to_idx)
    # for i in range(csv_data.shape[0]):
    #     sentence = str(csv_data.loc[[i], 'question1'].iat[0])
    #     train_x.append(Utilities.prepare_sequence(sentence, word_to_idx))

    # for i in range(csv_data.shape[0]):
    #     sentence = str(csv_data.loc[[i], 'question2'].iat[0])
    #     train_y.append(Utilities.prepare_sequence(sentence, word_to_idx))
    # label = Utilities.build_tag_from_csv(csv_data, 'is_duplicate')
    # save dictionay
    with open('ESIM_Dic', 'wb') as f:
        pickle.dump(word_to_idx, f)
    # build data loader
    traindataX = Dataset.ESIMDataSet(train_x)
    data_loaderX = data.DataLoader(
        traindataX,
        batch_size=8,
        shuffle=False,
        collate_fn=Utilities.padd_sentence)

    traindataY = Dataset.ESIMDataSet(train_y)
    data_loaderY = data.DataLoader(
        traindataY,
        batch_size=8,
        shuffle=False,
        collate_fn=Utilities.padd_sentence)
    # build model
    model = Model.ESIM(len(word_to_idx) + 2, 100, 128, 3)
    criterion = nn.CrossEntropyLoss()
    optimize = torch.optim.Adam(model.parameters(), lr=0.0001)
    for i in tqdm(range(100)):
        running_loss = 0.0
        iter1 = data.dataloader._DataLoaderIter(data_loaderX)
        iter2 = data.dataloader._DataLoaderIter(data_loaderY)
        for i in range(len(data_loaderX)):
            optimize.zero_grad()
            s1, l1 = iter1.next()
            s2, l2 = iter2.next()
            target = torch.tensor(
                label[i*s1.shape[0]:i*s1.shape[0]+s1.shape[0]], dtype=torch.long)
            p = model(s1, l1, s2, l2)
            loss = criterion(p, target)
            loss.backward()
            optimize.step()
            running_loss += loss.item()
        print(running_loss)
    torch.save(model, 'ESIMModel')
    print('finished')
