import torch
import Dataset
import Utilities
import pickle
import torch.utils.data as data


if __name__ == "__main__":
    t1 = []
    t2 = []
    word_to_idx = {}
    with open('ESIM_Dic', 'rb') as f:
        word_to_idx = pickle.load(f)
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
    model = torch.load('ESIMModel')
    for i in range(len(data_loaderX)):
        s1, l1 = iter1.next()
        s2, l2 = iter2.next()
        p = model(s1, l1, s2, l2)
        print(p.data)