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
    # test_sentence1 = 'How can I be a good geologist?'
    # test_sentence2 = 'What should I do to be a great geologist?'
    # t1.append(Utilities.prepare_sequence(test_sentence1, word_to_idx))
    # t2.append(Utilities.prepare_sequence(test_sentence2, word_to_idx))
    csv_data = Utilities.read_csv('/NLPTool/ESIM/Data/train_test.csv')
    label = Utilities.build_tag_from_csv(csv_data, 'is_duplicate')
    for i in range(csv_data.shape[0]):
        sentence = str(csv_data.loc[[i], 'question1'].iat[0])
        t1.append(Utilities.prepare_sequence(sentence, word_to_idx))

    for i in range(csv_data.shape[0]):
        sentence = str(csv_data.loc[[i], 'question2'].iat[0])
        t2.append(Utilities.prepare_sequence(sentence, word_to_idx))
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
    correct = 0.0
    for i in range(len(data_loaderX)):
        s1, l1 = iter1.next()
        s2, l2 = iter2.next()
        p = model(s1, l1, s2, l2)
        tag, idx = torch.max(p, dim=-1)
        print(idx)
        if label[i] == idx.data:
            correct = correct + 1
    print(correct/len(t1))