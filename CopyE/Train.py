import Preprocess as pre
import SentencesAndRelation as sar
import torch.utils.data as d

if __name__ == "__main__":
    data = pre.load_data_from_json('/CopyE/Data/train.json')
    training_data = pre.buildSentenceFromJson(data)
    training_loader = d.DataLoader(
        training_data,
        batch_size=2,
        shuffle=False,
        collate_fn=pre.pad_sentence)
    for batch_sentence, batch_label, batch_length in training_loader:
        print(batch_sentence)
