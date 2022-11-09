from torch.utils.data import Dataset, DataLoader
import torch
from sns_v3.sequence.load_sequence_dataset import load_sequence_dataset

VOCAB = ["[START]", "[END]",
         "in0", "in1", "in2", "in3", "in4", "in5", "in6", "in7",
         "out0", "out1", "out2", "out3", "out4", "out5", "out6", "out7",
         "and", "or", "not"]


class Seq2SeqDataset(Dataset):
    def __init__(self, dataset_dir, max_len: int):
        X, y = load_sequence_dataset(dataset_dir, max_len)
        input_ids = []
        for x in X:
            seq = [i[1] for i in x]
            input_ids.append(seq)
        self.X = torch.tensor(input_ids, dtype=torch.long)
        self.y = y
        print(y)

    def __getitem__(self, idx):
        return self.X[idx]

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    ds = Seq2SeqDataset('dataset_100_100', 100)
    dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    X = next(iter(dl))
    print(X.shape)
