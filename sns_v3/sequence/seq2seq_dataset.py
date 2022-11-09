from torch.utils.data import Dataset, DataLoader
import torch
from sns_v3.sequence.load_sequence_dataset import load_sequence_dataset

VOCAB = ["[UNK]", "[START]",
         "in0", "in1", "in2", "in3", "in4", "in5", "in6", "in7",
         "out0", "out1", "out2", "out3", "out4", "out5", "out6", "out7",
         "and", "or", "not", "[END]"]


class LogicSeqTokenizer:
    def __init__(self, vocab_set=VOCAB, pad_len=512):
        self.vocab = vocab_set
        self.vocab_size = len(vocab_set)
        self.v2i = {word: idx for idx, word in enumerate(vocab_set)}
        self.pad_len = pad_len

    def encode(self, seq):
        raw_seq = [self.v2i[word] for word in seq]
        # pad to max length
        pad_seq = raw_seq + [0] * (self.pad_len - len(raw_seq))
        attn_mask = [1] * len(raw_seq) + [0] * (self.pad_len - len(raw_seq))
        pad_seq = torch.tensor(pad_seq)
        attn_mask = torch.tensor(attn_mask)
        return pad_seq[0:self.pad_len], attn_mask[0:self.pad_len]

    def decode(self, seq):
        return [self.vocab[idx] for idx in seq]

    def __call__(self, seq):
        return self.encode(seq)


class Seq2SeqDataset(Dataset):
    def __init__(self, dataset_dir, max_len: int):
        X, y = load_sequence_dataset(dataset_dir, max_len)
        input_ids = []
        for x in X:
            seq = [i[1] for i in x]
            input_ids.append(seq)
        self.X = torch.tensor(input_ids, dtype=torch.long)
        self.X_mask = torch.ones_like(self.X)

        tokenizer = LogicSeqTokenizer()
        self.tokenizer = tokenizer
        enc_seq = []
        enc_mask = []
        for seq in y:
            s, mask = tokenizer(seq)
            enc_seq.append(s)
            enc_mask.append(mask)
        self.y = torch.stack(enc_seq)
        self.mask = torch.stack(enc_mask)

    def __getitem__(self, idx):
        return self.X[idx], self.X_mask[idx], self.y[idx], self.mask[idx]

    def __len__(self):
        return len(self.X)

    def get_tokenizer(self):
        return self.tokenizer


if __name__ == '__main__':
    ds = Seq2SeqDataset('dataset_100_100', 100)
    dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    X, X_mask, y, y_mask = next(iter(dl))
    print(X.shape)
    print(X_mask.shape)
    print(y.shape)
    print(y_mask.shape)
