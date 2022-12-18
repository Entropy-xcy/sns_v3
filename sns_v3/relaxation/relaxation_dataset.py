#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset, DataLoader
from sns_v3.sequence.load_sequence_dataset import load_sequence_dataset
from sns_v3.dataset.load_dataset import load_dataset_from_dir, load_dataset_from_dir_ray
import ray
from typing import List, Tuple
import torch
from tqdm import tqdm


class RelaxationDataset(Dataset):

    @staticmethod
    def io_examples_to_tensor(io_examples: List[List[str]]):

        def str_to_tensor(s: str):
            return torch.tensor([int(c) for c in s], dtype=torch.float32)

        X = []
        y = []
        for io_example in io_examples:
            X.append(str_to_tensor(io_example[0]))
            y.append(str_to_tensor(io_example[1]))
        return torch.stack(X), torch.stack(y)

    def __init__(self, ds):
        X = []
        y = []
        for _, io_examples in tqdm(ds):
            this_X, this_y = self.io_examples_to_tensor(io_examples)
            X.append(this_X)
            y.append(this_y)
        self.X = torch.stack(X)
        self.y = torch.stack(y)
        self.idx = 0

    def set_idx(self, idx):
        self.idx = idx

    def __len__(self):
        return self.X[self.idx].shape[0]

    def __getitem__(self, idx):
        return self.X[self.idx][idx], self.y[self.idx][idx]


if __name__ == '__main__':
    dataset = RelaxationDataset('dataset_100_100', 100)
    print(len(dataset))
