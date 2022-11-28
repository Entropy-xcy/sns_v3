#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from math import ceil
from sns_v3.relaxation.relaxation_dataset import RelaxationDataset


class Gate(nn.Module):

    def __init__(self):
        super().__init__()
        self.lut_tpe = nn.Parameter(torch.rand(16, 1))

    def forward(self, x):
        a = x[:, 0]
        not_a = 1 - a
        b = x[:, 1]
        not_b = 1 - b

        not_a_not_b = not_a * not_b
        not_a_b = not_a * b
        a_not_b = a * not_b
        a_b = a * b
        #  and_val = a * b
        #  nand_val = 1 - and_val
        #  or_val = a + b - and_val
        #  nor_val = 1 - or_val
        #  xor_val = a + b - 2 * and_val
        #  xnor_val = 1 - xor_val

        lut0001 = a_b
        lut0010 = a_not_b
        lut0100 = not_a_b
        lut1000 = not_a_not_b
        lut0011 = lut0001 + lut0010
        lut0101 = lut0001 + lut0100
        lut0110 = lut0010 + lut0100
        lut1001 = lut0001 + lut1000
        lut1010 = lut0010 + lut1000
        lut1100 = lut0100 + lut1000
        lut0111 = lut0001 + lut0010 + lut0100
        lut1011 = lut0001 + lut0010 + lut1000
        lut1101 = lut0001 + lut0100 + lut1000
        lut1110 = lut0010 + lut0100 + lut1000
        lut0000 = torch.zeros_like(lut0001)
        lut1111 = torch.ones_like(lut0001)

        lut = torch.stack(
            [
                lut0000,
                lut0001,
                lut0010,
                lut0011,
                lut0100,
                lut0101,
                lut0110,
                lut0111,
                lut1000,
                lut1001,
                lut1010,
                lut1011,
                lut1100,
                lut1101,
                lut1110,
                lut1111,
            ],
            dim=1,
        )

        lut_tpe = F.softmax(self.lut_tpe, dim=0)
        x = torch.matmul(lut, lut_tpe)
        return x


class BinaryTreeLayer(nn.Module):

    def __init__(self, in_bits: int):
        super().__init__()
        self.in_bits = in_bits
        self.gates = nn.ModuleList()
        width = ceil(in_bits / 2)
        for i in range(width):
            self.gates.append(Gate())
        self.out_bits = width

    def forward(self, x):
        pad_width = self.in_bits
        if self.in_bits % 2 == 1:
            x = F.pad(x, (0, 1))
            pad_width += 1
        x = x.view(-1, pad_width, 1)
        x = torch.cat([x[:, ::2], x[:, 1::2]], dim=2)
        x = torch.cat([self.gates[i](x[:, i]) for i in range(x.shape[1])],
                      dim=1)
        return x


class N2OneBinaryTree(pl.LightningModule):

    def __init__(self, in_bits: int):
        super().__init__()
        self.save_hyperparameters()
        self.tree = nn.ModuleList()
        current_width = in_bits
        while current_width > 1:
            l = BinaryTreeLayer(current_width)
            self.tree.append(l)
            current_width = l.out_bits

    def forward(self, x):
        for l in self.tree:
            x = l(x)
        return x


class RelaxationNetwork(pl.LightningModule):

    def __init__(self, in_bits: int, out_bits: int):
        super().__init__()
        self.save_hyperparameters()
        self.trees = nn.ModuleList()
        for i in range(out_bits):
            self.trees.append(N2OneBinaryTree(in_bits))

    def forward(self, x):
        x = x.view(-1, self.hparams.in_bits)
        x = torch.cat([self.trees[i](x) for i in range(self.hparams.out_bits)],
                      dim=1)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    ds = RelaxationDataset('dataset_100_100', 100)
    dl = torch.utils.data.DataLoader(ds, batch_size=1)
    model = RelaxationNetwork(8, 8)
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, dl)

