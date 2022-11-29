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

        lut_t = F.softmax(self.lut_tpe, dim=0)
        result = torch.matmul(lut, lut_t)
        return result

    def regularization_loss(self):
        lut_tpe = F.softmax(self.lut_tpe, dim=0)
        neg_lut_tpe = 1.0 - lut_tpe
        #  print(lut_tpe)
        #  print(neg_lut_tpe)
        sum_lst = []
        for i in range(16):
            prod_lst = [lut_tpe[i]]
            for j in range(16):
                if i != j:
                    #  prod *= neg_lut_tpe[j]
                    prod_lst.append(neg_lut_tpe[j])
            prod = torch.prod(torch.stack(prod_lst))
            #  sum += prod
            sum_lst.append(prod)
        sum_result = torch.sum(torch.stack(sum_lst))
        #  print(sum)
        neg_log_sum = -torch.log(sum_result)
        return neg_log_sum


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
    
    def regularization_loss(self):
        return sum([gate.regularization_loss() for gate in self.gates])


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

    def regularization_loss(self):
        return sum([l.regularization_loss() for l in self.tree])


class RelaxationNetwork(pl.LightningModule):

    def __init__(self, in_bits: int, out_bits: int, regularization_weight: float=1.0):
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

    def regularization_loss(self):
        return sum([tree.regularization_loss() for tree in self.trees])
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc_loss = F.binary_cross_entropy(y_hat, y)
        reg_loss = self.regularization_loss()
        loss = self.hparams.regularization_weight * reg_loss + acc_loss
        self.log('acc_loss', acc_loss, prog_bar=True)
        self.log('reg_loss', reg_loss, prog_bar=True)
        self.log('total_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    #  gate = Gate()
    #  x = torch.tensor([[0, 0]], dtype=torch.float32)
    #  y = torch.tensor([[0]], dtype=torch.float32)
    #  out = gate(x)
    #  reg = gate.regularization_loss()
    #  print(reg)
    #  loss = F.binary_cross_entropy(out, y) + reg
    #  loss.backward()
    #  print(loss)
#
    #  exit()
    torch.autograd.set_detect_anomaly(True)
    ds = RelaxationDataset('dataset_100_100', 100)
    dl = torch.utils.data.DataLoader(ds, batch_size=2)
    model = RelaxationNetwork(8, 8)
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, dl)

