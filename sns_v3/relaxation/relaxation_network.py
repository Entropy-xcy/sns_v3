#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch

class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.lut_tpe = nn.Parameter(torch.rand(16, 1))

    def forward(self, x):
        a = x[:, 0]
        not_a = 1 - a
        b = x[:, 1]
        not_b = 1 - b

        print(a, not_a, b, not_b)

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
        for i in range(in_bits // 2):
            self.gates.append(Gate())

    def forward(self, x):
        assert x.shape[1] == self.in_bits
        x = x.view(-1, self.in_bits // 2, 2)
        x = torch.stack([gate(x[:, i]) for i, gate in enumerate(self.gates)], dim=1)
        x = x.view(-1, self.in_bits // 2)
        return x

class N2OneBinaryTree(pl.LightningModule):
    def __init__(self, in_bits: int):
        super().__init__()
        self.save_hyperparameters()
        self.tree = nn.ModuleList()
        current_width = in_bits // 2
        for i in range(in_bits):
            for j in range(current_width):
                self.tree.append(Gate())
            current_width = current_width // 2


    def forward(self, x):
        pass

if __name__ == "__main__":
    model = N2OneBinaryTree(4)
    x = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], dtype=torch.float)
    print(x.shape)
    out = model(x)
    print(out)

