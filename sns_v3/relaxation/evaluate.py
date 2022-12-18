#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('relaxation_results.json') as f:
        data = json.load(f)
    tensor_dict = {
        "acc_loss": [],
        "total_loss": [],
        "exact_loss": [],
        "reg_loss": [],
    }
    for k, _ in tensor_dict.items():
        lst = []
        for v in data:
            row = []
            for vv in v:
                row.append(vv[k])
            lst.append(row)
        tensor_dict[k] = lst
    for k, v in tensor_dict.items():
        tensor_dict[k] = torch.tensor(v)

    mean_epoch_loss = {}
    for k, v in tensor_dict.items():
        mean_epoch_loss[k] = torch.mean(v, dim=0)

    mean_epoch_loss["reg_loss"] = mean_epoch_loss["reg_loss"] / mean_epoch_loss["reg_loss"].max() / 2.0

    for k, v in mean_epoch_loss.items():
        # set minimum v to 0.0000001
        v[v < 0.0000001] = 0.0000001
        print(k, v)
        plt.plot(v * 100.0, label=k)
    plt.legend()
    plt.title("Relaxation Method Training Results")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (%)")
    plt.yscale("log")
    plt.savefig('relaxation_results.pdf')

