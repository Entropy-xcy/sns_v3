#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import torch
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(5.5, 5))

def plot_subfig(ax, data_name, legend=False, title=None, xlabel=None):
    with open(data_name) as f:
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
        ax.plot(v * 100.0, label=k)
    if legend:
        ax.legend()
    if title is not None:
        ax.set_title(title)
    # plt.title("Relaxation Method Training Results")
    if xlabel is not None:
        ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (%)")
    ax.set_yscale("log")
    # plt.savefig('relaxation_results.pdf')


if __name__ == "__main__":
    # subplot 3 x1
    fig, ax = plt.subplots(3, 1)
    print(ax)
    plot_subfig(ax[0], 'relaxation_results_dataset_10_10.json', title="$M=10, N=10$")
    plot_subfig(ax[1], 'relaxation_results_dataset_100_100.json', title="$M=50, N=100$")
    plot_subfig(ax[2], 'relaxation_results_dataset_50_100.json', legend=True, title="$M=100, N=100$", xlabel="Epoch")
    # plt.show()
    fig.tight_layout()
    plt.savefig('relaxation_results.pdf')

