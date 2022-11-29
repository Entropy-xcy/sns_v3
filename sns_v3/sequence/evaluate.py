#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

import ray

from sns_v3.dataset.logic_dag_sim import simulate
from sns_v3.dataset.random_dag_gen import draw_logic_dag
from sns_v3.sequence.sequentialize import sequence_to_dag, untokenize_graph
from typing import List, AnyStr, Optional
from sns_v3.dataset.logic_dag_sim import evaluate
from tqdm import tqdm
import matplotlib.pyplot as plt


def pre_process_gen(gen: List[AnyStr]):
    # filter out [UNK]
    gen = [x for x in gen if x != '[UNK]']
    # delete second occurrence of [START]
    if gen.count('[START]') > 1:
        gen = gen[1:]
    return gen


def int_to_bool_list(num: int, align=8) -> List[bool]:
    ret = []
    for i in range(align):
        ret.append(bool(num & 1))
        num >>= 1
    ret.reverse()
    return ret


def table_to_io_examples(table: List[int]) -> List[List[bool]]:
    ret = []
    for i in range(len(table)):
        o = table[i]
        # i to bool list
        i_bin = int_to_bool_list(i)
        o_bin = int_to_bool_list(o)
        tup = (i_bin, o_bin)
        ret.append(tup)
    return ret


@ray.remote(num_cpus=1)
def evaluate_seq(gen: List[AnyStr], io: List[AnyStr]):
    io_examples = table_to_io_examples(io)
    sample = gen
    sample = pre_process_gen(sample)
    try:
        dag = sequence_to_dag(sample)
        dag = untokenize_graph(dag)
        # print(f"Step {step}, Batch {batch_idx}")
        # print(sample)
        success = True
    except:
        success = False
    if success:
        io_examples = table_to_io_examples(io)
        wrong_count_total, bit_wrong_count_total, accuracy_loss_total = evaluate(dag, io_examples)
        return wrong_count_total, bit_wrong_count_total, accuracy_loss_total
    else:
        return None


def evaluate_all():
    ray.init()
    generated_sequences = json.load(open("seq2seq.json", "r"))
    success_count = 0
    total_count = 0
    eval_result = []
    for step, step_v in generated_sequences.items():
        if step == '0':
            continue
        step_res = []
        for batch_idx, v in step_v.items():
            gen_res = []
            gens = v['gen']
            io = v['X'][0]
            for gen in gens:
                this = evaluate_seq.remote(gen, io)
                gen_res.append(this)
            step_res.append(gen_res)
        eval_result.append(step_res)
    for step in tqdm(range(len(eval_result))):
        for batch_idx in range(len(eval_result[step])):
            for gen_idx in range(len(eval_result[step][batch_idx])):
                eval_result[step][batch_idx][gen_idx] = ray.get(eval_result[step][batch_idx][gen_idx])
    print(eval_result)
    json.dump(eval_result, open("eval_result.json", "w"))


if __name__ == "__main__":
    #  evaluate_all()
    #  exit()
    # load json
    eval_result = json.load(open("eval_result.json", "r"))
    bit_wrong_count = []
    for step in range(len(eval_result)):
        sum = 0
        for batch in range(len(eval_result[step])):
            best_wrong_bits = 8 * 128
            best_wrong_count = 256
            for gen in range(len(eval_result[step][batch])):
                if eval_result[step][batch][gen] is not None:
                    wrong_count_total, bit_wrong_bit_count_total, accuracy_loss_total = eval_result[step][batch][gen]
                    if bit_wrong_bit_count_total < best_wrong_bits:
                        best_wrong_bits = bit_wrong_bit_count_total
                    if wrong_count_total < best_wrong_count:
                        best_wrong_count = wrong_count_total
            sum += best_wrong_bits
        bit_wrong_count.append(sum / len(eval_result[step]) / 2048 * 100.0)
    plt.xlabel("Training Step")
    plt.ylabel("%")
    plt.title("Bit Error Rate")
    plt.plot(bit_wrong_count)
    X_max = len(bit_wrong_count)
    plt.plot([0, X_max], [50, 50])
    plt.legend(["Bit Error", "Random Guess"])
    #  plt.show()
    plt.savefig("bit_error_rate.png")

