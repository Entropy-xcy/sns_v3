#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

from sns_v3.dataset.logic_dag_sim import simulate
from sns_v3.dataset.random_dag_gen import draw_logic_dag
from sns_v3.sequence.sequentialize import sequence_to_dag, untokenize_graph
from typing import List, AnyStr


def pre_process_gen(gen: List[AnyStr]):
    # filter out [UNK]
    gen = [x for x in gen if x != '[UNK]']
    # delete second occurrence of [START]
    if gen.count('[START]') > 1:
        gen = gen[1:]
    return gen


if __name__ == "__main__":
    generated_sequences = json.load(open("seq2seq_gen.json", "r"))
    success_count = 0
    total_count = 0
    for step, step_v in generated_sequences.items():
        if step == '0':
            continue
        for batch_idx, gen in step_v.items():
            # print(f"Step {step}, Batch {batch_idx}")
            for sample in gen:
                sample = pre_process_gen(sample)
                sample_set = set(sample)
                # print(sample)
                try:
                    dag = sequence_to_dag(sample)
                    dag = untokenize_graph(dag)
                    print(f"Step {step}, Batch {batch_idx}")
                    print(sample)
                    # draw_logic_dag(dag)
                    success = True
                except:
                    success = False
                if success:
                    success_count += 1
                    sim_output_values = simulate(dag, [True, False, True, False, True, False, True, False])
                    print(len(sim_output_values))
                    print(sim_output_values)
                total_count += 1
    print(f"Success rate: {success_count / total_count}")
