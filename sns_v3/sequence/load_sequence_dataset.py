from sns_v3.dataset.dataset_gen import str_io_samples_to_bool_seq
from sns_v3.dataset.load_dataset import load_dataset_from_dir, load_dataset_from_dir_ray
from typing import List, Tuple, Dict, Any

from sns_v3.sequence.sequentialize import tokenize_graph, dag_to_sequence
from tqdm import tqdm
import ray


@ray.remote(num_cpus=1)
def load_one_sequence(in_tup):
    dag, io_examples = in_tup
    io = str_io_samples_to_bool_seq(io_examples)
    tok_g = tokenize_graph(dag)
    seq = dag_to_sequence(tok_g)

    int_seq = []
    for i in io_examples:
        this_tup = (int(i[0], 2), int(i[1], 2))
        int_seq.append(this_tup)
    return (int_seq, seq, dag)


def load_sequence_dataset(fname: str, num: int):
    ds = load_dataset_from_dir_ray(fname, num)
    X = []
    y = []
    dags = []
    results = []
    for in_tup in ds:
        result = load_one_sequence.remote(in_tup)
        results.append(result)
    results = ray.get(results)
    for io, seq, dag in results:
        X.append(io)
        y.append(seq)
        dags.append(dag)
    return X, y, dags


if __name__ == "__main__":
    X, y, dags = load_sequence_dataset('dataset_10_10', 100)
    for i in X:
        outputs = [i[1] for i in i]
        unique_out = len(set(outputs))
        if unique_out != 1:
            print(unique_out)
