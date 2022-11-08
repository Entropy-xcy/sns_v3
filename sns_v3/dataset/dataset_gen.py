from sns_v3.dataset.random_dag_gen import *
from sns_v3.dataset.logic_dag_sim import *
from networkx.readwrite import json_graph
import json
import ray
import os


def bool_seq_to_str(values: List[bool]) -> str:
    return ''.join(['1' if v else '0' for v in values])


def str_to_bool_seq(s: str) -> List[bool]:
    return [True if x == '1' else False for x in s]


def str_io_samples_to_bool_seq(io_samples: List[Tuple[str, str]]) -> List[Tuple[List[bool], List[bool]]]:
    ret = []
    for i, o in io_samples:
        ret.append([str_to_bool_seq(i), str_to_bool_seq(o)])
    return ret

def gen_logic_dag_io_examples(dag: nx.DiGraph, sim_func) -> List[Tuple[str, str]]:
    input_nodes = [n for n in dag.nodes if dag.nodes[n]['op'] == 'in']

    # generate all possible input sequences
    ret = []
    for i in range(2 ** len(input_nodes)):
        input_values = [bool(int(x)) for x in bin(i)[2:].zfill(len(input_nodes))]
        output_values = sim_logic_dag(dag, input_values, sim_func)
        input_values_str = bool_seq_to_str(input_values)
        output_values_str = bool_seq_to_str(output_values)
        ret.append([input_values_str, output_values_str])
    return ret


def export_dag_io_examples(dag: nx.DiGraph, io_examples, filename: str):
    data = json_graph.adjacency_data(dag)
    data['io_examples'] = io_examples
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_dag_io_examples(filename: str) -> Tuple[nx.DiGraph, List[Tuple[str, str]]]:
    data = json.load(open(filename, "r"))
    io_examples = data['io_examples']
    del data['io_examples']
    dag = json_graph.adjacency_graph(data)
    return dag, io_examples


@ray.remote
def generate_save_dataset(outfile_name: str, num_nodes: int, num_edges: int, sim_func, probability_dict):
    dag = random_logic_dag_gen(num_nodes, num_edges, num_inputs=8, num_outputs=8, prob_dict=probability_dict)
    io_examples = gen_logic_dag_io_examples(dag, sim_func)
    export_dag_io_examples(dag, io_examples, outfile_name)
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--num_edges', type=int, default=100)
    parser.add_argument('--num_dags', type=int, default=10000)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    args.output_dir = args.output_dir or f'./dataset_{args.num_nodes}_{args.num_edges}'
    sim_func = {
        'and': and_func,
        'or': or_func,
        'not': not_func,
        'out': pass_func,
    }
    probability_dict = {
        "and": (0.33, 2),
        "or": (0.33, 2),
        "not": (1 - 0.33 - 0.33, 1),
    }
    ray.init()
    # create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    results = []
    for i in range(args.num_dags):
        outfile_name = os.path.join(args.output_dir, f'{i}.json')
        r = generate_save_dataset.remote(outfile_name, args.num_nodes, args.num_edges, sim_func, probability_dict)
        results.append(r)
    ray.get(results)
    print("Finished!")
