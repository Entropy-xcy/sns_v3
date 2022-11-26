import networkx as nx
from typing import List, Dict, Any
import copy
from sns_v3.dataset.dataset_gen import str_io_samples_to_bool_seq
from sns_v3.dataset.load_dataset import load_dataset_from_dir
from sns_v3.dataset.logic_dag_sim import evaluate, simulate
from sns_v3.dataset.random_dag_gen import draw_logic_dag
from tqdm import tqdm
START_NODE = 999999


def tokenize_graph(g: nx.DiGraph) -> nx.DiGraph:
    g = nx.DiGraph(g)
    for n in g.nodes:
        op = g.nodes[n]['op']
        if "idx" in g.nodes[n]:
            idx = g.nodes[n]['idx']
            token = f'{op}{idx}'
            del g.nodes[n]['idx']
        else:
            token = op
        g.nodes[n]['token'] = token
        del g.nodes[n]['op']
    return g


def untokenize_graph(g: nx.DiGraph) -> nx.DiGraph:
    g = nx.DiGraph(g)
    in_nodes = {}
    for n in list(g.nodes):
        token = g.nodes[n]['token']
        if token == "and" or token == "or" or token == "not":
            g.nodes[n]['op'] = token
        elif token.startswith('in'):
            if token not in in_nodes:
                in_nodes[token] = n
                g.nodes[n]['op'] = 'in'
                g.nodes[n]['idx'] = int(token[2:])
            else:
                for succ in list(g.successors(n)):
                    g.add_edge(in_nodes[token], succ)
                g.remove_node(n)
        elif token.startswith('out'):
            g.nodes[n]['op'] = 'out'
            g.nodes[n]['idx'] = int(token[3:])
        elif token == "[START]":
            g.remove_node(n)
    return g


def dag_to_sequence(dag: nx.DiGraph) -> List[str]:
    def dfs_visit(n: int) -> List[str]:
        # get and sort all successors
        successors = list(dag.successors(n))
        successors.sort()
        seq = [dag.nodes[n]['token']]
        for s in successors:
            seq += dfs_visit(s)
        seq.append('[END]')
        return seq

    dag = nx.DiGraph(dag)
    dag = dag.reverse()

    # Find starting nodes
    starting_nodes = [n for n in dag.nodes if len(list(dag.predecessors(n))) == 0]

    dag.add_node(START_NODE, token='[START]')
    for n in starting_nodes:
        dag.add_edge(START_NODE, n)
    return dfs_visit(START_NODE)


def sequence_to_dag(sequence):
    g = nx.DiGraph()
    cursor_stack = []
    max_node = 0

    def get_new_node():
        nonlocal max_node
        max_node += 1
        return max_node

    def push_cursor(node):
        cursor_stack.append(node)

    def pop_cursor():
        return cursor_stack.pop()

    def get_cursor():
        return cursor_stack[-1]

    for s in sequence:
        if s == '[START]':
            n = get_new_node()
            g.add_node(n, token=s)
            push_cursor(n)
        elif s == '[END]':
            pop_cursor()
        elif s == "and" or s == "or" or s == "not" or s.startswith('in') or s.startswith('out'):
            n = get_new_node()
            g.add_node(n, token=s)
            g.add_edge(get_cursor(), max_node)
            push_cursor(n)
        else:
            raise ValueError(f'Unknown token {s}')
    g = g.reverse()
    return g


def evaluate_and_print(dag: nx.DiGraph, bool_seq: List[List[bool]]):
    seq = copy.deepcopy(bool_seq)
    wrong_count_total, bit_wrong_count_total, accuracy_loss_total = evaluate(dag, seq)
    # print(f'wrong_count_total: {wrong_count_total}')
    # print(f'bit_wrong_count_total: {bit_wrong_count_total}')
    # print(f'accuracy_loss_total: {accuracy_loss_total}')
    assert wrong_count_total == 0
    assert bit_wrong_count_total == 0
    assert accuracy_loss_total == 0


def fuzzing_seq_dag_io(dag, io_examples):
    io_examples = str_io_samples_to_bool_seq(io_examples)
    tok_g = tokenize_graph(dag)
    seq = dag_to_sequence(tok_g)
    rec_g = sequence_to_dag(seq)
    retok_g = untokenize_graph(rec_g)
    evaluate_and_print(dag, io_examples)
    evaluate_and_print(retok_g, io_examples)


if __name__ == "__main__":
    ds = load_dataset_from_dir('dataset_100_100', 100)
    for dag, io_examples in tqdm(ds):
        fuzzing_seq_dag_io(dag, io_examples)

