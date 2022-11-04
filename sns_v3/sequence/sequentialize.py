import networkx as nx
from typing import List, Dict, Any

from sns_v3.dataset.dataset_load_example import load_dataset_from_dir
from sns_v3.dataset.random_dag_gen import draw_logic_dag


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


# Lexicographical Topological Sort
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

    # Find starting nodes
    starting_nodes = [n for n in dag.nodes if len(list(dag.predecessors(n))) == 0]
    start_node = 999999
    dag.add_node(start_node, token='[START]')
    for n in starting_nodes:
        dag.add_edge(start_node, n)
    return dfs_visit(start_node)


# Lexicographical Topological Sort
def sequence_to_dag(sequence):
    g = nx.DiGraph()
    cursor_stack = []
    max_node = 0

    def get_new_node():
        nonlocal max_node
        max_node += 1
        return max_node

    def push_cursor():
        cursor_stack.append(max_node)

    def pop_cursor():
        return cursor_stack.pop()

    def get_cursor():
        return cursor_stack[-1]

    for s in sequence:
        if s == '[START]':
            cursor = get_new_node()
            g.add_node(cursor, token=s)
        elif s == '[END]':
            continue
        elif s.startswith('in'):
            pass
        elif s.startswith('out'):
            pass
        elif s == "and":
            pass
        elif s == "or":
            pass
        elif s == "not":
            pass
        else:
            raise ValueError(f'Unknown token {s}')
    return g


if __name__ == "__main__":
    ds = load_dataset_from_dir('dataset_10_10', 100)
    dag, io_examples = ds[2]
    draw_logic_dag(dag)
    tok_g = tokenize_graph(dag)
    seq = dag_to_sequence(tok_g)
    print(seq)
    # print(tok_g.nodes)
    exit()
