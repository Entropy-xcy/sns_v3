import networkx as nx
from typing import List, Dict, Any, Tuple
from sns_v3.dataset.random_dag_gen import *


def _sim(g: nx.DiGraph, sim_func: Dict[str, Any]):
    # simulated nodes
    visited_nodes = [n for n in g.nodes if 'sim_value' in dict(g.nodes[n])]
    # find a node with all parents simulated
    node_to_sim = None
    for n in set(g.nodes) - set(visited_nodes):
        parents = list(g.predecessors(n))
        if set(parents) <= set(visited_nodes):
            node_to_sim = n
            break
    if node_to_sim is not None:
        # simulate the node
        parents = list(g.predecessors(node_to_sim))
        values = [g.nodes[p]['sim_value'] for p in parents]
        g.nodes[node_to_sim]['sim_value'] = sim_func[g.nodes[node_to_sim]['op']](values)
        # recursively simulate the rest
        _sim(g, sim_func)
    else:
        # all nodes simulated
        return


def sim_logic_dag(g: nx.DiGraph, input_values: List[bool], sim_func: Dict[str, Any]) -> List[bool]:
    # g = nx.DiGraph(g)
    # Add sort here
    input_nodes = [n for n in g.nodes if g.nodes[n]['op'] == 'in']
    # sort input nodes with their idx
    input_nodes = sorted(input_nodes, key=lambda n: g.nodes[n]['idx'])
    output_nodes = [n for n in g.nodes if g.nodes[n]['op'] == 'out']
    # sort output nodes with their idx
    output_nodes = sorted(output_nodes, key=lambda n: g.nodes[n]['idx'])

    assert len(input_nodes) == len(input_values), 'input_nodes: {}, input_values: {}'.format(input_nodes, input_values)

    # Step 1: assign input values to input nodes
    for i, n in enumerate(input_nodes):
        g.nodes[n]['sim_value'] = input_values[i]

    _sim(g, sim_func)

    # Step 2: get output values
    for n in output_nodes:
        assert 'sim_value' in g.nodes[n]
    sim_output_values = [g.nodes[n]['sim_value'] for n in output_nodes]
    return sim_output_values


def and_func(values: List[bool]) -> bool:
    assert len(values) == 2
    return all(values)


def or_func(values: List[bool]) -> bool:
    assert len(values) == 2
    return any(values)


def not_func(values: List[bool]) -> bool:
    assert len(values) == 1
    return not values[0]


def pass_func(values: List[bool]) -> bool:
    assert len(values) == 1
    return values[0]


def bit_wrong_count(a: List[bool], b: List[bool]) -> int:
    assert len(a) == len(b)
    wrong_count = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            wrong_count += 1
    return wrong_count


def accuracy_loss(a: List[bool], b: List[bool]) -> int:
    loss = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            loss += i ** 2
    return loss


def simulate(g: nx.DiGraph, input_values: List[bool]) -> List[bool]:
    sim_func = {
        'and': and_func,
        'or': or_func,
        'not': not_func,
        'out': pass_func,
    }
    return sim_logic_dag(g, input_values, sim_func)


def evaluate(g: nx.DiGraph, io_examples: List[List[bool]]) -> Tuple[int, int, int]:
    wrong_count_total = 0
    bit_wrong_count_total = 0
    accuracy_loss_total = 0
    for input_values, output_values in io_examples:
        sim_output_values = simulate(g, input_values)
        if sim_output_values != output_values:
            wrong_count_total += 1
        bit_wrong_count_total += bit_wrong_count(sim_output_values, output_values)
        accuracy_loss_total += accuracy_loss(sim_output_values, output_values)
    return wrong_count_total, bit_wrong_count_total, accuracy_loss_total


if __name__ == "__main__":
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
    dag = random_logic_dag_gen(10, 5, num_inputs=8, num_outputs=8, prob_dict=probability_dict)
    input_values = [True, False, True, False, True, False, True, False]
    output_values = sim_logic_dag(dag, input_values, sim_func)
    draw_logic_dag(dag)
