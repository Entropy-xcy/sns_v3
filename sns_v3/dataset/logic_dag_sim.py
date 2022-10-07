import networkx as nx
from typing import List, Dict, Any
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
    input_nodes = [n for n in g.nodes if g.nodes[n]['op'] == 'in']
    output_nodes = [n for n in g.nodes if g.nodes[n]['op'] == 'out']
    assert len(input_nodes) == len(input_values)

    # Step 1: assign input values to input nodes
    for i, n in enumerate(input_nodes):
        g.nodes[n]['sim_value'] = input_values[i]

    _sim(g, sim_func)


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
    dag = random_logic_dag_gen(50, 80, num_inputs=8, num_outputs=8, prob_dict=probability_dict)
    input_values = [True, False, True, False, True, False, True, False]
    output_values = sim_logic_dag(dag, input_values, sim_func)
    draw_logic_dag(dag)
