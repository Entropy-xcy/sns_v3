import random
from typing import Dict, Tuple

import networkx as nx
from pyvis.network import Network


def find_io_nodes(g):
    # find input with no incoming edges
    input_nodes = [n for n in g.nodes() if g.in_degree(n) == 0 and g.out_degree(n) > 0]

    # find output with no outgoing edges
    output_nodes = [n for n in g.nodes() if g.out_degree(n) == 0 and g.in_degree(n) > 0]
    return input_nodes, output_nodes


def find_remove_invalid_nodes(g):
    # find invalid node with more than 2 incoming edges
    invalid_node = None
    for n in g.nodes:
        if g.in_degree(n) > 2:
            invalid_node = n
            break

    if invalid_node is not None:
        g.remove_node(invalid_node)
        return find_remove_invalid_nodes(g)
    else:
        return g


def random_dag(nodes, edges):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    G = nx.DiGraph()
    for i in range(nodes):
        G.add_node(i)
    while edges > 0:
        a = random.randint(0, nodes - 1)
        b = a
        while b == a:
            b = random.randint(0, nodes - 1)
        G.add_edge(a, b)
        if nx.is_directed_acyclic_graph(G):
            edges -= 1
        else:
            # we closed a loop!
            G.remove_edge(a, b)
    return G


def sample_gate(in_degree: int, prob_dict: Dict[str, Tuple[float, int]]):
    """Sample a gate type based on the probability distribution."""
    # select in degree matches
    prob_dict = {k: v for k, v in prob_dict.items() if v[1] == in_degree}
    # if prob_dict is empty, return None
    if len(prob_dict) == 0:
        return None
    # sample with weights
    weights = [v[0] for k, v in prob_dict.items()]
    gate_type = random.choices(list(prob_dict.keys()), weights=weights, k=1)[0]
    return gate_type


def assign_logic_type(g, prob_dict: Dict[str, Tuple[float, int]], ignored_nodes: list):
    for n in set(g.nodes) - set(ignored_nodes):
        in_degree = g.in_degree(n)
        gate_type = sample_gate(in_degree, prob_dict)
        if gate_type is None:
            g.remove_node(n)
        else:
            g.nodes[n]['op'] = gate_type
    return g


def random_logic_dag_gen(n: int, m: int, num_inputs: int, num_outputs: int, prob_dict: Dict[str, Tuple[float, int]]):
    # Step 1: generate a random DAG
    g = random_dag(n, m)

    # Step 2: Remove invalid nodes
    g = find_remove_invalid_nodes(g)

    # Step 3: Find input and output nodes
    input_nodes_dag, output_nodes_dag = find_io_nodes(g)
    input_nodes = []
    output_nodes = []
    for i in range(num_inputs):
        input_node = n + i
        g.add_node(input_node)
        input_nodes.append(input_node)
        g.nodes[input_node]['op'] = 'in'
        g.nodes[input_node]['idx'] = i

    for i in range(num_outputs):
        output_node = n + num_inputs + i
        g.add_node(output_node)
        output_nodes.append(output_node)
        g.nodes[output_node]['op'] = 'out'
        g.nodes[output_node]['idx'] = i

    if len(input_nodes_dag) == 0 or len(output_nodes_dag) == 0:
        # rerun
        return random_logic_dag_gen(n, m, num_inputs, num_outputs, prob_dict)

    # randomly assign input nodes to input_nodes_dag
    # find sum of probabilities of 2 in degree gates
    p_2_input = sum([v[0] for k, v in prob_dict.items() if v[1] == 2])
    for n in input_nodes_dag:
        if random.random() < p_2_input:
            # randomly Connect to two input_nodes
            input_node_1 = random.choice(input_nodes)
            input_node_2 = random.choice(input_nodes)
            g.add_edge(input_node_1, n)
            g.add_edge(input_node_2, n)
        else:
            # randomly Connect to one input_node
            input_node = random.choice(input_nodes)
            g.add_edge(input_node, n)

    # randomly assign output nodes to output_nodes_dag
    for n in output_nodes:
        output_node_dag = random.choice(output_nodes_dag)
        g.add_edge(output_node_dag, n)

    # assign logic type to each node
    g = assign_logic_type(g, prob_dict, input_nodes + output_nodes)

    return g


def draw_logic_dag(dag: nx.DiGraph, fname='dag.html'):
    for n in dag.nodes:
        if 'op' not in dag.nodes[n]:
            dag.nodes[n]['op'] = 'unknown'
        # if input
        if dag.nodes[n]['op'] == 'in':
            dag.nodes[n]['color'] = 'red'
            dag.nodes[n]['label'] = 'in' + str(dag.nodes[n]['idx']) + "\n" + str(n)
        # if output
        elif dag.nodes[n]['op'] == 'out':
            dag.nodes[n]['color'] = 'green'
            dag.nodes[n]['label'] = 'out' + str(dag.nodes[n]['idx']) + "\n" + str(n)
        else:
            dag.nodes[n]['label'] = dag.nodes[n]['op'] + "\n" + str(n)
        if 'sim_value' in dag.nodes[n]:
            # change outgoing edge color
            if dag.nodes[n]['sim_value'] == True:
                color = "red"
            else:
                color = "black"
            for e in dag.out_edges(n):
                dag.edges[e]['color'] = color
    net = Network(directed=True)
    net.from_nx(dag)
    net.show(fname)


if __name__ == "__main__":
    probability_dict = {
        "and": (0.33, 2),
        "or": (0.33, 2),
        "not": (1 - 0.33 - 0.33, 1),
    }
    dag = random_logic_dag_gen(50, 80, num_inputs=8, num_outputs=8, prob_dict=probability_dict)
    draw_logic_dag(dag)
