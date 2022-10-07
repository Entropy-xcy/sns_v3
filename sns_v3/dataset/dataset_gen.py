from sns_v3.dataset.random_dag_gen import *
from sns_v3.dataset.logic_dag_sim import *


def bool_seq_to_str(values: List[bool]) -> str:
    return ''.join(['1' if v else '0' for v in values])


def gen_logic_dag_io_examples(dag: nx.DiGraph, sim_func):
    input_nodes = [n for n in dag.nodes if dag.nodes[n]['op'] == 'in']

    for i in range(2 ** len(input_nodes)):
        input_values = [bool(int(x)) for x in bin(i)[2:].zfill(len(input_nodes))]
        output_values = sim_logic_dag(dag, input_values, sim_func)
        input_values_str = bool_seq_to_str(input_values)
        output_values_str = bool_seq_to_str(output_values)
        print(f'{input_values_str} -> {output_values_str}')


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
    gen_logic_dag_io_examples(dag, sim_func)
