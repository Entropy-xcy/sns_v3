from sns_v3.dataset.dataset_gen import *
from sns_v3.dataset.random_dag_gen import draw_logic_dag


def load_dataset_from_dir(dir_path: str, num_samples: int) -> List[Tuple[nx.DiGraph, List[Any]]]:
    # load dataset from dir
    dataset = []
    for i in range(num_samples):
        json_name = f'{i}.json'
        dag, io_examples = load_dag_io_examples(os.path.join(dir_path, json_name))
        dataset.append((dag, io_examples))
    return dataset


if __name__ == "__main__":
    ds = load_dataset_from_dir('dataset', 100)
    dag = ds[0][0]
    io_examples = ds[0][1]
    for i, o in io_examples:
        print(i, "->", o)
    for n in dag.nodes:
        print(n, dag.nodes[n])
