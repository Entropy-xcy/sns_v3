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
    ds = load_dataset_from_dir('dataset_10_10', 100)
    print("Verifying Correctness of Dataset...")
    for dag, io_samples in ds:
        bool_seq = str_io_samples_to_bool_seq(io_samples)
        wrong_count_total, bit_wrong_count_total, accuracy_loss_total = evaluate(dag, bool_seq)
        assert wrong_count_total == 0
        assert bit_wrong_count_total == 0
        assert accuracy_loss_total == 0

