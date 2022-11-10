# SNS v3: Synthesis Neural Symbolically
Why v3? Because v1 and v2 has been taken

## Installation
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

## Dataset Generation
Execute the following commands:
```
python -m sns_v3.dataset.dataset_gen --num_nodes 50 --num_edges 100 --num_dags 100000
```
The dataset will be generated at `./dataset_50_100`.

## Dataset Utilities
### Load Raw dataset with Graphs and IO examples
* Checkout `sns_v3/dataset/load_dataset.py:33`
* IO Examples are encoded as a tuple of string.
* Logic DAGs are encoded as networkx graphs. (See function `draw_logic_dag` at `sns_v3/dataset/random_dag_gen.py`)

### Evaluate an Generated Solution
* See function `evaluate` at `sns_v3/dataset/logic_dag_sim`.
* The return of this function composed of three parts:
  * Number of wrong IO examples
  * Number of wrong bits in all IO examples
  * Sum of integer value distance of all IO examples

## Sequence Model
### DAG Sequentialize
* Convert DAG to sequence: see function `dag_to_sequence` in `sns_v3/sequence/sequentialize.py`.
* Conver generated sequence back to DAG: see function `sequence_to_dag` in `sns_v3/sequence/sequentialize.py`.

### Token Encoding
* Vocab set and tokenizer definition: `LogicSeqTokenizer` in `sns_v3/sequence/seq2seq_dataset.py`.

### Running Seq2Seq Model
```
python -m sns_v3.sequence.seq2seq_model
```
