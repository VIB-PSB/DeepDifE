# DeepDifE

DeepDifE is a deep learning model that predicts whether a gene is differentially expressed under a specific stress condition, based on the cis-regulatory elements in its sequence. This repository contains examples of how to train the model and how to tune its hyperparameters.

## Getting started

### Installation

Clone the repository and install the dependencies listed in `requirements.txt`:

```bash
git clone https://github.com/VIB-PSB/DeepDifE
cd DeepDifE
pip install -r requirements.txt
```

Or, in a conda environment:

```bash
conda create -n deepdife python=3.10
conda activate deepdife
pip install -r requirements.txt
```

### Input data

The training data is a CSV file with the following columns:

| Column        | Description |
|---------------|-------------|
| `GeneId`      | Unique gene identifier |
| `gene_family` | Gene family the gene belongs to |
| `Label`       | Differential expression class. 1 = differentially expressed, 0 = not |
| `Sequence`    | Promoter sequence of the gene. All sequences must have the same length. |

### Training

`train_model.py` trains the model using k-fold cross-validation:

1. The dataset is parsed, one-hot encoded and split into N equally sized folds (set with the `splits` parameter).
2. N models are trained, each one holding out a different fold as its test set.
3. Each model takes two inputs: the one-hot-encoded sequence and its reverse complement, both with a fixed shape set by the `shape` parameter.

```bash
e.g.
python scripts/train_model.py --data data/example_data_set.csv --splits 5 --shape 600 --name ABADL
```

### Hyperparameter optimization

Hyperparameters are tuned with [Optuna](https://optuna.org/) using the Tree-structured Parzen Estimator (TPE) sampler. `hyperparameter_optimisation.py` shows an example that tunes a selection of hyperparameters:

```bash
python scripts/hyperparameter_optimisation.py --data data/example_data_set.csv --trials 50 --splits 5 --shape 600
```