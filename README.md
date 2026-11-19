# DeepDiFE

DeepDifE is the core model to classify differential expression of genes under specific stresses, based on cis-regulatory elements. In this repo you can find examples on how to train this model and perform hyperparameter tuning.


## Getting started

Use the `requirements.txt` file to install the necessary python packages using *pip* or *conda*.

### Training
In the `train_model.py`, the dataset is first parsed, one-hot-encoded and split into N equal chunks N (`splits`). Subsequently N models are trained using each a different chunk as hold-out test set. The input of these models is a forward and reverse complement one-hot-encoded version of a sequence of a predetermined shape (`shape`). 

The expected input file should be a CSV file containing the following columns:
- GeneId,
- gene_family
- Label
- Sequence


### Hyperparameter optimization
The python package *Optuna* was imported to optimize the hyperparameters using the Tree-structured Parzen Estimator algorithm. In the script `hyperparameter_optimisation.py` an example can be found where a couple of hyperparameters were tuned.



