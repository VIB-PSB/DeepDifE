# DeepDiFE

## Getting started
This repository will include several notebooks which can be used as a starting point to experiment with the DeepDiFE analysis toolkit.
In order to run these notebooks on the PSB cluster the "start_jupyterlab_cluster.sh" should be submitted in the following way
```
sbatch -p all -c 1 --mem <MEMORY>G start_jupyterlab_cluster.sh <PORT>
```

## Notebooks
### logo_tutorial
In this tutorial you start from a trained model and perform DeepExplainer on a number of samples to render logo figures with the possibility of also generating an in silico mutagenesis scatterplot.