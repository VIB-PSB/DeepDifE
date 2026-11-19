#!/bin/bash
# quickly launch jupyter lab on a computing node
# qsub -pe serial 8 -l h_vmem=70G start_jupyterlab_cluster.sh 8895
# or
# sbatch -p all -c 1 --mem 70G start_jupyterlab_cluster.sh 8894

source activate deepdife2

python3 scripts/train_model.py --data /data/gent/vo/001/gvo00149/vsc43300/DeepDifE_github/data/example_data_set.csv --splits 2 --shape 600 --name test