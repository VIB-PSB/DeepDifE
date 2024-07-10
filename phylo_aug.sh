#!/bin/bash
# quickly launch jupyter lab on a computing node
# qsub -pe serial 8 -l h_vmem=70G start_jupyterlab_cluster.sh 8895
# or
# sbatch -p all -c 1 --mem 70G start_jupyterlab_cluster.sh 8894

module load regquest/x86_64/1.0
module unload python/x86_64/3.8.0

python3 scripts/train_model_phylo_aug.py