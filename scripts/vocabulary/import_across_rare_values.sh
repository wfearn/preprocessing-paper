#!/bin/bash

corpus=$1
corpus_size=$(($2 / 2))
methods=$3

rare_values=($(python3 -c "import numpy as np; print([int(np.ceil(n)) for n in np.geomspace(1, $corpus_size, num=10)])" | tr -d ',[]'))

for val in "${rare_values[@]}"
do
    if [ "$methods" = "" ]; then
        sbatch -C rhel7 -a 0-5 $HOME/preprocess/scripts/vocabulary/start_vocabulary_creation.slurm $corpus "r$val"
    else
        sbatch -C rhel7 -a 0-5 $HOME/preprocess/scripts/vocabulary/start_vocabulary_creation.slurm $corpus "${methods},r${val}"
    fi
done
