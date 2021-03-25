#!/bin/bash

hashing_values=(60000 40000 20000 10000 8000 6000 4000 2000 1000 500)

corpus=$1
methods=$2

if [ "$methods" = "" ]; then
    for val in "${hashing_values[@]}"
    do
        sbatch -C rhel7 -a 0-5 $HOME/preprocess/scripts/vocabulary/start_vocabulary_creation.slurm $corpus "h$val"
    done
else
    for val in "${hashing_values[@]}"
    do
        sbatch -C rhel7 -a 0-5 $HOME/preprocess/scripts/vocabulary/start_vocabulary_creation.slurm $corpus "${methods},h${val}"
    done
fi
