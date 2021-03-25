#!/bin/bash

train=(5000 10000 20000 40000 60000 80000 100000)

corpus=$1
methods=$2

for i in {0..6}; do
    sbatch -C rhel7 -a 0-4 $HOME/preprocess/scripts/q_builder/start_q_builder_process.slurm $corpus ${train[$i]} $methods
done
