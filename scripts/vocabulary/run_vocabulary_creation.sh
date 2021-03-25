#!/bin/bash

corpus=$1
methods=$2
seed=$SLURM_ARRAY_TASK_ID

# Very necessary to assure reproducible hashing
export PYTHONHASHSEED=$seed

echo "corpus: $corpus"
echo "methods: $methods"
echo "seed: $seed"

PYTHONPATH=~/preprocess python3 $HOME/preprocess/scripts/vocabulary/create_vocabulary.py -c $corpus -s $seed -m $methods
