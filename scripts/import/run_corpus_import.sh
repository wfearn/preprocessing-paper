#!/bin/bash

corpus=$1
train=$2

# Methods must be last
methods=$3
seed=$SLURM_ARRAY_TASK_ID

echo $corpus
echo $methods
echo $train
echo $seed

PYTHONPATH=~/preprocess python3 /fslhome/wfearn/preprocess/scripts/import/import_corpus.py -c $corpus -s $seed -tr $train -m $methods
