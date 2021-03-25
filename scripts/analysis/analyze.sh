#!/bin/bash

# value from 0 to array size
seed=$SLURM_ARRAY_TASK_ID
corpus=$1
train_size=$2
model=$3
#Methods must be last in order for command line parsing to work correctly
methods=$4

# This shouldn't be necessary
export PYTHONHASHSEED=$seed

echo $corpus
echo $train_size
echo $model
echo $methods
echo $seed

PYTHONPATH=~/preprocess python3 $HOME/preprocess/scripts/analysis/run_analysis.py -c $corpus -tr $train_size -mo $model -s $seed -m $methods
