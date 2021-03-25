#!/bin/bash

# value from 0 to array size
seed=$SLURM_ARRAY_TASK_ID
corpus=$1
train_size=$2
#Methods must be last in order for command line parsing to work correctly
methods=$3

# This shouldn't be necessary
export PYTHONHASHSEED=$seed

echo $corpus
echo $train_size
echo $methods
echo $seed

PYTHONPATH=~/preprocess python3 $HOME/preprocess/scripts/q_builder/build_q.py -c $corpus -tr $train_size -s $seed -m $methods
