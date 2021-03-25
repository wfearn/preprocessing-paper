#!/bin/bash

corpus=$1
offset=$2
methods=$3

array_id=$SLURM_ARRAY_TASK_ID
job_id=$SLURM_PROCID
job_num=$((($array_id * 30) + $job_id))

echo "Corpus: $corpus"
echo "Offset: $offset"
echo "Job Number: $job_num"
echo "Methods: $methods"

PYTHONPATH=~/preprocess python3 ~/preprocess/scripts/preprocess/preprocess_corpus.py -c $corpus -o $offset -r $job_num -m $methods
