#!/bin/bash

hashing_values=(60000 40000 20000 10000 8000 6000 4000 2000 1000 500)

corpus=$1
model=$2
methods=$3

if [ "$methods" = "" ]; then
    for val in "${hashing_values[@]}"
    do
        bash $HOME/preprocess/scripts/analysis/perform_training_spread.sh $corpus $model "h$val"
    done
else
    for val in "${hashing_values[@]}"
    do
        bash $HOME/preprocess/scripts/analysis/perform_training_spread.sh $corpus $model "${methods},h${val}"
    done
fi
