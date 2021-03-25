#!/bin/bash

hashing_values=(60000 40000 20000 10000 8000 6000 4000 2000 1000 500)

corpus=$1
methods=$2

if [ "$methods" = "" ]; then
    for val in "${hashing_values[@]}"
    do
        bash $HOME/preprocess/scripts/import/import_training_spread.sh $corpus "h$val"
    done
else
    for val in "${hashing_values[@]}"
    do
        bash $HOME/preprocess/scripts/import/import_training_spread.sh $corpus "${methods},h${val}"
    done
fi
