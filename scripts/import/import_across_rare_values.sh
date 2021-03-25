#!/bin/bash

corpus=$1
corpus_size=$(($2 / 2))
methods=$3

rare_values=($(python3 -c "import numpy as np; print([int(np.ceil(n)) for n in np.geomspace(1, $corpus_size, num=10)])" | tr -d ',[]'))

for val in "${rare_values[@]}"
do
    if [ "$methods" = "" ]; then
        bash $HOME/preprocess/scripts/import/import_training_spread.sh $corpus "r$val"
    else
        bash $HOME/preprocess/scripts/import/import_training_spread.sh $corpus "${methods},r${val}"
    fi
done
