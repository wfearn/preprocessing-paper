#!/bin/bash

corpus=$1
methods=$2

# This shouldn't be necessary
export PYTHONHASHSEED=$seed

echo $corpus
echo $methods
echo $seed

PYTHONPATH=~/preprocess python3 $HOME/preprocess/scripts/sample/sample.py -c $corpus -m $methods
