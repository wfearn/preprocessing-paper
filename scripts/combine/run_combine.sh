#!/bin/bash

corpus=$1
PYTHONPATH=~/preprocess python3 ~/preprocess/preprocess/combiner.py -c $corpus
