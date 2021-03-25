#!/bin/bash

train=(5000 10000 20000 40000 60000 80000 100000)

corpus=$1
model=$2
methods=$3

if [ $model = "ankura" ]
then
    runtime=(01:30:00 03:00:00 06:00:00 12:00:00 24:00:00 48:00:00 72:00:00)
elif [ $model = "naive" ]
then
    runtime=(00:30:00 00:30:00 00:30:00 2:00:00 3:00:00 4:00:00 6:00:00)
elif [ $model = "svm" ]
then
    runtime=(00:30:00 00:30:00 00:30:00 2:00:00 3:00:00 4:00:00 6:00:00)
else # using knn
    runtime=(00:30:00 00:30:00 00:30:00 2:00:00 3:00:00 4:00:00 6:00:00)
fi


for i in {0..6}; do
    sbatch -C rhel7 -a 0-4 --time=${runtime[$i]} $HOME/preprocess/scripts/analysis/start_analysis_process.slurm $corpus ${train[$i]} $model $methods
done
