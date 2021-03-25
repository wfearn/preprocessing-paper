#!/bin/bash

#SBATCH --ntasks=1 # number of processor cores (i.e. tasks)
#SBATCH --exclusive
#SBATCH --mem=128G   # total memory to use

# Compatibility variables for PBS. Delete if not needed.
export PBS_NODEFILE=`/fslapps/fslutils/generate_pbs_nodefile`
export PBS_JOBID=$SLURM_JOB_ID
export PBS_O_WORKDIR="$SLURM_SUBMIT_DIR"
export PBS_QUEUE=batch

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

corpus=$1
train_size=$2
model=$3
#Methods must be last in order for command line parsing to work correctly
methods=$4

bash ~/preprocess/scripts/analysis/analyze.sh $corpus $train_size $model $methods
