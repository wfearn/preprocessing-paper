#!/bin/bash

#SBATCH --time=01:00:00   # walltime
#SBATCH --ntasks=30 # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=10G   # total memory to use

# Compatibility variables for PBS. Delete if not needed.
export PBS_NODEFILE=`/fslapps/fslutils/generate_pbs_nodefile`
export PBS_JOBID=$SLURM_JOB_ID
export PBS_O_WORKDIR="$SLURM_SUBMIT_DIR"
export PBS_QUEUE=batch

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

numjobs=1001
numsubjobs=30
corpus=$1
methods=$2

if [ $corpus = "amazon" ] || [ $corpus = "reddit" ] || [ $corpus = "testamazon" ] || [ $corpus = "testreddit" ]
then
    ext="json"
else
    ext="txt"
fi

filename="$HOME/.preprocess/corpora/$corpus/$corpus.$ext.gz"
lines=$(gunzip -c $filename | wc -l | tr -dc '0-9')
linesperjob=$((($lines + ($numjobs - 1)) / $numjobs)) # we want to round up to overshoot
linespersubjob=$((($linesperjob + ($numsubjobs - 1)) / $numsubjobs))

srun -o out/job%a-%2t.out ~/preprocess/scripts/preprocess/run_corpus_preprocess.sh $corpus $linespersubjob $methods
