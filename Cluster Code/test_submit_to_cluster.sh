#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=estimator_aucs
#SBATCH --output=%x-%j.out 
#SBATCH --error=%x-%j.err
#SBATCH --account=tipes
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --workdir=/home/andreasm/RedNoiseEstimatorComparison/Cluster Code/

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

module load anaconda
source activate env_andreasm

i=0
j=0
echo " - windowsize_counter  $i"
echo " - observation_length_counter $j"
echo " "
srun python run_part.py  $i $j





