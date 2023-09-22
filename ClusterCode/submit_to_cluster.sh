#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=estimator_aucs
#SBATCH --output=%x-%j.out 
#SBATCH --error=%x-%j.err
#SBATCH --account=tipes
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --workdir=/home/andreasm/RedNoiseEstimatorComparison/ClusterCode/

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

module load anaconda
source activate env_andreasm

export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

srun -n 2 python run_part.py





