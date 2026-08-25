#!/bin/sh
#SBATCH --job-name=Concordant             # Job name
#SBATCH --partition=batch               # Partition (queue) name
#SBATCH --constraint=EPYC|Intel
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1             # Number of MPI ranks
#SBATCH --ntasks-per-node=1    # How many tasks on each node
#SBATCH --cpus-per-task=1     # Number of cores per MPI rank 
#SBATCH --mem=120GB        # Memory per processor
#SBATCH --gres=lscratch:800
#SBATCH --time=4:00:00
#SBATCH --output="%x.%j".out     # Standard output log
#SBATCH --error="%x.%j".err      # Standard error log

cd $SLURM_SUBMIT_DIR
export NSLOTS=1
export THREADS=1

set -eE
trap 'cleanup' EXIT

function cleanup(){
  echo "Exiting. Performing Cleanup"
  rm $PSI_SCRATCH -r
}
source ~/.bashrc
conda activate psi4_dlpno
export PSI_SCRATCH=/lscratch/$USER/tmp/$SLURM_JOB_ID
mkdir -p $PSI_SCRATCH
psi4 -n $NSLOTS -o output.dat

#ignored line -- do not remove
