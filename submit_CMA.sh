#!/bin/bash
#SBATCH --partition=highmem_p
#SBATCH --job-name=benchmarker
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=10G

source /home/mel64643/.bashrc
source activate CMA
#python -u /home/mel64643/github/CMA_Benchmarker/exec_cma_database.py > CMA.out
python -u exec_cma_database.py > CMA.out



