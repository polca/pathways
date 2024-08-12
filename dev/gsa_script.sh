#!/bin/bash
#SBATCH --cluster=merlin6                  # Cluster name
#SBATCH --partition=hourly                 # Specify one or multiple partitions
#SBATCH --time=01:00:00                    # Strongly recommended
#SBATCH --hint=multithread                 # Mandatory for multithreaded jobs
#SBATCH --job-name=sweet_sure_gsa          # Name your job
#SBATCH --ntasks=1                         # Number of parallel tasks (one for each year)
#SBATCH --cpus-per-task=4                  # Request one core per task
#SBATCH --output=sweet_sure_gsa.out # Define your output file
#SBATCH --error=sweet_sure_gsa.err  # Define your error file
#SBATCH --mem-per-cpu=20000

module use unstable
module load anaconda
conda activate /data/user/sacchi_r/conda-envs/salib
srun python /data/user/sacchi_r/python_scripts/gsa.py