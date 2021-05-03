#!/bin/bash
#SBATCH --job-name sq_logistic     # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition short          # Slurm partition to use
#SBATCH --array=1-15               # how many tasks in the array
#SBATCH --time 0-06:00             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=1000         # Memory limit for each tasks (in MB)
#SBATCH -o out/sq_logistic.out     # File to which STDOUT will be written
#SBATCH -e out/sq_logistic.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ronakdm@uw.edu # Email to which notifications will be sent

module load Python
source ./bin/activate
python logistic_regression.py $SLURM_ARRAY_TASK_ID