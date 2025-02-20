#!/bin/bash
# Configure the resources required
#SBATCH -p dm
#SBATCH -N 1 # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH -c 8                # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=00:00:10         # time allocation, which has the format (D-HH:MM), here set to 1 hour

# Configure notifications
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yunxiang.liu@adelaide.edu.au


a=$1
b=$2

echo $a
echo $b
