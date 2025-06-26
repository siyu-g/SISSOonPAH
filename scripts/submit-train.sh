#!/bin/bash
#SBATCH -J relaxation# Job name
#SBATCH -n 54 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH -o job_%j.out # File to which STDOUT will be written %j is the job #
#SBATCH -p cpu
#SBATCH --time=04:00:00
#SBATCH --mail-user=luo2@andrew.cmu.edu
#SBATCH --mail-type=FAIL,REQUEUE,STAGE_OUT

# Define all bins here
SISSO_BIN="/home/marom_group/SF/Siyu/SISSO/bin/SISSO"

source /home/marom_group/SF/Siyu/SISSO/src/ENV.txt

ulimit -s unlimited

echo "Job starts at:"
date 
mpirun -np 54 $SISSO_BIN &> SISSO_test.out
echo "Job ends at:"
date
