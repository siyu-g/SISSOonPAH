#!/bin/bash
#SBATCH -J relaxation# Job name
#SBATCH -n 54 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH -o job_%j.out # File to which STDOUT will be written %j is the job #
#SBATCH -p cpu
#SBATCH --time=00:10:00

# Define all bins here
SISSO_BIN="/home/marom_group/SF/Siyu/SISSO/bin/SISSO_predict"

source /home/marom_group/SF/Siyu/SISSO/src/ENV.txt

ulimit -s unlimited
 
mpirun -np 1 $SISSO_BIN &> SISSO_predict.out
