#!/bin/bash
#SBATCH --job-name=wegd_n
#SBATCH -o wegd_n%a.out     
#SBATCH -e wegd_n%a.err
#SBATCH -c 16
#SBATCH -p netsi_standard       
#SBATCH --mem=135GB

work=/scratch/klein.br/wegd/cluster/
cd $work

declare -a commands
readarray -t commands < wegd-n.conf # Exclude newline.
eval ${commands[$SLURM_ARRAY_TASK_ID]}
