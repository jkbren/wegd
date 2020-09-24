#!/bin/bash
#SBATCH --job-name=wegd_gnk_k
#SBATCH -o wegd_gnk_k%a.out     
#SBATCH -e wegd_gnk_k%a.err
#SBATCH -c 16
#SBATCH -p netsi_standard       
#SBATCH --mem=135GB

work=/scratch/klein.br/wegd/cluster/
cd $work

declare -a commands
readarray -t commands < wegd-gnk-k.conf # Exclude newline.
eval ${commands[$SLURM_ARRAY_TASK_ID]}
