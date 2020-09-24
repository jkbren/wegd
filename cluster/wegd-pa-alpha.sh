#!/bin/bash
#SBATCH --job-name=wegd_pa_alpha
#SBATCH -o wegd_pa_alpha%a.out     
#SBATCH -e wegd_pa_alpha%a.err
#SBATCH -c 16
#SBATCH -p netsi_standard       
#SBATCH --mem=135GB

work=/scratch/klein.br/wegd/cluster/
cd $work

declare -a commands
readarray -t commands < wegd-pa-alpha.conf # Exclude newline.
eval ${commands[$SLURM_ARRAY_TASK_ID]}
