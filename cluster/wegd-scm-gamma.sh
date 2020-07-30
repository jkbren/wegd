#!/bin/bash
#SBATCH --job-name=wegd_scm_gamma
#SBATCH -o wegd_scm_gamma%a.out     
#SBATCH -e wegd_scm_gamma%a.err
#SBATCH -c 16
#SBATCH -p netsi_standard       
#SBATCH --mem=135GB

work=/scratch/klein.br/wegd/cluster/
cd $work

declare -a commands
readarray -t commands < wegd-scm-gamma.conf # Exclude newline.
eval ${commands[$SLURM_ARRAY_TASK_ID]}
