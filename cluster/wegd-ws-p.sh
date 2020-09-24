#!/bin/bash
#SBATCH --job-name=wegd_ws_p
#SBATCH -o wegd_ws_p%a.out     
#SBATCH -e wegd_ws_p%a.err
#SBATCH -c 16
#SBATCH -p netsi_standard       
#SBATCH --mem=135GB

work=/scratch/klein.br/wegd/cluster/
cd $work

declare -a commands
readarray -t commands < wegd-ws-p.conf # Exclude newline.
eval ${commands[$SLURM_ARRAY_TASK_ID]}
