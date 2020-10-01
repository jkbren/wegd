# Scripts for batch calculations of WEGDs

This folder contains functionality for performing large-scale batch 
simulations, where two networks are generated from a random graph
ensemble and then their distance is calculated. 

Each experiment consists of three files. These have the following 
format (using the soft configuration model as an example)

* `wegd-scm-gamma.py`: The core logic implementing the experiment, for 
    different parameterizations of the model. The parameter values
    are read from the command line.
* `wegd-scm-gamma.conf`: A file that contains the specific command-line
   calls to `python` to run the core script for all parameterizaitons.
* `wegd-scm-gamma.sh`: A shell script that deploys the above files 
   on a [SLURM](https://en.wikipedia.org/wiki/Slurm_Workload_Manager)
   cluster.

On a cluster, run the experiment with `sbatch --array=0-600 wegd-scm-gamma.sh`
