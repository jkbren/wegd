# Scripts for batch calculations of WEGDs

This folder contains functionality for performing large-scale batch 
simulations, where two networks are generated from a random graph
ensemble and then their distance is calculated. It consists of 
three files, all for performing the WEGD calculation using the 
soft configuration model with varying gamma.

* `wegd-scm-gamma.py`: The core logic implementing the experiments, for 
    different parameterizations of the model. The parameter values
    are read from the command line.
* `wegd-scm-gamma.conf`: A file that contains the specific command-line
   calls to `python` to run the core script for all parameterizaitons.
* `wegd-scm-gamma.sh`: A shell script that deploys the above files 
   on a [SLURM](https://en.wikipedia.org/wiki/Slurm_Workload_Manager)
   cluster.
