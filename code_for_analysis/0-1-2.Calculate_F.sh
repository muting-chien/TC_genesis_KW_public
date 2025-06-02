#!/bin/bash
# ## bash users replace /tcsh with /bash -l
# PBS -N save_mean_state
# PBS -A UWAS0114
# PBS -l walltime=01:00:00
# PBS -q regular
# PBS -j oe
# PBS -k eod
# PBS -m abe
# PBS -M muting@uw.edu
# PBS -l select=2:ncpus=36:mpiprocs=36

python3 Calculate_F_qsub.py 


