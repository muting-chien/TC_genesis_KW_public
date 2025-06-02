#!/bin/bash

#SBATCH --partition=bar_all
#SBATCH --ntasks=8
#SBATCH --nodes=1 #1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-use=Mu-Ting.Chien@colostate.edu
#SBATCH --output=../txt/Tropical_wave.txt

python Tropical_wave_ace2_100yr.py

