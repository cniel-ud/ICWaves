#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --job-name=generate_cmmn_filters
# The below is maximum time for the job.
#SBATCH --time=7-00:00:00
# SBATCH --time-min=0-01:00:00
#SBATCH --mail-user='ajmeek@udel.edu'
# this could be --mail-type=END, FAIL, TIME_LIMIT_90. but I thought it was too many emails
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --export=NONE
#SBATCH --mem=128G
#export UD_JOB_EXIT_FN_SIGNALS="SIGTERM EXIT"

vpkg_devrequire intel-python
source ../venv/bin/activate

# Run bash / python script below

python generate_cmmn_filters.py
