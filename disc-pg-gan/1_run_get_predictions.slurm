#!/bin/bash

#SBATCH --job-name=pg-gan-get-predictions
#SBATCH --output=outfiles/%x-%j.out
#SBATCH --error=outfiles/%x-%j.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:10:00

eval "$(micromamba shell hook --shell=bash)"
micromamba activate disc-pg-gan

mkdir -p results/predictions
mkdir -p results/scram_data

python 1_get_predictions.py