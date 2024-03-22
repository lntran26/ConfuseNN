#!/bin/bash

#SBATCH --job-name=flagel_cnn_format_data_scramble
#SBATCH --output=outfiles/format_data_scramble_job_array/%x-%A-%a.out
#SBATCH --error=outfiles/format_data_scramble_job_array/%x-%A-%a.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --time=1:00:00
#SBATCH --array=1-10

# micromamba activate flagel_demography_cnn
date

# i out of 10 experimental reps
i=${SLURM_ARRAY_TASK_ID}

msFilePath="results/simulations/ms_simulated_data_${i}/sims/three_epoch.msOut.gz"
demogParamPath="results/simulations/ms_simulated_data_${i}/popSizeHistories/three_epoch.popSize"

outFileDir="results/scramble_test_data/formatted_rep_${i}"
mkdir -p $outFileDir
outFilePath="${outFileDir}/formatted_test_10000_scramble"

python scripts/extractDataSetScramble.py $msFilePath $demogParamPath $outFilePath

date
