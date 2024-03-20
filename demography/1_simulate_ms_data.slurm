#!/bin/bash

#SBATCH --job-name=flagel_cnn_sim_ms_three_epoch
#SBATCH --output=outfiles/simulation_job_array/%x-%A-%a.out
#SBATCH --error=outfiles/simulation_job_array/%x-%A-%a.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --array=1-10

# micromamba activate flagel_demography_cnn

date
# i out of 10 experimental reps
i=${SLURM_ARRAY_TASK_ID}

outdir="results/simulations/ms_simulated_data_${i}"
mkdir -p $outdir

filePrefix='three_epoch'
numReps='100000'
msPath='/xdisk/rgutenk/lnt/software/msdir/ms'

python scripts/simulatePopSizeTrajectoriesWithMs3EpochDSPriors.py $outdir $filePrefix $numReps $msPath

date