#!/bin/bash

#SBATCH --job-name=flagel_cnn_format_data
#SBATCH --output=outfiles/format_data_job_array/%x-%A-%a.out
#SBATCH --error=outfiles/format_data_job_array/%x-%A-%a.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --time=3:00:00
#SBATCH --array=1-10

# micromamba activate flagel_demography_cnn

# i out of 10 experimental reps
i=${SLURM_ARRAY_TASK_ID}

date

msFilePath="results/simulations/ms_simulated_data_${i}/sims/three_epoch.msOut.gz"
demogParamPath="results/simulations/ms_simulated_data_${i}/popSizeHistories/three_epoch.popSize"

outFileDir="results/formatted_simulations/formatted_ms_data_${i}"
mkdir -p $outFileDir
outFilePath=${outFileDir}/three_epoch_ms_formatted

python scripts/extractDataSet.py $msFilePath $demogParamPath $outFilePath

date