#!/bin/bash

####SBATCH --job-name=flagel_cnn_train_convsize4
#SBATCH --output=outfiles/cnn_train_convsize4/%x-%A-%a.out
#SBATCH --error=outfiles/cnn_train_convsize4/%x-%A-%a.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --time=5:00:00
#SBATCH --array=1-10

# micromamba activate flagel_demography_cnn

# i out of 10 experimental reps
i=${SLURM_ARRAY_TASK_ID}

date

convDim=1d
convSize=4
poolSize=2
logTransformY=True
intAllele=True
sortRows=True
useDropout=True
lossThreshold=1

inDir="results/formatted_simulations/formatted_ms_data_${i}"
results_dir="results/trained_models/convSize_4/rep_${i}"
mkdir -p $results_dir
plots_dir="results/plots/convSize_4_original/rep_${i}"
mkdir -p $plots_dir

weightFileName=${results_dir}/trained_weight
modelFileName=${results_dir}/trained_model
resultFileName=${results_dir}/cnn_pred

python scripts/demogConvRegMerge.py $convDim $convSize $poolSize \
            $logTransformY $intAllele $sortRows $useDropout $lossThreshold \
            $inDir $weightFileName $modelFileName $resultFileName $plots_dir

date