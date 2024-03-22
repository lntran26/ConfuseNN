#!/bin/bash

#SBATCH --job-name=flagel_cnn_predict_scramble_plot
#SBATCH --output=outfiles/predict_scramble_plot_job_array/%x-%A-%a.out
#SBATCH --error=outfiles/predict_scramble_plot_job_array/%x-%A-%a.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --time=1:00:00
#SBATCH --array=1-10

# micromamba activate flagel_demography_cnn
date

# i out of 10 experimental reps
i=${SLURM_ARRAY_TASK_ID}

weightFileName2="results/trained_models/convSize_2/rep_${i}/trained_weight"
weightFileName4="results/trained_models/convSize_4/rep_${i}/trained_weight"

testPlotDir2="results/plots/convSize_2_scramble/rep_${i}"
testPlotDir4="results/plots/convSize_4_scramble/rep_${i}"

mkdir -p $testPlotDir2
mkdir -p $testPlotDir4

for mode in col colI free right left; do

    npzFileName="results/scramble_test_data/formatted_rep_${i}/formatted_test_10000_scramble_${mode}.npz"
    outFileName="results/scramble_test_data/formatted_rep_${i}/formatted_test_10000_scramble_${mode}_sorted.npz"

    testPredFileName2="results/trained_models/convSize_2/rep_${i}/scramble_${mode}_pred"
    testPredFileName4="results/trained_models/convSize_4/rep_${i}/scramble_${mode}_pred"

    testPlotPrefix2="${testPlotDir2}/pred_scramble_${mode}"
    testPlotPrefix4="${testPlotDir4}/pred_scramble_${mode}"
    
    python scripts/predict_scramble_plot.py $npzFileName $outFileName $weightFileName2 $testPredFileName2 $testPlotPrefix2 2
    python scripts/predict_scramble_plot.py $npzFileName $outFileName $weightFileName4 $testPredFileName4 $testPlotPrefix4 4
    
done

date
