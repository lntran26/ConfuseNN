#!/bin/bash

#SBATCH --job-name=pg_gan_slim_sim_test_CEU
#SBATCH --output=outfiles/slim_sim_test_CEU/%x-%A-%a.out
#SBATCH --error=outfiles/slim_sim_test_CEU/%x-%A-%a.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=150:00:00
#SBATCH --array=0-4

# micromamba activate disc-pg-gan

s_coeffs=("0.0" "0.01" "0.025" "0.05" "0.1")

i=${SLURM_ARRAY_TASK_ID} # for job array, the job that we are on
printf -v j "%03d" $i # change $i to $j for index 3

s_coeff=${s_coeffs[${i}]}

mkdir -p output

if [ "${s_coeff}" = "0.0" ]; then
    bash bash_scripts/make_exp_trees.sh 3000 reco_rates.txt 22552,3313,0.00535,3589,1050 \
            ./output/ CEU_neutral slim_scripts/exp_neutral.slim ${s_coeff}
else
    bash bash_scripts/make_exp_trees.sh 600 reco_rates.txt 22552,3313,0.00535,3589,1050 \
            ./output/ CEU_sel_${j} slim_scripts/exp_selection.slim ${s_coeff}
fi