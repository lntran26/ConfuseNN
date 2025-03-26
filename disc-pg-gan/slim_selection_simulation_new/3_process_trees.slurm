#!/bin/bash

#SBATCH --job-name=pg_gan_process_trees_CEU
#SBATCH --output=outfiles/process_trees_CEU/%x-%A-%a.out
#SBATCH --error=outfiles/process_trees_CEU/%x-%A-%a.err
#SBATCH --account=rgutenk
#SBATCH --partition=standard
####SBATCH --partition=high_priority
####SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --time=120:00:00
#SBATCH --array=0-4

## neutral set (600 trees) took bout 18 hrs to run
## each selection set (120 trees) took about 4 hr to run
## ran on Ocelote standard nodes, which is 6GB memory/cpu, max 28c/node
# micromamba activate disc-pg-gan

s_coeffs=("0.0" "0.01" "0.025" "0.05" "0.1")

i=${SLURM_ARRAY_TASK_ID} # for job array, the job that we are on
printf -v j "%03d" $i # change $i to $j for index 3

s_coeff=${s_coeffs[${i}]}

mkdir -p output

if [ "${s_coeff}" = "0.0" ]; then
    # python python_scripts/process_trees.py -i all_CEU_neutral_trees.txt -o CEU_neutral_3000 -r reco_rates.txt --Ne 22552
    echo "skip neutral case"
else
    python python_scripts/process_trees.py -i all_CEU_sel_${j}_trees.txt -o CEU_sel${j}_600 -r reco_rates.txt --Ne 22552
    # echo "skip selection case"
fi