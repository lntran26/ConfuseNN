#!/bin/bash

#SBATCH --job-name=pg_gan_process_trees_neutral_CEU_split
#SBATCH --output=outfiles/process_trees_neutral_CEU_split/%x-%A-%a.out
#SBATCH --error=outfiles/process_trees_neutral_CEU_split/%x-%A-%a.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --time=24:00:00
#SBATCH --array=0-9

# micromamba activate disc-pg-gan

# split -l 300 all_CEU_neutral_trees.txt
# this command split the CEU neutral trees path into 10 files starting with "x"

# save split file names to files
files=($(ls x*))

i=${SLURM_ARRAY_TASK_ID} # for job array, the job that we are on
fn=${files[${i}]} # pair the job array id with one file in files

printf -v j "%02d" $i # change $i to $j for index 2

python python_scripts/process_trees.py -i $fn -o CEU_neutral_300_${j} -r reco_rates.txt --Ne 22552
