#!/bin/bash

#SBATCH --job-name=imagene_cnn_scramble_test_multi_3epoch
#SBATCH --output=outfiles/test_scramble/%x-%A-%a.out
#SBATCH --error=outfiles/test_scramble/%x-%A-%a.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=1:00:00
#SBATCH --array=1-5

# 1-5 for each scramble test level
# each job took about 8 minutes to run
# micromamba activate ImaGene

echo "running job array ${SLURM_ARRAY_TASK_ID}"
python scripts/test_scramble.py ${SLURM_ARRAY_TASK_ID}
