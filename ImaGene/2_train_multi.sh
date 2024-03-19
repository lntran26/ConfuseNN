#!/bin/bash

#SBATCH --job-name=imagene_cnn_train_multi
#SBATCH --output=outfiles/%x-%j.out
#SBATCH --error=outfiles/%x-%j.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=3:00:00

# micromamba activate ImaGene

# train and test for multiclass-classification for the original 3-epoch model

python scripts/train_multi.py 3
