#!/bin/bash

#SBATCH --job-name=imagene_cnn_generate_data
#SBATCH --output=outfiles/%x-%j.out
#SBATCH --error=outfiles/%x-%j.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --time=2:00:00

# micromamba activate ImaGene

# simulate binary/multiclass (option 'Binary') or (semi)-continuous (option 'Continuous')
# MODE=$1 # Binary or Continuous

MODE=Binary

# path to software (msms)
DIRMSMS=/xdisk/rgutenk/lnt/software/msms3.2rc-b163.jar
# path to store simulations
DIRDATA=/xdisk/rgutenk/lnt/projects/ConfuseNN/ImaGene/results/simulations/$MODE

date

# number of batches of simulations
EACH=10
# each batch is simulated 3 times, for each demographic model
for INDEX in {0..29}
do
	A=$(($INDEX / $EACH))
	model=$(( A + 1 ))
	repetition=$(( $(( $INDEX - $(( A*EACH )) )) + 1 ))

	FNAME=$DIRDATA/Simulations$repetition.Epoch$model
	echo $FNAME
	mkdir -p $FNAME
	bash scripts/simulate.sh $DIRMSMS $FNAME $model $MODE
done

date


