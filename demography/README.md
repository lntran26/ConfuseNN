### Instruction for reproducing Flagel et al. demographic inference CNN results

Original study with instruction that we followed to reproduce the CNN:
https://github.com/flag0010/pop_gen_cnn/tree/master/demography

1. Create a virtual environment and install dependencies required to reproduce results
```console
$ micromamba env create -f environment.yml
$ micromamba activate flagel_demography_cnn
```
2. Run the numerically-labeled scripts in order:

`1_simulate_ms_data.slurm` is a bash script that uses `ms` to generate the simulated data. 
Simulation also requires separate installation of `ms` (more details [here](https://home.uchicago.edu/~rhudson1/source/mksamples.html)).
After installation, the path to `ms` needs to be specified in simulation scripts.
This script runs `scripts/simulatePopSizeTrajectoriesWithMs3EpochDSPriors.py`, which originates from https://github.com/flag0010/pop_gen_cnn/blob/master/demography/simulatePopSizeTrajectoriesWithMs3EpochDSPriors.py

`2_format_data.slurm` is a bash script to run `scripts/extractDataSet.py`, which originates from https://github.com/flag0010/pop_gen_cnn/blob/master/demography/extractDataSet.py. This script formats the simulated data for CNN training.

`3_train_flagel_cnn_convsize2.slurm` and `3_train_flagel_cnn_convsize4.slurm` are scripts to reproduce the original trained CNNs.
`_convsize2` is for the best performning CNN reported in the original study and evaluated in ours.
`_convsize4` is another CNN with a bigger convolution kernel size that we also tested. This CNN yielded similar results and was not formally reported in our paper.
These scripts run `scripts/demogConvRegMerge.py`, which originates from https://github.com/flag0010/pop_gen_cnn/blob/master/demography/demogConvRegMerge.py

`4_format_data_scramble.slurm` is the modified version of `2_format_data.slurm`, and was used to generate permuted test data.

`5_predict_scramble_plot.slurm` takes the trained CNN generated in step 3 above and gets its predictions for the permuted test data generated in step 4. It also generates the resulting plots.
