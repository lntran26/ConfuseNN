### Instruction for reproducing ImaGene results

Original study:
https://github.com/mfumagalli/ImaGene

1. Create a virtual environment and install dependencies required to reproduce results
```console
$ micromamba create -n ImaGene python=3 tensorflow=2 keras=2 numpy scipy scikit-image scikit-learn matplotlib pydot arviz jupyterlab -y -c conda-forge
$ micromamba activate ImaGene
```

2. Run the numerically-labeled scripts in order:

`1_generate_data.sh` is a bash script that uses msms to generate the simulated data.

Simulation also requires a separate installation of msms (more details [here](https://www.mabs.at/publications/software-msms/downloads/)).

For all analyses done here, I used the pure jar file `msms3.2rc-b163.jar`.
The path to the msms jar file needs to be specified in the script used for simulation.

`2_train_multi.sh` runs `scripts/train_multi.py`, which originates from https://github.com/mfumagalli/ImaGene/blob/master/Reproduce/train_multi.py with modified file paths.
This step reproduce the original trained multiclass classifier CNN.

`3_plot_multi_result.py` originates from https://github.com/mfumagalli/ImaGene/blob/master/Reproduce/plot_multi.py with modified file paths, and plots the original trained multiclass classifier CNN's performance.

`4_scramble_test.sh` runs `scripts/test_scramble.py` to perform all permutations described in the paper and plot the results.

### Notes
While not explicitly run, step 2,3,4 depends on `scripts/ImaGene_scramble.py`, which is the modified version of the original 
`ImaGene.py` [file](https://github.com/mfumagalli/ImaGene/blob/master/ImaGene.py).
This modified version contains the method necessary to perform our permutation tests on the ImaGene CNN (under the `scramble()` method that we added).
