Scramble test analysis for ImaGene CNN

Reference:
https://github.com/mfumagalli/ImaGene

Creating environment and installing python dependencies required to reproduce results
```console
$ micromamba create -n ImaGene python=3 tensorflow=2 keras=2 numpy scipy scikit-image scikit-learn matplotlib pydot arviz jupyterlab -y -c conda-forge
$ micromamba activate ImaGene
```

Simulation also requires separate installation of msms (more details [here](https://www.mabs.at/publications/software-msms/downloads/))
For all experiments here, I used the pure jar file `msms3.2rc-b163.jar`.

The path to the msms jar file needs to be specified in the script used for simulation.

`scripts/ImaGene_scramble.py` is the modified version of the original 
`ImaGene.py` [file](https://github.com/mfumagalli/ImaGene/blob/master/ImaGene.py).

The modified version contains the code necessary to perform scramble tests on the ImaGene CNN (under the `scramble()` method).