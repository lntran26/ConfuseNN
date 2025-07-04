### Instruction for reproducing disc-pg-gan results
1. Use `environmental.yml` to set up a virtual environment with all dependencies required for running scripts.

2. The original test data generated using SLiM are done in the `slim_selection_simulation_new` dir,
which reused scripts and followed instructions from https://github.com/mathiesonlab/disc-pg-gan/tree/main/slim_selection_simulation.
Further instruction is provided in the `README.md` inside this dir.

3. Once the original test data are generated, return to this dir and run the scripts starting with numbers e.g. "1_*" and "2_*" sequentially.

`1_run_get_predictions.slurm` is a bash script to run `1_get_predictions.py`, which generates the discriminator CNN predictions for the original and permuted test data.

`2_plot_predictions.py` is a Python script for plotting the ROC curve shown in the paper, using output generated from the step above.

### Extra notes
The trained and fine-tuned discriminator was provided by Riley et al., which includes the `CEU_19_230410.out` file and the `CEU_19_230410_230830_finetuneAug23` dir.

All other `.py` scripts in this dir are code dependencies copied as is from https://github.com/mathiesonlab/disc-pg-gan to run trained the model.
