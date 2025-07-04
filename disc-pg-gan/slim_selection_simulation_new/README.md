### Instruction for reproducing SLiM simulation
1. Activate the virtual environment for disc-pg-gan described in `environmental.yml` file in the previous dir.
2. Run the numerically-labeled scripts in order:

`1_run_slim_sim.slurm` generates all neutral and selected trees data using SLiM.

`2_make_file_lists.sh` generates `.txt` files with paths to the trees generated above,
which is required for the following processing step. The resulting `.txt` files are in the current repository.

`3_process_trees.slurm` and `3_process_trees_neutral_split.slurm` are scripts to process the simulated data from trees into arrays format 
that can be input into the trained discriminator CNN. One script is for processing the trees with selection, and one for the neutral trees.
`3_process_trees.slurm` could be used to process all trees with the code at line #32 uncommented.
In practice, since there are a lot of neutral trees, I made a separate script (`3_process_trees_neutral_split.slurm`) to further split up 
these trees for more efficient parallelization.

All other scripts in this dir are code dependencies copied from https://github.com/mathiesonlab/disc-pg-gan/tree/main/slim_selection_simulation
