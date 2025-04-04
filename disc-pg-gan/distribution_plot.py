"""
Create seaborn plot for discriminator predictions on real data
Author: Rebecca Riley
Date: 03/15/2023
"""

import math
import matplotlib.pyplot as plt
import numpy as np
# print(np.version.version) # 1.21.6 (1.17 as in environment.yml also works)
import sys

sys.path.insert(1, "../")

from discriminator import OnePopModel
from generator import Generator
from global_vars import DEFAULT_SAMPLE_SIZE
from parse import parse_output
import prediction_utils
import real_data_random
import simulation
import tensorflow as tf

SIMULATED_BATCH_SIZE = 500

# =============================================================================
# COLOR SETTINGS
# =============================================================================
REAL_DATA_COLORS = {"CEU": ["grey", "dodgerblue", "midnightblue", "slateblue"],
                    "GBR": ["grey", "dodgerblue", "midnightblue", "slateblue"],
                    "YRI": ["grey", "yellow", "sienna", "darkorange"],
                    "ESN": ["grey", "yellow", "sienna", "darkorange"],
                    "CHB": ["grey", "limegreen", "darkgreen", "olivedrab"],
                    "CHS": ["grey", "limegreen", "darkgreen", "olivedrab"]}

# =============================================================================
# PLOT UTILS
# =============================================================================
def get_title(title_data):
    return "train: {train_pop}, test: {test_pop}, seed: {seed}".format(\
        train_pop=title_data["train"], test_pop=title_data["test"],
        seed=title_data["seed"])

def save_violin_plot(data, colors, labels, output, title_data, use_pdf):
    RANGE = range(len(colors))

    quantiles = []
    quantile_colors = []
    for i in RANGE:
        quantiles.append([0.05, 0.95])
        quantile_colors.extend([colors[i], colors[i]])

    parts = plt.violinplot(data, RANGE, showmeans=True, showextrema=False, quantiles=quantiles)
        
    for i in RANGE:
        parts["bodies"][i].set_facecolor(colors[i])
    parts["cmeans"].set_color(colors)
    parts["cquantiles"].set_color(quantile_colors)

    # final plotting
    plt.xticks(RANGE, labels)
    plt.ylim([0,1])
    plt.ylabel("discriminator prediction")
    plt.title(get_title(title_data))
    plt.tight_layout()

    if use_pdf:
        plt.savefig(output+"_violin.pdf", format='pdf', dpi=350)
    else:
        plt.savefig(output+"_violin.png", format='png', dpi=300)

    plt.clf() # important

# =============================================================================
# DATA PROCESSING
# =============================================================================
def get_prob(x):
    return 1 / (1 + math.exp(-x))

def process_regions(disc, regions, positions=None):
    # print(tf.executing_eagerly())
    
    # inference = disc.signatures["serving_default"]
    # print(inference) # <tensorflow.python.saved_model.load._WrapperFunction object at 0x2b246e4c0590>
    # preds = inference(regions)
    # print(type(preds)) # <class 'dict'>
    # print(preds) # {'output_1': <tf.Tensor 'StatefulPartitionedCall_1:0' shape=(1000, 1) dtype=float32>}
    # print(preds['output_1']) # Tensor("StatefulPartitionedCall:0", shape=(1000, 1), dtype=float32)
    
    # print(tf.executing_eagerly()) # true

    preds = disc(regions, training=False).numpy()
    
    # print(disc(regions, False)) # Tensor("StatefulPartitionedCall_1:0", shape=(1000, 1), dtype=float32)
    # result = tf.reduce_sum(disc(regions, False))
    # print(result) # Tensor("Sum:0", shape=(), dtype=float32)
    
    probs = [get_prob(pred[0]) for pred in preds]
    return probs

def get_seed(name):
    i = 0

    while not name[i].isnumeric():
        i += 1

    seed = ""
    while name[i].isnumeric():
        seed = seed + name[i]
        i += 1

    return seed

def process_outfiles(outfiles_list):
    seed_param_disc_dict = {}
    outfiles = open(outfiles_list).readlines()

    for outfile in outfiles:
        final_params, trial_data = parse_output(outfile[:-1])

        trained_disc = tf.saved_model.load("saved_model/" + trial_data["disc"] + "/")
        # trained_disc = OnePopModel(num_samples, saved_model=trained_disc)

        seed_param_disc_dict[trial_data["seed"]] = final_params, trained_disc

    sample_size = DEFAULT_SAMPLE_SIZE if trial_data["sample_size"] \
        is None else trial_data["sample_size"]
    generator = prediction_utils.get_generator(trial_data, sample_size)

    return seed_param_disc_dict, generator

# =============================================================================
# LOAD FILES
# =============================================================================
'''
Accepts a file containing a list of prepared prediction files 
(see analysis/genome_disc.py), a bed file corresponding to 
positive selection in the test population, and a signal to save the final
as a pdf or a png.
Saves violin representations of the predictions on neutral test data and
test data under selection.
'''
def plot_real(prediction_list, outfiles_list, pos_sel_bed, use_pdf):
    seed_param_disc_dict, generator = process_outfiles(outfiles_list)

    pos_sel_mask = real_data_random.read_mask(pos_sel_bed)
    pred_files = open(prediction_list).readlines()

    tokens = pred_files[0].split('.')[-2].split("_")
    train_pop_name = tokens[-2]
    test_pop_name = tokens[-1]

    labels = ["simulated (msprime)", "neutral ("+test_pop_name+")", 
        "pos. sel. ("+test_pop_name+")"]
    colors_all = REAL_DATA_COLORS[test_pop_name]
    colors = [colors_all[0], colors_all[1], colors_all[3]]
    
    title_data = {"train": train_pop_name, "test": test_pop_name}

    # iterate through each discriminator's predictions
    for pred_file in pred_files:
        seed = get_seed(pred_file)
        outfile = train_pop_name+"_"+test_pop_name+"_"+seed
        title_data["seed"] = seed

        predictions = np.loadtxt(pred_file[:-1], delimiter="\t")

        neutrals = []
        sels = []

        for row in predictions:
            chrom, start, end, pred = row[0], row[1], row[2], row[3]
            
            region = real_data_random.Region(int(chrom), start, end)
            if region.inside_mask(pos_sel_mask):
                sels.append(pred)
            else:
                neutrals.append(pred)

        params, disc = seed_param_disc_dict[seed]
        generator.update_params(params)
        simulated_batch = generator.simulate_batch(SIMULATED_BATCH_SIZE)
        sim_preds = process_regions(disc, simulated_batch)

        save_violin_plot([sim_preds, neutrals, sels], colors, labels, outfile, title_data, use_pdf)

if __name__ == "__main__":
    prediction_list = sys.argv[1]
    outfiles_list = sys.argv[2]
    pos_sel_bed = sys.argv[3]

    options = sys.argv[-1]
    use_pdf = ("png" not in options)

    plot_real(prediction_list, outfiles_list, pos_sel_bed, use_pdf)

