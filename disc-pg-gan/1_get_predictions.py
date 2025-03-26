import tensorflow as tf
import numpy as np
from parse import parse_output
import prediction_utils
from slim_iterator import SlimIterator
import distribution_plot
import pickle

def get_params_trial_data(trial_file):
    if trial_file[-4:] == ".txt": # list of TRIAL FILES containing the data we want
        files = open(trial_file, 'r').readlines()
        params, trial_data = parse_output(files[0][:-1])
    else:
        files = [trial_file+"\n"] # add newline to be discarded, matches list format
        params, trial_data = parse_output(trial_file)
    return params, trial_data, files
    

def scramble_data(regions, mode):

    scram_regions = []
    
    for j in regions: # number of j depends on specified batch size for slim iterator
        # take only the genotype matrix dim to scramble
        gt_matrix = j[:,:,0]
        
        if mode == "col":
            # scramble column only
            temp_arr = gt_matrix.T
            np.random.shuffle(temp_arr)
            scram_gt_matrix = temp_arr.T
        
        elif mode == "colI":
            # scramble column and inside columns
            temp_arr = gt_matrix.T
            np.random.shuffle(temp_arr)
            for r in range(temp_arr.shape[0]): # iterate through each row, which are the snp columns 
                np.random.shuffle(temp_arr[r]) # shuffle within each rows, which are snp columns
            scram_gt_matrix = temp_arr.T
        
        elif mode == "free":
            # scramble free
            temp_arr = gt_matrix
            d1, d2 = temp_arr.shape
            # reshape to 1D arr, then shuffle, which is equivalent to free shuffle
            temp_arr = temp_arr.reshape(d1 * d2)
            np.random.shuffle(temp_arr)
            scram_gt_matrix = temp_arr.reshape(d1, d2)
            
        elif mode == "left":
            # scramble block left
            temp_arr = gt_matrix
            rows, cols = temp_arr.shape
            num_ones = np.count_nonzero(temp_arr == 1.)
            temp_arr = np.concatenate([np.ones(num_ones), -1 * np.ones(rows * cols - num_ones)])
            scram_gt_matrix = temp_arr.astype('float32').reshape((cols, rows)).T
        
        # create new tensor from the scrambled gt matrix and same dist matrix
        dist_matrix = j[:,:,1]
        scram_tensor = np.stack((scram_gt_matrix, dist_matrix), axis=-1)
        
        # append to new scram_regions_i
        scram_regions.append(scram_tensor)

    return np.array(scram_regions)

'''
Plots and saves discriminator predictions on SLiM data provided by arguments

ARGUMENTS: 5 files whose contents are lists of numpy arrays corresponding to
the selection strengths: neutral (0.0), 0.01, 0.025, 0.05, 0.10
'''
def plot_selection(in_trial_data, files):
    sel_paths = [   "data_paths/CEU_neutrals_0.txt",
                    "data_paths/CEU_neutrals_1.txt",
                    "data_paths/CEU_neutrals_2.txt",
                    "data_paths/CEU_neutrals_3.txt",
                    "data_paths/CEU_neutrals_4.txt",
                    "data_paths/CEU_neutrals_5.txt",
                    "data_paths/CEU_neutrals_6.txt",
                    "data_paths/CEU_neutrals_7.txt",
                    "data_paths/CEU_neutrals_8.txt",
                    "data_paths/CEU_neutrals_9.txt",
                    "data_paths/CEU_sel_1.txt", 
                    "data_paths/CEU_sel_2.txt", 
                    "data_paths/CEU_sel_3.txt", 
                    "data_paths/CEU_sel_4.txt"]

    regions = []
    for sel_path in sel_paths:
        if "neutrals" in sel_path:
            # reduce batch size by a factor of 10 because we are using 10 neutral files
            # 3000 trees total, taking 300 regions from each neutral file
            region = SlimIterator(sel_path).real_batch(300)
        else:
            region = SlimIterator(sel_path).real_batch(600)
        
        regions.append(region)
    
    with open(f"results/scram_data/original.pickle", 'wb') as handle:
        pickle.dump(regions, handle)

    for f in files:
        infile = f[:-1]
        params, trial_data = parse_output(infile)
        outfile = infile[:-4]+"_sel"

        trained_disc = tf.saved_model.load("CEU_19_230410_230830_finetuneAug23")

        sel_labels = ["s=0.01", "s=0.025", "s=0.05", "s=0.10"]
        all_preds = {}

        neutral_pred = []
        sel_pred = []

        for i, sel_path in enumerate(sel_paths):
            # get prediction
            preds = distribution_plot.process_regions(trained_disc, regions[i])
            
            # append to either neutral or selection
            if "neutrals" in sel_path:
                neutral_pred.append(preds)
            else:
                sel_pred.append(preds)
        
        neutral_pred = np.array(neutral_pred).flatten().tolist()
        all_preds["neutral"] = neutral_pred
        
        for j, sel_label in enumerate(sel_labels):
            all_preds[sel_label] = sel_pred[j]
            
        print(all_preds.keys())
        print(len(all_preds["neutral"]))
        print(len(all_preds["s=0.10"]))
    
        with open('results/predictions/original_preds.pickle', 'wb') as handle:
            pickle.dump(all_preds, handle)
            
            
        # scramble data

        for j, scram_mode in enumerate(['col', 'colI', 'free', 'left']):
            
            # get prediction for each scramble case
            sel_labels = ["s=0.01", "s=0.025", "s=0.05", "s=0.10"]
            all_preds = {}
    
            neutral_pred = []
            sel_pred = []
            scram_regions = []
    
            for i, sel_path in enumerate(sel_paths): # 10 neutrals and 4 sels
                scram_region = scramble_data(regions[i], scram_mode)
                scram_regions.append(scram_region)
            
                # get prediction
                preds = distribution_plot.process_regions(trained_disc, scram_region)
                
                # append to either neutral or selection
                if "neutrals" in sel_path:
                    neutral_pred.append(preds)
                else:
                    sel_pred.append(preds)
            
            neutral_pred = np.array(neutral_pred).flatten().tolist()
            all_preds["neutral"] = neutral_pred
            
            for j, sel_label in enumerate(sel_labels):
                all_preds[sel_label] = sel_pred[j]
            
            print(all_preds.keys())
            print(len(all_preds["neutral"]))
            print(len(all_preds["s=0.10"]))
        
            with open(f"results/predictions/scram_{scram_mode}_preds.pickle", 'wb') as handle:
                pickle.dump(all_preds, handle)
                
            with open(f"results/scram_data/scram_{scram_mode}.pickle", 'wb') as handle:
                pickle.dump(scram_regions, handle)


if __name__ == "__main__":

    tf.compat.v1.enable_eager_execution() # this work to turn on eager execution

    trial_files = "CEU_19_230410.out"
    params, trial_data, files = get_params_trial_data(trial_files)
    plot_selection(trial_data, files)

    