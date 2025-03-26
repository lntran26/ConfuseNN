import numpy as np
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt

def process_roc(data):
    # make y_true for roc_curve (0 for 1000 first neutral, 1 for the rest)
    y_true = []
    for i, a in enumerate(data):
        for _ in a:
            if i==0:
                y_true.append(0)
            else:
                y_true.append(1)
    
    # flatten the predicted probs y_pred
    y_score = []
    for probs_list in data:
        y_score += probs_list
    
    assert len(y_true) == len(y_score) # 5 x 1000
    
    fpr, tpr, _ = metrics.roc_curve(np.array(y_true), np.array(y_score))
    auc = metrics.roc_auc_score(np.array(y_true), np.array(y_score))
    
    return fpr, tpr, auc
    

def yellow_dot_counter(regions):
    data = []
    
    for j in regions: # 1000 in batch
        # take only the genotype matrix dim and ignore the distance dim
        gt_matrix = j[:,:,0]
        num_ones = np.count_nonzero(gt_matrix == 1.)
        rows, cols = gt_matrix.shape
        
        # since num_ones is expected to be larger for neutral case, which is true negative
        # need to inverse it so that it is more consistent with the lower score = negative
        data.append(1 - (num_ones / (rows * cols) ))
    
    return data
    
    
def save_violin_plot(data, colors, labels, output, title_data, use_pdf):
    """Copy from plotting/distribution_plot.py, removing dependency for get_title()"""
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
    plt.title(title_data)
    plt.tight_layout()

    if use_pdf:
        plt.savefig(output+"_violin.pdf", format='pdf', dpi=350)
    else:
        plt.savefig(output+"_violin.png", format='png', dpi=300)

    plt.clf() # important
    

if __name__ == "__main__":
    scram_labels = ['Original', 'Disrupt spatial patterns', 'Disrupt LD', 'Disrupt AFS', 'Disrupt total diversity']
    fpr, tpr, auc = [],[],[]

    # load predictions
    for i, mode in enumerate(["original",  "scram_col",  "scram_colI",  "scram_free",  "scram_left"]):
        with open(f'results/predictions/{mode}_preds.pickle', 'rb') as handle:
            preds_dict = pickle.load(handle) # keys: ['neutral', 's=0.01', 's=0.025', 's=0.05', 's=0.10']

        labels = ["neutral", "s=0.01", "s=0.025", "s=0.05", "s=0.10"]
        data = [None for s in labels] # same length
        for j, case in enumerate(list(preds_dict.keys())):
            if j==0 :
                assert case == "neutral" 
            data[j] = preds_dict[case]
            
        # plot violin plot for each case
        colors = ["grey", "pink", "salmon", "red", "darkred"]
        output = f"results/{mode}"
        title_data = f"train: CEU, test: SLiM, seed: 19, {scram_labels[i]}"
        save_violin_plot(data, colors, labels, output, title_data, use_pdf=False)
        
        # save roc curve data for each case
        fpr_i, tpr_i, auc_i = process_roc(data)
        fpr.append(fpr_i)
        tpr.append(tpr_i)
        auc.append(auc_i)
    
    for i in range(len(fpr)):
        plt.plot(fpr[i], tpr[i], label=f"{scram_labels[i]}, AUC="+str(round(auc[i],3)))
    
    # simple pixel counter model for ROC curve (optional)
    # load original input data, list of 10x100 neutral and 4x1000 selection
    with open('results/scram_data/original.pickle', 'rb') as handle:
        orig_data = pickle.load(handle)
    # merging 10 neutral sets into one and combine with 4 selection sets
    neutral_set = [orig_data[i] for i in range(10)]
    neutral_set_merge = [j for i in neutral_set for j in i]
    regions = orig_data[-4:]
    regions.insert(0, neutral_set_merge)
    # count pixels & plot ROC
    data = [yellow_dot_counter(region) for region in regions]
    fpr, tpr, auc = process_roc(data)
    plt.plot(fpr, tpr, label=f"Pixel counter, AUC="+str(round(auc,3)))
    
    
    # plot ROC curve cont.
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.savefig("results/ROC.png", format='png', dpi=300)
    plt.clf()
        
  