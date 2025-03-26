from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import glob

def root_mean_squared_error(pred_pre, true_pre):
    # exclude nans
    pred_post = pred_pre[~np.isnan(pred_pre)]
    true_post = true_pre[~np.isnan(pred_pre)]
    return ((pred_post - true_post) ** 2).mean() ** 0.5

def plot_accuracy_single(
    x,
    y,
    size=(8, 2, 20),
    x_label="Simulated",
    y_label="Inferred",
    log=False,
    r2=None,
    rho=None,
    rmse=None,
    c=None,
    title=None,
):
    """
    Plot a single x vs. y scatter plot panel, with correlation scores

    x, y = lists of x and y values to be plotted, e.g. true, pred
    size = [dots_size, line_width, font_size],
        e.g size = [8,2,20] for 4x4, size= [20,4,40] for 2x2
    log: if true will plot in log scale
    r2: r2 score for x and y
    rmse: rmse score for x and y
    rho: rho score for x and y
    c: if true will plot data points in a color range with color bar
    """
    font = {"size": size[2]}
    plt.rc("font", **font)
    ax = plt.gca()
    # make square plots with two axes the same size
    ax.set_aspect("equal", "box")

    # plot data points in a scatter plot
    if c is None:
        # 's' specifies dots size
        plt.scatter(x, y, s=size[0] * 2**3, alpha=0.8)

    # axis label texts
    plt.xlabel(x_label, labelpad=size[2] / 2)
    plt.ylabel(y_label, labelpad=size[2] / 2)
    
    # only plot in log scale if log specified for the param
    x=x.tolist()
    y=y.tolist()
    if log:
        plt.xscale("log")
        plt.yscale("log")
        # axis scales customized to data
        plt.xlim([min(x + y) * 10**-0.5, max(x + y) * 10**0.5])
        plt.ylim([min(x + y) * 10**-0.5, max(x + y) * 10**0.5])
        plt.xticks(ticks=[1e-2, 1e0, 1e2])
        plt.yticks(ticks=[1e-2, 1e0, 1e2])
        plt.minorticks_off()
    else:
        # axis scales customized to data
        if max(x + y) > 1:
            plt.xlim([min(x + y) - 0.5, max(x + y) + 0.5])
            plt.ylim([min(x + y) - 0.5, max(x + y) + 0.5])
        else:
            plt.xlim([min(x + y) - 0.05, max(x + y) + 0.05])
            plt.ylim([min(x + y) - 0.05, max(x + y) + 0.05])
    plt.tick_params("both", length=size[2] / 2, which="major")

    # plot a line of slope 1 (perfect correlation)
    intercept = 0
    slope = 1
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, linewidth=size[1] / 2, color="black", zorder=-100)

    # plot scores if specified
    if rho is not None:
        plt.text(
            # 0.25,
            # 0.82,
            0.55,
            0.92,
            "ρ: " + str(round(rho, 3)),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=size[2]-1.5,
            transform=ax.transAxes,
        )
    if rmse is not None:
        plt.text(
            0.6,
            0.08,
            "RMSE: " + str(round(rmse, 3)),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=size[2]-1.5,
            transform=ax.transAxes,
        )
    if title is not None:
        ax.text(0.05, 0.98, title, transform=ax.transAxes, va="top")
    plt.tight_layout()
    
def parse_results(file_path, reorder=True):
    """Read in prediction result text file
    Output: true and pred floats for N0 N1 N2 and T1"""
    
    with open(file_path) as fh:
        raw_results = []
        for line in fh:
            # Split the line by spaces
            values = line.split()   
            # Append the values to the array
            raw_results.append(values)

    # remove "vs" columns
    columns_to_exclude = [1, 4, 7, 10, 13]
    results = np.delete(np.array(raw_results), columns_to_exclude, axis=1)
    
    # separate true vs predict then reorder columns (if reorder=True)
    # original results order: N0 T1 N1 T2 N2 --> new order for plotting: N0 N1 N2 T1
    true = results.astype('float')[:, [0,2,4,6,8]]
    pred = results.astype('float')[:, [1,3,5,7,9]]
    
    if reorder:
        true_reorder = true[:, [0, 2, 4, 1]]
        pred_reorder = pred[:, [0, 2, 4, 1]]
    else:
        true_reorder = true
        pred_reorder = pred
        
    return true_reorder, pred_reorder
    
files = sorted(glob.glob("/xdisk/rgutenk/lnt/projects/ConfuseNN/demography/results/trained_models/convSize_2/rep_1/*_pred"))
files_reorder = np.array(files)[[0,2,1,3,5,4]] 
fig = plt.figure(1, figsize=(12, 16), dpi=150)

for j, filename in enumerate(files_reorder):
    true_reorder, pred_reorder = parse_results(filename)
    
    for i, param in enumerate(["$N_0$", "$N_1$", "$N_2$", "$T_1$"]):
        rho = spearmanr(true_reorder[:, i], pred_reorder[:, i])[0]
        rmse = root_mean_squared_error(true_reorder[:, i], pred_reorder[:, i])
        ax = fig.add_subplot(6,4,i+1+j*4, aspect='equal')
        ax = plot_accuracy_single(  true_reorder[:, i],
                                    pred_reorder[:, i],
                                    size=(0.25, 2, 15),
                                    x_label="Simulated" if j==5 else None,
                                    y_label="Inferred" if (i+j*4)%4==0 else None,
                                    log=False,
                                    r2=None,
                                    rho=rho,
                                    rmse=rmse,
                                    title=param,)

plt.savefig("flagel_all_accuracy.png", transparent=True, dpi=150)
    
    
