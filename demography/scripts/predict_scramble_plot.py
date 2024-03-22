import sys, os
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.pooling import MaxPooling2D, AveragePooling1D
from keras import backend as K
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

npzFileName, outFileName, weightFileName, testPredFileName, testPlotPrefix, convSize = sys.argv[1:]
sortRows = True
useLog = True
useInt = True

def resort_min_diff(amat):
    ###assumes your snp matrix is indv. on rows, snps on cols
    mb = NearestNeighbors(len(amat), metric='manhattan').fit(amat)
    v = mb.kneighbors(amat)
    smallest = np.argmin(v[0].sum(axis=1))
    return amat[v[1][smallest]]

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
            0.25,
            0.82,
            "p: " + str(round(rho, 3)),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
    if rmse is not None:
        plt.text(
            0.7,
            0.08,
            "rmse: " + str(round(rmse, 3)),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=size[2],
            transform=ax.transAxes,
        )
    if title is not None:
        ax.text(0.05, 0.98, title, transform=ax.transAxes, va="top")
    plt.tight_layout()

X = []
y = []
print("reading data")

# for npzFileName in os.listdir(inDir):
u = np.load(npzFileName)
currX, curry = [u[i] for i in  ('X', 'y')]
ni,nr,nc = currX[:10000].shape # this should be (10000, 51, 729)
newCurrX = []
for i in range(ni):
    currCurrX = [currX[i,0]]
    if sortRows:
        currCurrX.extend(resort_min_diff(currX[i,1:]))
    else:
        currCurrX.extend(currX[i,1:])
    currCurrX = np.array(currCurrX)
    newCurrX.append(currCurrX.T)
currX = np.array(newCurrX)
assert currX.shape == (ni,nc,nr)
X.extend(currX)
y.extend(curry)

y = np.array(y)
print(y.shape)
numParams=y.shape[1]
if useLog:
    y[y == 0] = 1e-6#give zeros a very tiny value so they don't break our log scaling
    y = np.log(y)

print("formatting data arrays")
X = np.array(X)
posX=X[:,:,0]
X=X[:,:,1:]
imgRows, imgCols = X.shape[1:]

print(imgRows) # 729
print(imgCols) # 50
print(X.shape) # (10000, 729, 50)
print(posX.shape) # (10000, 729)

if useInt:
    X = X.astype('int8') 
    #this ^^^ is a bug. the intent was to converts to 0/255, but as it it converts to 0/-1
    #see: https://github.com/flag0010/pop_gen_cnn/issues/4
    #leaving as is so the code represents what we actually did. Bugs and all
    #if you want to fix to get 0/255, please see the suggestion at the link above
else:
    X = X.astype('float32')/127.5-1

posX = posX.astype('float32')/127.5-1

# save output after sorting for genetic similarity
np.savez_compressed(outFileName, X=X, posX=posX)

yMeans=np.mean(y, axis=0)
yStds=np.std(y, axis=0)
y = (y-yMeans)/yStds

################ CNN
inputShape = (imgRows, imgCols)
# inputShape = (729, 50)
convFunc = Conv1D
poolFunc = AveragePooling1D

useDropout = True
convSize = int(convSize)
poolSize = 2

b1 = Input(shape=inputShape)
conv11 = convFunc(128, kernel_size=convSize, activation='relu')(b1)
pool11 = poolFunc(pool_size=poolSize)(conv11)
if useDropout:
    pool11 = Dropout(0.25)(pool11)
conv12 = convFunc(128, kernel_size=2, activation='relu')(pool11)
pool12 = poolFunc(pool_size=poolSize)(conv12)
if useDropout:
    pool12 = Dropout(0.25)(pool12)
conv13 = convFunc(128, kernel_size=2, activation='relu')(pool12)
pool13 = poolFunc(pool_size=poolSize)(conv13)
if useDropout:
    pool13 = Dropout(0.25)(pool13)
conv14 = convFunc(128, kernel_size=2, activation='relu')(pool13)
pool14 = poolFunc(pool_size=poolSize)(conv14)
if useDropout:
    pool14 = Dropout(0.25)(pool14)
flat11 = Flatten()(pool14)

b2 = Input(shape=(imgRows,))
dense21 = Dense(32, activation='relu')(b2)
if useDropout:
    dense21 = Dropout(0.25)(dense21)

merged = concatenate([flat11, dense21])
denseMerged = Dense(256, activation='relu', kernel_initializer='normal')(merged)
if useDropout:
    denseMerged = Dropout(0.25)(denseMerged)
denseOutput = Dense(numParams)(denseMerged)

model = Model(inputs=[b1, b2], outputs=denseOutput)
print(model.summary())

# load the weights for the previously trained model
# weightFileName='results/trained_weight'
model.load_weights(weightFileName)
model.compile(loss='mean_squared_error', optimizer='adam')

# get the loss for trained model on the scrambled test set and emit predictions
testLoss = model.evaluate([X, posX], y)
print(testLoss)

preds = model.predict([X, posX])

with open(testPredFileName, "w") as outFile:
    for i in range(len(preds)):
        outStr = []
        for j in range(len(preds[i])):
            outStr.append("%f vs %f" %(y[i][j], preds[i][j]))
        outFile.write("\t".join(outStr) + "\n")

for i, param in enumerate(['n0', 't1', 'n1', 't2', 'n2']):
    rho = spearmanr(y[:, i], preds[:, i])[0]
    rmse = root_mean_squared_error(y[:, i], preds[:, i])
    fig = plt.figure()
    plot_accuracy_single(
    y[:, i],
    preds[:, i],
    size=(1, 2, 20),
    log=False,
    r2=None,
    rho=rho,
    rmse=rmse,
    title=param,)
    
    plot_name = testPlotPrefix + '_' + param + '_accuracy.png'
    plt.savefig(plot_name)
