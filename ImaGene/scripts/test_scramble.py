import os
import gzip
import _pickle as pickle

import numpy as np
import scipy.stats

import skimage.transform
from keras import models, layers, activations, optimizers, regularizers
from keras.utils import plot_model
from keras.models import load_model

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

exec(open('/xdisk/rgutenk/lnt/projects/ConfuseNN/ImaGene/scripts/ImaGene_scramble.py').read())

import pathlib
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys

scramble_mode_idx = int(sys.argv[1]) - 1
scramble_modes = ['col', 'colI', 'free', 'right', 'left']

# folder to save processed scrambled data and results
folder = '/xdisk/rgutenk/lnt/projects/ConfuseNN/ImaGene/results/scramble_tests'
print(folder)
pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

# load and process scramble test data (epoch 3)
e=str(3)
i=10 # simulation 10 is used for test (1-9 were used for training)
m = 'RowsCols' 

myfile = ImaFile(simulations_folder='/xdisk/rgutenk/lnt/projects/ConfuseNN/ImaGene/results/simulations/Binary/Simulations' + str(i) + '.Epoch' + str(3), 
                    nr_samples=128, model_name='Marth-' + str(3) + 'epoch-CEU')
mygene = myfile.read_simulations(parameter_name='selection_coeff_hetero', max_nrepl=5000)
    
# first process as the original paper: polarizing by freq and filter
mygene.majorminor()
mygene.filter_freq(0.01)

# SCRAMBLE HERE
mygene.scramble(scramble_modes[scramble_mode_idx])

# then process as the original paper for the rest of the steps
if (m =='Rows') | (m == 'RowsCols'):
    mygene.sort('rows_freq')
if (m =='Cols') | (m == 'RowsCols'):
    mygene.sort('cols_freq')
mygene.resize((128, 128))
mygene.convert()

mygene.classes = np.array([0,200,400])
mygene.subset(get_index_classes(mygene.targets, mygene.classes))

mygene.subset(get_index_random(mygene))

mygene.targets = to_categorical(mygene.targets)

# save new scramble testing data
mygene.save(folder + '/mygene_scramble_' + scramble_modes[scramble_mode_idx] + '_epoch_3')

# generate results

fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
c = -1
classes = [0,200,400]
for e in [3]: # test only CNN trained on 3-epoch data
    cnn_folder = '/xdisk/rgutenk/lnt/projects/ConfuseNN/ImaGene/results/trained_models/Multi/Epoch' + str(e)
    
    # load previously trained model(s) and my net results
    model = load_model(cnn_folder + '/model.h5')
    mynet = load_imanet(cnn_folder + '/mynet')
    
    # test on new scramble data and update mynet values with predict()
    mynet.test = model.evaluate(mygene.data, mygene.targets, batch_size=None, verbose=0)
    mynet.predict(mygene, model)
    
    # save new network output
    mynet.save(folder + '/mynet_scramble_' + scramble_modes[scramble_mode_idx] + '_cnn_epoch_' + str(e))

    # print and plot results
    c += 1
    print(mynet.test)
    cm = confusion_matrix(mynet.values[0,:], mynet.values[1,:])
    accuracy = np.trace(cm) / float(np.sum(cm))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax[c].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax[c].set_xticks(np.arange(len(classes)))
    ax[c].set_yticks(np.arange(len(classes)))
    ax[c].set_xticklabels(classes)
    ax[c].set_yticklabels(classes)
    ax[c].set_title('Trained on ' + str(e) + '-epoch model')
    ax[c].set_xlabel('Predicted')
    if c == 0:
        ax[c].set_ylabel('Tested on 3-epoch model\nTrue')
    else:
        ax[c].set_ylabel('')
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax[c].text(j, i, "{:0.3f}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

fig.show()
plt.savefig(fname = folder + '/Figure_scramble_' + scramble_modes[scramble_mode_idx] + '.pdf')
