# reproduce multi analysis, effect of demographic model

# train on 3 different models, test on 3-epoch

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

exec(open('/xdisk/rgutenk/lnt/projects/ConfuseNN/ImaGene/scripts/ImaGene_scramble.py').read())

import pathlib

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys

e = str(sys.argv[1]) # epoch
m = 'RowsCols' 

folder = '/xdisk/rgutenk/lnt/projects/ConfuseNN/ImaGene/results/trained_models/Multi/Epoch' + str(e)
print(folder)
pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

i = 1
while i <= 10:

    if i < 10:
        myfile = ImaFile(
            simulations_folder='/xdisk/rgutenk/lnt/projects/ConfuseNN/ImaGene/results/simulations/Binary/Simulations' + str(i) + '.Epoch' + str(e), 
                    nr_samples=128, model_name='Marth-' + str(e) + 'epoch-CEU')
    else:
        # take 3-epoch for testing
        myfile = ImaFile(
            simulations_folder='/xdisk/rgutenk/lnt/projects/ConfuseNN/ImaGene/results/simulations/Binary/Simulations' + str(i) + '.Epoch' + str(3), 
                    nr_samples=128, model_name='Marth-' + str(3) + 'epoch-CEU')
   
    mygene = myfile.read_simulations(parameter_name='selection_coeff_hetero', max_nrepl=5000)

    mygene.majorminor()
    mygene.filter_freq(0.01)
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

    # first iteration
    if i == 1:

        model = models.Sequential([
                    layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005), padding='valid', input_shape=mygene.data.shape[1:4]),
                    layers.MaxPooling2D(pool_size=(2,2)),
                    layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005), padding='valid'),
                    layers.MaxPooling2D(pool_size=(2,2)),
                    layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005), padding='valid'),
                    layers.MaxPooling2D(pool_size=(2,2)),
                    layers.Flatten(),
                    layers.Dense(units=len(mygene.classes), activation='softmax')])
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        plot_model(model, folder + '/model.png')

        mynet = ImaNet(name='[C32+P]+[C64+P]x2')

    # training
    if i < 10:
        score = model.fit(mygene.data, mygene.targets, batch_size=32, epochs=1, verbose=0, validation_split=0.10)
        mynet.update_scores(score)
    else:
        # testing
        mynet.test = model.evaluate(mygene.data, mygene.targets, batch_size=None, verbose=0)
        mynet.predict(mygene, model)

    i += 1

# save final (trained) model
model.save(folder + '/model.h5')

# save testing data
mygene.save(folder + '/mygene')

# save final network
mynet.save(folder + '/mynet')

print(mynet.test)

