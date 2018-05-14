#!/usr/bin/python
# -*- coding: utf-8 -

# Multilayer perceptron network.
# Requisites
import os, sys
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten, MaxPooling2D, Reshape, InputLayer
from PIL import Image


# Set seed value to stop randomness
seed = 128
rng = np.random.RandomState(seed)

# Set paths
root_dir = "./"
data_dir = "./data"
train_thumbnails = "./data/thumbnails/train/"

train = pd.read_csv(root_dir + 'labels_train.csv')

#Check image sizes

size = (128, 128)

#Create thumbnails of images
for img_name in train.filename:
    outfile = train_thumbnails+img_name
    if img_name != outfile:
        try:
            im = Image.open(data_dir+img_name)
            resized = im.resize(size, Image.ANTIALIAS)
            resized.save(outfile, "JPEG")
        except IOError:
            print ("cannot create thumbnail for '%s'" % img_name)

for img_name in train.filename:
    im = Image.open(train_thumbnails + img_name)
    print (img_name, im.format, "%dx%d" % im.size, im.mode)
    #try:

    #except IOError:
    #    pass