#!/usr/bin/python
# -*- coding: utf-8 -

# CNN built with Keras, trained with 128x128 RGB images
# Requisites
import os, sys
import h5py
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D, Reshape, InputLayer
from PIL import Image


# Set seed value to stop randomness
seed = 128
rng = np.random.RandomState(seed)

# Set paths
# Set paths
root_dir = "./"
train_labels = root_dir + "labels_train.csv"
valid_labels = root_dir + "labels_valid.csv"
test_labels = root_dir + "labels_test.csv"

print("Getting training and validation labels")
#Train and validation labels
train_y = keras.utils.np_utils.to_categorical(train[["labels"]])

val_y = keras.utils.np_utils.to_categorical(valid[["labels"]])


#get number of labels to predict
print(val_y.shape)
unique_labels = val_y.shape[1]

train_thumbnails = root_dir+"thumbnails/train/"
valid_thumbnails = root_dir+"thumbnails/valid/"
test_thumbnails = root_dir+"thumbnails/test/"

# read test and train files and convert them

train = pd.read_csv(train_labels)
valid = pd.read_csv(valid_labels)
test = pd.read_csv(test_labels)

temp = []
for img_name in test.filename:
  image_path = test_thumbnails + img_name
  img = imread(image_path, flatten=False)
  img = img.astype('float32')
  temp.append(img)

test_x = np.stack(temp)

test_x /= 255.0

print("Done.\nGetting the train set")
temp = []
for img_name in train.filename:
  image_path = train_thumbnails + img_name
  img = imread(image_path, flatten=False)
  img = img.astype('float32')
  temp.append(img)

train_x = np.stack(temp)

train_x /= 255.0
#train_x = train_x.reshape(train_x.shape[0],-1).astype('float32')

print(train_x.shape)

print("Done.\nGetting the validation set")

temp = []
for img_name in valid.filename:
  image_path = valid_thumbnails + img_name
  img = imread(image_path, flatten=False)
  img = img.astype('float32')
  temp.append(img)

val_x = np.stack(temp)

val_x /= 255.0



#Set parameters of the neural network and hidden layers. This one is wide:

input_shape = train_x.shape[1:]
output_num_units = unique_labels
epochs = 50
batch_size = 128
dropout_ratio = 0.2

print("Done.\nCreating model")

# Deep and wide model.

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(input_shape)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(output_num_units))
model.add(Activation('softmax'))

print("Done.\nCompiling model")

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Done.\nFitting model")

model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))

print("Done.\nSaving model")

preds = model.predict(test_x,batch_size=batch_size)

model.save(root_dir + "cnn_model")

np.savetxt(root_dir+"preds.csv", delimiter=',')

print("Done. Exiting now")

#model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

#trained_model_deep_n_wide = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))

