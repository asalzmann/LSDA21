#!/usr/bin/python
# -*- coding: utf-8 -

# Multilayer perceptron network.
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
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten, MaxPooling2D, Reshape, InputLayer
from PIL import Image


# Set seed value to stop randomness
seed = 128
rng = np.random.RandomState(seed)

# Set paths
# Set paths
root_dir = "/home/people/joserh/C1/data/"
train_labels = root_dir + "labels_train.csv"
valid_labels = root_dir + "labels_valid.csv"

# read test and train files and convert them

train = pd.read_csv(train_labels)
valid = pd.read_csv(valid_labels)

#sample_submission = pd.read_csv(os.path.join(data_dir, 'Sample_Submission.csv'))

#Check image sizes
# for img_name in train.filename:
#     try:
#         im = Image.open(img_name)
#         print img_name, im.format, "%dx%d" % im.size, im.mode
#     except IOError:
#         pass

print("Getting training and validation labels")
#Train and validation labels
train_y = keras.utils.np_utils.to_categorical(train[["labels"]])

val_y = keras.utils.np_utils.to_categorical(valid[["labels"]])


#get number of labels to predict
print(val_y.shape)
unique_labels = val_y.shape[1]


train_thumbnails = root_dir+"thumbnails/train/"
valid_thumbnails = root_dir+"thumbnails/valid/"

print("Done.\nGetting the train set")
temp = []
for img_name in train.filename:
  image_path = train_thumbnails + img_name
  img = imread(image_path, flatten=True)
  img = img.astype('float32')
  temp.append(img)

train_x = np.stack(temp)

#train_x /= 255.0
train_x = train_x.reshape(train_x.shape[0],-1).astype('float32')

print(train_x.shape)

print("Done.\nGetting the validation set")

temp = []
for img_name in valid.filename:
  image_path = valid_thumbnails + img_name
  img = imread(image_path, flatten=True)
  img = img.astype('float32')
  temp.append(img)

val_x = np.stack(temp)

#valid_x /= 255.0
val_x = val_x.reshape(val_x.shape[0],-1).astype('float32')

#Set parameters of the neural network and hidden layers. This one is wide:

input_num_units = train_x.shape[1]
hidden_num_units = 500
output_num_units = unique_labels
batch_size = 128

# # Add dropout and increase epochs

# epochs = 50
# dropout_ratio = 0.2

# model = Sequential([
#  Dense(output_dim=hidden1_num_units, input_dim=input_num_units, activation='relu'),
#  Dropout(dropout_ratio),
#  Dense(output_dim=hidden2_num_units, input_dim=hidden1_num_units, activation='relu'),
#  Dropout(dropout_ratio),
# Dense(output_dim=output_num_units, input_dim=hidden5_num_units, activation='softmax'),
#  ])

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# trained_model_5d_with_drop = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))

# model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

# trained_model_5d_with_drop = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))

print("Done.\nCreating model")

# Deep and wide model.

input_num_units = train_x.shape[1]
hidden1_num_units = 100
hidden2_num_units = 100
hidden3_num_units = 100
hidden4_num_units = 100
hidden5_num_units = 100
output_num_units = unique_labels

epochs = 500
batch_size = 128
dropout_ratio = 0.2

model = Sequential([
 Dense(output_dim=hidden1_num_units, input_dim=input_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden2_num_units, input_dim=hidden1_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden3_num_units, input_dim=hidden2_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden4_num_units, input_dim=hidden3_num_units, activation='relu'),
 Dropout(dropout_ratio),
 Dense(output_dim=hidden5_num_units, input_dim=hidden4_num_units, activation='relu'),
 Dropout(dropout_ratio),

Dense(output_dim=output_num_units, input_dim=hidden5_num_units, activation='softmax'),
 ])

print("Done.\nCompiling model")

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Done.\nFitting model")

model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))

print("Done.\nSaving model")

model.save(data_dir + "deep_wide_model")

print("Done. Exiting now")

#model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

#trained_model_deep_n_wide = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))

