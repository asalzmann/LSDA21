import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import json

import h5py


def hog_dataset(Xtrain, Xtest):
    Xtrain_hog = []
    Xdev_hog = []

    size = 56

    for idx, x in enumerate(Xtrain):
        x = x.reshape(size, size)
        xg = hog(x, orientations=8, pixels_per_cell=(8, 8),
                 cells_per_block=(1, 1), feature_vector=True)
        Xtrain_hog.append(xg)
        del xg
        if idx % 10000 == 0:
            print(idx)
    Xtrain_hog = np.array(Xtrain_hog).astype('float32')
    print(Xtrain_hog.data.shape)

    for idx, x in enumerate(Xtest):
        x = x.reshape(size, size)
        xg = hog(x, orientations=8, pixels_per_cell=(8, 8),
                 cells_per_block=(1, 1), feature_vector=True)
        Xdev_hog.append(xg)
        del xg
        if idx % 1000 == 0:
            print(idx)
    Xdev_hog = np.array(Xdev_hog).astype('float32')
    print(Xdev_hog.data.shape)

    return Xtrain_hog, Xdev_hog


def predict_test_set(Xtrain_hog, ytrain, Xtest_hog, n_components, n_neighbors):
    pca = PCA(n_components=n_components)
    pca.fit(Xtrain_hog)
    Xtrain_pca = pca.transform(Xtrain_hog)
    Xdev_pca = pca.transform(Xtest_hog)
    print("n_components", n_components)
    print("Captured variance", np.cumsum(pca.explained_variance_ratio_[:n_components])[-1])

    model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='kd_tree')
    model.fit(Xtrain_pca, ytrain)

    # compute accuracy
    preds = model.predict(Xdev_pca)

    image_ids = np.arange(len(preds)) + 1

    submission_format = (np.column_stack((image_ids, preds))).astype(int)
    np.savetxt("knn_56_sub.csv", submission_format, delimiter=',', fmt='%i', header="id,predicted", comments="")
    print("Predictions saved to file!")


def acc_on_dev_set(Xtrain_hog, ytrain, Xval_hog, yval, n_components, n_neighbors):
    pca = PCA(n_components=n_components)
    pca.fit(Xtrain_hog)
    Xtrain_pca = pca.transform(Xtrain_hog)
    Xdev_pca = pca.transform(Xval_hog)
    print("n_components", n_components)
    print("Captured variance", np.cumsum(pca.explained_variance_ratio_[:n_components])[-1])

    model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='kd_tree')
    model.fit(Xtrain_pca, ytrain)

    # compute accuracy
    preds = model.predict(Xdev_pca)
    acc = accuracy_score(yval, preds)
    print("Final accuracy: {}".format(acc), "n_neighbors", n_neighbors)


def pca_grid_search(Xtrain, ytrain, n_components):
    Xtrain_split, Xdev, ytrain_split, ydev = train_test_split(Xtrain, ytrain, test_size=0.03, random_state=42)

    Xtrain_hog, Xdev_hog = hog_dataset(Xtrain_split, Xdev)

    for k in range(15):
        pca = PCA(n_components=n_components)
        pca.fit(Xtrain_hog)
        Xtrain_pca = pca.transform(Xtrain_hog)
        Xdev_pca = pca.transform(Xdev_hog)
        print("n_components", n_components)
        print("Captured variance", np.cumsum(pca.explained_variance_ratio_[:n_components])[-1])
        n_neighbors = 3
        for i in range(2):
            model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='kd_tree')
            model.fit(Xtrain_pca, ytrain_split)

            # compute accuracy
            preds = model.predict(Xdev_pca)
            acc = accuracy_score(ydev, preds)
            print("Final accuracy: {}".format(acc), "n_neighbors", n_neighbors)

            n_neighbors = n_neighbors + 2

        n_components = n_components + 1


if __name__ == "__main__":
    d_train = h5py.File('train_56.h5', 'r')
    d_test = h5py.File('test_56.h5', 'r')
    # d_val = h5py.File('validation_56.h5', 'r')

    Xtrain = np.array(d_train['train']['images'])
    ytrain = np.array(d_train['train']['labels'])

    # Xval = d_val['validation']['images']
    # yval = d_val['validation']['labels']

    Xtest = np.array(d_test['test']['images'])

    n_components = 90
    n_neighbors = 5

    # pca_grid_search(Xtrain, ytrain, n_components)

    # Xtrain_hog, Xval_hog = hog_dataset(Xtrain, Xval)
    # acc_on_dev_set(Xtrain_hog, ytrain, Xval_hog, yval, n_components, n_neighbors)

    Xtrain_hog, Xtest_hog = hog_dataset(Xtrain, Xtest)
    predict_test_set(Xtrain_hog, ytrain, Xtest_hog, n_components, n_neighbors)





