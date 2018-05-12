import json
import numpy as np
from PIL import Image, ImageOps
from os import listdir
import os
import matplotlib.pyplot as plt
import h5py

# dataset_path = "validation/valid/"
# json_data_file = "train.json"
# json_cleared_file = 'validation_clean.json'
# json_obj = "annotations"
# extension = ".jpeg"

dataset_path = "test/"
json_data_file = "test.json"
json_obj = "images"
extension = ".jpg"


def read_json_file(file):
    available_image_ids = np.array(listdir(dataset_path))
    available_image_ids_set = set(available_image_ids)

    with open(file) as json_data:
        data = json.load(json_data)
        new_data = {json_obj: []}
        for i, doc in enumerate(data[json_obj]):
            image_id = str(doc["image_id"]) + extension
            if image_id in available_image_ids_set:
                new_data[json_obj].append(doc)

            if i % 500 == 0:
                print("Processed", i)
        with open(json_cleared_file, 'w') as outfile:
            json.dump(new_data, outfile, indent=4)


def load_resize_image(filename, size):
    img = Image.open(filename)
    img.load()
    img = img.resize((size, size), resample=Image.LANCZOS)
    img = ImageOps.grayscale(img)

    data = np.asarray(img, dtype="uint8")
    del img
    data = data.reshape(-1)
    return data


def save_file(set, X, y, id):
    '''
        set in {"train","test"}
        id = file number
        X = train_images...
        y = train_labels...
    '''
    with h5py.File('{}_{}.h5'.format(set, id), 'w') as hf:
        group = hf.create_group(set)
        group.create_dataset('images', data=X)
        group.create_dataset('labels', data=y)


def save_file_test(set, X, id):
    with h5py.File('{}_{}.h5'.format(set, id), 'w') as hf:
        group = hf.create_group(set)
        group.create_dataset('images', data=X)


def images2data(file, size):
    with open(file) as json_data:
        data = json.load(json_data)
        len_data_annotations = len(data[json_obj])

        images = np.empty((len_data_annotations, size * size))
        labels = np.empty((len_data_annotations,))

        for i, doc in enumerate(data[json_obj]):
            images[i] = load_resize_image(dataset_path + str(doc["image_id"]) + extension, size)
            labels[i] = int(doc["label_id"])
            if i % 500 == 0:
                print("Processed", i, "/", len_data_annotations)

        return images, labels


def images2data_test(file, size):
    with open(file) as json_data:
        data = json.load(json_data)
        len_data_annotations = len(data[json_obj])

        images = np.empty((len_data_annotations, size * size))

        for i, doc in enumerate(data[json_obj]):
            try:
                images[i] = load_resize_image(dataset_path + str(doc["image_id"]) + extension, size)
            except:
                images[i] = np.zeros(size * size)
            finally:
                if i % 500 == 0:
                    print("Processed", i, "/", len_data_annotations)

        return images


if __name__ == "__main__":
    # Create a json file with info about the available images at dataset_path
    # read_json_file(json_data_file)

    size = 28

    # # Process train data
    # images, labels = images2data(json_cleared_file, size)
    # save_file("validation", images, labels, size)

    # Process test data
    images = images2data_test(json_data_file, size)
    save_file_test("test", images, 3)

    # Example of reading the h5py files
    # d_val = h5py.File('test_2.h5', 'r')
    # labels = np.array(d_val['validation']['labels'])
    # images = np.array(d_val['test']['images'])
