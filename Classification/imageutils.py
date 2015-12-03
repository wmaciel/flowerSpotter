from __future__ import division
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

from os.path import exists, isdir, basename, join, splitext
import sift
from glob import glob

EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]

def get_classes(base_dir):
    class_paths = [files for files in glob(base_dir + "/*") if isdir(files)]
    class_paths.sort()
    classes = [basename(class_path) for class_path in class_paths]
    return classes


def get_files(path):
    all_files = []
    all_files.extend([join(path, basename(filename)) for filename in glob(path + "/*") if splitext(filename)[-1].lower() in EXTENSIONS])
    return all_files


def load_data(base_dir):
    classes = get_classes(base_dir)

    class_dict = { class_name: index for (index, class_name) in enumerate(classes)}


    # Figure out the dimensions of the images (all assumed to be the same)
    first_class_path = join(base_dir, classes[0])
    first_file_path = get_files(first_class_path)[0]
    first_image = np.array(Image.open(first_file_path))

    # Number of rows (dim 0) is the height, and number of cols is the width
    image_height = first_image.shape[0]
    image_width = first_image.shape[1]


    xs = []
    ys = []

    for class_name in classes:
        class_path = join(base_dir, class_name)
        file_paths = get_files(class_path)

        # We look up the class number based on the name of the folder the image is in.
        # This maps a folder name like 'daffodil' to a class number like 0.
        class_index = class_dict[class_name]
    
        X = np.empty([len(file_paths), image_width, image_height, 3])
        y = []
    
        for (index, file_path) in enumerate(file_paths):
            I = np.array(Image.open(file_path))
            X[index] = I
            y.append(class_index)
    
        X_combined = X
        y_array = np.array(y)
        xs.append(X_combined)
        ys.append(y_array)
    
    X_all = np.concatenate(xs)
    y_all = np.concatenate(ys)

    return (X_all, y_all, classes, class_dict)

        