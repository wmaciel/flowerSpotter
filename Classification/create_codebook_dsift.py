from os.path import exists, isdir, basename, join, splitext
import sift
from glob import glob
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
import scipy.cluster.vq as vq
from cPickle import dump, HIGHEST_PROTOCOL
import numpy as np


size = 20
step = 10
dataset_path = '../flower_rec1/square_images128_dsift/train'
num_clusters = 300
K_THRESH = 1
codebook_file = "codebook_dsift_{0}_{1}_{2}.file".format(size, step, num_clusters)


def get_categories(datasetpath):
    cat_paths = [files for files in glob(datasetpath + "/*") if isdir(files)]
    cat_paths.sort()
    cats = [basename(cat_path) for cat_path in cat_paths]
    return cats


def get_dsift_files(path):
    all_files = []
    all_files.extend([join(path, basename(fname)) for fname in glob(path + "/*") if splitext(fname)[-1].lower() == ".dsift_{0}_{1}".format(size,step)])
    return all_files


def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
    return histogram_of_words



categories = get_categories(dataset_path)

all_sift_files = []

for category in categories:
    category_path = join(dataset_path, category)
    sift_file_list = get_dsift_files(category_path)
    all_sift_files += sift_file_list


all_sift_files = sorted(all_sift_files)
print(len(all_sift_files))


file_descriptors = dict()
descriptors = []

for sift_file in all_sift_files:
    desc = sift.read_features_from_file(sift_file)[1]
    descriptors.append(desc)
    file_descriptors[sift_file] = desc


all_sift_features = np.vstack(descriptors)

print(all_sift_features.shape)

codebook, distortion = vq.kmeans(all_sift_features, num_clusters, thresh=K_THRESH)

with open(codebook_file, 'wb') as f:
    dump(codebook, f, protocol=HIGHEST_PROTOCOL)

