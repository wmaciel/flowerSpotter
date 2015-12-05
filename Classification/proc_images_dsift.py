from os.path import exists, isdir, basename, join, splitext
import dsift
from glob import glob

EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]

size = 20
step = 10

def get_categories(datasetpath):
    cat_paths = [files for files in glob(datasetpath + "/*") if isdir(files)]
    cat_paths.sort()
    cats = [basename(cat_path) for cat_path in cat_paths]
    return cats


def get_imgfiles(path):
    all_files = []
    all_files.extend([join(path, basename(fname)) for fname in glob(path + "/*") if splitext(fname)[-1].lower() in EXTENSIONS])
    return all_files


dataset_path = '../flower_rec1/square_images128_dsift/test'
categories = get_categories(dataset_path)

for category in categories:
    category_path = join(dataset_path, category)
    image_file_list = get_imgfiles(category_path)

    num_images = len(image_file_list)
    feature_file_list = [image_file_list[i][:-3]+"dsift_{0}_{1}".format(size,step) for i in range(num_images)]

    for i in range(num_images):
        dsift.process_image_dsift(image_file_list[i], feature_file_list[i], size, step)
