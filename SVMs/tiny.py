from PIL import Image
import glob, os, sys


top_folder = sys.argv[1]
size = 32, 32

input_image_dir = top_folder + '/jpg'
output_image_dir = top_folder + '/tiny_images_32'

if not os.path.isdir(output_image_dir):
    os.mkdir(output_image_dir)

for infile in glob.glob(input_image_dir + '/*.jpg'):
    filename, ext = os.path.splitext(infile)
    basename = os.path.basename(filename)
    im = Image.open(infile)
    resized_im = im.resize(size)
    resized_im.save(output_image_dir + '/' + basename + '_tiny.jpg', 'JPEG')