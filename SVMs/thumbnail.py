import Image
import os, sys
from glob import glob


top_folder = sys.argv[1]
size = 256, 256

input_image_dir = top_folder + '/jpg'
output_image_dir = top_folder + '/thumbnails'

if not os.path.isdir(output_image_dir):
    os.mkdir(output_image_dir)

for infile in glob(input_image_dir + '/*.jpg'):
    filename, ext = os.path.splitext(infile)
    basename = os.path.basename(filename)
    im = Image.open(infile)
    im.thumbnail(size)
    im.save(output_image_dir + '/' + basename + '_thumb.jpg', 'JPEG')





