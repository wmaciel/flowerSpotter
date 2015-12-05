from __future__ import division

import math
import sys, glob, os
import Image


top_folder = sys.argv[1]
side_length = 256


input_image_dir = top_folder + '/jpg'
output_image_dir = top_folder + '/square_images'
#output_square_dir = 'square_images128'

if not os.path.isdir(output_image_dir):
    os.mkdir(output_image_dir)

for infile in glob.glob(input_image_dir + '/*.jpg'):
    filename, ext = os.path.splitext(infile)
    basename = os.path.basename(filename)

    im = Image.open(infile)

    [width, height] = im.size

    if width < height:
        ratio = height / width
        new_width = side_length
        new_height = int(math.ceil(ratio * new_width))
    else:
        ratio = width / height
        new_height = side_length
        new_width = int(math.ceil(ratio * new_height))

    resized_im = im.resize((new_width, new_height), Image.ANTIALIAS)

    if new_width != side_length:
        left_bound = int((new_width - side_length) / 2)
        upper_bound = 0
        right_bound = int((new_width + side_length) / 2)
        lower_bound = side_length
    else:
        left_bound = 0
        upper_bound = int((new_height - side_length) / 2)
        right_bound = side_length
        lower_bound = int((new_height + side_length) / 2)

    cropped_im = resized_im.crop((left_bound, upper_bound, right_bound, lower_bound))

    cropped_im.save(output_image_dir + '/' + basename + '_square.jpg', 'JPEG')