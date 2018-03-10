import sys
import os
import numpy as np
import random
from skimage.io import imread, imshow, imsave

COLORS = ['f06292', '7986cb', '4dd0e1', 'aed581', 'ffd54f', 'ff8a65', 'e040fb']

def get_random_rgb():
    color = COLORS[random.randint(0, len(COLORS) - 1)]
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)

def get_images(fname):
    images = set()
    f = open(fname)
    for line in f:
        if 'ImageId' in line:
            continue
        img, _ = line.strip().split(',')
        images.add(img)
    f.close()
    return images

def get_rles(fname):
    #key is image, value is dict.
    # Each value has two keys, offsets and limits
    # Each of the two values is an array of arrays, each array
    # corresponding to one mask
    image_2_rles = {}
    f = open(fname)
    for line in f:
        if 'ImageId' in line:
            continue
        img, rle = line.strip().split(',')
        offsets, limits = [], []
        for idx, part in enumerate(rle.split()):
           if idx % 2 == 0:
               offsets.append(int(part))
           else:
               limits.append(int(part))
        image_2_rles.setdefault(img, {})
        image_2_rles[img].setdefault('offsets', []).append(offsets)
        image_2_rles[img].setdefault('limits', []).append(limits)
    f.close()
    return image_2_rles

def get_image_sizes(images):
    sizes = {}
    for image in images:
        x = imread('../input/stage1_train/' + image + '/images/' + image + '.png')
        sizes[image] = (x.shape[0], x.shape[1])
    return sizes

def find_mask_image(size, rle_list):
    w, h = size #That's how skimage read/writes image: first dimension is height
    img = np.zeros((w, h, 3), dtype=np.uint8)
    offset_arrays = rle_list['offsets']
    limit_arrays  = rle_list['limits']
    assert len(offset_arrays) == len(limit_arrays)

    # one loop for one mask
    for i in range(len(offset_arrays)):
        r, g, b = get_random_rgb()
        offsets = offset_arrays[i]
        limits = limit_arrays[i]
        assert len(offsets) == len(limits)
        for j in range(len(offsets)):
            offset = offsets[j]
            limit = limits[j] 
            for k in range(limit):
                ox = (offset + k - 1) % w
                oy = (offset + k - 1) // w
                #print(ox, oy, r, g, b)
                try:
                    img[ox,oy] = [r, g, b]
                except:
                    print(offset, k, ox, oy, 'for size', w, h)
                    #print(offsets)
                    #print(limits)
    return img

rle_fname = sys.argv[1]
images = get_images(rle_fname)
sizes = get_image_sizes(images)
rle_lists = get_rles(rle_fname)

directory = rle_fname.split('.')[0]
if not os.path.exists(directory):
    os.makedirs(directory)

for image in images:
    size = sizes[image]
    rle_list = rle_lists[image]
    output_image = find_mask_image(size, rle_list)
    imsave(directory + '/' + image + '.png', output_image)
    print('processed ' + image)
