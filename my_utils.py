import numpy as np
import pandas as pd
from skimage.io import imread

def distance(p1, p2):
    x, y = p1
    a, b = p2
    return (x - a) * (x - a) + (y - b) * (y - b)

def get_image_names_from_rle_file(fname):
    images = set()
    f = open(fname)
    for line in f:
        if 'ImageId' in line:
            continue
        img, _ = line.strip().split(',')
        images.add(img)
    f.close()
    return images

def get_image_sizes(images, mode):
    sizes = {}
    for image in images:
        x = imread('../input/stage1_' + mode + '/' + image + '/images/' + image + '.png')
        sizes[image] = (x.shape[0], x.shape[1])
    return sizes

def get_train_image_sizes(images):
    return get_image_sizes(images, 'train')

def get_test_image_sizes(images):
    return get_image_sizes(images, 'test')

def get_one_img_2_masks(pixel_array, size):
    w, h = size
    masks = np.zeros((w, h, len(pixel_array)), dtype=np.uint8)
    for id, one_array in enumerate(pixel_array):
        for elem in one_array:
            ox = (elem - 1) % w
            oy = (elem - 1) // w
            masks[ox, oy, id] = 1
    return masks

def get_img_2_masks(img_2_pixels, sizes):
    img_2_masks = {}
    for img, pixel_array in img_2_pixels.items():
        img_2_masks[img] = get_one_img_2_masks(pixel_array, sizes[img])
    return img_2_masks

def write_rle_to_file(test_ids, rles, filename):
    sub = pd.DataFrame()
    sub['ImageId'] = test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(filename, index=False)

def get_img_2_pixels_from_rle_file(input_rle_file):
    #key is image, value is an array of sets. One set for each mask 
    img_2_pixels = {}
    fin = open(input_rle_file)
    for line in fin:
        if 'ImageId' in line:
            continue
        img, rle = line.strip().split(',')
        offsets, limits = [], []
        for idx, part in enumerate(rle.split()):
           if idx % 2 == 0:
               offsets.append(int(part))
           else:
               limits.append(int(part))
        img_2_pixels.setdefault(img, [])
        pixels = set()
        for i in range(len(offsets)):
            img_2_pixels[img]
            offset = offsets[i]
            limit = limits[i]
            for j in range(limit):
                pixels.add(offset + j)
        img_2_pixels[img].append(pixels)
    fin.close()
    return img_2_pixels

