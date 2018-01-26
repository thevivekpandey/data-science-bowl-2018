import sys
import os
from itertools import izip
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from scipy.misc import imsave
from sklearn.model_selection import train_test_split
import random
import constants

class DataGeneratorB(object):
    def __init__(self, train_or_test):
        assert train_or_test in ('train', 'test')
        if train_or_test == 'train':
            self.init_erroneous_image_ids()
            self.load_train_data()
            self.X_train, self.X_validate, self.Y_train, self.Y_validate = train_test_split(self.X, self.Y, test_size=0.1, random_state=7)
        else:
            self.load_test_data()

    def init_erroneous_image_ids(self):
        self.image_ids_to_ignore = []
        f = open('erroneous_image_ids.txt')
        for line in f:
            self.image_ids_to_ignore.append(line.strip())
        f.close()

    def generator(self, train_or_val, batch_size):
        if train_or_val == 'train' or train_or_val == 'temp':
            X_local, Y_local = self.X_train, self.Y_train
        else:
            X_local, Y_local = self.X_val, self.Y_val
   
        while True:
            idxs, X_batch, Y_batch = [], [], []
            for i in range(batch_size):
                idx = random.randint(0, len(X_local) - 1)
                size_x, size_y = X_local[idx].shape[0], X_local[idx].shape[1]
                my_init_x = random.randint(0, size_x - constants.IMG_WIDTH) 
                my_init_y = random.randint(0, size_y - constants.IMG_HEIGHT)
                one_x = X_local[idx][my_init_x:my_init_x + constants.IMG_WIDTH, my_init_y: my_init_y + constants.IMG_HEIGHT, :]
                one_y = Y_local[idx][my_init_x:my_init_x + constants.IMG_WIDTH, my_init_y: my_init_y + constants.IMG_HEIGHT] / 255.0
                idxs.append(idx)
                X_batch.append(one_x)
                Y_batch.append(one_y)
                imsave('temp_images/x-' + str(i) + '.png', one_x)
                imsave('temp_images/y-' + str(i) + '.png', one_y.reshape(256, 256))
            if train_or_val == 'temp':
                yield idxs, np.array(X_batch), np.array(Y_batch)
            else:
                yield np.array(X_batch), np.array(Y_batch)

    def load_train_data(self):
        train_ids = next(os.walk(constants.TRAIN_PATH))[1][0:10]
        self.X, self.Y = [], []
        print('Getting train images and masks ... ')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            path = constants.TRAIN_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:,:,:constants.IMG_CHANNELS]
            self.X.append(img)
            mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file).reshape((img.shape[0], img.shape[1], 1))
                mask = np.maximum(mask, mask_)
            self.Y.append(mask)
        print('Done!')
        
    def load_test_data(self):
        self.test_ids = next(os.walk(constants.TEST_PATH))[1]
        # Get and resize test images
        self.X_test = np.zeros((len(self.test_ids), constants.IMG_HEIGHT, constants.IMG_WIDTH, constants.IMG_CHANNELS), dtype=np.uint8)
        self.sizes_test = []
        print('Getting and resizing test images ... ')
        for n, id_ in tqdm(enumerate(self.test_ids), total=len(self.test_ids)):
            path = constants.TEST_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:,:,:constants.IMG_CHANNELS]
            self.sizes_test.append([img.shape[0], img.shape[1]])
            img = resize(img, (constants.IMG_HEIGHT, constants.IMG_WIDTH), mode='constant', preserve_range=True)
            self.X_test[n] = img
        print('Done!')

    def get_test_data(self):
        return self.test_ids, self.X_test, self.sizes_test

    def get_train_data(self):
        return self.train_ids, self.X, self.sizes_train
