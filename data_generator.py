import sys
import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import constants

class DataGenerator(object):
    def __init__(self, train_or_test):
        assert train_or_test in ('train', 'test')
        if train_or_test == 'train':
            self.get_train_data()
            self.X_train, self.X_validate, self.Y_train, self.Y_validate = train_test_split(self.X, self.Y, test_size=0.1)
        else:
            self.load_test_data()
       
    def load_train_data(self):
        train_ids = next(os.walk(constants.TRAIN_PATH))[1]

        self.X = np.zeros((len(train_ids), constants.IMG_HEIGHT, constants.IMG_WIDTH, constants.IMG_CHANNELS), dtype=np.uint8)
        self.Y = np.zeros((len(train_ids), constants.IMG_HEIGHT, constants.IMG_WIDTH, 1), dtype=np.bool)
        print('Getting and resizing train images and masks ... ')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            path = constants.TRAIN_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:,:,:constants.IMG_CHANNELS]
            img = resize(img, (constants.IMG_HEIGHT, constants.IMG_WIDTH), mode='constant', preserve_range=True)
            self.X[n] = img
            mask = np.zeros((constants.IMG_HEIGHT, constants.IMG_WIDTH, 1), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (constants.IMG_HEIGHT, constants.IMG_WIDTH), mode='constant', 
                                              preserve_range=True), axis=-1)
                mask = np.maximum(mask, mask_)
            self.Y[n] = mask
        
    def load_test_data(self):
        self.test_ids = next(os.walk(constants.TEST_PATH))[1]
        # Get and resize test images
        self.X_test = np.zeros((len(self.test_ids), constants.IMG_HEIGHT, constants.IMG_WIDTH, constants.IMG_CHANNELS), dtype=np.uint8)
        self.sizes_test = []
        print('Getting and resizing test images ... ')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(self.test_ids), total=len(self.test_ids)):
            path = constants.TEST_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:,:,:constants.IMG_CHANNELS]
            self.sizes_test.append([img.shape[0], img.shape[1]])
            img = resize(img, (constants.IMG_HEIGHT, constants.IMG_WIDTH), mode='constant', preserve_range=True)
            self.X_test[n] = img
        
        print('Done!')

    def generate(self, t):
        assert t in ['train', 'validate']
        while True:
            if t == 'train':
                yield self.X_train, self.Y_train
            else:
                yield self.X_validate, self.Y_validate

    def get_test_data(self):
        return self.test_ids, self.X_test, self.sizes_test
