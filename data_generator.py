import sys
import os
from itertools import izip
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from scipy.misc import imsave
from sklearn.model_selection import train_test_split
import constants
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator(object):
    def __init__(self, train_or_test):
        assert train_or_test in ('train', 'test')
        if train_or_test == 'train':
            self.load_train_data()
            self.X_train, self.X_validate, self.Y_train, self.Y_validate = train_test_split(self.X, self.Y, test_size=0.1, random_state=7)
        else:
            self.load_test_data()

    def generator(self, batch_size):
        xtr = self.X_train
        xval = self.X_validate
        ytr = self.Y_train
        yval = self.Y_validate
        data_gen_args = dict(horizontal_flip=True,
                             vertical_flip=True,
                             rotation_range=360.,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2)
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        image_datagen.fit(xtr, seed=7)
        mask_datagen.fit(ytr, seed=7)
        image_generator = image_datagen.flow(xtr, batch_size=batch_size, seed=7)
        mask_generator = mask_datagen.flow(ytr, batch_size=batch_size, seed=7)
        train_generator = izip(image_generator, mask_generator)
    
        val_gen_args = dict()
        image_datagen_val = ImageDataGenerator(**val_gen_args)
        mask_datagen_val = ImageDataGenerator(**val_gen_args)
        image_datagen_val.fit(xval, seed=7)
        mask_datagen_val.fit(yval, seed=7)
        image_generator_val = image_datagen_val.flow(xval, batch_size=batch_size, seed=7)
        mask_generator_val = mask_datagen_val.flow(yval, batch_size=batch_size, seed=7)
        val_generator = izip(image_generator_val, mask_generator_val)
    
        return train_generator, val_generator
       
    def load_train_data(self):
        self.train_ids = next(os.walk(constants.TRAIN_PATH))[1]

        self.X = np.zeros((len(self.train_ids), constants.IMG_HEIGHT, constants.IMG_WIDTH, constants.IMG_CHANNELS), dtype=np.uint8)
        self.Y = np.zeros((len(self.train_ids), constants.IMG_HEIGHT, constants.IMG_WIDTH, 1), dtype=np.bool)
        print('Getting and resizing train images and masks ... ')
        self.sizes_train = []
        for n, id_ in tqdm(enumerate(self.train_ids), total=len(self.train_ids)):
            path = constants.TRAIN_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:,:,:constants.IMG_CHANNELS]
            imsave('input_images/' + id_ + '.png', img)
            self.sizes_train.append([img.shape[0], img.shape[1]])
            img = resize(img, (constants.IMG_HEIGHT, constants.IMG_WIDTH), mode='constant', preserve_range=True)
            self.X[n] = img
            mask = np.zeros((constants.IMG_HEIGHT, constants.IMG_WIDTH, 1), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (constants.IMG_HEIGHT, constants.IMG_WIDTH), mode='constant', 
                                              preserve_range=True), axis=-1)
                mask = np.maximum(mask, mask_)
            reshaped_mask = mask.reshape(constants.IMG_HEIGHT, constants.IMG_WIDTH) * 255
            imsave('actual_masks/' + id_ + '.png', reshaped_mask)
            self.Y[n] = mask
        
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

    def generate(self, t):
        assert t in ['train', 'validate']
        while True:
            if t == 'train':
                yield self.X_train, self.Y_train
            else:
                yield self.X_validate, self.Y_validate

    def get_test_data(self):
        return self.test_ids, self.X_test, self.sizes_test

    def get_train_data(self):
        return self.train_ids, self.X, self.sizes_train
