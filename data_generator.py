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
            self.init_erroneous_image_ids()
            self.load_train_data()
            self.X_train, self.X_validate, self.Y_train, self.Y_validate = train_test_split(self.X, self.Y_cg, test_size=0.1, random_state=7)
        else:
            self.load_test_data()

    def init_erroneous_image_ids(self):
        self.image_ids_to_ignore = []
        f = open('erroneous_image_ids.txt')
        for line in f:
            self.image_ids_to_ignore.append(line.strip())
        f.close()

    def generator(self, batch_size):
        xtr = self.X_train
        xval = self.X_validate
        ytr = self.Y_train
        yval = self.Y_validate
        data_gen_args = dict(horizontal_flip=True,
                             vertical_flip=True,
                             rotation_range=360.,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             zoom_range=0.0)
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
        for id in self.image_ids_to_ignore:
            self.train_ids.remove(id)

        self.X = np.zeros((len(self.train_ids), constants.IMG_HEIGHT, constants.IMG_WIDTH, constants.IMG_CHANNELS), dtype=np.uint8)
        self.Y = np.zeros((len(self.train_ids), constants.IMG_HEIGHT, constants.IMG_WIDTH, 1), dtype=np.bool)
        self.Y_cg = np.zeros((len(self.train_ids), constants.IMG_HEIGHT, constants.IMG_WIDTH), dtype=np.uint8)
        print('Getting and resizing train images and masks ... ')
        self.sizes_train = []
        for n, id_ in tqdm(enumerate(self.train_ids), total=len(self.train_ids)):
            path = constants.TRAIN_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:,:,:constants.IMG_CHANNELS]
            imsave('input_images/' + id_ + '.png', img)
            self.sizes_train.append([img.shape[0], img.shape[1]])
            img = resize(img, (constants.IMG_HEIGHT, constants.IMG_WIDTH), mode='constant', preserve_range=True)
            self.X[n] = img
            mask    = np.zeros((constants.IMG_HEIGHT, constants.IMG_WIDTH, 1), dtype=np.bool)
            cg_mask = np.zeros((constants.IMG_HEIGHT, constants.IMG_WIDTH), dtype=np.uint8)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (constants.IMG_HEIGHT, constants.IMG_WIDTH), mode='constant', 
                                              preserve_range=True), axis=-1)

                # We are finding CG in resized image
                cg_x, cg_y, _ = self.find_cg(mask_)
                #Sometimes, masks get lost. Perhaps because of image resizing. We lose some masks for around ~25 images
                if cg_x > -1:
                    cg_mask[int(cg_x)][int(cg_y)] = constants.MAX_MASK_VAL
                mask = np.maximum(mask, mask_)
            reshaped_mask = mask.reshape(constants.IMG_HEIGHT, constants.IMG_WIDTH) * 255
            imsave('actual_masks/' + id_ + '.png', reshaped_mask)
            imsave('cgs/' + id_ + '.png', cg_mask)
            self.Y[n] = mask
            self.Y_cg[n] = cg_mask
        
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

    def find_cg(self, mask):
        indices = np.argwhere(mask > constants.MAX_MASK_VAL - 10)
        if len(indices) == 0:
            return -1, -1, 0
        return np.average(indices, axis=0)

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
