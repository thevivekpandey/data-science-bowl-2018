import sys
import os
from itertools import izip
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from scipy.misc import imsave, imrotate
from sklearn.model_selection import train_test_split
import random
import constants

class DataGeneratorB(object):
    def __init__(self, train_or_test):
        assert train_or_test in ('train', 'test')
        if train_or_test == 'train':
            self.init_erroneous_image_ids()
            self.load_train_data()
            self.X_train, self.X_val, self.Y_train, self.Y_val= train_test_split(self.X, self.Y, test_size=0.1, random_state=7)
        else:
            self.load_test_data()

    def init_erroneous_image_ids(self):
        self.image_ids_to_ignore = []
        f = open('erroneous_image_ids.txt')
        for line in f:
            self.image_ids_to_ignore.append(line.strip())
        f.close()

    def modify_image(self, one_x, one_y):
        #ops = ['none', 'rot_90', 'rot_180', 'rot_270', 'horiz_flip', 'vert_flip', 'diag_flip_1', 'diag_flip_2']
        ops = ['none', 'rot_90', 'rot_180', 'rot_270', 'horiz_flip', 'vert_flip', 'diag_flip_1']
        op = random.choice(ops)
        if op == 'none':
            return op, one_x, one_y
        elif op == 'rot_90':
            return op, np.rot90(one_x, 1), np.rot90(one_y, 1)
        elif op == 'rot_180':
            one_x, one_y = np.rot90(one_x, 1), np.rot90(one_y, 1)
            one_x, one_y = np.rot90(one_x, 1), np.rot90(one_y, 1)
            return op, one_x, one_y
        elif op == 'rot_270':
            one_x, one_y = np.rot90(one_x, 1), np.rot90(one_y, 1)
            one_x, one_y = np.rot90(one_x, 1), np.rot90(one_y, 1)
            one_x, one_y = np.rot90(one_x, 1), np.rot90(one_y, 1)
            return op, one_x, one_y
        elif op == 'horiz_flip':
            return op, np.flip(one_x, 1), np.flip(one_y, 1)
        elif op == 'vert_flip':
            return op, np.flip(one_x, 0), np.flip(one_y, 0)
        elif op == 'diag_flip_1':
            tmp_x = np.transpose(one_x, (1, 0, 2))
            tmp_y = np.transpose(one_y, (1, 0, 2))
            return op, tmp_x, tmp_y
        else: #diag_flip_2
            return op, one_x, one_y

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
                op, one_x, one_y = self.modify_image(one_x, one_y)
                idxs.append(idx)
                X_batch.append(one_x)
                Y_batch.append(one_y)
                #r = random.randint(1, 100000000)
                #print one_x.shape
                #imsave('temp_images/' + str(r) + '-x-' + op + '.png', one_x)
                #imsave('temp_images/' + str(r) + '-y-' + op + '.png', one_y.reshape(256, 256))
            if train_or_val == 'temp':
                yield idxs, np.array(X_batch), np.array(Y_batch)
            else:
                yield np.array(X_batch), np.array(Y_batch)

    def load_train_data(self):
        self.train_ids = next(os.walk(constants.TRAIN_PATH))[1][0:10]
        self.X, self.Y = [], []
        print('Getting train images and masks ... ')
        sys.stdout.flush()
        self.sizes_train = []
        for n, id_ in tqdm(enumerate(self.train_ids), total=len(self.train_ids)):
            path = constants.TRAIN_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:,:,:constants.IMG_CHANNELS]
            self.X.append(img)
            self.sizes_train.append([img.shape[0], img.shape[1]])
            mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file).reshape((img.shape[0], img.shape[1], 1))
                mask = np.maximum(mask, mask_)
            self.Y.append(mask)
            imsave('actual_masks/' + id_ + '.png', mask.reshape((mask.shape[0], mask.shape[1])))
        print('Done!')
        
    def load_test_data(self):
        self.test_ids = next(os.walk(constants.TEST_PATH))[1]

        self.X_test = []
        self.sizes_test = []
        print('Getting and resizing test images ... ')
        for n, id_ in tqdm(enumerate(self.test_ids), total=len(self.test_ids)):
            path = constants.TEST_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:,:,:constants.IMG_CHANNELS]
            self.sizes_test.append([img.shape[0], img.shape[1]])
            if img.shape[0] < constants.IMG_HEIGHT or img.shape[1] < constants.IMG_WIDTH:
                img = resize(img, (constants.IMG_HEIGHT, constants.IMG_WIDTH), mode='constant', preserve_range=True)
            self.X_test.append(img)
        print('Done!')
        self.X_test = np.array(self.X_test)

    def get_test_data(self):
        return self.test_ids, self.X_test, self.sizes_test

    def get_train_data(self):
        return self.train_ids, self.X, self.sizes_train

if __name__ == '__main__':
    dg = DataGeneratorB('train')
