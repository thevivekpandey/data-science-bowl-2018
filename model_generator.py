import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate, Lambda
from keras.layers import Conv2D, MaxPooling2D, Convolution1D, MaxPooling1D, Add, GlobalMaxPooling1D, Conv2D
from keras.layers import Conv2DTranspose
#from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.merge import concatenate
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras import regularizers
from keras.layers import Input
#from keras.layers.convolutional import Conv2D, Conv2DTranspose
#from keras.layers.pooling import MaxPooling2D
from keras.layers import BatchNormalization, UpSampling2D
import constants

class ModelGenerator():
    def get_unet_1(self):
        inputs = Input((constants.IMG_HEIGHT, constants.IMG_WIDTH, constants.IMG_CHANNELS))
        s = Lambda(lambda x: x / 255) (inputs)
        
        base = 16
        c1 = Conv2D(base, (3, 3), activation='relu', padding='same') (s)
        c1 = Conv2D(base, (3, 3), activation='relu', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)
        
        c2 = Conv2D(2*base, (3, 3), activation='relu', padding='same') (p1)
        c2 = Conv2D(2*base, (3, 3), activation='relu', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)
        
        c3 = Conv2D(4*base, (3, 3), activation='relu', padding='same') (p2)
        c3 = Conv2D(4*base, (3, 3), activation='relu', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)
        
        c4 = Conv2D(8*base, (3, 3), activation='relu', padding='same') (p3)
        c4 = Conv2D(8*base, (3, 3), activation='relu', padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
        
        c5 = Conv2D(64*base, (3, 3), activation='relu', padding='same') (p4)
        c5 = Conv2D(64*base, (3, 3), activation='relu', padding='same') (c5)
        
        u6 = Conv2DTranspose(8*base, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(8*base, (3, 3), activation='relu', padding='same') (u6)
        c6 = Conv2D(8*base, (3, 3), activation='relu', padding='same') (c6)
        
        u7 = Conv2DTranspose(4*base, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(4*base, (3, 3), activation='relu', padding='same') (u7)
        c7 = Conv2D(4*base, (3, 3), activation='relu', padding='same') (c7)
        
        u8 = Conv2DTranspose(2*base, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(2*base, (3, 3), activation='relu', padding='same') (u8)
        c8 = Conv2D(2*base, (3, 3), activation='relu', padding='same') (c8)
        
        u9 = Conv2DTranspose(base, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(base, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(base, (3, 3), activation='relu', padding='same') (c9)
        
        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
        
        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    def downstream_convolve_1(self, input, f):
        conv = Conv2D(f, (3, 3), padding='same', kernel_initializer='he_uniform')(input)
        conv = PReLU()(conv)
        conv = BatchNormalization()(conv)
        conv = Conv2D(f, (3, 3), padding='same', kernel_initializer='he_uniform')(conv)
        conv = PReLU()(conv)
        conv = BatchNormalization()(conv)
        return conv
    
    def upstrem_convolve_1(self, prev_input, lateral_input, f):
        up = concatenate([UpSampling2D(size=(2, 2))(prev_input), lateral_input],  axis=3)
        conv = Conv2D(f, (3, 3), padding='same', kernel_initializer='he_uniform')(up)
        conv = PReLU()(conv)
        conv = BatchNormalization()(conv)
        conv = Conv2D(f, (3, 3), padding='same', kernel_initializer='he_uniform')(conv)
        conv = PReLU()(conv)
        conv = BatchNormalization()(conv)
        return conv

    def get_unet_8a(self):
        inputs = Input((constants.IMG_HEIGHT, constants.IMG_WIDTH, constants.IMG_CHANNELS))

        conv1 = self.downstream_convolve_1(inputs, 32)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.downstream_convolve_1(pool1, 64)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.downstream_convolve_1(pool2, 128)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.downstream_convolve_1(pool3, 256)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5_u = self.downstream_convolve_1(pool4, 512)

        conv4_u = self.upstrem_convolve_1(conv5_u, conv4, 256)
        conv3_u = self.upstrem_convolve_1(conv4_u, conv3, 128)
        conv2_u = self.upstrem_convolve_1(conv3_u, conv2, 64)
        conv1_u = self.upstrem_convolve_1(conv2_u, conv1, 32)
        output = Conv2D(1, (1, 1), activation='sigmoid')(conv1_u)

        return Model(input=inputs, outputs=output)

    def get_unet_8_for_k_means(self):
        inputs = Input((constants.IMG_HEIGHT, constants.IMG_WIDTH, constants.IMG_CHANNELS))

        conv1 = self.downstream_convolve_1(inputs, 8)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.downstream_convolve_1(pool1, 16)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.downstream_convolve_1(pool2, 32)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        flat = Flatten()(pool3)
        dense = Dense(128, activation='relu')(flat)
        output = Dense(1, activation='relu')(dense)

        return Model(input=inputs, outputs=output)

if __name__ == '__main__':
    n_mels = 40
    raw_wav = Input(shape=(16000, 1))
    mel_spec = Input(shape=(n_mels, 32, 1))
    mg = ModelGenerator()
    x = mg.get_1d_part(raw_wav)
    y = mg.get_mel_part(mel_spec)
    z = keras.layers.concatenate([x, y])
    z = Dropout(0.5)(z)
    z = Dense(128, activation='relu')(z)
    z = Dropout(0.5)(z)
    z = Dense(12, activation='softmax')(z)
    model = Model(inputs=[raw_wav, mel_spec], outputs=[z])
    print model.summary()
