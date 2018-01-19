import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, Add, GlobalMaxPooling1D, Conv2D
from keras.layers import Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.merge import concatenate
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras import regularizers
import constants

class ModelGenerator():
    def get_unet_1(self):
        inputs = Input((constants.IMG_HEIGHT, constants.IMG_WIDTH, constants.IMG_CHANNELS))
        s = Lambda(lambda x: x / 255) (inputs)
        
        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)
        
        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)
        
        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)
        
        c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
        c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
        
        c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
        c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)
        
        u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)
        
        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)
        
        u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)
        
        u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)
        
        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
        
        model = Model(inputs=[inputs], outputs=[outputs])
        return model

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
