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
from keras.layers import BatchNormalization, UpSampling2D, merge
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

    def get_unet8(elf):
        inputs = Input((constants.IMG_HEIGHT, constants.IMG_WIDTH, constants.IMG_CHANNELS))
        conv1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(inputs)
        conv1 = PReLU()(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(conv1)
        conv1 = PReLU()(conv1)
        conv1 = BatchNormalization()(conv1)
    
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
        conv2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(pool1)
        conv2 = PReLU()(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(conv2)
        conv2 = PReLU()(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
        conv3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(pool2)
        conv3 = PReLU()(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(conv3)
        conv3 = PReLU()(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
        conv4 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(pool3)
        conv4 = PReLU()(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(conv4)
        conv4 = PReLU()(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
        conv5 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(pool4)
        conv5 = PReLU()(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(conv5)
        conv5 = PReLU()(conv5)
        conv5 = BatchNormalization()(conv5)
    
        up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
        conv6 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(up6)
        conv6 = PReLU()(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(conv6)
        conv6 = PReLU()(conv6)
        conv6 = BatchNormalization()(conv6)
    
        up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
        conv7 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(up7)
        conv7 = PReLU()(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(conv7)
        conv7 = PReLU()(conv7)
        conv7 = BatchNormalization()(conv7)
    
        up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
        conv8 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(up8)
        conv8 = PReLU()(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(conv8)
        conv8 = PReLU()(conv8)
        conv8 = BatchNormalization()(conv8)
    
        up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
        conv9 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(up9)
        conv9 = PReLU()(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(conv9)
        conv9 = PReLU()(conv9)
        conv9 = BatchNormalization()(conv9)
        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
        model = Model(input=inputs, output=conv10)
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
