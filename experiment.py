import sys
import random
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from data_generator import DataGenerator
from model_generator import ModelGenerator

seed = 42
random.seed = seed
np.random.seed = seed

def run_keras(model, model_name):
    generator = DataGenerator()
    training_generator = generator.generate('train')
    test_generator = generator.generate('validate')

    opt = Adam(lr=0.001, decay=0)
    #opt = keras.optimizers.Adadelta()
    #model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam', loss_weights=[1.0])
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
    #filepath = "models/model-" + model_name + "-{epoch:03d}-{val_acc:.4f}.h5"
    filepath = "models/model-" + model_name + "-{epoch:03d}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, mode='max')
    reduce_lr = ReduceLROnPlateau(verbose=1, min_lr = 1e-8, patience=5, factor=0.3)
    callbacks = [checkpoint, reduce_lr]

    model.fit_generator(generator=training_generator, validation_data=test_generator, 
                        steps_per_epoch=20, validation_steps=2,
                        epochs=200,
                        callbacks=callbacks)
    return model

model_name = sys.argv[1]
model_generator = ModelGenerator()
model = model_generator.get_unet_1()
#print model.summary()
#model_json = model.to_json()
#with open('models/model-' + model_number + '.json', 'w') as f:
#    f.write(model_json)
model = run_keras(model, model_name)
