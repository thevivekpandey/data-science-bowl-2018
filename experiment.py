import sys
import random
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from data_generator import DataGenerator
from data_generator_b import DataGeneratorB
from model_generator import ModelGenerator
import metric
import losses

#seed = 42
#random.seed = seed
#np.random.seed = seed

def run_keras(model, model_name):
    #data_generator = DataGenerator('train')
    #train_generator, validate_generator = data_generator.generator(16)
    data_generator = DataGeneratorB('train')
    train_generator    = data_generator.generator('train', 16)
    validate_generator = data_generator.generator('validate', 16)

    opt = Adam(lr=0.001, decay=0)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[metric.mean_iou])
    filepath = "models/model-" + model_name + "-{epoch:03d}-{val_mean_iou:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_mean_iou', verbose=0, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(verbose=1, min_lr = 1e-8, patience=3, factor=0.3)
    callbacks = [checkpoint, reduce_lr]

    model.fit_generator(generator=train_generator, validation_data=validate_generator,
                        steps_per_epoch=40, validation_steps=4,
                        epochs=200,
                        callbacks=callbacks)
    return model

model_name = sys.argv[1]
model_generator = ModelGenerator()
model = model_generator.get_unet_1()
print model.summary()
model_json = model.to_json()
with open('models/model-' + model_name + '.json', 'w') as f:
    f.write(model_json)
model = run_keras(model, model_name)
