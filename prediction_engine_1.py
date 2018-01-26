import sys
import numpy as np
import pandas as pd
from scipy.misc import imsave
from skimage.transform import resize
from skimage.morphology import label
from keras.models import model_from_json, load_model
import constants
from data_generator import DataGenerator

class PredictionEngine:
    def __init__(self, train_or_test, model_name, data_generator):
        self.train_or_test = train_or_test
        self.model_name = model_name
        self.data_generator = data_generator

        assert train_or_test in ('train', 'test')
        self.model = load_model('models/model-' + model_name + '.h5')
        
    # Run-length encoding taken from 
    # https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    def rle_encoding(self, x):
        dots = np.where(x.T.flatten() == 1)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b>prev+1): run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths
    
    def prob_to_rles(self, x, cutoff=0.5):
        lab_img = label(x > cutoff)
        for i in range(1, lab_img.max() + 1):
            yield self.rle_encoding(lab_img == i)

    def predict(self):
        if self.train_or_test == 'train':
            ids, X, sizes = data_generator.get_train_data()
        else:
            ids, X, sizes = data_generator.get_test_data()

        preds_test_upsampled = []
        for i, (id, x, size) in enumerate(zip(ids, X, sizes)):
            preds_test = self.model.predict(np.array([x]), verbose=1)
    
            preds_test_upsampled.append(resize(np.squeeze(preds_test[0]), 
                                               (size[0], size[1]), 
                                               mode='constant', preserve_range=True))
    
            #preds_test_t = (preds_test > 0.50).astype(np.uint8)

        for i in range(len(preds_test_upsampled)):
            imsave('output_images/' + ids[i] + '.png', preds_test_upsampled[i]) 

        new_test_ids = []
        rles = []
        for n, id_ in enumerate(ids):
            rle = list(self.prob_to_rles(preds_test_upsampled[n]))
            rles.extend(rle)
            new_test_ids.extend([id_] * len(rle))
    
        # Create submission DataFrame
        sub = pd.DataFrame()
        sub['ImageId'] = new_test_ids
        sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        ofile_name = 'models/sub-' + train_or_test + '-' + self.model_name + '.csv' 
        sub.to_csv(ofile_name, index=False)

if __name__ == '__main__':
    train_or_test = sys.argv[1]
    model_name = sys.argv[2]

    #data_generator = DataGenerator(train_or_test)
    data_generator = None
    prediction_engine = PredictionEngine(train_or_test, model_name, data_generator)
    prediction_engine.predict()
