import sys
from skimage.transform import resize
from skimage.morphology import label
from keras.models import model_from_json
import pandas as pd
import numpy as np
from data_generator import DataGenerator

class PredictionEngine:
    def __init__(self, train_or_test, model_name, data_generator):
        self.train_or_test = train_or_test
        self.model_name = model_name
        self.data_generator = data_generator

        assert train_or_test == 'test'
        json_file = open('models/model-' + model_name + '.json')
        self.model = model_from_json(json_file.read())
        json_file.close()
        self.model.load_weights('models/model-' + model_name + '.h5')
        
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
        test_ids, X_test, sizes_test = data_generator.get_test_data()
        preds_test = self.model.predict(X_test, verbose=1)
        preds_test_t = (preds_test > 0.5).astype(np.uint8)

        # Create list of upsampled test masks
        preds_test_upsampled = []
        for i in range(len(preds_test)):
            preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                               (sizes_test[i][0], sizes_test[i][1]), 
                                               mode='constant', preserve_range=True))

        new_test_ids = []
        rles = []
        for n, id_ in enumerate(test_ids):
            rle = list(self.prob_to_rles(preds_test_upsampled[n]))
            rles.extend(rle)
            new_test_ids.extend([id_] * len(rle))

        # Create submission DataFrame
        sub = pd.DataFrame()
        sub['ImageId'] = new_test_ids
        sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        sub.to_csv('models/sub-' + self.model_name + '.csv', index=False)
if __name__ == '__main__':
    train_or_test = sys.argv[1]
    model_name = sys.argv[2]

    data_generator = DataGenerator('test')
    prediction_engine = PredictionEngine(train_or_test, model_name, data_generator)
    prediction_engine.predict()
    #model = load_model('models/model-' + model_name + '.h5', custom_objects={'mean_iou': mean_iou})
