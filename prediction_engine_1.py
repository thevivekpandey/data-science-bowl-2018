import sys
import numpy as np
import pandas as pd
from scipy.misc import imsave
from skimage.transform import resize
from skimage.morphology import label
from keras.models import model_from_json, load_model
import constants
import metric
from data_generator import DataGenerator
from data_generator_b import DataGeneratorB

class PredictionEngine:
    def __init__(self, train_or_test, model_name, data_generator):
        self.train_or_test = train_or_test
        self.model_name = model_name
        self.data_generator = data_generator

        assert train_or_test in ('train', 'test')
        self.model = load_model('models/model-' + model_name + '.h5', custom_objects={'mean_iou': metric.mean_iou, 'new_iou': metric.new_iou})
        
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

    def get_indices_work(self, x, block):
        indices = []
        for i in range(0, 51200, block):
            if i + block < x:
                indices.append(i)
            if i > x:
                indices.append(x - block)
                break
        return indices
  
    def get_indices(self, x, y):
        x_indices = self.get_indices_work(x, constants.IMG_WIDTH)
        y_indices = self.get_indices_work(y, constants.IMG_WIDTH)
        return [(x, y) for x in x_indices for y in y_indices]

    def predict(self, chunked):
        if self.train_or_test == 'train':
            ids, X, sizes = data_generator.get_train_data()
        else:
            ids, X, sizes = data_generator.get_test_data()

        preds_test_final = []
        for i, (id, x, size) in enumerate(zip(ids, X, sizes)):
            print(i, ":", size, ":", x.shape)

            if chunked:
                counts = np.zeros((x.shape[0], x.shape[1]), dtype=np.uint8)
                probs  = np.zeros((x.shape[0], x.shape[1]), dtype=np.float64)
                blocks = self.get_indices(x.shape[0], x.shape[1])
                print(blocks)
                for block in blocks:
                    a, b = block
                    w, h = constants.IMG_WIDTH, constants.IMG_HEIGHT
                    part = x[a:a+w, b:b+h,:]
                    preds = self.model.predict(np.array([part]))
                    counts[a:a+w, b:b+h] += 1
                    probs[a:a+w, b:b+h]  += preds[0, :, :, 0]
                    chunk_name = ids[i] + '-' + str(a) + '-' + str(b) + '.png'
                    print(preds.shape)
                    imsave('output_masks_chunked/' + chunk_name, preds.reshape((constants.IMG_WIDTH, constants.IMG_HEIGHT)) > 0.5)
                probs = probs / counts
                preds_test_final.append(probs)
                imsave('output_masks_chunked/' + ids[i] + '.png', preds_test_final[i] > 0.5)

            else:
                resized_x = resize(x, (constants.IMG_HEIGHT, constants.IMG_WIDTH),
                              mode='constant', preserve_range=True)
                preds_test = self.model.predict(np.array([resized_x]), verbose=1)
    
                preds_test_final.append(resize(np.squeeze(preds_test[0]), 
                                                   (size[0], size[1]), 
                                                   mode='constant', preserve_range=True))
                imsave('output_masks_unchunked/' + ids[i] + '.png', preds_test_final[i] > 0.5)

        new_test_ids = []
        rles = []
        for n, id_ in enumerate(ids):
            rle = list(self.prob_to_rles(preds_test_final[n]))
            rles.extend(rle)
            new_test_ids.extend([id_] * len(rle))
    
        # Create submission DataFrame
        sub = pd.DataFrame()
        sub['ImageId'] = new_test_ids
        sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        if chunked:
            ofile_name = 'models/sub-' + train_or_test + '-' + self.model_name + '-chunked.csv' 
        else:
            ofile_name = 'models/sub-' + train_or_test + '-' + self.model_name + '-unchunked.csv' 
        sub.to_csv(ofile_name, index=False)

if __name__ == '__main__':
    train_or_test = sys.argv[1]
    model_name = sys.argv[2]
    chunked = True if sys.argv[3] == 'chunked' else False
    data_generator = DataGeneratorB(train_or_test)
    prediction_engine = PredictionEngine(train_or_test, model_name, data_generator)
    prediction_engine.predict(chunked)

