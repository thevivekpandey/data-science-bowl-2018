import keras.backend as K

# Component 1: cross entropy
# Component 2: Predict right number of means
# Component 3: Have biased predictions rather than diffused
def cg_loss(y_true, y_pred):
    #part1 = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    #part2 = (K.abs(K.sum(y_pred, axis=-1) - K.sum(y_true, axis=-1))) / K.sum(y_true, axis = -1)
    #part3 = K.sum(y_true, axis=-1) / K.sum(K.pow(y_pred, 2), axis=-1)
    part2 = K.mean(K.square(y_pred - y_true), axis=-1)

    #return K.sum(y_true, axis=-1)
    #return part1 + part2 + part3
    return part2

if __name__ == '__main__':
    y_a = K.variable([[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]])
    y_b = K.variable([[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]])
    print K.int_shape(y_a)
    print K.eval(cg_loss(y_a, y_b))

