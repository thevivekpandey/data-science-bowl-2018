def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def new_iou(y_true, y_pred):
    smooth = 1
    threshold = 0.5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    iou = (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return iou
   
def part1(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def part2(y_true, y_pred):
    return K.abs(K.sum(y_pred, axis=-1) - K.sum(y_true, axis=-1)) / K.sum(y_true, axis=-1)

def part3(y_true, y_pred):
    return K.sum(y_true, axis=-1) / K.sum(K.pow(y_pred, 2), axis=-1)

def cg_loss(y_true, y_pred):
    #part1 = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    #part2 = K.abs(K.sum(y_pred, axis=-1) - K.sum(y_true, axis=-1))
    #part3 = 1.0 / K.sum(K.pow(y_pred, 2), axis=-1)
    return part1(y_true, y_pred) + part3(y_true, y_pred)

