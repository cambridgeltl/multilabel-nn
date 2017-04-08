from keras import backend as K


if K._BACKEND == 'theano':
    _floatX = 'float32'
else:
    _floatX = 'float'

def _float(x):
    return K.cast(x, _floatX)

def tp_tn_fp_fn(y_true, y_pred):
    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)

    correct = K.equal(y_true, y_pred)
    incorrect = K.not_equal(y_true, y_pred)
    pos_pred = K.not_equal(y_pred, K.zeros_like(y_pred))
    neg_pred = K.equal(y_pred, K.zeros_like(y_pred))

    tp = K.sum(K.cast(correct & pos_pred, 'int32'))
    tn = K.sum(K.cast(correct & neg_pred, 'int32'))
    fp = K.sum(K.cast(incorrect & pos_pred, 'int32'))
    fn = K.sum(K.cast(incorrect & neg_pred, 'int32'))

    return tp, tn, fp, fn

def prec(y_true, y_pred):
    tp, tn, fp, fn = tp_tn_fp_fn(y_true, y_pred)
    return _float(tp) / (tp + fp)

def rec(y_true, y_pred):
    tp, tn, fp, fn = tp_tn_fp_fn(y_true, y_pred)
    return _float(tp) / (tp + fn)

def f1(y_true, y_pred):
    tp, tn, fp, fn = tp_tn_fp_fn(y_true, y_pred)
    p = _float(tp) / (tp + fp)
    r = _float(tp) / (tp + fn)
    return 2*p*r / (p+r)
