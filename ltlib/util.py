import numpy as np

from types import GeneratorType
from itertools import chain, izip_longest

from keras.models import model_from_json


def unique(iterable):
    """Return unique values from iterable."""
    seen = set()
    return [i for i in iterable if not (i in seen or seen.add(i))]

def binarize_sparse(a, dtype=np.int32):
    """Return array where maximum value in a is one and others zero."""
    a = np.asarray(a)
    b = np.zeros(a.shape)
    b[np.argmax(a)] = 1
    return b

def lookaround(iterable):
    "s -> (None,s0,s1), (s0,s1,s2), ..., (sn-1,sn,None), (sn,None,None)"
    a, b, c = iter(iterable), iter(iterable), iter(iterable)
    next(c, None)
    return izip_longest(chain([None], a), b, c)

def as_scalar(iterable):
    """Return scalar equivalent of (optionally) one-hot value."""
    if isinstance(iterable, GeneratorType):
        iterable = list(iterable)
    a = np.asarray(iterable)
    if a.ndim == 0:
        return a.item()    # scalar
    elif a.ndim == 1:
        return np.argmax(a)    # one-hot
    else:
        raise ValueError('cannot map array of shape %s' % str(a.shape))

def dict_argmax(d):
    """Return key giving maximum value in dictionary."""
    m = max(d.values())
    for k, v in d.items():
        if v == m:
            return k

def save_keras(model,path):
    model_json = model.to_json()
    with open(path+"model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(path+"model.h5")
    print "saved keras model to: " + path

def load_keras(path):
    # load json and create model
    json_file = open(path+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path+"model.h5")
    print("Loaded model from" + path)
    return loaded_model
