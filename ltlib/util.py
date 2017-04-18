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


def calculate_micro_scores(tot):
    p = tot[0] / (tot[0] + tot[1])
    r = tot[0] / (tot[0] + tot[3])
    f = 2.0 * p * r / (p + r)
    a = (tot[0] + tot[2]) / (tot[0] + tot[1] + tot[2] + tot[3])
    s = tot[0] + tot[1] + tot[2] + tot[3]
    tp = np.sum(tot[0])
    fp = np.sum(tot[1])
    tn = np.sum(tot[2])
    fn = np.sum(tot[3])
    p_micro = tp / (tp + fp)
    r_micro = tp / (tp + fn)
    f_micro = 2 * p_micro * r_micro / (p_micro + r_micro)
    # print("tot: " + str(tot))
    #print("f: " + str(f_micro))
    #print("r: " + str(r_micro))
    #print("p: " + str(p_micro))
    #print("a: " + str(np.average(a)))
    # print("s: "+ str(s))
    res = {}
    res["acc"] = np.average(a)
    res["fscore"] = f_micro  # np.average(f)
    res["p"] = p_micro  # np.average(p)
    res["r"] = r_micro  # np.average(r)
    res["tp"] = tp  # np.average(tot[0])
    res["fp"] = fp  # np.average(tot[1])
    res["tn"] = tn  # np.average(tot[2])
    res["fn"] = fn  # np.average(tot[3])
    return res


