import numpy as np
from multiLabelDataReader import MultiLabelDataReader
from collections import Counter
import config


def UB(shape):
    nh = shape[0]
    no = shape[1]
    return np.sqrt(6)/np.sqrt(nh +no)

def count_cooc(targets):
    targetTuples = [tuple(t) for t in targets]
    c = Counter(targetTuples)
    return c


def initilize(targets,shape):
    c = count_cooc(targets)
    coocArray = np.array(c.keys())
    initW = UB(shape)
    print "init weight is: " + str(initW)
    initArray = coocArray * initW
    #randComp = np.random.uniform(low=0.0, high=1.0, size=(shape[0] - initArray.shape[0], initArray.shape[1]))
    randComp = np.zeros((shape[0] - initArray.shape[0], initArray.shape[1]))
    result = np.concatenate((initArray, randComp))
    return result

if __name__ == '__main__':
    datasets = MultiLabelDataReader(config.Defaults.input_path).load(datasetNames=["train"])
    targets = datasets.train.get_targets()
    #print str(len(targets))
    c =count_cooc(targets)
    coocArray = np.array(c.keys())

    for k in c:
        print str(k) + " : " + str(c[k])
    print "unique co-occurances: "+ str(len(c.keys()))

    initArray = coocArray*UB()
    print "init array shape: " + str(initArray.shape)
    randComp = np.random.uniform(low=0.0, high=1.0, size=(config.Defaults.multilabel_output_layer_hidden_units-initArray.shape[0],initArray.shape[1]))
    print str(randComp.shape)

    #print str(coocArray[0]) + "\n" + str(initArray[0])

    result = np.concatenate((initArray,randComp))
    print "final array shape = " + str(result.shape)
    print str(result)





