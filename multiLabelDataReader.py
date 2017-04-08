import io
import re
import random
from os import path
from ltlib.defaults import  Defaults
import numpy as np



class MultiLabelDataReader(object):



    #takes dir path (to load train,devel,and test)
    def __init__(self, dirPathName,config=Defaults):
        self.dirPathName=dirPathName
        self.config=config

    def getTargets(self,labelList,instantLabels):
        labelList = np.array(labelList)
        targets = []
        for y in instantLabels:
            target = np.zeros(labelList.shape)
            target[[int(np.nonzero(labelList == i)[0]) for i in y]] = 1.0
            targets.append(target)
        return targets



    def loadData(self,datasetName):
        tokens =[]
        labels = []
        allLabels = set()
        with open(self.dirPathName+datasetName+".txt") as f:
            for line in f:
                t, l = self.processLine(line)
                tokens.append(t)
                labels.append(l)
                allLabels = allLabels.union(set(l))
        labelsList = list(allLabels)
        labelsList.sort()
        targets = self.getTargets(labelsList,labels)
        return tokens,labels,targets



    # datasetType is "train", "devel", "test", or "all" (default)
    def load(self, datasetNames=["train", "devel"], randomize=False):
        dataetTypeList = ["train","devel","test"]
        if not all([ds in dataetTypeList for ds in datasetNames]):
            raise BaseException("dataset argument must be 'train', 'devel', or 'test'")

        for ds in datasetNames:
            return  self.loadData(ds)


    def processLine(self, line):
        splits = line.split("\t")
        text = splits[0].strip()
        token_regex = re.compile(self.config.token_regex, re.UNICODE)
        tokens = [t for t in token_regex.split(text) if t]
        tokens = [t for t in tokens if t and not t.isspace()]
        labels = splits[1].strip().split()
        return tokens, labels




if __name__ == '__main__':
    reader = MultiLabelDataReader("/homes/sb895/multilabel-nn/data/doc/hoc/")

    x,l,y = reader.load()
    print str(l[222])
    print str(y[222])
