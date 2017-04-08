import io
import re
import random
from os import path
from ltlib.defaults import  Defaults



class MultiLabelDataReader(object):



    #takes dir path (to load train,devel,and test)
    def __init__(self, dirPathName,config=Defaults):
        self.dirPathName=dirPathName
        self.config=config



    def loadData(self,datasetName):
        ret = []
        with open(self.dirPathName+datasetName+".txt") as f:
            for line in f:
                ret.append(self.processLine(line))
        return ret




    # datasetType is "train", "devel", "test", or "all" (default)
    def load(self, datasetNames=["train", "devel"], randomize=False):
        dataetTypeList = ["train","devel","test"]
        if not all([ds in dataetTypeList for ds in datasetNames]):
            raise BaseException("dataset argument must be 'train', 'devel', or 'test'")

        for ds in datasetNames:
            return   self.loadData(ds)


    def processLine(self, line):
        splits = line.split("\t")
        text = splits[0]
        token_regex = re.compile(self.config.token_regex, re.UNICODE)
        tokens = [t for t in token_regex.split(text) if t]
        tokens = [t for t in tokens if t and not t.isspace()]
        labels = splits[1]
        return [tuple(tokens),tuple(labels)]




if __name__ == '__main__':
    reader = MultiLabelDataReader("/home/sb/multilabel-nn/data/doc/hoc/")

    print(str(reader.load()))
