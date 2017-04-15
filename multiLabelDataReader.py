import io
import re
import random
from os import path
from ltlib.defaults import  Defaults
from ltlib.data import Document, Sentence, Token, Dataset,Datasets

import numpy as np



class MultiLabelDataReader(object):



    #takes dir path (to load train,devel,and test)
    def __init__(self, dirPathName,config=Defaults):
        self.dirPathName=dirPathName
        self.config=config

    def getTargets(self,labelList,instantLabels):
        labelList = np.array(labelList)
        label_to_index = { l: i for i, l in enumerate(labelList) }
        targets = []
        for y in instantLabels:
            target = np.zeros(labelList.shape)
            for l in y:
                target[label_to_index[l]] = 1
            targets.append(target)
            #print str(target.shape)
        return targets



    def loadData(self,datasetName,target_indx=None):
        tokens =[]
        labels = []
        ids = []
        allLabels = set()
        with open(self.dirPathName+datasetName+".txt") as f:
            for line in f:
                id,t, l = self.processLine(line)
                tokens.append(t)
                labels.append(l)
                ids.append(id)
                allLabels = allLabels.union(set(l))
        labelsList = list(allLabels)
        labelsList.sort()
       # print ("getting targets")
        targets = self.getTargets(labelsList,labels)
        #print ("finished getting targets")
        docs = []
        #print "read: " + str(len(targets))
        for i in range(len(tokens)):
            doc = self.make_document(ids[i],tokens[i],labels[i],targets[i],target_indx=target_indx)
            docs.append(doc)
        #print "finished making docs"
        return docs


    # datasetType is "train", "devel", "test", or "all" (default)
    def load(self,target_indx=None, datasetNames=["train", "devel","test"], randomize=False):
        dataetTypeList = ["train","devel","test"]
        if not all([ds in dataetTypeList for ds in datasetNames]):
            raise BaseException("dataset argument must be 'train', 'devel', or 'test'")
        datasets = {}

        for dname in datasetNames:
            #print "loading: " + dname
            docs =  self.loadData(dname,target_indx)
            datasets[dname]=Dataset(documents=docs, name=dname)
        return Datasets(**datasets)


    def processLine(self, line):
        splits = line.split("\t")
        id = splits[0].strip()
        text = splits[1].strip()
        token_regex = re.compile(self.config.token_regex, re.UNICODE)
        tokens = [t for t in token_regex.split(text) if t]
        tokens = [t for t in tokens if t and not t.isspace()]
        labels = splits[2].strip().split()
        return id, tokens, labels

    def make_document(self,docid, tokens, labels, target,target_indx=None):
        """Return Document object initialized with given token texts."""
        tokens = [Token(t) for t in tokens]
        # We don't have sentence splitting, but the data structure expects
        # Documents to contain Sentences which in turn contain Tokens.
        # Create a dummy sentence containing all document tokens to work
        # around this constraint.
        sentences = [Sentence(tokens=tokens)]

        doc =  Document(id=docid,target_idx=target_indx,target_str=str(labels), sentences=sentences)
        doc.set_target(target)
        return doc


