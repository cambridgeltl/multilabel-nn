from multiLabelDataReader import MultiLabelDataReader
from config import Defaults
import numpy as np

import utility

def jaccard(data,i,j):
    union = data[i] + data[j]
    intersection = np.dot(data[i],data[j])
    unionCount = np.sum(union)
    intersectionCount = np.sum(intersection)
    ret = float(intersectionCount)/float(unionCount)
    #print (str(ret))
    return ret

def average_jaccard(data):
    data = np.transpose(data)
    jaccardList = []
    #print str(len(data))
    for i in range(len(data)):
        for j in range(i):
            if i == j:continue
            jaccardList.append(jaccard(data,i,j))
    #print str(len(jaccardList))
    #print str(jaccardList)
    jaccardA = np.array(jaccardList)
    return [np.average(jaccardA), np.max(jaccardA)]




data = MultiLabelDataReader(Defaults.input_path).load()


devTar = data.devel.documents.targets
testTar = data.test.documents.targets
trainTar = data.train.documents.targets
allTar = np.concatenate((devTar,testTar,trainTar))


#print (str(devTar.shape))
develJ = average_jaccard(devTar)
print ("devel\t" + str(develJ[0]) + "\t"+str(develJ[1]))

testJ = average_jaccard(testTar)
print ("test\t" + str(testJ[0]) + "\t"+str(testJ[1]))

trainJ = average_jaccard(trainTar)
print ("train\t" + str(trainJ[0]) + "\t"+str(trainJ[1]))

allJ = average_jaccard(allTar)
print ("all\t" + str(allJ[0]) + "\t"+str(allJ[1]))