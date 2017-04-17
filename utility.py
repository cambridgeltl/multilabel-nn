
#import num2words

__author__ = 'Simon'

import os
import codecs
import subprocess
import string
utf8 = False
import sys
import json

reload(sys)
sys.setdefaultencoding('utf8')

stopwords = ["a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","less","let","like","likely","may","me","might","more","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","such","than","that","the","their","them","then","there","therefore","these","they","this","tis","to","too","twas","us","wants","was","we", "well","were","what","when","where","which","while","who","whom","why","will","with","within","without","would","yet","you","your"]
#numbersList = [num2words.num2words(i) for i in range(0,10)]
#numbersList.extend([num2words.num2words(i) for i in range(0,100,10)])
numbersList = []
numbersList.extend([str(i) for i in range (0,10)])
numbersList.extend([str(i) for i in range (0,100,10)])
numbersList = set(numbersList)
punctuations = set(string.punctuation)

def displayProgresstDone(text, numberDone, total):

    percent = round(100.0*float(numberDone)/float(total),2)
    sys.stdout.write("\r%s: %d/%d (%d%%)" % (text,numberDone,total,percent))
    sys.stdout.flush()
    if numberDone==total:
        print "\n"



def isStopWord(word):
    return word.lower() in stopwords

def isPunctuation(word):
    return word.lower() in punctuations

def isNumber(word):
    return word in numbersList


def writeToFile(text,filepath, supressScreenOutput=False):
    if not supressScreenOutput:
        print "writing: " + filepath
    if(utf8):
        try:
            with codecs.open(filepath, 'w', 'UTF-8') as f:
                f.write(text)
                f.close()
        except UnicodeDecodeError:
            with open(filepath, 'w') as f:
                f.write(text)
                f.close()

    else:
        with open(filepath, 'w') as f:
            f.write(text)
            f.close()

def appendToFile(text,filepath):
    print "writing: " + filepath
    if(utf8):
        try:
            with codecs.open(filepath, 'a', 'UTF-8') as f:
                f.write(text)
                f.close()
        except UnicodeDecodeError:
            with open(filepath, 'a') as f:
                f.write(text)
                f.close()

    else:
        with open(filepath, 'a') as f:
            f.write(text)
            f.close()

def appendToFile(text,filepath):
    print "appending: " + filepath
    if(utf8):

        with codecs.open(filepath, 'a', 'UTF-8') as f:
            f.write(text)
            f.close()
    else:
        with open(filepath, 'a') as f:
            f.write(text)
            f.close()

def convertDicToList(dic): #sorts the keys in order then returns a list of values according to that order
    retList = []
    sortedKeys = sorted(dic.keys())
    #print "size of sorted keyes: " + str(len(sortedKeys))
    for key in sortedKeys:
        retList.append(dic[key])
    #print "size of returnList " + str(len(retList))
    return retList


def writeListAsStringFile(list,path):
    writeToFile(listToString(list,"\n"),path)

def writeDictAsStringFile(dict,path):

    writeToFile(json.dumps(dict),path)

def readDictFromStringFile(path):
    return json.loads(readFileAsString(path))

def readListFromStringFile(path):
    return stringToList(readFileAsString(path),"\n")

def openFileAsLines(filepath):
    return  readFileAsString(filepath).splitlines()

def fileExisits(filePath):
    return os.path.isfile(filePath)

def readFileAsString(filePath):
    if(utf8):
        try:
            stream = codecs.open(filePath, "r", 'utf-8')
            str = stream.read()
            stream.close()
            return str
        except UnicodeDecodeError:
            stream = codecs.open(filePath, "r", 'utf-8')
            str = stream.read()
            stream.close()
            return str
    else:
        stream = codecs.open(filePath, "r", 'utf-8')
        str = stream.read()
        stream.close()
        return str

def removeIfExisits(filePath):
    if os.path.isfile(filePath):
        os.remove(filePath)
        print "removed : " + filePath
    else:
        print filePath + " doesn't exist"


def incrmentDic(dic,keyToAdd,value=1):
    if(keyToAdd in dic.keys()):
        value = dic[keyToAdd] + value
    dic[keyToAdd] = value

def listToString(inputlist, delmiter):
    retStr = ""
    for item in inputlist:
        if(not item):continue
        retStr += item + delmiter
    return retStr.strip(delmiter)

def dicToStr(dic):
    retStr = ""
    keys = dic.keys()
    for i in range(0,len(keys)):
        retStr += str(i) + "\t" + str(dic[keys[i]]) + "\t" + str(keys[i]) + "\n"
    return retStr

def removeEmptyItemsFromList(originalList):
    newList = []
    for item in originalList:
        item = item.strip()
        if(item==None or item == ""):continue
        newList.append(item)
    return newList

def stringToDict(dicStr):
    retDic = {}
    for line in dicStr.splitlines():
        splits = line.split("\t")
        key = splits[2].strip()
        value = splits[1].strip()
        retDic[key] = value
        #retStr += str(i) + "\t" + str(dic[keys[i]]) + "\t" + keys[i] + "\n"
    return retDic

def stringToList(inputStr,delemiter, delemiterGoesFirst=False):
    retList = []
    for item in inputStr.split(delemiter):
        item = item.strip()
        retList.append(item)
    if(delemiterGoesFirst):
        retList = retList[1:] #remove first item which is always empty string
    else :
        retList = retList[:-1] #remove last item which is always empty string
    return retList

def isEmptyList(alist):
    if not isinstance(alist,list): raise ValueError("instance must be a list")
    if not alist: return 1
    if(len(alist)==0): return 1
    for item in alist:
        if item.strip() != "":
            return 0
    return 1

def createDirIfNotExist(path):
    if not os.path.exists(path):
        print "creating dir :" + path
        os.makedirs(path)

def runCommand(cmd):
    print cmd
    p = subprocess.Popen(cmd, stdin=None, stdout=None, shell=True)
    os.popen(cmd)
    p.wait()

#splits the list by divisor (n),  returns a list of lists, with the size of n lists,and a remainder list (if any)
def splitList(mainList, divisor):
    retLists = []
    segmentSize = len(mainList)/divisor
    remainder = len(mainList)%divisor
    for i in range(0,divisor):
        retLists.append(mainList[i*segmentSize:(i+1)*segmentSize])
    if remainder>0:
        remainderList = mainList[divisor*segmentSize:divisor*segmentSize+remainder]
        retLists.append(remainderList)
    return retLists




#test = range(1,106)
#print splitList(test,10)
