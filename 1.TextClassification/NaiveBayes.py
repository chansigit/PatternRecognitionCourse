import numpy as np
import pandas as pd
import math
import time
import pickle

# ========================= Load Train Data =============================
docLabel = []
for line in open("data/train.label"):
    docLabel.append(int(line))


def getDocLabel(docID):
    return docLabel[docID - 1]


docList = []
wordList = []
wcList = []
for line in open("data/train.data"):
    (docID, wordID, wordCount) = (int(line.split()[0]), int(
        line.split()[1]), int(line.split()[2]))
    docList.append(docID)
    wordList.append(wordID)
    wcList.append(wordCount)

wordTable = pd.DataFrame()
wordTable["docID"] = docList
wordTable["wordID"] = wordList
wordTable["wordCount"] = wcList
classLabel = list(map(getDocLabel, list(wordTable.docID)))
wordTable["classLabel"] = classLabel

# Total Document Count
N = len(docLabel)

# Total Unique Word (dimension of the word-vector)
D = len(set(wordList))


# ====================== Compute the prior distribution ==================
prior = []
for i in range(1, 21):
    prior.append(docLabel.count(i) / float(N))
logPrior = list(map(math.log, prior))


def getLogPrior(classLabel):
    return logPrior[classLabel - 1]


# ===================   Compute conditional distribution =================
def numeOfTheta(classNum, wordID):
    # select wordCount from wordTable where `classLable`=classNum and
    # `wordID`=wordID
    return sum(list(wordTable[(wordTable.classLabel == classNum) & (wordTable.wordID == wordID)].wordCount))


def denomOfTheta(classNum):
    # select wordCount from wordTable where `classLable`=classNum
    return sum(list(wordTable[wordTable.classLabel == classNum].wordCount))

denomOfThetaCache=[]
for classNum in range(1, 1+20):
    denomOfThetaCache.append(denomOfTheta(classNum=classNum))
def getDenomOfTheta(classNum):
    return denomOfThetaCache[classNum-1]

def Theta(classNum, wordID):
    #return numeOfTheta(classNum=classNum, wordID=wordID) / denomOfTheta(classNum=classNum)
    # cache speed up
    return numeOfTheta(classNum=classNum, wordID=wordID) / getDenomOfTheta(classNum=classNum)


def LogTheta(classNum, wordID):
    #return math.log(numeOfTheta(classNum=classNum, wordID=wordID) + 1) - math.log(denomOfTheta(classNum=classNum) + D)
    # cache speed up
    return ( math.log(numeOfTheta(classNum=classNum, wordID=wordID) + 1) \
           - math.log(getDenomOfTheta(classNum=classNum) + D))


# ==================== Store value of theta in a matrix ==================
# Extremely heavy computation
# Load precomputed value instead of compute it again
# start=time.time()
# thetaMatrix=[]
# for classNum in range(1, 1+20):
#    row=[]
#    for wordID in range(1, 1+D):
#        row.append(LogTheta(classNum=classNum, wordID=wordID))
#        if wordID % 1000 == 0:
#            print("class=%d word=%d" % (classNum,wordID) )
#    thetaMatrix.append(row)
# end=time.time()
# print(end-start)
#pickle.dump(thetaMatrix, open("theta.mat.data","wb")  )

# Load precomputed thetaMatrix stored in the serialized file
thetaMatrix = pickle.load(open("theta.mat.data", "rb"))


def getLogTheta(classNum, wordID):
    try:
        val = thetaMatrix[classNum - 1][wordID - 1]
    except IndexError:  # For unknown word
        val = -math.log(getDenomOfTheta(classNum=classNum) + D)
    return val


def getLogPosterior(classNum, wordsInDoc, wcInDoc):
    ret = getLogPrior(classLabel=classNum)
    for i in range(0, 0 + len(wordsInDoc)):
        w = wordsInDoc[i]
        c = wcInDoc[i]
        ret = ret + c * getLogTheta(classNum=classNum, wordID=w)
    return ret

# ==================== Training Dataset Validation =======================
print("Training Dataset Validation Begins!")
trainCorrect = 0
trainWrong = 0
trainValidateResult = open("trainValidation.txt", "w")
trainValidateResult.write("Training Dataset Validation: \n")
start = time.time()
for docID in range(1, 1 + N):
    w = list(wordTable[wordTable.docID == docID].wordID)
    wc = list(wordTable[wordTable.docID == docID].wordCount)
    logLikelihood = []
    for i in range(1, 1 + 20):
        logLikelihood.append(getLogPosterior(
            classNum=i, wordsInDoc=w, wcInDoc=wc))
    prediction = 1 + logLikelihood.index(max(logLikelihood))
    trainValidateResult.write("docID=  %d\tPrediction=  %d\t  ActualLabel=  %d\n" %
                              (docID, prediction, getDocLabel(docID)))
    if prediction == getDocLabel(docID):
        trainCorrect += 1
    else:
        trainWrong += 1
    if docID % 100 == 0:
        print("%d out of %d have finished." % (docID, N))
end = time.time()

print("Classification finished in %f seconds" % (end - start))
print("Training Dataset: %d correctly identified, %d wrong. Accuracy=%f"
      % (trainCorrect, trainWrong, trainCorrect / (trainCorrect + trainWrong)))
trainValidateResult.write("Training Dataset: %d correctly identified, %d wrong. Accuracy=%f"
                          % (trainCorrect, trainWrong, trainCorrect / (trainCorrect + trainWrong)))
trainValidateResult.close()

# ========================== Load Test Data =================================
tdocLabel = []
for line in open("data/test.label"):
    tdocLabel.append(int(line))


def getTestDocLabel(docID):
    return tdocLabel[docID - 1]
tdocList = []
twordList = []
twcList = []
for line in open("data/test.data"):
    (docID, wordID, wordCount) = (int(line.split()[0]),
                                  int(line.split()[1]),
                                  int(line.split()[2]))
    tdocList.append(docID)
    twordList.append(wordID)
    twcList.append(wordCount)

twordTable = pd.DataFrame()
twordTable["docID"] = tdocList
twordTable["wordID"] = twordList
twordTable["wordCount"] = twcList
tclassLabel = list(map(getTestDocLabel, list(twordTable.docID)))
twordTable["classLabel"] = tclassLabel

# Total Document Count
tN = len(tdocLabel)

# ======================== Test Dataset Validation =======================
print("Test Dataset Validation Begins!")
testCorrect = 0
testWrong = 0
testValidateResult =open("testValidation.txt", "w")
testValidateResult.write("Test Dataset Validation: \n")

start = time.time()
for docID in range(1, 1 + tN):
    w = list(twordTable[twordTable.docID == docID].wordID)
    wc = list(twordTable[twordTable.docID == docID].wordCount)
    logLikelihood = []
    for i in range(1, 1 + 20):
        logLikelihood.append(getLogPosterior(
            classNum=i, wordsInDoc=w, wcInDoc=wc))
    prediction = 1 + logLikelihood.index(max(logLikelihood))
    testValidateResult.write("docID=  %d\tPrediction=  %d\t  ActualLabel=  %d\n" %
                             (docID, prediction, getDocLabel(docID)))
    if prediction == getTestDocLabel(docID):
        testCorrect += 1
    else:
        testWrong += 1
    if docID % 100 == 0:
        print("%d out of %d have finished." % (docID, tN))
end = time.time()

print("Classification finished in %f seconds" % (end - start))
print("Test Dataset: %d correctly identified, %d wrong. Accuracy=%f"
      % (testCorrect, testWrong, testCorrect / (testCorrect + testWrong)))
testValidateResult.write("Test Dataset: %d correctly identified, %d wrong. Accuracy=%f"
                         % (testCorrect, testWrong, testCorrect / (testCorrect + testWrong)))
testValidateResult.close()
