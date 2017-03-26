import numpy as np
import pandas as pd
import math
import time
import pickle
import copy
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

# Total Class Number
K = 20

# Load Data
trainX=np.zeros((N,D))
for i in range(0,0+len(docList)):
    trainX[ docList[i]-1 ][ wordList[i]-1 ] = wcList[i]
print("Data Loaded!")
# ====================== Define Training Function ==================
def classify(x, W):
    # Note that ClassNum is from 1 to 20
    # but our matrix begins with 0
    logNumerator = [np.dot(x, W[ClassNum-1]) for ClassNum in range(1, K-1+1)]
    logNumerator.append(0)
    return logNumerator.index(max(logNumerator))+1

def regularError(W, X, Y, lamda=0.001, method="one"):
    # every item in X is a D-dimension word vector
    sum= 0
    lenX=len(X)
    sampleCnt=1
    #for i in range(0, 0+len(X)):
    #    sum = sum + (Y[i] - classify(X[i], W)) ** 2
    if method== "one":
        choice = np.random.randint(0, lenX - 1)
        sum = sum + int(Y[choice] == classify(X[choice], W)) * int(Y[choice] == classify(X[choice], W))
        sampleCnt=1
    elif method == "mini_batch":
        for i in np.random.choice(lenX, 10):
            sum = sum + (Y[i] - classify(X[i], W)) ** 2
        sampleCnt = 10
    elif method == "batch":
        for i in range(0, 0+lenX):
            sum = sum + (Y[i] - classify(X[i], W)) ** 2
        sampleCnt = lenX
    sum = sum /(2*sampleCnt)
    sum = sum + 0.5*lamda*(W.max())*(W.max())
    return sum

def dW(W, step, X, Y, lamda=0.001, method="one"):
    # dW=regErr(W1)-regErr(W)
    W1=copy.deepcopy(W)
    ret=copy.deepcopy(W)
    for i in range(0, 0+K-1):
        for j in np.random.choice(D,1):
        #for j in range(0, W.shape[1]):
            W1[i][j] = W1[i][j]+step
            ret[i][j] = regularError(W1,X,Y,lamda, method=method) - regularError(W,X,Y,lamda,method=method)
    return ret

# ============================= Model Training =========================
# generate a random parameter matrix as init value
#W = np.random.random(size=(K-1, D))
#learningRate = 0.0025
#learningStep = 5.0*1e-5
#for loop in range(1,100000):
#    previousErr=regularError(W=W, X=trainX, Y=docLabel, lamda=1, method="one")
#    W=W-learningRate*dW(W, step=learningStep, X=trainX, Y=docLabel, lamda=1, method="one")
#    currentErr=regularError(W=W, X=trainX, Y=docLabel, lamda=1, method="one")
#    print("loop=%d Error=%.16f"%(loop, currentErr))
#    if currentErr<1e-5:
#        break
#    #if abs(currentErr-previousErr)<1e-7:
#    #    break
#
#print("Training finished")
#pickle.dump(W,open("softmax.mat.data","wb"))
W=pickle.load( open("softmax.mat.data","rb") )
# ======================== Training Dataset Validation ===================
trainCorrect = 0
trainWrong = 0
for i in range(0,0+N):
    if classify(trainX[i], W) == docLabel[i]:
        trainCorrect += 1
    else:
        trainWrong += 1
print("Accuracy on training dataset: %f" % (trainCorrect/(trainCorrect+trainWrong)))
