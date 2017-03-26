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
trainX = np.zeros((N, D))
for i in range(0, 0 + len(docList)):
    trainX[docList[i] - 1][wordList[i] - 1] = wcList[i]
print("Data Loaded!")
# ====================== Define Training Function ==================


def classify(x, W):
    candidate = [np.dot(x, W[j]) for j in range(0, 0 + K)]
    return candidate.index(max(candidate)) + 1


def probability(Y, X, W):
    logNume = np.dot(W[Y - 1], X)
    L = [np.dot(W[j], X) for j in range(0, 0 + K)]
    A = max(L)
    L = L - A
    logDenom = A + math.log(sum([math.exp(Li) for Li in L]))
    logPr = logNume - logDenom
    return math.exp(logPr)


def pointUpdate(i, j, X, Y, W):
    return X[i - 1] * (int(Y[i - 1] == j) - probability(Y=j, X=X[i - 1], W=W))


def classUpdate(X, Y, W, lamda, j):
    #print("classupdate j=%d"%j)
    N = len(X)
    sampleNum = 1
    # return (-1/N)*sum(np.array([pointUpdate(i,j,X,Y,W) for i in
    # range(1,1+N)])) + lamda*W[j-1]
    return (-1 / sampleNum) * sum(np.array([pointUpdate(i, j, X, Y, W) for i in np.random.choice(range(1, 1 + N), sampleNum)])) + lamda * W[j - 1]


def dW(X, Y, W, lamda):
    #print("dw in")
    rows = [classUpdate(X, Y, W, lamda, j) for j in range(1, 1 + K)]
    return np.array(rows)


# ============================= Model Training =========================
# generate a random parameter matrix as init value
#W = 1e-4*np.random.random(size=(K, D))
W = pickle.load(open("softmax.mat.data", "rb"))
learningStep = 1.56 * 1e-8
start = end = time.time()
for loop in range(1, 1000):
    W = W - learningStep * dW(X=trainX, Y=docLabel, W=W, lamda=0.25)
    if loop % 200 == 0:
        end = time.time()
        trainCorrect = 0
        trainWrong = 0
        for i in range(0, 0 + N):
            if classify(trainX[i], W) == docLabel[i]:
                trainCorrect += 1
            else:
                trainWrong += 1
        print("Loop %d: Accuracy on training dataset: %f, taking %d secs" %
              (loop, trainCorrect / (trainCorrect + trainWrong), (end - start)))
        #pickle.dump(W, open("softmax.mat.data", "wb"))
        start = time.time()
    #if loop %500 ==0
    #    pickle.dump(W, open("softmax.mat.data", "wb"))

pickle.dump(W, open("softmax.mat.data", "wb"))
print("Training finished. W matrix has been serialized.")

#W=pickle.load( open("softmax.mat.data","rb") )
print("Test Dataset Validation Begins!")
testCorrect = 0
testWrong = 0
testValidateResult =open("testLogisticValidation.txt", "w")
testValidateResult.write("Test Dataset Validation: \n")


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
    if wordID <= D: # ignore the existance of novel words
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

# Test Dataset
testX= np.zeros((tN,D))
for i in range(0,len(tdocList)):
    testX[ tdocList[i]-1  ][ twordList[i]-1 ]=twcList[i]

# ======================== Test Dataset Validation =======================
start = time.time()
for docID in range(1, 1 + tN):
    prediction = classify(x=testX[docID-1],W=W)
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
