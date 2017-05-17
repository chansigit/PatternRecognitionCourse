import GiniTree
import pickle
import sys
import copy
import time
import scipy.io
import numpy as np
import pandas as pd


batchNum = [1,2,3,4,5,6,7,8,9,10]

predicted=[]
for num in batchNum:
    predictedName='predicted%d.pkl'%num
    fp = open(predictedName, 'rb')
    predicted.append( pickle.load(fp) )
    fp.close()


fp = open("test.dataframe.pkl", 'rb')
test = pickle.load(fp)
fp.close()


combinedPredicted = []
for sampleID in range(len(predicted[0])):
    votePool=[]
    for classifier in range(10):
        votePool.append(predicted[classifier][sampleID])

    votePool=np.array(votePool)
    counts = np.bincount(votePool)
    mode=np.argmax(counts)
    combinedPredicted.append(mode)


correct=list(test['label'].values)
accuracy=sum([combinedPredicted[labelID]==correct[labelID] for labelID in range(len(combinedPredicted))])/len(correct)
print(combinedPredicted)
print(correct)
print("validating accuracy=%f"%accuracy)
