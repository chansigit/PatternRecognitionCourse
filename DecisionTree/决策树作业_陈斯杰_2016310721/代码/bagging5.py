import GiniTree
import pickle
import sys
import copy
import time
import scipy.io
import numpy as np
import pandas as pd


batchNum = 5
trainBatchName="train%d.dataframe.pkl"%batchNum

fp = open(trainBatchName, 'rb')
train = pickle.load(fp)
fp.close()

fp = open("test.dataframe.pkl", 'rb')
test = pickle.load(fp)
fp.close()


t1=time.time()
model = GiniTree.treeGeneration(Dataset=train, Attributes=list(range(1,1+1200)))
t2=time.time()
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print("training takes %f secs"%(t2-t1))

predicted = GiniTree.predict(model, test)
correct=list(test['label'].values)
accuracy=sum([predicted[labelID]==correct[labelID] for labelID in range(len(predicted))])/len(correct)
print("validating accuracy=%f"%accuracy)

fp = open('predicted%d.pkl'%batchNum, 'wb')
pickle.dump(predicted, fp)
fp.close()