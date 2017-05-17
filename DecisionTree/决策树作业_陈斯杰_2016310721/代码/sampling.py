# ===============================================================
#
#                      Train Data and Test Data
#
# ===============================================================
import sys
import copy
import time
import scipy.io
import random
rawData = scipy.io.loadmat('Sogou_data/Sogou_webpage.mat')
docLabels   = rawData['doclabel']
wordVectors = rawData['wordMat']
docCnt=len(docLabels)
wordCnt =wordVectors.shape[1]


import numpy as np
import pandas as pd
df_wordVectors = pd.DataFrame(data=docLabels, columns=['label'])
df_docLabels   = pd.DataFrame(wordVectors,columns=[str(x) for x in range(1,1+wordCnt)])
cleanData = pd.concat([df_wordVectors, df_docLabels], axis=1)


labelCnt =len(set(cleanData['label']))


import random
# 取80%数据训练，留20%数据测试
train = cleanData.sample(frac=0.8)
test =  cleanData.drop(train.index)

random.seed(1)
train1 = train.sample(frac=0.2)
random.seed(2)
train2 = train.sample(frac=0.2)
random.seed(3)
train3 = train.sample(frac=0.2)
random.seed(4)
train4 = train.sample(frac=0.2)
random.seed(5)
train5 = train.sample(frac=0.2)
random.seed(6)
train6 = train.sample(frac=0.2)
random.seed(7)
train7 = train.sample(frac=0.2)
random.seed(8)
train8 = train.sample(frac=0.2)
random.seed(9)
train9 = train.sample(frac=0.2)
random.seed(10)
train10= train.sample(frac=0.2)

import pickle
fp = open('train1.dataframe.pkl', 'wb')
pickle.dump(train1, fp)
fp.close()

fp = open('train2.dataframe.pkl', 'wb')
pickle.dump(train2, fp)
fp.close()

fp = open('train3.dataframe.pkl', 'wb')
pickle.dump(train3, fp)
fp.close()

fp = open('train4.dataframe.pkl', 'wb')
pickle.dump(train4, fp)
fp.close()

fp = open('train5.dataframe.pkl', 'wb')
pickle.dump(train5, fp)
fp.close()

fp = open('train6.dataframe.pkl', 'wb')
pickle.dump(train6, fp)
fp.close()

fp = open('train7.dataframe.pkl', 'wb')
pickle.dump(train7, fp)
fp.close()

fp = open('train8.dataframe.pkl', 'wb')
pickle.dump(train8, fp)
fp.close()

fp = open('train9.dataframe.pkl', 'wb')
pickle.dump(train9, fp)
fp.close()

fp = open('train10.dataframe.pkl', 'wb')
pickle.dump(train10, fp)
fp.close()

fp = open('test.dataframe.pkl', 'wb')
pickle.dump(test, fp)
fp.close()

print("batch sizes:")
print([len(train1), len(train2),len(train3),len(train4),len(train5),len(train6),len(train7),len(train8),len(train9),len(train10)])
