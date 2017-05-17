# ===============================================================
#
#                         Tree Definition
#
# ===============================================================
class TreeNode:
    def __init__(self, attrName):
        self.attr = attrName
        self.son = dict()
        self.isLeaf = False
        self.leafLabel = None

    def getAttr(self):
        return self.attr

    def setAttr(self, attrName):
        self.attr = attrName

    def addSon(self, attrVal, subTree):
        self.son[attrVal] = subTree

    def getAllSon(self):
        return list(self.son.values())

    def getSonByAttrVal(self, attrVal):
        return self.son[attrVal]

    def getCurrentAttrDivision(self):
        return list(self.son.keys())

    def setLeaf(self, label):
        self.isLeaf = True
        self.leafLabel = label
        self.attr = str(label)


class Tree:
    def __init__(self, rootNode):
        self.root = rootNode

    def add(self, nextAttrName, currentVal):
        if self.root == None:
            print("warning: attempt to add a subtree to a null tree")
            return
        subTree = Tree(TreeNode(nextAttrName))
        self.root.addSon(currentVal, subTree)

    def printTree(self):
        if self.root == None:
            return
        if self.root.isLeaf:
            print("Reach Leaf:%s" % self.root.leafLabel)
            return
        print("Node %s has sons:" % (self.root.getAttr()))

        for subtree in self.root.getAllSon():
            if subtree.root == None:
                continue
            print(">" + subtree.root.getAttr())
        subtrees = self.root.getAllSon()
        for subtree in subtrees:
            subtree.printTree()
        if subtrees == []:
            print(">None")
        print("--------------")

    def deleteTree(self):
        self.root = None

# ===============================================================
#
#                         Construct Tree v2
#
# ===============================================================
import math
def Ent(Dataset):
    labels = list(set(Dataset['label']))
    nrow = float(len(Dataset))
    def ratioItem(label):
        ratio= (float(sum(Dataset['label'] == label)) / nrow)
        if abs(ratio)<1e-5:
            return 0
        else:
            return ratio*math.log(ratio,2)
    entropy = - sum(map(ratioItem, labels))
    return entropy

def Gain(Dataset, attrName):
    gain = Ent(Dataset)
    nrow = float(len(Dataset))

    subDataset = Dataset[Dataset[attrName] == 0]
    nrowSub = float(len(subDataset))
    gain -= Ent(subDataset) * nrowSub/nrow

    subDataset = Dataset[Dataset[attrName] == 1]
    nrowSub = float(len(subDataset))
    gain -= Ent(subDataset) * nrowSub/nrow

    return gain

def attrSelection(Dataset, Attributes, cheatMode=False):
    if cheatMode == True:
        Dataset = Dataset.sample(frac=0.3)

    def fucktion(attr):
        return Gain(Dataset, str(attr))

    optAttr = max(Attributes, key=fucktion)
    return (optAttr, Gain(Dataset, str(optAttr)))
# =========================================================
# =========================================================

def identicalRows(df):
    for rowID in range(len(df) - 1):
        if not df.iloc[rowID, 1:].equals(df.iloc[rowID + 1, 1:]):
            return False
    return True


def treeGeneration(Dataset, Attributes):
    labels = Dataset['label']

    # 1. 生成空节点node
    node = TreeNode(attrName="unset")
    # tree = Tree(rootNode=node)

    # 2. Boundary Case 1
    if len(set(labels)) == 1:  # 如果Dataset中的样本全属于同一类别C
        C = labels.iloc[0]
        node.setLeaf(label=C)  # 将node标记成C类叶子节点
        tree = Tree(rootNode=node)
        return tree

    # 3. Boundary Case 2
    # 找到Dataset中样本数最多的类
    try:
        C = pd.Series.mode(labels)[0]
    except KeyError:
        C = labels.iloc[0]
    except IndexError:
        C = labels.iloc[0]
    print("most label=%d"%C)
    if Attributes == [] or identicalRows(Dataset.iloc[:, Attributes]):  # 如果Attributes=∅,或者Dataset在Attributes的这几个属性上取值相同
        node.setLeaf(label=C)  # 将node标记成叶子节点，其类别是Dataset中样本数最多的类
        tree = Tree(rootNode=node)
        return tree

    # 3.5 pre剪枝 (myYY)
    # if len(Dataset) < 21:
    #     node.setLeaf(label=C)  # 将node标记成叶子节点，其类别是Dataset中样本数最多的类
    #     tree = Tree(rootNode=node)
    #     return tree

    # 4. 选择最优划分属性: optAttr
    t1 = time.time()
    optAttr, gainVal = attrSelection(Dataset, Attributes, cheatMode=False)
    t2 = time.time()
    print("selected attr: %d with gain=%f,   size:%d, taking %f secs" % (optAttr,gainVal, len(Dataset), t2 - t1))
    sys.stdout.flush()
    node.setAttr(str(optAttr))

    # 4.5 剪枝 太小的增益直接标成叶子节点，已经比较纯了
    if gainVal < 0.003:
        node.setLeaf(label=C)  # 将node标记成叶子节点，其类别是Dataset中样本数最多的类
        tree = Tree(rootNode=node)
        return tree

    # 5. 为optAttr属性的每一个值
    for optAttrVal in [0, 1]:
        subDataset = Dataset[Dataset[str(optAttr)] == optAttrVal]
        if len(subDataset) == 0:  # 为node加上一个分支，分支为叶子节点，叶子节点类别标记为Dataset中样本数最多的类
            leafNode = TreeNode(attrName="")
            leafNode.setLeaf(label=C)
            leafNodeTree = Tree(rootNode=leafNode)
            node.addSon(attrVal=optAttrVal, subTree=leafNodeTree)
        else:
            reducedAttributes = copy.deepcopy(Attributes)
            reducedAttributes.remove(optAttr)
            # import pdb;pdb.set_trace();
            sonTree = treeGeneration(Dataset=subDataset, Attributes=reducedAttributes)
            node.addSon(attrVal=optAttrVal, subTree=sonTree)

    tree = Tree(rootNode=node)
    return tree

# ===============================================================
#
#                      Train Data and Test Data
#
# ===============================================================
import sys
import copy
import time
import scipy.io
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


train = cleanData.sample(frac=0.08)
test =  cleanData.drop(train.index)

print("train size=%d"%len(train))
print("test size=%d"%len(test))

# ===============================================================
#
#                         Train Model
#
# ===============================================================

trainBegin=time.time()
dct1=treeGeneration(Dataset=train, Attributes=list(range(1,1+1200)))
trainEnd = time.time()
print("train finished, taking %f secs"%(trainEnd-trainBegin))
dct1.printTree()

# ===============================================================
#
#                         Predictor Definition
#
# ===============================================================
def predict(decisionTree, Dataset):
    predicted= []
    for rowID in range(len(Dataset)):
        DataItem = Dataset.iloc[rowID,:]
        currNode = decisionTree.root
        while not currNode.isLeaf:
            judgeAttrName=currNode.getAttr()
            judgeVal = DataItem[judgeAttrName]
            currNode = currNode.getSonByAttrVal(judgeVal).root
        predicted.append(currNode.leafLabel)
    return predicted


res=predict(dct1, test)
correct=list(test['label'].values)
accuracy=sum([res[labelID]==correct[labelID] for labelID in range(len(res))])/len(correct)
print(accuracy)