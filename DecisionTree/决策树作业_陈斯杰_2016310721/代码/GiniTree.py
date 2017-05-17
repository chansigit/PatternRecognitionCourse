import sys
import copy
import time
import scipy.io
import numpy as np
import pandas as pd
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
def gini(Dataset):
    labels = list(set(Dataset['label']))
    nrow = float(len(Dataset))

    def ratioSqr(label):
        return (float(sum(Dataset['label'] == label)) / nrow) ** 2

    giniVal = 1.0 - sum(map(ratioSqr, labels))
    return giniVal


def giniIndex(Dataset, attrName):
    giniIndexVal = 0.0
    nrow = float(len(Dataset))

    subDataset = Dataset[Dataset[attrName] == 0]
    nrowSub = float(len(subDataset))
    giniIndexVal += gini(subDataset) * nrowSub

    subDataset = Dataset[Dataset[attrName] == 1]
    nrowSub = float(len(subDataset))
    giniIndexVal += gini(subDataset) * nrowSub

    return giniIndexVal / nrow


def attrSelection(Dataset, Attributes, cheatMode=False):
    if cheatMode == True:
        Dataset = Dataset.sample(frac=0.3)

    def fucktion(attr):
        return giniIndex(Dataset, str(attr))

    optAttr = min(Attributes, key=fucktion)
    return (optAttr, giniIndex(Dataset, str(optAttr)))


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
    optAttr, giniCoef = attrSelection(Dataset, Attributes, cheatMode=False)
    t2 = time.time()
    print("selected attr: %d with gini=%f,   size:%d, taking %f secs" % (optAttr, giniCoef, len(Dataset), t2 - t1))
    sys.stdout.flush()
    node.setAttr(str(optAttr))

    # 4.5 剪枝 太小的基尼系数直接标成叶子节点，已经比较纯了
    if giniCoef < 0.1:
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