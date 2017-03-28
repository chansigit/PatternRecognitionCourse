import numpy as np
import matplotlib.pyplot as plt

def linearlySeperablePoints(category, slope, intercept, expScale, xmin, xmax,margin=1):
    x=np.random.uniform(xmin,xmax)
    y=slope*x+intercept + category*(margin+np.random.exponential(scale=expScale))
    return (x,y)

# ------------------------ parameter ------------------------
xmin, xmax = (1,100)
slope, intercept = (3,15)
expScale, margin = (70,60)



# ------------------- generate points -----------------------
cluster1 = []
cluster2=[]
dataset=[]
for i in range(0,200):
    pt1 = linearlySeperablePoints(category=1, expScale=expScale,
                                 slope=slope, intercept=intercept,
                                 xmin=xmin, xmax=xmax, margin=margin)
    cluster1.append(pt1)
    dataset.append((pt1,1))
    pt2 = linearlySeperablePoints(category=-1, expScale=expScale,
                                 slope=slope, intercept=intercept,
                                 xmin=xmin, xmax=xmax, margin=margin)
    cluster2.append(pt2)
    dataset.append((pt2, -1))



# ---------------------- draw points ------------------------
plt.figure()
plt.scatter([pt[0] for pt in cluster1], [pt[1] for pt in cluster1],
            color='r', label='category 1', s=13)
plt.scatter([pt[0] for pt in cluster2], [pt[1] for pt in cluster2],
            color='b', label='category 2', s=13)
plt.xlabel("x1")
plt.ylabel("x2")


# ------------------------------
def scalarMul(scalar, vec):
    return tuple(scalar*elem for elem in vec)

def Perceptron(dataset):
    # augmented weight vector = (w0, w_x1, w_x2)
    # augmented sample vector = (1,  x1,   x2  )
    # the sign of the dot product of wv and sv determines the class

    # Point Regularization
    regAugDataset=[]
    for item in dataset:
        point=item[0]
        category = item[1]
        augPoint = scalarMul(scalar=category, vec=(1, point[0], point[1]))
        #print(point)
        #print(augPoint)
        regAugDataset.append((augPoint, category))
    # Training
    wv=[0,0,0]
    while True:
        misclassCnt=0
        for item in regAugDataset:
            regAugPt= item[0]
            if np.dot(regAugPt, wv)<=0:
                wv=list(np.add(wv, regAugPt))
                misclassCnt += 1
        if misclassCnt==0:
            break
    return wv

def MarginPerceptron(dataset,gamma):
    # augmented weight vector = (w0, w_x1, w_x2)
    # augmented sample vector = (1,  x1,   x2  )
    # the sign of the dot product of wv and sv determines the class

    # Point Regularization
    regAugDataset=[]
    for item in dataset:
        point=item[0]
        category = item[1]
        augPoint = scalarMul(scalar=category, vec=(1, point[0], point[1]))
        #print(point)
        #print(augPoint)
        regAugDataset.append((augPoint, category))
    # Training
    wv=[0,0,0]
    adjustCnt=0
    while True:
        misclassCnt=0
        for item in regAugDataset:
            regAugPt= item[0]

            if np.dot(regAugPt, wv)<=gamma:
                wv=list(np.add(wv, regAugPt))
                misclassCnt += 1
                adjustCnt+=1
        if misclassCnt==0:
            break
    print("converge after %d rounds"%adjustCnt)
    return wv


gamma=20000000
(w0,w_x1,w_x2)=Perceptron(dataset=dataset)
(a0,a_x1,a_x2)=MarginPerceptron(dataset=dataset, gamma=gamma)
print((w0,w_x1,w_x2))
print((a0,a_x1,a_x2))
sep_x = np.arange(xmin,xmax+1)
sep_y = (-w_x1*sep_x-w0)/w_x2
margin_x = np.arange(xmin,xmax+1)
margin_y= (-a_x1*margin_x-a0)/a_x2
plt.plot(sep_x, sep_y, 'g-', lw=1, label="classic")
plt.plot(sep_x, margin_y, 'b-', lw=1, label="margin")


plt.legend(loc="best")
filename="scatterFig_gamma=%d.png"%gamma
plt.savefig(filename, dpi=1200)
plt.show()