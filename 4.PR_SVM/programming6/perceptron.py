import numpy as np
import matplotlib.pyplot as plt

def linearlySeperablePoints(category, slope, intercept, expScale, xmin, xmax,margin=1):
    x=np.random.uniform(xmin,xmax)
    y=slope*x+intercept + category*(margin+np.random.exponential(scale=expScale))
    return (x,y)

# ------------------------ parameter ------------------------
xmin, xmax = (1,100)
slope, intercept = (3,15)
expScale, margin = (70,20)



# ------------------- generate points -----------------------
cluster1 = []
cluster2=[]
dataset=[]
for i in range(0,100):
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
plt.legend(loc="best")
plt.savefig("scatterFig.png")
plt.show()


# ------------------------------

