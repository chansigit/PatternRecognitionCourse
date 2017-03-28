import numpy as np
import matplotlib.pyplot as plt


def linearlySeperablePoints(category, slope, intercept, expScale, xmin, xmax, margin=1):
    x = np.random.uniform(xmin, xmax)
    y = slope * x + intercept + category * \
        (margin + np.random.exponential(scale=expScale))
    return (x, y)

# ------------------------ parameter ------------------------
xmin, xmax = (0, 1)
slope, intercept = (1.96, 0.15)
expScale, margin = (0.05, 0.195)


# ------------------- generate points -----------------------
cluster1 = []
cluster2 = []
dataset = []
for i in range(0, 200):
    pt1 = linearlySeperablePoints(category=1, expScale=expScale,
                                  slope=slope, intercept=intercept,
                                  xmin=xmin, xmax=xmax, margin=margin)
    cluster1.append(pt1)
    dataset.append((pt1, 1))
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
    return tuple(scalar * elem for elem in vec)


def Perceptron(dataset):
    # augmented weight vector = (w0, w_x1, w_x2)
    # augmented sample vector = (1,  x1,   x2  )
    # the sign of the dot product of wv and sv determines the class

    # Point Regularization
    regAugDataset = []
    for item in dataset:
        point = item[0]
        category = item[1]
        augPoint = scalarMul(scalar=category, vec=(1, point[0], point[1]))
        # print(point)
        # print(augPoint)
        regAugDataset.append((augPoint, category))
    # Training
    wv = [0, 0, 0]
    while True:
        misclassCnt = 0
        for item in regAugDataset:
            regAugPt = item[0]
            if np.dot(regAugPt, wv) <= 0:
                wv = list(np.add(wv, regAugPt))
                misclassCnt += 1
        if misclassCnt == 0:
            break
    return wv


def MarginPerceptron(dataset, gamma):
    # augmented weight vector = (w0, w_x1, w_x2)
    # augmented sample vector = (1,  x1,   x2  )
    # the sign of the dot product of wv and sv determines the class

    # Point Regularization
    regAugDataset = []
    for item in dataset:
        point = item[0]
        category = item[1]
        augPoint = scalarMul(scalar=category, vec=(1, point[0], point[1]))
        # print(point)
        # print(augPoint)
        regAugDataset.append((augPoint, category))
    # Training
    wv = [0, 0, 0]
    adjustCnt = 0
    while True:
        misclassCnt = 0
        for item in regAugDataset:
            regAugPt = item[0]

            if np.dot(regAugPt, wv) <= gamma:
                wv = list(np.add(wv, regAugPt))
                misclassCnt += 1
                adjustCnt += 1
        if misclassCnt == 0:
            break
    print("γ=%f, converge after %d rounds" % (gamma,adjustCnt))
    return wv



(w0, w_x1, w_x2) = Perceptron(dataset=dataset)
(a0, a_x1, a_x2) = MarginPerceptron(dataset=dataset, gamma=0.005)
(b0, b_x1, b_x2) = MarginPerceptron(dataset=dataset, gamma=0.010)
(c0, c_x1, c_x2) = MarginPerceptron(dataset=dataset, gamma=0.050)
(d0, d_x1, d_x2) = MarginPerceptron(dataset=dataset, gamma=0.100)
(e0, e_x1, e_x2) = MarginPerceptron(dataset=dataset, gamma=0.250)
(f0, f_x1, f_x2) = MarginPerceptron(dataset=dataset, gamma=0.500)
(g0, g_x1, g_x2) = MarginPerceptron(dataset=dataset, gamma=1.500)

print((w0, w_x1, w_x2))
print((a0, a_x1, a_x2))
sep_x = np.arange(xmin, xmax + 1)
sep_y = (-w_x1 * sep_x - w0) / w_x2
margin_ya = (-a_x1 * sep_x - a0) / a_x2
margin_yb = (-b_x1 * sep_x - b0) / b_x2
margin_yc = (-c_x1 * sep_x - c0) / c_x2
margin_yd = (-d_x1 * sep_x - d0) / d_x2
margin_ye = (-e_x1 * sep_x - e0) / e_x2
margin_yf = (-f_x1 * sep_x - f0) / f_x2
margin_yg = (-g_x1 * sep_x - g0) / g_x2

plt.plot(sep_x, margin_ya, 'g',   lw=1, label="margin $\gamma$=0.005")
plt.plot(sep_x, margin_yb, 'lime',    lw=1, label="margin $\gamma$=0.010")
plt.plot(sep_x, margin_yc, 'springgreen', lw=1, label="margin $\gamma$=0.050")
plt.plot(sep_x, margin_yd, 'aqua',        lw=1, label="margin $\gamma$=0.100")
plt.plot(sep_x, margin_ye, 'steelblue',   lw=1, label="margin $\gamma$=0.250")
plt.plot(sep_x, margin_yf, 'royalblue',   lw=1, label="margin $\gamma$=0.500")
plt.plot(sep_x, margin_yg, 'blue',        lw=1, label="margin $\gamma$=1.500")
plt.plot(sep_x, sep_y, 'magenta', lw=2, label="classic")

plt.legend(loc="best")
filename = "scatterFig.png" 
plt.savefig(filename, dpi=1200)
plt.show()
