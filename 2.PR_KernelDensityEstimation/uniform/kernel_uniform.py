import numpy as np
import matplotlib as mp
from matplotlib import pyplot
from math import sqrt
import random
μ1 = -1
μ2 = 1
σ1 = σ2 = 1
c1 = 0.2


def genSample():
    if random.uniform(0, 1) < c1:
        return np.random.normal(loc=μ1, scale=σ1)
    else:
        return np.random.normal(loc=μ2, scale=σ2)
X = []
for i in range(0, 10000):
    X.append(genSample())


def winFunc(x, a):
    if -a / 2 <= x <= a / 2:
        return 1 / a
    else:
        return 0


def prazenKDE(x, sample, width, kernel):
    return sum([kernel((x - xi) / width, a=width) for xi in sample]) / (len(sample) * width)


mp.pyplot.figure()
sampleSize=10000
x_axis = np.linspace(-10, 10, sampleSize)

a=0.1
labelstr="n=%d a=%.2f"%(sampleSize,a)
y_axis = [prazenKDE(x=x, sample=X, width=a, kernel=winFunc)
          for x in x_axis]
mp.pyplot.plot(x_axis, y_axis, color="red", label=labelstr, linewidth=1)


a=1
labelstr="n=%d a=%.2f"%(sampleSize,a)
y_axis = [prazenKDE(x=x, sample=X, width=a, kernel=winFunc)
          for x in x_axis]
mp.pyplot.plot(x_axis, y_axis, color="gold", label=labelstr, linewidth=1)


a=2
labelstr="n=%d a=%.2f"%(sampleSize,a)
y_axis = [prazenKDE(x=x, sample=X, width=a, kernel=winFunc)
          for x in x_axis]
mp.pyplot.plot(x_axis, y_axis, color="springgreen", label=labelstr, linewidth=1)


a=3
labelstr="n=%d a=%.2f"%(sampleSize,a)
y_axis = [prazenKDE(x=x, sample=X, width=a, kernel=winFunc)
          for x in x_axis]
mp.pyplot.plot(x_axis, y_axis, color="dodgerblue",label=labelstr, linewidth=1)


a=4
labelstr="n=%d a=%.2f"%(sampleSize,a)
y_axis = [prazenKDE(x=x, sample=X, width=a, kernel=winFunc)
          for x in x_axis]
mp.pyplot.plot(x_axis, y_axis, color="purple",label=labelstr, linewidth=1)


mp.pyplot.xlabel('$x$')
mp.pyplot.ylabel('$KDE$')
mp.pyplot.legend(loc="upper left")
#mp.pyplot.show()
mp.pyplot.savefig(filename="kernel_uniform_n=%d.png"%sampleSize, dpi=1200)