import numpy as np
import matplotlib as mp
from math import sqrt
import math
import scipy.stats
from scipy.integrate import simps
from matplotlib import pyplot
import random
μ1 = -1
μ2 = 1
σ1 = σ2 = 1
c1 = 0.2
c2 = 0.8

def genSample():
    if random.uniform(0, 1) < c1:
        return np.random.normal(loc=μ1, scale=σ1)
    else:
        return np.random.normal(loc=μ2, scale=σ2)
X = []
for i in range(0, 1000):
    X.append(genSample())

def truePDF(x):
    return c1*scipy.stats.norm.pdf(x, loc=μ1, scale=σ1)+c2*scipy.stats.norm.pdf(x, loc=μ2, scale=σ2)


def winFunc(x, a):
  if -a<x<a:
    return (a-abs(x))/(a*a)
  else:
    return 0


def prazenKDE(x, sample, width, kernel):
  return sum([kernel((x - xi) / width, a=width) for xi in sample]) / (len(sample) * width)




sampleSize = 100
x_axis = np.linspace(-10, 10, sampleSize)
true_y = [truePDF(x) for x in x_axis]


def errMeanVar(width):
  a = width
  errList = []
  for i in range(0, 10):
    y_axis = [prazenKDE(x=x, sample=X, width=a, kernel=winFunc)
              for x in x_axis]
    err = [(true_y[i] - y_axis[i])**2 for i in range(0, len(y_axis))]
    I = simps(err, x_axis)
    errList.append(I)
  return (np.mean(errList), np.var(errList))

f=open('error_uniform.result.txt','w')
import sys
old=sys.stdout # Save stdout
sys.stdout=f   # Redirect output to file

mp.pyplot.figure()
sampleSize = 5
aList = []
errList = []
for a in np.linspace(start=0.80, stop=1.06, num=25):
  err = errMeanVar(width=a)
  print("n=%d,a=%.2f, MeanErr&VarErr=%s" %
        (sampleSize, a, err))
  aList.append(a)
  errList.append(err)
mp.pyplot.plot(aList, errList, color="red", label="n=%d" % sampleSize, linewidth=1)

sampleSize = 10
aList = []
errList = []
for a in np.linspace(start=0.80, stop=1.05, num=25):
  err = errMeanVar(width=a)
  print("n=%d,a=%.2f, MeanErr&VarErr=%s" %
        (sampleSize, a, err))
  aList.append(a)
  errList.append(err)
mp.pyplot.plot(aList, errList, color="gold", label="n=%d" % sampleSize, linewidth=1)

sampleSize = 50
aList = []
errList = []
for a in np.linspace(start=0.80, stop=1.04, num=25):
  err = errMeanVar(width=a)
  print("n=%d,a=%.2f, MeanErr&VarErr=%s" %
        (sampleSize, a, err))
  aList.append(a)
  errList.append(err)
mp.pyplot.plot(aList, errList, color="springgreen", label="n=%d" % sampleSize, linewidth=1)

sampleSize = 100
aList = []
errList = []
for a in np.linspace(start=0.80, stop=1.07, num=25):
  err = errMeanVar(width=a)
  print("n=%d,a=%.2f, MeanErr&VarErr=%s" %
        (sampleSize, a, err))
  aList.append(a)
  errList.append(err)
mp.pyplot.plot(aList, errList, color="dodgerblue", label="n=%d" % sampleSize, linewidth=1)

sampleSize = 500
aList = []
errList = []
for a in np.linspace(start=0.80, stop=1.08, num=25):
  err = errMeanVar(width=a)
  print("n=%d,a=%.2f, MeanErr&VarErr=%s" %
        (sampleSize, a, err))
  aList.append(a)
  errList.append(err)
mp.pyplot.plot(aList, errList, color="purple", label="n=%d" % sampleSize, linewidth=1)

mp.pyplot.xlabel('$a$')
mp.pyplot.ylabel('$error$')
mp.pyplot.legend(loc="best")
mp.pyplot.savefig(filename="error_uniform.png", dpi=1200)

sys.stdout=old # Resume output to screen
f.close() 
