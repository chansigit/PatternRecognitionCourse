import numpy as np
import matplotlib as mp
import math
from matplotlib import pyplot
from math import sqrt
import scipy.stats

def pn(x,h,a):
  if x<0:
    return 0
  elif x<=a:
    return (1/a)*(1-math.exp(-x/h))
  else:
    return (1/a)*(math.exp(a/h)-1)*math.exp(-x/h)

a=1
mp.pyplot.figure()
X = np.linspace(-1,2,50)
Y= [pn(x=x,a=a, h=1) for x in X]
mp.pyplot.plot(X, Y, color="red", label="h=1", linewidth=1)
Y= [pn(x=x,a=a, h=1/4) for x in X]
mp.pyplot.plot(X, Y, color="gold", label="h=1/4", linewidth=1)
Y= [pn(x=x,a=a, h=1/16) for x in X]
mp.pyplot.plot(X, Y, color="springgreen", label="h=1/16", linewidth=1)



mp.pyplot.xlabel('$x$')
mp.pyplot.ylabel('$pn$')
mp.pyplot.legend(loc="best")
mp.pyplot.savefig(filename="problem2_1.png", dpi=1200)

mp.pyplot.figure()
X= np.linspace(0,0.05,100)
Y= [pn(x=x,a=1, h=0.00217) for x in X]
mp.pyplot.plot(X, Y, color="red", label="h=0.00217", linewidth=1)
mp.pyplot.xlabel('$x$')
mp.pyplot.ylabel('$pn$')
mp.pyplot.legend(loc="best")
mp.pyplot.savefig(filename="problem2_2.png", dpi=1200)