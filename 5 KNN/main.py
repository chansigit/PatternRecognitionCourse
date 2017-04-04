import scipy.io as sio
import numpy as np

matFileName = "mnist_521303078.mat"
data=sio.loadmat(matFileName)
trainX=data['train_X']
trainY=data['train_Y']
testX=data['test_X']
testY=data['test_Y']
print(trainX.shape)
print(trainY.shape)
x=np.reshape(trainX[30010,] , (28,28))
#print()


from matplotlib import pyplot as plt
plt.imshow(x, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
plt.show()
