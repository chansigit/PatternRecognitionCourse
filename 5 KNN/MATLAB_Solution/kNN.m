% % images = 255*loadMNISTImages('mnist_dataset/train-images.idx3-ubyte');
% % labels = 255*loadMNISTLabels('mnist_dataset/train-labels.idx1-ubyte');
% % display_network(images(:,1:400));
% % images(:,1)
% % disp(labels(1:20));
% 
% mex tangentDistanceCImpl/tangentDist.c  tangentDistanceCImpl/ortho.c tangentDistanceCImpl/td.c
% 
% 
% %Train
% Xtrain = 256*loadMNISTImages('mnist_dataset/train-images.idx3-ubyte');
% Ytrain = loadMNISTLabels('mnist_dataset/train-labels.idx1-ubyte');
% Xtrain=Xtrain';
% 
% %Test
% Xtest = 256*loadMNISTImages('mnist_dataset/t10k-images.idx3-ubyte');
% Ytest = loadMNISTLabels('mnist_dataset/t10k-labels.idx1-ubyte');
% Xtest=Xtest';
% 
% size(Xtrain)
% 
% X1=Xtrain(51,:);
% X2=Xtrain(20,:);
% imagesc(reshape(X1,28,28),'EraseMode','none',[-1 1])
% imagesc(reshape(X2,28,28),'EraseMode','none',[-1 1])
% 
% z=tangentDist(X1,X2, 28,28,[1,1,1,1,1,1,1,0,0],0.0)
clear
clc
mex tangentDistanceCImpl/tangentDist.c  tangentDistanceCImpl/ortho.c tangentDistanceCImpl/td.c
load("mnist_dataset/mnist_521303078.mat")

x1=train_X(1,:);
x2=train_X(12,:);
%imshowpair(reshape(x1,28,28), reshape(x2,28,28), 'montage')


tic
minkow1Dist=minkowskiDist(x1,x2,1)
toc

tic
minkow1o5Dist=minkowskiDist(x1,x2,1.5)
toc

tic
minkow2Dist=minkowskiDist(x1,x2,2)
toc 

tic
minkow3Dist=minkowskiDist(x1,x2,3)
toc

tic
tangentDist=tangentDist(x1, x2, 28,28,[1,1,1,1,1,1,1,1,1],0.0)
toc

tic
getCategory(train_X, x1, 10)
toc
