clear;
clc;
profile_master = parallel.importProfile('LocalProfile1');
parallel.defaultClusterProfile(profile_master)


mex tangentDistanceCImpl/tangentDist.c  tangentDistanceCImpl/ortho.c tangentDistanceCImpl/td.c
load("mnist_dataset/mnist_521303078.mat");

x1=train_X(1,:);
x2=train_X(12,:);
%imshowpair(reshape(x1,28,28), reshape(x2,28,28), 'montage')


[n, wid] = size(test_X);
validate = zeros(1,n);


sample = 6000;
k      = 3;
method = 'mink1.5';
TaskName=sprintf("TrainNum%d-k%d-%s",sample,k,method);

tic
parfor i = 1:n
    validate(i)= test_Y(i) == getCategory(train_X, train_Y, test_X(i,:), sample, k, method);
end
elapse=toc

save(TaskName+".res.mat","validate");
acc=sum(validate)/length(validate);


fprintf(fopen(TaskName+".txt", 'w'), "TrainNum=%d  k=%d  %s\ntime=%f secs, accuracy=%.10f",sample,k,method, elapse, acc);

poolobj = gcp('nocreate');
delete(poolobj);