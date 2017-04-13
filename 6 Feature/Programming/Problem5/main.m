%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML114
% Project Title: Implementation of Linear Discriminant Analysis in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

clc;
clear;
close all;

%% Load Data

load('orl_faces.mat')
X= data;
T= label;

[Y, W, lambda] = LDA(X,T)

for classID= 1:40
    find(label==classID)
end
 label(label==39 | label==40)


tic
tot=0;
ClassCnt = 40
for classI = 1:ClassCnt-1
    for classJ = classI+1:ClassCnt
        X=data(find(label==classI | label==classJ),:);
        Y=label(label==classI | label==classJ);
        tot= tot+1;
    end
end
tot
toc


%% Plot Results

figure;

D = size(X,2);
for d=1:D
    % Original Data
    subplot(D,2,2*d-1);
    plot(X(:,d));
    ylabel(['x_' num2str(d)]);
    if d==D
        xlabel('Sample Index');
    end
    if d==1
        title('Original Data');
    end
    grid on;
    
    % Transformed Data
    subplot(D,2,2*d);
    plot(Y(:,d));
    ylabel(['y_' num2str(d)]);
    if d==D
        xlabel('Sample Index');
    end
    if d==1
        title('LDA Output');
    end
    grid on;
    
end
