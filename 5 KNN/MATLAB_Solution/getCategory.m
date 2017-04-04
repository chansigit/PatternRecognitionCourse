function Score = getCategory(trainX, test,  sample)
    if sample ~= 0
        trainX = trainX(randsample(1:length(trainX), sample) ,:);
    end
    
    [len,wid]=size(trainX);
    Score= zeros(1,len);
    for i= 1:len
        Score(i)=tangentDist(trainX(i,:) , test,28,28, [1,1,1,1,1,1,1,1,1],0.0);
    end
end

