function class = getCategory(trainX, trainY, test,  sample,k)
    if sample ~= 0
        rowID=randsample(1:length(trainX), sample);
        trainX = trainX(rowID,:);
        trainY = trainY(:,rowID);
    end
    
    [len,wid]=size(trainX);
    Score= zeros(1,len);
    for i= 1:len
        Score(i)=tangentDist(trainX(i,:) , test,28,28, [1,1,1,1,1,1,1,1,1],0.0);
    end
    
    [scoreSort scoreIdx] = getNElements(Score, k);
    class=mode(trainY(scoreIdx));
end

