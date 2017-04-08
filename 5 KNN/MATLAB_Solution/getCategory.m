function class = getCategory(trainX, trainY, test, sample, k, method)
    if sample ~= 0
        rowID=randsample(1:length(trainX), sample);
        trainX = trainX(rowID,:);
        trainY = trainY(:,rowID);
    end
    
    [len,wid]=size(trainX);
    Score= zeros(1,len);
    if method == "mink1"
        for i= 1:len
            Score(i)=minkowskiDist(trainX(i,:), test, 1);
        end
        [scoreSort scoreIdx] = getNElements(Score, k);
        class=mode(trainY(scoreIdx));    
    elseif method == "mink1.5"
        for i= 1:len
            Score(i)=minkowskiDist(trainX(i,:), test, 1.5);
        end  
        [scoreSort scoreIdx] = getNElements(Score, k);
        class=mode(trainY(scoreIdx));
    elseif method == "euc"
        for i= 1:len
            Score(i)=minkowskiDist(trainX(i,:), test, 2);
        end        
        [scoreSort scoreIdx] = getNElements(Score, k);
        class=mode(trainY(scoreIdx));
    elseif method == "tan"
        for i= 1:len
            Score(i)=tangentDist(trainX(i,:) , test,28,28, [1,1,1,1,1,1,1,1,1],0.0);
        end
        [scoreSort scoreIdx] = getNElements(Score, k);
        class=mode(trainY(scoreIdx));
    else
        disp("invalid method")
        class=[NaN, NaN];
    end
end

