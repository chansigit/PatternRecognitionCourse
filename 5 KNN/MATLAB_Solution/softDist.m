function dist = softDist(A, X, i, j)
    if i==j
        dist=0;
        return 
    end

    nume= exp(-(norm(A*(X(i,:)-X(j,:))'))^2);
    [H,~]=size(X);
    denom=0;
    for k=1:H
        if k==i
            continue
        end
        
        denom= denom+exp(-(norm(A*(X(i,:)-X(k,:))'))^2);
    end
    dist=nume/denom;

end

