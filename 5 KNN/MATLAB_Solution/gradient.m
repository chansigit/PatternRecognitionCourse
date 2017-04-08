function grad = gradient( A, X, C ,train_Y)

    [N,~]=size(X);
    sum=0;
    i=randperm(N,1);
    %for i=1:N
        % calc pi and latter term
        pi=0;
        latterTerm=0;
        
        for j =C{1+train_Y(i)}
            
            softD=softDist(A, X, i, j{1});
            pi=pi+softD;
            latterTerm=latterTerm+softD*((X(i,:)-X(j{1},:))*(X(i,:)-X(j{1},:))');
        end
        % calc gradient
        res=0;
        for k = 1:length(X)
            res=res+ softDist(A, X,i,k)*((X(i,:)-X(k,:))*(X(i,:)-X(k,:))');
        end
        res=res*pi-latterTerm;
        sum=sum+res;
    %end
    grad=sum*2*A;
end

