function dist= minkowskiDist(x1,x2,p)
    diff = (abs(x1 - x2)).^p;
    dist = (diff * diff')^(1/p);
end

