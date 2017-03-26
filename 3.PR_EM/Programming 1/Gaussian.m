function [ y ] = Gaussian( x )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
for i=1:size(x,1)
y(i)=(1/(2*pi)).*exp(-x(i,:)*x(i,:)'/2);

end
end

