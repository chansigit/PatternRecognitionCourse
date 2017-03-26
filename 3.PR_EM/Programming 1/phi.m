function [ y ] = phi( x )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
y=0;
y(-1/2<=x(:,1) & x(:,1)<=1/2 & -1/2<=x(:,2) & x(:,2)<=1/2)=1;
y(-1/2>x(:,1) | x(:,1)>1/2 | -1/2>x(:,2) | x(:,2)>1/2)=0;

end

