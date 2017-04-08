clear;
clc;
%profile_master = parallel.importProfile('LocalProfile1');
%parallel.defaultClusterProfile(profile_master)


%mex tangentDistanceCImpl/tangentDist.c  tangentDistanceCImpl/ortho.c tangentDistanceCImpl/td.c
load("mnist_dataset/mnist_521303078.mat");

train_X=train_X(1:6000,:);
train_Y=train_Y(1:6000);

for i=0:9
    C{1+i}=num2cell(find(train_Y==i));
end



% termination tolerance
tol = 1e-6;

% maximum number of allowed iterations
maxiter = 1000;

% minimum allowed perturbation
dxmin = 1e-6;

% step size ( 0.33 causes instability, 0.2 quite accurate)
alpha = 0.1;

% initialize gradient norm, optimization vector, iteration counter, perturbation
gnorm = inf; 
A =zeros(1,784);
niter = 0; 
dx = inf;

% gradient descent algorithm:
while and(gnorm>=tol, and(niter <= maxiter, dx >= dxmin))
    % calculate gradient:
    g = gradient(A,train_X,C,train_Y);
    gnorm = norm(g)
    % take step:
    Anew = A - alpha*g;
    % check step
    if sum(~isfinite(Anew))
        display(['Number of iterations: ' num2str(niter)])
        error('A is inf or NaN')
    end

    % update termination metrics
    niter = niter + 1;
    dx = norm(Anew-A);
    A = Anew;
end

