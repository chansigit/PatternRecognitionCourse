function [param,history,ll] = em_mix(data,number_of_components, eps)
%  [param] = em_mix(data,number_of_components)
% 
%  runs EM to estimate a Gaussian mixture model with
%  NUMBER_OF_COMPONENTS components, based on DATA. 

% set stopping  criterion
if (nargin < 3), eps = min(1e-3,1/(size(data,1)*100));  end

param = initialize_mixture(data,number_of_components);
plot_all(data,param); 

history = {}; ll = [];


cont = 1; it = 1; log_likel = 0; 
while (cont), 
   [param,new_log_likel] = one_EM_iteration(data,param); 
   
   history{length(history)+1}=param;
   ll(length(ll)+1)=new_log_likel;
   
   plot_all(data,param); 

   cont = (new_log_likel - log_likel)>eps*abs(log_likel);
   cont = cont | it<10; it = it + 1;

   log_likel = new_log_likel; 
   
   % uncomment if you wish to monitor the likelihood convergence
   %fprintf('%4d %f\n',it,log_likel);
   
   pause(0.1);
end;

% --------------------------------------------------------
function [] = plot_all(data,param,dim1,dim2)

if (nargin<3), dim1 = 1; dim2 = 2; end;

st = ['b','r','g','c','y','m'];
while(length(st)<length(param)), st = [st,st]; end;

set(0,'DefaultLineLineWidth',2)    
set(0,'DefaultAxesFontSize',14)        

plot(data(:,dim1),data(:,dim2),'ko'); hold on;
myaxis = axis;

for i=1:length(param),
  plot_gaussian(param(i),dim1,dim2,st(i));
  axis(myaxis);
end;

hold off;

% --------------------------------------------------------
function [] = plot_gaussian(param,dim1,dim2,st);

[V,E] = eig(param.cov);
V = V';

s = diag(sqrt(E)); % standard deviations

t=(0:0.05:2*pi)'; 
X = s(dim1)*cos(t)*V(dim1,:)+s(dim2)*sin(t)*V(dim2,:); 
X = X + repmat(param.mean,length(t),1);

plot(X(:,1),X(:,2),st);

% --------------------------------------------------------
function [param,log_likel] = one_EM_iteration(data,param);

[n,d] = size(data);

log_prob = zeros(n,length(param)); 
for i=1:length(param), 
 log_prob(:,i) = gaussian_log_prob(data,param(i))+log(param(i).p);
end;

log_prob_max = max(log_prob,[],2); % per point
log_prob = log_prob - repmat(log_prob_max,1,length(param));

log_likel = sum(log_prob_max+log(sum(exp(log_prob),2)))+log_prior(param);

post_prob = exp(log_prob); % posterior component assignments 
post_prob = post_prob./repmat(sum(post_prob,2),1,length(param));
% Uncomment this section to resume the original covariance setting
% for i=1:length(param),
%   post_n = sum(post_prob(:,i));
%   prior_p= param(i).prior_p;
%   prior_n= param(i).prior_n;
%     
%   param(i).p = (post_n+prior_n*prior_p)/(n+prior_n); 
%   param(i).mean = post_prob(:,i)'*data/post_n; 
% 
%   Z = data-repmat(param(i).mean,n,1);
%   weighted_cov = (repmat(post_prob(:,i),1,d).*Z)'*Z;
%   param(i).cov = (weighted_cov + prior_n*param(i).prior_cov)/(post_n+prior_n);
% end;
temp_nume=0*eye(d);
temp_denom=0;
for i=1:length(param),
  length(param)
  post_n = sum(post_prob(:,i));
  prior_p= param(i).prior_p;
  prior_n= param(i).prior_n;
    
  param(i).p = (post_n+prior_n*prior_p)/(n+prior_n); 
  param(i).mean = post_prob(:,i)'*data/post_n; 
  
  Z = data-repmat(param(i).mean,n,1);
  weighted_cov = (repmat(post_prob(:,i),1,d).*Z)'*Z;
  temp_nume  = temp_nume  + weighted_cov
  temp_denom = temp_denom + post_n
end;
for i=1:length(param),
    param(i).cov =temp_nume/temp_denom;
end;

% --------------------------------------------------------
function [log_prob] = gaussian_log_prob(data,param)

[n,d] = size(data);
Ci = inv(param.cov);

Z = data-repmat(param.mean,n,1); 
log_prob = (-sum( (Z*Ci).*Z, 2 ) + log(det(Ci))-d*log(2*pi))/2;

% --------------------------------------------------------
function [log_prob] = log_prior(param)

log_prob = 0; 
for i=1:length(param),
  prior_n = param(i).prior_n; % equivalent sample size
 
  log_prob = log_prob + prior_n*param(i).prior_p*log(param(i).p); % Dirichlet

  Ci = inv(param(i).cov); 
  Cprior = param(i).prior_cov;

  log_prob = log_prob - 0.5*prior_n*(trace(Ci*Cprior)-log(det(Ci)));

end;

% --------------------------------------------------------
function [param] = initialize_mixture(data,number_of_components)

[n,d] = size(data);

e = eig(cov(data)); 
C = median(e)*eye(d); % initial spherical covariance matrix

[rt,I] = sort(rand(n,1)); % random ordering of the examples

param=[];
for i=1:number_of_components,

  prm.mean = data(I(i),:); % random point as the mean
  prm.cov  = C; % spherical covariance
  prm.p    = 1/number_of_components; % uniform freq

  prm.prior_cov = C;
  prm.prior_p   = 1/number_of_components;
  
  prm.prior_n   = 1; % equivalent sample size
  
  param = [param;prm];
end;



