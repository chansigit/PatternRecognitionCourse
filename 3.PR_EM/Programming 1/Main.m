clc;clear;

mu1 = [-1 0]'; Sigma1 = [1 0; 0 1];%f1参数
mu2 = [1 0]'; Sigma2 = [2 0; 0 1];%f2参数
p=0.5;%f1的混合比例

n=1000;%样本数
%Draw n samples from p(x|1) and p(x|2) with labels respectively.
x1=mvnrnd(mu1,Sigma1,n);
x2=mvnrnd(mu2,Sigma2,n);
% figure
% plot(x1(:,1),x1(:,2),'r.',x2(:,1),x2(:,2),'b.');

%产生2n个混合分布的随机数
y=fix(rand(2*n,1)+p);
x =repmat(y,size(mu1')).*mvnrnd(mu1,Sigma1,2*n)+repmat((1-y),size(mu1')).*mvnrnd(mu2,Sigma2,2*n);
% figure
% plot(x(:,1),x(:,2),'.');

%实际概率密度图
mu=mean(x);
s=std(x);
xmin=mu-5*s;
xmax=mu+5*s;
nn=100;  ;%作图区间分段数
dx=(xmax(1)-xmin(1))/nn;
dy=(xmax(2)-xmin(2))/nn;
xx(:,1)=(xmin(1):dx:xmax(1))'; 
xx(:,2)=(xmin(2):dy:xmax(2))'; 
Pe1=0;Pe2=0;
Pe1_phi=0;Pe2_phi=0;
Pe1_Gaussian=0;Pe2_Gaussian=0;
a=1;%窗宽
for i=1:size(xx,1)
    for j=1:size(xx,1)
        f1(i,j)=1/(2*pi)/det(Sigma1)^(1/2)*exp(-1/2*([xx(i,1),xx(j,2)]'-mu1)'*Sigma1^(-1)*([xx(i,1),xx(j,2)]'-mu1));
        f2(i,j)=1/(2*pi)/det(Sigma2)^(1/2)*exp(-1/2*([xx(i,1),xx(j,2)]'-mu2)'*Sigma2^(-1)*([xx(i,1),xx(j,2)]'-mu2));
        
        pn1_phi(i,j)=1/size(x1,1)*sum(1/a*phi((repmat([xx(i,1),xx(j,2)],size(x1,1),1)-x1)/a)); %方窗  
        pn2_phi(i,j)=1/size(x2,1)*sum(1/a*phi((repmat([xx(i,1),xx(j,2)],size(x2,1),1)-x2)/a)); %方窗  
        
        pn1_Gaussian(i,j)=1/size(x1,1)*sum(1/a^2*Gaussian((repmat([xx(i,1),xx(j,2)],size(x1,1),1)-x1)/a)); %高斯窗
        pn2_Gaussian(i,j)=1/size(x2,1)*sum(1/a^2*Gaussian((repmat([xx(i,1),xx(j,2)],size(x2,1),1)-x2)/a)); %高斯窗
        
        if p*f1(i,j)>=(1-p)*f2(i,j)
            Pe2=Pe2+(1-p)*f2(i,j)*dx*dy;
        else
            Pe1=Pe1+p*f1(i,j)*dx*dy;
        end
        
        if p*pn1_phi(i,j)>=(1-p)*pn2_phi(i,j)
            Pe2_phi=Pe2_phi+(1-p)*pn2_phi(i,j)*dx*dy;
        else
            Pe1_phi=Pe1_phi+p*pn1_phi(i,j)*dx*dy;
        end
        
        if p*pn1_Gaussian(i,j)>=(1-p)*pn2_Gaussian(i,j)
            Pe2_Gaussian=Pe2_Gaussian+(1-p)*pn2_Gaussian(i,j)*dx*dy;
        else
            Pe1_Gaussian=Pe1_Gaussian+p*pn1_Gaussian(i,j)*dx*dy;
        end
    end
end
f=p*f1+(1-p)*f2;
Pe=Pe1+Pe2
Pe_phi=Pe1_phi+Pe2_phi
Pe_Gaussian=Pe1_Gaussian+Pe2_Gaussian
figure
% meshc(xx(:,1),xx(:,2),f);
meshc(xx(:,1),xx(:,2),p*f1),hold on,
meshc(xx(:,1),xx(:,2),(1-p)*f2)
hidden off,hold off,

components= em_gmm(x, 2)
Pe1_EM=0;Pe2_EM=0;
for i=1:size(xx,1)
    for j=1:size(xx,1)
        p2n1(i,j)=1/(2*pi)/det(components.sigma(:,:,1))^(1/2)*exp(-1/2*([xx(i,1),xx(j,2)]'-components.mu(1,:)')'*components.sigma(:,:,1)^(-1)*([xx(i,1),xx(j,2)]'-components.mu(1,:)'));
        p2n2(i,j)=1/(2*pi)/det(components.sigma(:,:,2))^(1/2)*exp(-1/2*([xx(i,1),xx(j,2)]'-components.mu(2,:)')'*components.sigma(:,:,2)^(-1)*([xx(i,1),xx(j,2)]'-components.mu(2,:)'));

        if p*p2n1(i,j)>=(1-p)*p2n2(i,j)
            Pe2_EM=Pe2_EM+(1-p)*p2n2(i,j)*dx*dy;
        else
            Pe1_EM=Pe1_EM+p*p2n1(i,j)*dx*dy;
        end
    end
end
Pe_EM=Pe1_EM+Pe2_EM