% DEMO_LOGITGP  Demonstration of Logistic-Gaussian Process density estimate
%               for 1D and 2D data 
%
%    Description 
%
%    Logistic-Gaussian Process (LOGITGP) is a model for density estimation.
%    For the samples from continuous distribution, the space is discretized
%    into n intervals with equal lengths covering the interesting region. The
%    following model is used in estimation
%    
%        p(y_i|f_i) ~ exp(f_i) / Sum_j^n exp(f_j),
%
%    where a zero mean Gaussian process prior is placed for f =
%    [f_1, f_2,...,f_n] ~ N(0, K). K is the covariance matrix, whose
%    elements are given as K_ij = k(x_i, x_j | th). The function
%    k(x_i, x_j| th) is covariance function and th its parameters,
%    hyperparameters. We place a hyperprior for hyperparameters, p(th).
%
%    The inference is conducted via Laplace and the last example compares
%    the results of Laplace approximation to MCMC. 
%
%    See also  DEMO_LGCP

% Copyright (c) 2011 Jaakko Riihimäki and Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

%addpath /proj/bayes/software/jtriihim/gp_density/
%stream = RandStream('mrg32k3a');
%RandStream.setDefaultStream(stream);

% =====================================
% 1) 1D-examples
% =====================================

figure(1)
subplot(2,2,1)
% t_4
stream.Substream = 1;
x=[trnd(4,1,100)]';
[N,X]=hist(x);
h=bar(X,N/sum(N.*diff(X(1:2))),'hist');set(h,'FaceColor',[.4 .6 1]);
xt=linspace(-7,7,400)';
[p,pq]=logitgp(x,xt,'int_method','mode');
line(xt,p,'color','r','marker','none','linewidth',2)
line(xt,pq,'color','r','marker','none','linewidth',1,'linestyle','--')
xlim([-7 7])
title('t_4')
% correct density
p0=t_pdf(xt,4,0,1);
line(xt,p0,'color','k')
%sum(p0.*log(p))

subplot(2,2,2)
% Mixture of two t_4
stream.Substream = 1;
x=[trnd(4,1,100) 3+trnd(4,1,50)*0.1]';
[N,X]=hist(x);
h=bar(X,N/sum(N.*diff(X(1:2))),'hist');set(h,'FaceColor',[.4 .6 1]);
xt=linspace(-6,6,400)';
[p,pq]=logitgp(x,xt,'int_method','mode');
line(xt,p,'color','r','marker','none','linewidth',2)
line(xt,pq,'color','r','marker','none','linewidth',1,'linestyle','--')
title('Mixture of two t_4')
% correct density
p0=t_pdf(xt,4,0,1)*2/3+t_pdf(xt,4,3,.1)*1/3;
line(xt,p0,'color','k')
%sum(p0.*log(p))

subplot(2,2,3)
% Galaxy data
x=load('demos/galaxy.txt');
[N,X]=hist(x);
h=bar(X,N/sum(N.*diff(X(1:2))),'hist');set(h,'FaceColor',[.4 .6 1]);
xt=linspace(0,40000,200)';
[p,pq]=logitgp(x,xt,'int_method','mode');
line(xt,p,'color','r','marker','none','linewidth',2)
line(xt,pq,'color','r','marker','none','linewidth',1,'linestyle','--')
title('Galaxy data')

subplot(2,2,4)
% Gamma(1,1)
stream.Substream = 1;
x=gamrnd(1,1,100,1);
[N,X]=hist(x);
h=bar(X,N/sum(N.*diff(X(1:2))),'hist');set(h,'FaceColor',[.4 .6 1]);
xt=linspace(0,6,400)';
[p,pq]=logitgp(x,xt,'int_method','mode','gpcf',@gpcf_rq);
line(xt,p,'color','r','marker','none','linewidth',2)
line(xt,pq,'color','r','marker','none','linewidth',1,'linestyle','--')
title('Gamma(1,1)')

% =====================================
% 1) 2D-examples
% =====================================

figure(2)
clf
subplot(2,2,1)
% Mixture of two Gaussians
n=200;
stream.Substream = 1;
x=[randn(n/2,2);bsxfun(@plus,randn(n/2,2)*0.5,[2 2])];
logitgp(x,'int_method','mode','gridn',20);
line(x(:,1),x(:,2),'LineStyle','none','Marker','.')
title('Mixture of two Gaussians')

subplot(2,2,2)
% Truncated t_8
Sigma = [1 .7; .7 1];R = chol(Sigma);
stream.Substream = 1;
x=trnd(8,200,2)*R;x(x(:,2)<-1,2)=-1;
logitgp(x,'int_method','mode','gridn',20);
line(x(:,1),x(:,2),'LineStyle','none','Marker','.')
title('Truncated t_8')

subplot(2,2,3)
% Banana-shaped
stream.Substream = 1;
b=0.02;x=randn(200,2);x(:,1)=x(:,1)*10;x(:,2)=x(:,2)+b*x(:,1).^2-10*b;
logitgp(x,'int_method','mode','gridn',20);
line(x(:,1),x(:,2),'LineStyle','none','Marker','.')
title('Banana-shaped')


subplot(2,2,4)
% Mixture of two Gaussians
n=200;
stream.Substream = 1;
x=[randn(n/2,2);bsxfun(@plus,randn(n/2,2)*0.5,[2 2])];
%x=[randn(n,2)];%bsxfun(@plus,randn(n/2,2)*0.5,[2 2])];
%xt=linspace(-3,4,26)';
logitgp(x,'range',[-3 4 -3 4],'int_method','mode','gridn',26);
line(x(:,1),x(:,2),'LineStyle','none','Marker','.')
%p=logitgp(x,'range',[-3 4 -3 4],'int_method','mode','gridn',26);
%p1=logitgp(x(:,1),'range',[-3 4],'int_method','mode','gridn',26);
%p2=logitgp(x(:,2),'range',[-3 4],'int_method','mode','gridn',26);
%plot(xt,p1),ylim([0 .5])
%plot(xt,p2),ylim([0 .5])
%pp=reshape(p,26,26);
%sum(sum(pp.*(log(pp)-bsxfun(@plus,log(p1'),log(p2)))))
line(x(:,1),x(:,2),'LineStyle','none','Marker','.')
title('Mixture of two Gaussians')


% =====================================
% 1) 1D-example MCMC vs Laplace
% =====================================
figure(3)
subplot(2,1,1)
hold on
% t_4
stream.Substream = 1;
x=[trnd(4,1,100)]';
xt=linspace(-7,7,50)';
[N,X]=hist(x);
h=bar(X,N/sum(N.*diff(X(1:2))),'hist');set(h,'FaceColor',[.4 .6 1]);
[p,pq]=logitgp(x,xt,'int_method','mode');
pla=p;
line(xt,p,'color','r','marker','none','linewidth',2)
line(xt,pq,'color','r','marker','none','linewidth',1,'linestyle','--')
xlim([-7 7])
title('t_4 (Laplace)')
% correct density
p0=t_pdf(xt,4,0,1);
line(xt,p0,'color','k')

subplot(2,1,2)
hold on
h=bar(X,N/sum(N.*diff(X(1:2))),'hist');set(h,'FaceColor',[.4 .6 1]);
%[p,pq]=logitgp(x,xt,'int_method','mode');
[p,pq]=logitgp(x,xt,'latent_method','MCMC');
pmc=p;
line(xt,p,'color','r','marker','none','linewidth',2)
line(xt,pq,'color','r','marker','none','linewidth',1,'linestyle','--')
%xlim([-7 7])
title('t_4 (MCMC)')
line(xt,p0,'color','k')

[pks] = ksdensity(x,xt);

disp(['Laplace: ' num2str(sum(p0.*log(pla)))])
disp(['MCMC: ' num2str(sum(p0.*log(pmc)))])
disp(['ksdensity: ' num2str(sum(p0.*log(pks)))])

