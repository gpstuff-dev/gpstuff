function demo_classific1
%DEMO_CLAASIFIC1    Classification problem demonstration for 2 classes 
%
%      Description
%      The demonstration program is based on synthetic two 
%      class data used by B.D. Ripley (Pattern Regocnition and
%      Neural Networks, 1996}. The data consists of 2-dimensional
%      vectors that are divided into to classes, labeled 0 or 1.
%      Each class has a bimodal distribution generated from equal
%      mixtures of Gaussian distributions with identical covariance
%      matrices. A Bayesian aprouch is used to find the decision
%      line and predict the classes of new data points.

% Copyright (c) 2007 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% This demonstration is based on the dataset used in the book Pattern Recognition and
% Neural Networks by B.D. Ripley (1996) Cambridge University Press ISBN 0 521
% 46986 7

S = which('demo_classific1');
L = strrep(S,'demo_classific1.m','demos/synth.tr');
x=load(L);
y=x(:,end);
x(:,end)=[];
[n, nin] = size(x);

% Create covariance functions
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [0.9 0.9], 'magnSigma2', 40);

% Set the prior for the parameters of covariance functions 
gpcf1.p.lengthScale = gamma_p({3 7 3 7});
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% Create the likelihood structure
likelih = likelih_logit('init');

% Create the GP data structure
gp = gp_init('init', 'FULL', nin, likelih, {gpcf1}, []);   %{gpcf2}

% Set the approximate inference method
gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(y))'});

% Set the optimization parameters...
opt=gp_mcopt;
opt.repeat=15;
opt.nsamples=1;
opt.hmc_opt.steps=10;
opt.hmc_opt.stepadj=0.1;
opt.hmc_opt.nsamples=1;
opt.latent_opt.display=0;
opt.latent_opt.repeat = 20;
opt.latent_opt.sample_latent_scale = 0.5;
hmc2('state', sum(100*clock))
[r,g,rstate1]=gp_mc(opt, gp, x, y);

% Set the sampling options
opt.nsamples=1000;
opt.repeat=1;
opt.hmc_opt.steps=4;
opt.hmc_opt.stepadj=0.02;
opt.latent_opt.repeat = 5;
hmc2('state', sum(100*clock));

% Sample 
[rgp,g,rstate2]=gp_mc(opt, gp, x, y, [], [], r);

% Thin the sample chain (the thinning is not optimal)
rr=thin(r,150,4);

% Plot the posteriors of the hyperparameters
figure(1)
subplot(1,2,1)
hist(rgp.cf{1}.lengthScale)
title('the posterior of the length-scale')
subplot(1,2,2)
hist(rgp.cf{1}.magnSigma2)
title('the posterior of the magnitude')

% Print some figures that show results
% First create data for predictions
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

% Make predictions
p1 = mean(squeeze(logsig(gp_preds(rr, x, rr.latentValues', xstar))),2);

figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==0,1),x(y==0,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases', 'fontsize', 14)

% visualise predictive probability  p(ystar = 1) with contours
figure, hold on
[cs,h]=contour(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20),[0.025 0.25 0.5 0.75 0.975], 'linewidth', 3);
text_handle = clabel(cs,h);
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==1,1), x(y==1,2), 'rx', 'markersize', 8, 'linewidth', 2),
plot(x(y==0,1), x(y==0,2), 'bo', 'markersize', 8, 'linewidth', 2)
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('predictive probability contours', 'fontsize', 14)
