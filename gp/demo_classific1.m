function demo_classific1
%DEMO_CLAASIFIC1    Classification problem demonstration for 2 classes via MCMC
%
%      Description
%      The demonstration program is based on synthetic two 
%      class data used by B.D. Ripley (Pattern Regocnition and
%      Neural Networks, 1996}. The data consists of 2-dimensional
%      vectors that are divided into to classes, labeled 0 or 1.
%      Each class has a bimodal distribution generated from equal
%      mixtures of Gaussian distributions with identical covariance
%      matrices. A Bayesian aproach is used to find the decision
%      line and predict the classes of new data points.
%
%      The probability of y being one is assumed to be 
%
%            p(y=1|f) = 1 / ((1+exp(-f))
%
%      The latent values f are given a zero mean Gaussian process prior.
%      This implies that at the observed input locations latent values 
%      have prior 
%
%         f ~ N(0, K),
%
%      where K is the covariance matrix, whose elements are given as 
%      K_ij = k(x_i, x_j | th). The function k(x_i, x_j | th) is covariance 
%      function and th its parameters, hyperparameters. 
% 
%      Here we use MCMC methods to find the posterior of the latent values and 
%      hyperparameters. With these we can make predictions on the class 
%      probability of future observations. See Neal (1996) for the detailed 
%      treatment of the MCMC samplers.
%
%      NOTE! The class labels have to be {0,1} for logit likelihood 
%      (different from the probit likelihood).

% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% This demonstration is based on the dataset used in the book Pattern Recognition and
% Neural Networks by B.D. Ripley (1996) Cambridge University Press ISBN 0 521
% 46986 7

%========================================================
% PART 1 data analysis with full GP model
%========================================================

S = which('demo_classific1');
L = strrep(S,'demo_classific1.m','demos/synth.tr');
x=load(L);
y=x(:,end);
y = 2.*y-1;
x(:,end)=[];
[n, nin] = size(x);

% Create covariance functions
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [0.9 0.9], 'magnSigma2', 10);

% Set the prior for the parameters of covariance functions 
pl = prior_logunif('init');
gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);

% Create the likelihood structure
likelih = likelih_probit('init', y);

% Create the GP data structure
gp = gp_init('init', 'FULL', nin, likelih, {gpcf1}, [],'jitterSigmas', 0.1);   %{gpcf2}

% Set the approximate inference method
gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(y))', @scaled_mh});

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
opt.nsamples=100;
opt.repeat=1;
opt.hmc_opt.steps=4;
opt.hmc_opt.stepadj=0.02;
opt.latent_opt.repeat = 5;
hmc2('state', sum(100*clock));

% Sample 
[rgp,g,rstate2]=gp_mc(opt, gp, x, y, r);

% Thin the sample chain. 
% Note! the thinning is not optimal and the chain is too short. Run
% longer chain, if you want good analysis.
rr=thin(rgp,150,4);

% Plot the sample chains of the hyperparameters
figure(1)
subplot(1,2,1)
plot(rgp.cf{1}.lengthScale)
title('the posterior samples of the length-scale, full GP')
subplot(1,2,2)
plot(rgp.cf{1}.magnSigma2)
title('the posterior samples of the magnitude, full GP')

% Print some figures that show results
% First create data for predictions
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

% Make predictions
p1 = mean(squeeze(logsig(mc_pred(rr, x, rr.latentValues', xstar))),2);

figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases, full GP', 'fontsize', 14)

% visualise predictive probability  p(ystar = 1) with contours
figure, hold on
[cs,h]=contour(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20),[0.025 0.25 0.5 0.75 0.975], 'linewidth', 3);
text_handle = clabel(cs,h);
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==1,1), x(y==1,2), 'rx', 'markersize', 8, 'linewidth', 2),
plot(x(y==-1,1), x(y==-1,2), 'bo', 'markersize', 8, 'linewidth', 2)
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('predictive probability contours, full GP', 'fontsize', 14)


%========================================================
% PART 2 data analysis with FIC GP model
%========================================================

% Set the inducing inputs
[u1,u2]=meshgrid(linspace(-1.25, 0.9,10),linspace(-0.2, 1.1,10));
Xu=[u1(:) u2(:)];
%Xu = Xu([3 4 7:18 20:24 26:30 33:36],:);

likelih = likelih_probit('init', y);

% Set the prior for the parameters of covariance functions 
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [0.9 0.9], 'magnSigma2', 10);
pl = prior_logunif('init');
gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);

% Create the GP data structure
gp_fic = gp_init('init', 'FIC', nin, likelih, {gpcf1}, [], 'jitterSigmas', 0.1, 'X_u', Xu);

% Initialize the hyperparameters with Laplace approximation
    gp_fic = gp_init('set', gp_fic, 'latent_method', {'Laplace', x, y, 'hyper'});
   
    fe=str2fun('gpla_e');
    fg=str2fun('gpla_g');
    n=length(y);
    opt = scg2_opt;
    opt.tolfun = 1e-3;
    opt.tolx = 1e-3;
    opt.display = 1;
    opt.maxiter = 30;

    % do scaled conjugate gradient optimization 
    w = gp_pak(gp_fic, 'hyper');
    w = scg2(fe, w, opt, fg, gp_fic, x, y, 'hyper');
    gp=gp_unpak(gp_fic,w, 'hyper');


% Set the approximate inference method
gp_fic = gp_init('set', gp_fic, 'latent_method', {'MCMC', zeros(size(y))', @scaled_mh});

% Set the sampling options
opt=gp_mcopt;
opt.repeat=10;
opt.latent_opt.sample_latent_scale = 0.5;

opt.nsamples=20000;
opt.repeat=1;
% $$$ opt.hmc_opt.steps=5;
% $$$ opt.hmc_opt.stepadj=0.02;
opt.latent_opt.repeat = 5;

defaultStream = RandStream.getDefaultStream;
savedState = defaultStream.State;

%hmc2('state', sum(100*clock));

gp.tyyppi = 'FULL'
defaultStream.State = savedState;
[rgp_fic,gp_fic,rstate2]=gp_mc(opt, gp_fic, x, y);
gp.tyyppi = 'FIC1'
defaultStream.State = savedState;
[rgp_fic1,gp_fic,rstate2]=gp_mc(opt, gp_fic, x, y);
gp.tyyppi = 'FIC'
defaultStream.State = savedState;
[rgp_fic2,gp_fic,rstate2]=gp_mc(opt, gp_fic, x, y);


% Thin the sample chain. 
% Note! the thinning is not optimal and the chain is too short. Run
% longer chain, if you want good analysis.
rr_fic=thin(rgp_fic,100,2);

% Plot the sample chains of the hyperparameters
figure(1)
subplot(1,2,1)
plot(rgp_fic.cf{1}.lengthScale)
title('the posterior samples of the length-scale, FIC')
subplot(1,2,2)
plot(rgp_fic.cf{1}.magnSigma2)
title('the posterior samples of the magnitude, FIC')

% Print some figures that show results
% First create data for predictions
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

% Make predictions
p1_fic = mean(squeeze(logsig(mc_pred(rr_fic, x, rr_fic.latentValues', xstar))),2);

figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_fic,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases, FIC', 'fontsize', 14)

% visualise predictive probability  p(ystar = 1) with contours
figure, hold on
[cs,h]=contour(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_fic,20,20),[0.025 0.25 0.5 0.75 0.975], 'linewidth', 3);
text_handle = clabel(cs,h);
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==1,1), x(y==1,2), 'rx', 'markersize', 8, 'linewidth', 2),
plot(x(y==-1,1), x(y==-1,2), 'bo', 'markersize', 8, 'linewidth', 2)
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('predictive probability contours, FIC', 'fontsize', 14)



%========================================================
% PART 3 data analysis with CS+FIC GP model
%========================================================

% The CS+FIC model is not very  efficient for  this data, 
% since there are not additive pehonomenon in the data. Thus, 
% Full GP and FIC with just one covariance function work as good
% as  CS+FIC. This part is only for demonstrating how to use 
% CS+FIC with non Gaussian likelihood.

% Create covariance functions
gpcf2 = gpcf_ppcs2('init', nin, 'lengthScale', [0.9 0.9], 'magnSigma2', 10);

% Set the prior for the parameters of covariance functions 
gpcf2.p.lengthScale = gamma_p({3 7 3 7});
gpcf2.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% Create the GP data structure
gp_csfic = gp_init('init', 'CS+FIC', nin, likelih, {gpcf1, gpcf2}, [], 'jitterSigmas', 0.1, 'X_u', Xu);

% Set the approximate inference method
gp_csfic = gp_init('set', gp_csfic, 'latent_method', {'MCMC', zeros(size(y))'});

% Set the sampling options
opt.nsamples=100;
opt.repeat=1;
opt.hmc_opt.steps=5;
opt.hmc_opt.stepadj=0.02;
opt.latent_opt.repeat = 5;
hmc2('state', sum(100*clock));

hmc2('state', sum(100*clock))
[rgp_csfic,gp_csfic,rstate2]=gp_mc(opt, gp_csfic, x, y);

% Thin the sample chain. 
% Note! the thinning is not optimal and the chain is too short. Run
% longer chain, if you want good analysis.
rr_csfic=thin(rgp_csfic,150,4);

% Plot the sample chains of the hyperparameters
figure(1)
subplot(1,2,1)
plot(rgp_csfic.cf{1}.lengthScale)
hold on
plot(rgp_csfic.cf{2}.lengthScale)
title('the posterior samples of the length-scale')
subplot(1,2,2)
plot(rgp_csfic.cf{1}.magnSigma2)
hold on
plot(rgp_csfic.cf{2}.magnSigma2)
title('the posterior samples of the magnitude')

% Print some figures that show results
% First create data for predictions
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

% Make predictions
p1_csfic = mean(squeeze(logsig(gp_preds(rr_csfic, x, rr_csfic.latentValues', xstar))),2);

figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_csfic,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==0,1),x(y==0,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases, CS+FIC', 'fontsize', 14)

% visualise predictive probability  p(ystar = 1) with contours
figure, hold on
[cs,h]=contour(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_csfic,20,20),[0.025 0.25 0.5 0.75 0.975], 'linewidth', 3);
text_handle = clabel(cs,h);
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==1,1), x(y==1,2), 'rx', 'markersize', 8, 'linewidth', 2),
plot(x(y==0,1), x(y==0,2), 'bo', 'markersize', 8, 'linewidth', 2)
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('predictive probability contours, CS+FIC', 'fontsize', 14)
