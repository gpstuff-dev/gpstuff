%DEMO_BINOMIAL2     Demonstration for modeling age-period-cohort data
%                   by a binomial model combined with GP prior.
%
%      Description
%      Demonstration of estimating the unknown the population proportion
%      in binomial model from a sequence of success/failure trials. Data
%      consists of observations Y describing the number of successes in
%      a sequence of N iid (Bernoulli) trials, and of explanatory
%      variables X. The binomial model is 
%
%      Y_i ~ Binomial(Y_i | N_i, p_i),
%
%      where the parameter p_i represents the proportion of successes.
%      The total number of trials N_i is fixed in the model. A
%      Gaussian process prior is assumed for latent variables f
%
%      f = N(0, K),
%
%      which are linked to the p_i parameter using the logistic
%      transformation:
%       
%      p_i = logit^-1(f_i) = 1/(1+exp(-f_i)).
%
%      The elements of the covariance matrix K are given as 
%      K_ij = k(x_i, x_j | th). The function k(x_i, x_j | th) is covariance 
%      function and th its parameters, hyperparameters.  We place a
%      hyperprior for hyperparameters, p(th). The inference is done with
%      Laplace approximation.
%   
%      In this demonstration X is three dimensional, containing inputs for
%      age group, time period and cohort group (birth year) for each
%      data point. Y represents the number of disease cases among N
%      susceptibles. The data is simulated such that each input have
%      additionally there is a interaction effect between age group and
%      time period.
%
%      NOTE! In the prediction, the total number of trials Nt at the
%      test points Xt must be set additionally in the likelihood structure
%      when E(Yt), Var(Yt) or predictive densities p(Yt) are computed.

% Copyright (c) 2010 Jaakko Riihimäki, Jouni Hartikainen

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


%========================================================
% data analysis with full GP model
%========================================================

% Demonstration for modeling age-period-cohort data
% by a binomial model combined with GP prior.

% First load data
S = which('demo_binomial2');
L = strrep(S,'demo_binomial2.m','demos/binodata.txt');
binodata=load(L);

f = binodata(:,1);
f1 = binodata(:,2);
f2 = binodata(:,3);
f3 = binodata(:,4);
f4 = binodata(:,5);
nn = binodata(:,6);
xx = binodata(:,7:9);
yy = binodata(:,10);

% xx contains the three dimensional inputs for 1377 data points:
%   xx(:,1) - age group
%   xx(:,2) - time period
%   xx(:,3) - cohort (birth year)

% Use only a (random) proportion of original data points in training
inds = randperm(size(xx,1));
ntr = 300;
itr = sort(inds(1:ntr));
itst = sort(inds(ntr+1:end));

% Original points
xxo = xx; yyo = yy;
nno = nn; fo  = f;
f1o = f1; f2o = f2;
f3o = f3; f4o = f4;

% Test points
xt = xx(itst,:); yt = yy(itst,:);
nt = nn(itst,:); ft = f(itst,:);
f1t = f1(itst,:); f2t = f2(itst,:);
f3t = f3(itst,:); f4t = f4(itst,:);

% Training points
xx = xx(itr,:); yy = yy(itr,:);
nn = nn(itr,:); f  = f(itr,:);
f1 = f1(itr,:); f2 = f2(itr,:);
f3 = f3(itr,:); f4 = f4(itr,:);


% Initialization of covariance functions. We shall assume here
% that inputs have additive effect to disease risk (covariance
% functions gpcf1, gpcf2 and gpcf3) as well as interaction effect
% between age group and time period (gpcf4).

% First define priors for length scales and magnitudes
pl = prior_t('init');
pm = prior_t('init', 's2', 0.3);

gpcf1 = gpcf_sexp('init','magnSigma2', 1, 'magnSigma2_prior',pm);
metric1 = metric_euclidean('init', {[1]},'lengthScales',[10], 'lengthScales_prior', pl);
gpcf1 = gpcf_sexp('set', gpcf1, 'metric', metric1);

gpcf2 = gpcf_sexp('init', 'magnSigma2', 1, 'magnSigma2_prior',pm);
metric2 = metric_euclidean('init', {[2]},'lengthScales',[10], 'lengthScales_prior', pl);
gpcf2 = gpcf_sexp('set', gpcf2, 'metric', metric2);

gpcf3 = gpcf_sexp('init', 'magnSigma2', 1, 'magnSigma2_prior',pm);
metric3 = metric_euclidean('init',  {[3]},'lengthScales',[4], 'lengthScales_prior', pl);
gpcf3 = gpcf_sexp('set', gpcf3, 'metric', metric3);

gpcf4 = gpcf_sexp('init', 'magnSigma2', 1, 'magnSigma2_prior',pm);
metric4 = metric_euclidean('init',  {[1 2]},'lengthScales',[10 2], 'lengthScales_prior', pl);
gpcf4 = gpcf_sexp('set', gpcf4, 'metric', metric4);

% Initialize the likelihood structure
likelih = likelih_binomial('init', yy, nn);
    
% Initialize GP structure
gp = gp_init('init', 'FULL', likelih, {gpcf1,gpcf2,gpcf3,gpcf4}, [],'jitterSigma2',0.01^2);   %{gpcf2}
    
% Set the approximate inference method
gp = gp_init('set', gp, 'latent_method', {'Laplace', xx, yy, 'z', nn});

% Function handles for optimization
fe=str2fun('gpla_e');
fg=str2fun('gpla_g');

% Set the options for scaled conjugate optimazation
opt_scg = scg2_opt;
opt_scg.tolfun = 1e-2;
opt_scg.tolx = 1e-1;
opt_scg.display = 1;

% Do scaled conjugate gradient optimization 
w=gp_pak(gp);
[wopt, opt, flog]=scg2(fe, w, opt_scg, fg, gp, xx, yy, 'z',nn);
gp=gp_unpak(gp, wopt);

% Making predictions

% First with all components
[Ef,Varf,Ey,Vary,Py] = la_pred(gp,xx,yy,xt,'z',nn,'zt',nt,'yt',yt);
% Age group effect
[Ef_1,Varf_1] = la_pred(gp,xx,yy,xxo,'predcf',[1],'z',nn,'zt',nno);
% Time period effect
[Ef_2,Varf_2] = la_pred(gp,xx,yy,xxo,'predcf',[2],'z',nn,'zt',nno);
% Cohort effect
[Ef_3,Varf_3] = la_pred(gp,xx,yy,xxo,'predcf',[3],'z',nn,'zt',nno);
% Interaction effect between age group and time period
[Ef_4,Varf_3] = la_pred(gp,xx,yy,xxo,'predcf',[4],'z',nn,'zt',nno);

% Plotting predictions

% First some indexes needed for plotting
% the additive effect
[xx1 ind1] = unique(xxo(:,1));
[xx2 ind2] = unique(xxo(:,2));
[xx3 ind3] = unique(xxo(:,3));

% Age group effect
figure; subplot(3,1,1)
set(gcf, 'color', 'w'), hold on
color1=ones(1,3)*0.8; color2=ones(1,3)*0.5;
% Estimate
h1=fill([xx1' fliplr(xx1')], [(Ef_1(ind1)+1.96*sqrt(Varf_1(ind1)))' ...
    fliplr((Ef_1(ind1)-1.96*sqrt(Varf_1(ind1)))')], color1, 'edgecolor', color1);
h2=plot(xx1, Ef_1(ind1), 'color', color2, 'linewidth', 3);
% True function
h4=plot(xx1, f1o(ind1), 'color', 'r', 'linewidth', 2); hold off
title('Age group effect')
xlabel('Age group'); ylabel('logit(p)')

% Time period effect
subplot(3,1,2)
set(gcf, 'color', 'w'), hold on
h1=fill([xx2' fliplr(xx2')], [(Ef_2(ind2)+1.96*sqrt(Varf_2(ind2)))' ...
    fliplr((Ef_2(ind2)-1.96*sqrt(Varf_2(ind2)))')], color1, 'edgecolor', color1);
h2=plot(xx2, Ef_2(ind2), 'color', color2, 'linewidth', 3);
% true function
h4=plot(xx2, f2o(ind2), 'color', 'r', 'linewidth', 2);
title('Time period effect')
xlabel('Time period'); ylabel('logit(p)')

% Cohort effect
subplot(3,1,3)
set(gcf, 'color', 'w'), hold on
h1=fill([xx3' fliplr(xx3')], [(Ef_3(ind3)+1.96*sqrt(Varf_3(ind3)))' ...
    fliplr((Ef_3(ind3)-1.96*sqrt(Varf_3(ind3)))')], color1, 'edgecolor', color1);
h2=plot(xx3, Ef_3(ind3), 'color', color2, 'linewidth', 3);
% true function
h4=plot(xx3, f3o(ind3), 'color', 'r', 'linewidth', 2);
title('Cohort effect')
xlabel('Cohort effect'); ylabel('logit(p)')

% Plotting of interaction effect
figure; subplot(1,2,1)
imagesc(reshape(Ef_4,81,17)); colorbar;
title('Estimated interaction effect')
xlabel('Time period'); ylabel('Age group')
subplot(1,2,2);
imagesc(reshape(f4o,81,17)); colorbar;
title('True interaction')
xlabel('Time period'); ylabel('Age group')

