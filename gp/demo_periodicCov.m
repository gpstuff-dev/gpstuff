%DEMO_PERIODICCOV    Regression problem demonstration for periodic data
%                    
%
%    Description
%    A demonstration of the use of periodic covariance function
%    gpcf_periodic with two data sets, the Mauna Loa CO2 data (see for
%    example Rasmussen and Williams 2006) and the monthly Finnish
%    drowning statistics 2002-2008.
%    
%    For the Mauna Loa data, the standard Gaussian process regression
%    model is constructed as in DEMO_REGRESSION2 with additive
%    covariance functions and Gaussian noise
%
%    The observations y are assumed to satisfy
%
%         y = f + g + e,    where e ~ N(0, s^2).
%
%    f and g are underlying latent functions, which we are interested
%    in. We place a zero mean Gaussian process prior them, which
%    implies that at the observed input locations latent values have
%    prior
%
%         f ~ N(0, Kf) and g ~ N(0,Kg)
%
%    where K is the covariance matrix, whose elements are given as
%    K_ij = k(x_i, x_j | th). The function k(x_i, x_j | th) is
%    covariance function and th its parameters, hyperparameters.
%
%    Since both the likelihood and prior are Gaussian, we obtain a
%    Gaussian marginal likelihood
%
%        p(y|th) = N(0, Kf + Kg + I*s^2).
%    
%   By placing a hyperprior for hyperparameters, p(th), we can find
%   the maximum a posterior (MAP) estimate for them by maximizing
%
%       argmax   log p(y|th) + log p(th).
%         th
%   
%   If we want to find an approximation for the posterior of the
%   hyperparameters, we can sample them using Markov chain Monte Carlo
%   (MCMC) methods.
%
%   After finding MAP estimate or posterior samples of
%   hyperparameters, we can use them to make predictions for f.
%
%   For more detailed discussion of Gaussian process regression see,
%   for example, Rasmussen and Williams (2006).
%
%   For the drowning data, a different approach is needed as the
%   likelihood is no longer Gaussian.  The regression of counts is
%   implemented with a Poisson likelihood model with Expectation
%   Propagation as the latent optimisation method. 
%
%   For details on the implementation see GP_E, GP_G, GP_PRED for the
%   standard regression and GPEP_E, GPEP_G AND EP_PRED for the
%   expectation propagation.
%
%   See also  DEMO_REGRESSION2, DEMO_SPATIAL1
%
%   References:
%    Rasmussen, C. E. and Williams, C. K. I. (2006). Gaussian
%    Processes for Machine Learning. The MIT Press.

% Copyright (c) 2009 Heikki Peura

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


% This file is organised in two parts:
%  1) Mauna Loa data analysis with GP regression
%  2) Drowning data analysis with Poisson likelihood

%========================================================
% PART 1 Mauna Loa data analysis with full GP model
%========================================================

% Load the data
S = which('demo_periodicCov');
L = strrep(S,'demo_periodicCov.m','demos/maunaloa_data.txt');

data=load(L);
y = data(:, 2:13);
y=y';
y=y(:);
x = [1:1:length(y)]';
x = x(y>0);
y = y(y>0);
avgy = mean(y);
y = y-avgy;

[n,nin] = size(x);
% Now 'x' consist of the inputs and 'y' of the output. 
% 'n' and 'nin' are the number of data points and the 
% dimensionality of 'x' (the number of inputs).

% First, we will do the inference without the periodic covariance function
% (as in DEMO_REGRESSION2), then add the periodic term and compare the
% results

% ---------------------------
% --- Construct the model ---
% 
% First create squared exponential covariance function with ARD and 
% Gaussian noise data structures...
gpcf1 = gpcf_sexp('init', 'lengthScale', 5, 'magnSigma2', 3);
gpcf2 = gpcf_ppcs2('init', 'nin', nin, 'lengthScale', 2, 'magnSigma2', 3);
gpcfn = gpcf_noise('init', 'noiseSigma2', 1);

% ... Then set the prior for the parameters of covariance functions...
pl = prior_t('init', 's2', 3);
pm = prior_t('init', 's2', 0.3);

gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
gpcf2 = gpcf_ppcs2('set', gpcf2, 'lengthScale_prior', pm, 'magnSigma2_prior', pm);
%gpcf2 = gpcf_sexp('set', gpcf2, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
gpcfn = gpcf_noise('set', gpcfn, 'noiseSigma2_prior', pm);


% ... Finally create the GP data structure
gp = gp_init('init', 'FULL', 'gaussian', {gpcf1,gpcf2}, {gpcfn}, 'jitterSigma2', 0.001,'infer_params', 'covariance')    

% -----------------------------
% --- Conduct the inference ---
%
% We will make the inference first by finding a maximum a posterior estimate 
% for the hyperparameters via gradient based optimization.  

% --- MAP estimate using scaled conjugate gradient method ---
%     (see scg2 for more details)

fe=str2fun('gp_e');     % create a function handle to negative log posterior
fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior

% set the options for scg2
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do the optimization
w0 = gp_pak(gp);
w=scg2(fe, w0, opt, fg, gp, x, y);
gp = gp_unpak(gp,w);

% NOTICE here that when the hyperparameters are packed into vector with 'gp_pak'
% they are also transformed through logarithm. The reason for this is that they 
% are easier to sample with MCMC after log transformation.

% Make predictions. Below Ef_full is the predictive mean and Varf_full the 
% predictive variance.
x1=[1:800]';

[Ef_full, Varf_full] = gp_pred(gp, x, y, x1);
Varf_full = Varf_full + gp.noise{1}.noiseSigma2;

% Plot the prediction and data
figure;hold on
plot(x,y,'.', 'MarkerSize',7)
plot(x1,Ef_full,'k', 'LineWidth', 2)
plot(x1,Ef_full-2.*sqrt(Varf_full),'k--')
plot(x1,Ef_full+2.*sqrt(Varf_full),'k--')
axis tight
caption1 = sprintf('Full GP:  l_1= %.2f, s^2_1 = %.2f, \n l_2= %.2f, s^2_2 = %.2f \n s^2_{noise} = %.2f', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2, gp.cf{2}.lengthScale, gp.cf{2}.magnSigma2, gp.noise{1}.noiseSigma2);
title(caption1)
legend('Data point', 'predicted mean', '2\sigma error', 'Location', 'NorthWest')

% -------------------------------------------
% INFERENCE WITH PERIODIC COVARIANCE FUNCTION

% With the increasing number of hyperparameters, the optimisation takes
% longer, especially with period length optimisation included. The results
% are however significantly better. Both models fit the data well, yet only
% the one with the periodic component has real predictive power.

% ---------------------------
% --- Construct the model ---
% 
% First create a set of covariance functions: a long term squared
% exponential function, two short term ones, the periodic function and a
% noise structure
gpcf1 = gpcf_sexp('init', 'lengthScale', 67*12, 'magnSigma2', 66*66);
gpcfp = gpcf_periodic('init', 'nin', nin, 'lengthScale', 1.3, 'magnSigma2', 2.4*2.4);
gpcfp = gpcf_periodic('set', gpcfp, 'period', 12,'optimPeriod',1,'lengthScale_exp', 90*12, 'decay', 1);
gpcfn = gpcf_noise('init', 'noiseSigma2', 0.3);
gpcf2 = gpcf_sexp('init', 'lengthScale', 2, 'magnSigma2', 2);
%gpcf3 = gpcf_sexp('init', 'lengthScale', 1, 'magnSigma2', 1);

% ... Then set the prior for the parameters of covariance functions...
pl = prior_t('init', 's2', 10, 'nu', 3);
pn = prior_t('init', 's2', 10, 'nu', 4);

gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);
gpcf2 = gpcf_sexp('set', gpcf2, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);
%gpcf3 = gpcf_sexp('set', gpcf3, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);
gpcfp = gpcf_periodic('set', gpcfp, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);
gpcfp = gpcf_periodic('set', gpcfp, 'lengthScale_exp_prior', pl, 'period_prior', pn);
gpcfn = gpcf_noise('set', gpcfn, 'noiseSigma2_prior', pn);

% ... Finally create the GP data structure
gp = gp_init('init', 'FULL', 'gaussian', {gpcf1, gpcfp, gpcf2}, {gpcfn}, 'jitterSigma2', 0.003) 


% -----------------------------
% --- Conduct the inference ---
%
% We will make the inference first by finding a maximum a posterior estimate 
% for the hyperparameters via gradient based optimization.  

% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg2 for more details)

fe=str2fun('gp_e');     % create a function handle to negative log posterior
fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior

% set the options for scg2
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;
opt.maxiter = 50;

% do the optimization
w0 = gp_pak(gp);
w=scg2(fe, w0, opt, fg, gp, x, y);
gp = gp_unpak(gp,w);

% NOTICE here that when the hyperparameters are packed into vector with 'gp_pak'
% they are also transformed through logarithm. The reason for this is that they 
% are easier to sample with MCMC after log transformation.

% Make predictions. Below Ef_full is the predictive mean and Varf_full the 
% predictive variance.

x1=[1:800]';

[Ef_full, Varf_full, Ey, Vary] = gp_pred(gp, x, y, x1);
Varf_full = Vary;

% Plot the prediction and data
figure;hold on
plot(x1,Ef_full,'k')
plot(x1,Ef_full-2.*sqrt(Varf_full),'k--')
plot(x1,Ef_full+2.*sqrt(Varf_full),'k--')
plot(x,y,'.', 'MarkerSize',7)
axis tight
caption1 = sprintf('Full GP:  l_1= %.2f, s^2_1 = %.2f, \n l_2= %.2f, s^2_2 = %.2f, p=%.2f, s_exp^2 = %.2f, \n l_3= %.2f, s^2_3 = %.2f, \n l_4= %.2f, s^2_4 = %.2f, \n s^2_{noise} = %.2f', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2, gp.cf{2}.lengthScale, gp.cf{2}.magnSigma2, gp.cf{2}.period, gp.cf{2}.lengthScale_exp, gp.cf{3}.lengthScale, gp.cf{3}.magnSigma2, gp.noise{1}.noiseSigma2);
title(caption1)
legend(caption1, 'predicted mean', '2\sigma error','Location','NorthWest')

% Plot the components separately
[Ef_full, Varf_full, Ey_full, Vary_full] = gp_pred(gp, x, y, x);
[Ef_full1, Varf_full1] = gp_pred(gp, x, y, x, 'predcf', 1);
[Ef_full2, Varf_full2] = gp_pred(gp, x, y, x, 'predcf', [2 3]);

figure
[AX, H1, H2] = plotyy(x, Ef_full2, x, Ef_full1);
set(H2,'LineStyle','--')
set(H2, 'LineWidth', 2)
%set(H1, 'Color', 'k')
set(H1,'LineStyle','-')
set(H1, 'LineWidth', 0.8)
title('The long and short term trend')






%========================================================
% PART 2 Drowning data analysis with FULL GP
%========================================================

% Here we use a GP model with Poisson likelihood to analyse the
% monthly Finnish drowning mortality data from 2002-2008. Finland,
% with almost 200 000 lakes and a long coast on the Baltic sea, has a
% relatively high drowning mortality among the developed countries.
% It is well known that drownings exhibit a periodic behaviour within
% the year, peaking in the summer holiday season in July and coming to
% near zero in the winter when most lakes and the Baltic sea are
% frozen.

% The Poisson likelihood is chosen to deal with the regression of
% counts.  As the amount of drownings, although small in the
% wintertime, can never be negative, a Gaussian likelihood is not
% suitable. A negative binomial is another option, especially with
% overdispersed data (ie. with high variance), as it provides another
% parameter to control the dispersion.

% Load the data

S = which('demo_periodicCov');
L = strrep(S,'demo_periodicCov.m','demos/drowning.txt');
data=load(L);
y = data(:, 2:13);
y=y';
y=y(:);
y1=y;
y=y(1:72);
avgy = mean(y);
x = [1:length(y)]';

[n,nin] = size(x);

% ---------------------------
% --- Construct the model ---
% 

% Create covariance functions. Here we use a squared exponential and a
% neural network function to deal with long term change, another SE for
% short term effects and a periodic component for the cyclic nature of the
% data. The period of the cycle is not optimised as it is strongly believed
% to be exactly 12 months.

gpcf1 = gpcf_sexp('init', 'lengthScale', [67], 'magnSigma2', 1);
gpcfp = gpcf_periodic('init', 'nin', nin, 'lengthScale', [1.3], 'magnSigma2', 2.4*2.4,...
    'period', 12,'optimPeriod',0, 'lengthScale_exp', 50, 'decay', 1);
gpcfnn=gpcf_neuralnetwork('init', 'biasSigma2',10,'weightSigma2',3);
gpcf2 = gpcf_sexp('init', 'lengthScale', [2], 'magnSigma2', 2);

% ... Then set the prior for the parameters of covariance functions...
pl = prior_t('init', 's2', 1000, 'nu', 3);
pm = prior_t('init', 's2', 2, 'nu', 3);
pl2 = prior_t('init', 's2', 5, 'nu', 3);
pm2 = prior_t('init', 's2', 3, 'nu', 3);
ppl = prior_t('init', 's2', 100, 'nu', 3);
ppm = prior_t('init', 's2', 1, 'nu', 3);
pn = prior_t('init', 's2', 10, 'nu', 4);
ppp = prior_t('init', 's2', 100, 'nu', 4);

gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
gpcf2 = gpcf_sexp('set', gpcf2, 'lengthScale_prior', pl2, 'magnSigma2_prior', pm2);
gpcfnn = gpcf_neuralnetwork('set', gpcfnn, 'biasSigma2_prior', pn, 'weightSigma2_prior', ppp);
gpcfp = gpcf_periodic('set', gpcfp, 'lengthScale_prior', ppl, 'magnSigma2_prior', ppm,  'lengthScale_exp_prior', pl);
gpcfn = gpcf_noise('set', gpcfn, 'noiseSigma2_prior', pn);



% ... Create the GP data structure, Poisson likelihood with Expectation
% Propagation as approximation method
likelih = likelih_poisson('init');
z=repmat(mean(y),length(y),1);

gp = gp_init('init', 'FULL', likelih, {gpcf1,gpcfp,gpcf2,gpcfnn}, {}, 'jitterSigma2', 0.001,'infer_params', 'covariance')   
gp = gp_init('set', gp, 'latent_method', {'EP', x, y,'z',z});

fe=str2fun('gpep_e');     % create a function handle to negative log posterior
fg=str2fun('gpep_g');     % create a function handle to gradient of negative log posterior

% set the options for scg2
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;
opt.maxiter = 50;

% Learn the hyperparameters
w0 = gp_pak(gp);
w=scg2(fe, w0, opt, fg, gp, x, y, 'z', z);
gp = gp_unpak(gp,w);

% Prediction, has to be converted with exp
x1=[1:96]';
[Ef_full, Varf_full] = ep_pred(gp, x, y, x1);

% Plot results
x11=2001+23/24+1/12*x1;
figure;hold on
plot(x11(1:length(y),1),y,'k.', 'MarkerSize',20)
plot(x11(length(y)+1:length(y1),1),y1(length(y)+1:length(y1)),'k*', 'MarkerSize',7)
plot(x11(:,1),exp(Ef_full).*mean(y),'k', 'LineWidth', 2)
plot(x11(:,1),exp(Ef_full-2.*sqrt(Varf_full)).*mean(y),'k--')
plot(x11(:,1),exp(Ef_full+2.*sqrt(Varf_full)).*mean(y),'k--')


legend('Training data', 'Validation data','Predicted mean','2\sigma-error', 'Location', 'NorthWest')
line(2008,0:80,'LineWidth',2)
axis tight
