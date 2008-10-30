%DEMO_REGRESSION3    Regression problem demonstration for 2-input 
%                    function with Gaussian process
%
%    Description
%    The regression problem consist of a data with two input variables
%    and one output variable with Gaussian noise. The model constructed 
%    is following:
%
%    The observations y are assumed to satisfy
%
%         y = f + e,    where e ~ N(0, s^2)
%
%    where f is an underlying function, which we are interested in. 
%    We place a zero mean Gaussian process prior for f, which implies that
%    at the observed input locations latent values have prior
%
%         f ~ N(0, K),
%
%    where K is the covariance matrix, whose elements are given as 
%    K_ij = k(x_i, x_j | th). The function k(x_i, x_j | th) is covariance 
%    function and th its parameters, hyperparameters. 
%
%    Since both likelihood and prior are Gaussian, we obtain a Gaussian 
%    marginal likelihood
%
%        p(y|th) = N(0, K + I*s^2).
%    
%   By placing a hyperprior for hyperparameters, p(th), we can find the 
%   maximum a posterior (MAP) estimate for them by maximizing
%
%       argmax   log p(y|th) + log p(th).
%         th
%   
%   If we want to find an approximation for the posterior of the hyperparameters, 
%   we can sample them using Markov chain Monte Carlo (MCMC) methods.
%
%   After finding MAP estimate or posterior samples of hyperparameters, we can 
%   use them to make predictions for f:
%
%       p(f | y, th) = N(m, S),
%       m = 
%       S =
%   
%   For more detailed discussion of Gaussian process regression see for example
%   Vanhatalo and Vehtari (2008) or 
%
%
%   See also  DEMO_REGRESSION2

% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


% This file is organised in three parts:
%  1) Comparison of SSsexp and sexp covariance functions
%  2) Data analysis using sexp full GP
%  3) Data analysis using SSGP

%========================================================
% PART 1 Comparison of SSsexp and sexp covariance functions
%========================================================


xx = [-10:0.05:10]';
yy = sin(xx/3)*5 + 0.3*randn(size(xx));
[n,nin] = size(xx);

% ---------------------------
% --- Construct the model ---
% 
% First create squared exponential covariance function with ARD and 
% Gaussian noise data structures...
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 2, 'magnSigma2', 1);
gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);

% ... Then set the prior for the parameters of covariance functions...
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% ... Finally create the GP data structure
gp1 = gp_init('init', 'FULL', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.001)

% Create the squared exponential sparse spectral covariance function
gpcf3 = gpcf_SSsexp('init', nin, 'lengthScale', 2, 'magnSigma2', 1);
gpcf3 = gpcf_SSsexp('set', gpcf3, 'nfreq', 10);

% ... Then set the prior for the parameters of covariance functions...
gpcf3.p.lengthScale = gamma_p({3 7});  
gpcf3.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% ... Finally create the GP data structure
gp2 = gp_init('init', 'SSGP', nin, 'regr', {gpcf3}, {gpcf2})

% compare the covariance functions
[K1, C1] = gp_trcov(gp1, xx);
[Phi, S] = gp_trcov(gp2, xx);

subplot(2,2,1)
plot(xx,K1(201,:))
hold on
K2 = Phi*Phi';
plot(xx,K2(201,:), 'r')
title('10 spectral points')

gpcf3 = gpcf_SSsexp('set', gpcf3, 'nfreq', 50);
gp2 = gp_init('init', 'SSGP', nin, 'regr', {gpcf3}, {gpcf2})
[Phi, S] = gp_trcov(gp2, xx);
subplot(2,2,2)
plot(xx,K1(201,:))
hold on
K2 = Phi*Phi';
plot(xx,K2(201,:), 'r')
title('50 spectral points')

gpcf3 = gpcf_SSsexp('set', gpcf3, 'nfreq', 100);
gp2 = gp_init('init', 'SSGP', nin, 'regr', {gpcf3}, {gpcf2})
[Phi, S] = gp_trcov(gp2, xx);
subplot(2,2,3)
plot(xx,K1(201,:))
hold on
K2 = Phi*Phi';
plot(xx,K2(201,:), 'r')
title('100 spectral points')

gpcf3 = gpcf_SSsexp('set', gpcf3, 'nfreq', 200);
gp2 = gp_init('init', 'SSGP', nin, 'regr', {gpcf3}, {gpcf2})
[Phi, S] = gp_trcov(gp2, xx);
subplot(2,2,4)
plot(xx,K1(201,:))
hold on
K2 = Phi*Phi';
plot(xx,K2(201,:), 'r')
title('200 spectral points')

% ================================================
% PART 2 Data analysis using FULL GP
% ================================================

% Load the data
S = which('demo_regression1');
L = strrep(S,'demo_regression1.m','demos/dat.1');
data=load(L);
x = [data(:,1) data(:,2)];
y = data(:,3);
[n, nin] = size(x);

% Now 'x' consist of the inputs and 'y' of the output. 
% 'n' and 'nin' are the number of data points and the 
% dimensionality of 'x' (the number of inputs).


% ---------------------------
% --- Construct the model ---
% 
% First create squared exponential covariance function with ARD and 
% Gaussian noise data structures...
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);

% ... Then set the prior for the parameters of covariance functions...
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% ... Finally create the GP data structure
gp1 = gp_init('init', 'FULL', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.001)

% -----------------------------
% --- Conduct the inference ---
%
% We will make the inference first by finding a maximum a posterior estimate 
% for the hyperparameters via gradient based optimization. After this we will
% perform an extensive Markov chain Monte Carlo sampling for the hyperparameters.
% 

% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg for more details)

w1=gp_pak(gp1, 'hyper');  % pack the hyperparameters into one vector
fe=str2fun('gp_e');     % create a function handle to negative log posterior
fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior

% set the options
opt(1) = 1;
opt(2) = 1e-2;
opt(3) = 3e-1;
opt(9) = 0;
opt(10) = 0;
opt(11) = 0;
opt(14) = 0;

% do the optimization
[w1, opt, flog]=scg(fe, w1, opt, fg, gp1, x, y, 'hyper');

% Set the optimized hyperparameter values back to the gp structure
gp1=gp_unpak(gp1,w1, 'hyper');

% NOTICE here that when the hyperparameters are packed into vector with 'gp_pak'
% they are also transformed through logarithm. The reason for this is that they 
% are easier to sample with MCMC after log transformation.

% for the last make prections of the underlying function on a dense grid 
% and plot it. Below Ef_full is the predictive mean and Varf_full the predictive 
% variance.
[p1,p2]=meshgrid(-1.8:0.05:1.8,-1.8:0.05:1.8);
p=[p1(:) p2(:)];
[Ef_full, Varf_full] = gp_pred(gp1, x, y, p);

% Plot the prediction and data
figure(2)
mesh(p1, p2, reshape(Ef_full,73,73));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title('The predicted underlying function and the data points (MAP solution)');
 


% ================================================
% PART 3 Data analysis using SSGP
% ================================================

% Create the squared exponential sparse spectral covariance function
gpcf3 = gpcf_SSsexp('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);
gpcf3 = gpcf_SSsexp('set', gpcf3, 'nfreq', 200);

% ... Then set the prior for the parameters of covariance functions...
gpcf3.p.lengthScale = gamma_p({3 7});  
gpcf3.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% ... Finally create the GP data structure
gp2 = gp_init('init', 'SSGP', nin, 'regr', {gpcf3}, {gpcf2})


% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg for more details)

w2=gp_pak(gp2, 'hyper');  % pack the hyperparameters into one vector
[w2, opt, flog]=scg(fe, w2, opt, fg, gp2, x, y, 'hyper');

% Set the optimized hyperparameter values back to the gp structure
gp2=gp_unpak(gp2, w2, 'hyper');

% NOTICE here that when the hyperparameters are packed into vector with 'gp_pak'
% they are also transformed through logarithm. The reason for this is that they 
% are easier to sample with MCMC after log transformation.

% for the last make prections of the underlying function on a dense grid 
% and plot it. Below Ef_full is the predictive mean and Varf_full the predictive 
% variance.
[p1,p2]=meshgrid(-1.8:0.05:1.8,-1.8:0.05:1.8);
p=[p1(:) p2(:)];
[Ef2, Varf2] = gp_pred(gp2, x, y, p);

% Plot the prediction and data
figure(1)
mesh(p1, p2, reshape(Ef2,73,73));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title('The predicted underlying function and the data points (MAP solution)');
 
















































[x1, x2] = meshgrid([-5:0.5:5]', [-5:0.5:5]');
xx = [x1(:) x2(:)];
[n,nin] = size(xx);

% ---------------------------
% --- Construct the model ---
% 
% First create squared exponential covariance function with ARD and 
% Gaussian noise data structures...
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 4, 'magnSigma2', 1);
gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);

% ... Then set the prior for the parameters of covariance functions...
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% ... Finally create the GP data structure
gp1 = gp_init('init', 'FULL', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.001)

% Create the squared exponential sparse spectral covariance function
gpcf3 = gpcf_SSsexp('init', nin, 'lengthScale', 4, 'magnSigma2', 1);
gpcf3 = gpcf_SSsexp('set', gpcf3, 'nfreq', 10);

% ... Then set the prior for the parameters of covariance functions...
gpcf3.p.lengthScale = gamma_p({3 7});  
gpcf3.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% ... Finally create the GP data structure
gp2 = gp_init('init', 'SSGP', nin, 'regr', {gpcf3}, {gpcf2})

% compare the covariance functions
[K1, C1] = gp_trcov(gp1, xx);
[Phi, S] = gp_trcov(gp2, xx);

subplot(2,2,1)
plot(xx,K1(201,:))
hold on
K2 = Phi*Phi';
plot(xx,K2(201,:), 'r')
title('10 spectral points')

gpcf3 = gpcf_SSsexp('set', gpcf3, 'nfreq', 50);
gp2 = gp_init('init', 'SSGP', nin, 'regr', {gpcf3}, {gpcf2})
[Phi, S] = gp_trcov(gp2, xx);
subplot(2,2,2)
plot(xx,K1(201,:))
hold on
K2 = Phi*Phi';
plot(xx,K2(201,:), 'r')
title('50 spectral points')

gpcf3 = gpcf_SSsexp('set', gpcf3, 'nfreq', 100);
gp2 = gp_init('init', 'SSGP', nin, 'regr', {gpcf3}, {gpcf2})
[Phi, S] = gp_trcov(gp2, xx);
subplot(2,2,3)
plot(xx,K1(201,:))
hold on
K2 = Phi*Phi';
plot(xx,K2(201,:), 'r')
title('100 spectral points')

gpcf3 = gpcf_SSsexp('set', gpcf3, 'nfreq', 200);
gp2 = gp_init('init', 'SSGP', nin, 'regr', {gpcf3}, {gpcf2})
[Phi, S] = gp_trcov(gp2, xx);
subplot(2,2,4)
plot(xx,K1(201,:))
hold on
K2 = Phi*Phi';
plot(xx,K2(201,:), 'r')
title('200 spectral points')

