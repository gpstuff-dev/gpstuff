%DEMO_REGRESSION1    Regression problem demonstration for 2-input 
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
%  1) data analysis with MAP estimate
%  2) data analysis with LOO-CV estimate
%  3) data analysis with 10-fold-CV estimate
%  4) data analysis with 

%========================================================
% PART 1 data analysis with MAP estimate
%========================================================

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
gp = gp_init('init', 'FULL', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.001)    


% -----------------------------
% --- Conduct the inference ---
%
% We will make the inference first by finding a maximum a posterior estimate 
% for the hyperparameters via gradient based optimization. After this we will
% perform an extensive Markov chain Monte Carlo sampling for the hyperparameters.
% 

% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg for more details)

w=gp_pak(gp, 'hyper');  % pack the hyperparameters into one vector
fe=str2fun('gp_e');     % create a function handle to negative log posterior
fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior

% set the options
opt(1) = 1;
opt(2) = 1e-5;
opt(3) = 3e-5;
opt(9) = 0;
opt(10) = 0;
opt(11) = 0;
opt(14) = 0;

% do the optimization
[w, opt, flog]=scg(fe, w, opt, fg, gp, x, y, 'hyper');

S = sprintf(['The hyperparameter values, MAP estimate:                              \n'...
             ' magnSigma2:    %.6f\n length-scales: %.6f %.6f\n noiseSigma2:   %.6f \n'], exp(w(1)), exp(w(2)), exp(w(3)), exp(w(4)));
fprintf(S)

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w, 'hyper');

% NOTICE here that when the hyperparameters are packed into vector with 'gp_pak'
% they are also transformed through logarithm. The reason for this is that they 
% are easier to sample with MCMC after log transformation.

% for the last make prections of the underlying function on a dense grid 
% and plot it. Below Ef_full is the predictive mean and Varf_full the predictive 
% variance.
[p1,p2]=meshgrid(-1.8:0.05:1.8,-1.8:0.05:1.8);
p=[p1(:) p2(:)];
[Ef_full, Varf_full] = gp_pred(gp, x, y, p);

% Plot the prediction and data
figure(1)
mesh(p1, p2, reshape(Ef_full,73,73));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title('The predicted underlying function and the data points (MAP solution)');

%=============================================================
% PART 2 data analysis with full GP and LOO-CV energy function
%=============================================================

% Here we conduct the same analysis as in part 1, but this time we 
% use LOO-CV energy function

gp_cv = gp_init('init', 'FULL', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.001)    


% -----------------------------
% --- Conduct the inference ---
%
% We will make the inference first by finding a maximum a posterior estimate 
% for the hyperparameters via gradient based optimization. After this we will
% perform an extensive Markov chain Monte Carlo sampling for the hyperparameters.
% 

% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg for more details)

w_cv=gp_pak(gp_cv, 'hyper');  % pack the hyperparameters into one vector
fe=str2fun('gp_cve');     % create a function handle to negative log posterior
fg=str2fun('gp_cvg');     % create a function handle to gradient of negative log posterior

% set the options
opt(1) = 1;
opt(2) = 1e-5;
opt(3) = 3e-5;
opt(9) = 0;
opt(10) = 0;
opt(11) = 0;
opt(14) = 0;

% do the optimization
[w_cv, opt, flog]=scg(fe, w_cv, opt, fg, gp_cv, x, y, 'hyper');

S_cv = sprintf(['The hyperparameter values, LOO-CV error:                              \n'...
                ' magnSigma2:    %.6f\n length-scales: %.6f %.6f\n noiseSigma2:   %.6f \n'], exp(w_cv(1)), exp(w_cv(2)), exp(w_cv(3)), exp(w_cv(4)));

fprintf([S '\n' S_cv])


% Set the optimized hyperparameter values back to the gp structure
gp_cv = gp_unpak(gp_cv, w_cv, 'hyper');

% NOTICE here that when the hyperparameters are packed into vector with 'gp_pak'
% they are also transformed through logarithm. The reason for this is that they 
% are easier to sample with MCMC after log transformation.

% for the last make prections of the underlying function on a dense grid 
% and plot it. Below Ef_full is the predictive mean and Varf_full the predictive 
% variance.
[p1,p2]=meshgrid(-1.8:0.05:1.8,-1.8:0.05:1.8);
p=[p1(:) p2(:)];
[Ef_cv, Varf_cv] = gp_pred(gp_cv, x, y, p);

% Plot the prediction and data
figure(2)
mesh(p1, p2, reshape(Ef_cv,73,73));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title('The predicted underlying function and the data points (LOO-CV solution)');

w0 = randn(size(w_cv));
gradcheck(w0, @gp_cve, @gp_cvg, gp, x, y, 'hyper')

%=================================================================
% PART 3 data analysis with full GP and 10-fold-CV energy function
%=================================================================















[Ef, Varf] = gp_pred(gp, x(2:end,:), y(2:end), x(1,:));
Varf = Varf + gp.noise{1}.noiseSigmas2;
pp = norm_lpdf(y(1), Ef, sqrt(Varf));
for i = 2:n-1;
    xt = [x(1:i-1,:) ; x(i+1:end,:)];
    yt = [y(1:i-1,:) ; y(i+1:end,:)];
    [Ef, Varf] = gp_pred(gp, xt, yt, x(i,:));
    Varf = Varf + gp.noise{1}.noiseSigmas2;
    pp = pp + norm_lpdf(y(i), Ef, sqrt(Varf));              
end          
[Ef, Varf] = gp_pred(gp, x(1:end-1,:), y(1:end-1), x(end,:));
Varf = Varf + gp.noise{1}.noiseSigmas2;
pp = pp + norm_lpdf(y(end), Ef, sqrt(Varf));
