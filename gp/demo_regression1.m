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
%   For more detailed discussion of Gaussian process regression see,
%   for example, Rasmussen and Williams (2006 or Vanhatalo and Vehtari
%   (2008)
%
%
%   See also  DEMO_REGRESSION2

% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


% This file is organised in three parts:
%  1) data analysis with full GP model
%  2) data analysis with compact support (CS) GP model
%  3) data analysis with FIC approximation
%  4) data analysis with PIC approximation

%========================================================
% PART 1 data analysis with full GP model
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
%gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);
gpcf1 = gpcf_exp('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);

% ... Then set the prior for the parameters of covariance functions...
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% ... Finally create the GP data structure
gp = gp_init('init', 'FULL', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.001)    

% What has happend this far is following
% - we created data structures 'gpcf1' and 'gpcf2', which describe 
%   the properties of the covariance function and Gaussian noise
%   (see gpcf_sexp and gpcf_noise for more details)
% - we created data structures that describe the prior of the length-scale 
%   and magnitude of the squared exponential covariance function and the 
%   prior of the noise variance. These structures were set into 'gpcf1' and 
%   'gpcf2'
%   (see sinvchi2_p and gamma_p for more details)
% - we created a GP data structure 'gp', which has among others 'gpcf1' and 
%   'gpcf2' data structures.
%   (see gp_init for more details)

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
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do the optimization
[w, opt, flog]=scg2(fe, w, opt, fg, gp, x, y, 'hyper');

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w, 'hyper');

% NOTICE here that when the hyperparameters are packed into vector with 'gp_pak'
% they are also transformed through logarithm. The reason for this is that they 
% are easier to sample with MCMC after log transformation.

% for the last make prections of the underlying function on a dense grid 
% and plot it. Below Ef_full is the predictive mean and Varf_full the predictive 
% variance.
[p1,p2]=meshgrid(-1.8:0.1:1.8,-1.8:0.1:1.8);
p=[p1(:) p2(:)];
[Ef_full, Varf_full] = gp_pred(gp, x, y, p);

% Plot the prediction and data
figure(1)
mesh(p1, p2, reshape(Ef_full,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title('The predicted underlying function and the data points (MAP solution)');


% --- MCMC approach ---
%  (see gp_mc for details
% The hyperparameters are sampled with hybrid Monte Carlo 
% (see, for example, Neal (1996)). 

% The sampling options are set to 'opt' structure, which is given to
% 'gp_mc' sampler
opt=gp_mcopt;
opt.nsamples= 300;
opt.repeat=5;
opt.hmc_opt.steps=4;
opt.hmc_opt.stepadj=0.05;
opt.hmc_opt.persistence=0;
opt.hmc_opt.decay=0.6;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

% Do the sampling (this takes approximately 5-10 minutes)
% 'rfull'   will contain a record structure with all the sampls
% 'g'       will contain a GP structure at the current state of the sampler
% 'rstate1' will contain a structure with information of the state of the sampler
[rfull,g,rstate1] = gp_mc(opt, gp, x, y);

% After sampling we delete the burn-in and thin the sample chain
rfull = thin(rfull, 10, 2);

% Now we make the predictions. 'gp_preds' is a function that returns 
% the predictive mean of the latent function with every sampled 
% hyperparameter value. Thus, the returned Ef_sfull is a matrix of 
% size n x (number of samples). By taking the mean over the samples
% we do the Monte Carlo integration over the hyperparameters.
Ef_sfull = gp_preds(rfull, x, y, p);
meanEf_full = mean(squeeze(Ef_sfull)');

figure(1)
clf, subplot(1,2,1)
mesh(p1, p2, reshape(Ef_full,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title(['The predicted underlying function ';
       'and the data points (MAP solution)']);
subplot(1,2,2)
mesh(p1, p2, reshape(meanEf_full,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title(['The predicted underlying function  ';
       'and the data points (MCMC solution)']);
set(gcf,'pos',[93 511 1098 420])

% We can compare the posterior samples of the hyperparameters to the 
% MAP estimate that we got from optimization
figure(2)
clf, subplot(1,2,1)
plot(rfull.cf{1}.lengthScale)
title('The sample chain of length-scales')
subplot(1,2,2)
plot(rfull.cf{1}.magnSigma2)
title('The sample chain of magnitude')
set(gcf,'pos',[93 511 1098 420])

figure(3)
clf, subplot(1,4,1)
hist(rfull.cf{1}.lengthScale(:,1),20)
hold on
plot(gp.cf{1}.lengthScale(1), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 1')
subplot(1,4,2)
hist(rfull.cf{1}.lengthScale(:,2),20)
hold on
plot(gp.cf{1}.lengthScale(2), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 2')
subplot(1,4,3)
hist(rfull.cf{1}.magnSigma2,20)
hold on
plot(gp.cf{1}.magnSigma2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('magnitude')
subplot(1,4,4)
hist(rfull.noise{1}.noiseSigmas2,20)
hold on
plot(gp.noise{1}.noiseSigmas2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Noise variance')
legend('MCMC samples', 'MAP estimate')
set(gcf,'pos',[93 511 1098 420])



%========================================================
% PART 2 data analysis with compact support (CS) GP 
%========================================================

% Load the data
S = which('demo_regression1');
L = strrep(S,'demo_regression1.m','demos/dat.1');
data=load(L);
x = [data(:,1) data(:,2)];
y = data(:,3);
[n, nin] = size(x);

% Here we conduct the same analysis as in part 1, but this time we 
% use compact support covariance function

% Create the piece wise polynomial covariance function
gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);

gpcf3 = gpcf_ppcs2('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);
gpcf3.p.lengthScale = gamma_p({3 7});  
gpcf3.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});

% Create the GP data structure
gp_cs = gp_init('init', 'FULL', nin, 'regr', {gpcf3}, {gpcf2}, 'jitterSigmas', 0.001)

% -----------------------------
% --- Conduct the inference ---

% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg for more details)

param = 'hyper';

% set the options
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

w = gp_pak(gp_cs, param);          % pack the hyperparameters into one vector
[w, opt, flog]=scg2(fe, w, opt, fg, gp_cs, x, y, param);       % do the optimization
gp_cs = gp_unpak(gp_cs,w, param);     % Set the optimized hyperparameter values back to the gp structure

% Make the prediction
[Ef_cs, Varf_cs] = gp_pred(gp_cs, x, y, p);

% Plot the solution of full GP and CS
figure(1)
clf, subplot(1,2,1)
mesh(p1, p2, reshape(Ef_full,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title(['The predicted underlying function and data points (full GP)']);
xlim([-2 2]), ylim([-2 2])
subplot(1,2,2)
mesh(p1, p2, reshape(Ef_cs,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title(['The predicted underlying function and data points (CS)']);
xlim([-2 2]), ylim([-2 2])
set(gcf,'pos',[93 511 1098 420])


% --- MCMC approach ---
%  (see gp_mc for details
% The hyperparameters are sampled with hybrid Monte Carlo 
% the Inducing inputs are kept fixed at the optimized locations

% The sampling options are set to 'opt' structure, which is given to
% 'gp_mc' sampler
opt=gp_mcopt;
opt.nsamples= 300;
opt.repeat=5;
opt.hmc_opt.steps=3;
opt.hmc_opt.stepadj=0.02;
opt.hmc_opt.persistence=0;
opt.hmc_opt.decay=0.6;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

% Do the sampling (this takes approximately 3-5 minutes)
[rcs,g2,rstate2] = gp_mc(opt, gp_cs, x, y);

% After sampling we delete the burn-in and thin the sample chain
rcs = thin(rcs, 10, 2);

% Make the predictions. 
Ef_scs = gp_preds(rcs, x, y, p);
meanEf_cs = mean(squeeze(Ef_scs)');

% Plot the results
figure(1)
clf, subplot(1,2,1)
mesh(p1, p2, reshape(Ef_cs,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title(['The predicted underlying function and';
       'the data points (MAP solutionm, CS) ']);
subplot(1,2,2)
mesh(p1, p2, reshape(meanEf_cs,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title(['The predicted underlying function and';
       'the data points (MCMC solution, CS) ']);
set(gcf,'pos',[93 511 1098 420])


% Here we copare the hyperparameter posteriors of CS and full GP
figure(3)
clf, subplot(2,4,1)
hist(rcs.cf{1}.lengthScale(:,1),20)
hold on
plot(gp_cs.cf{1}.lengthScale(1), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 1 (CS)')
subplot(2,4,2)
hist(rcs.cf{1}.lengthScale(:,2),20)
hold on
plot(gp_cs.cf{1}.lengthScale(2), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 2 (CS)')
subplot(2,4,3)
hist(rcs.cf{1}.magnSigma2,20)
hold on
plot(gp_cs.cf{1}.magnSigma2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('magnitude (CS)')
subplot(2,4,4)
hist(rcs.noise{1}.noiseSigmas2,20)
hold on
plot(gp_cs.noise{1}.noiseSigmas2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Noise variance (CS)')
subplot(2,4,5)
hist(rfull.cf{1}.lengthScale(:,1),20)
hold on
plot(gp.cf{1}.lengthScale(1), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 1 (full GP)')
subplot(2,4,6)
hist(rfull.cf{1}.lengthScale(:,2),20)
hold on
plot(gp.cf{1}.lengthScale(2), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 2 (full GP)')
subplot(2,4,7)
hist(rfull.cf{1}.magnSigma2,20)
hold on
plot(gp.cf{1}.magnSigma2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('magnitude (full GP)')
subplot(2,4,8)
hist(rfull.noise{1}.noiseSigmas2,20)
hold on
plot(gp.noise{1}.noiseSigmas2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Noise variance (full GP)')
set(gcf,'pos',[93 511 1098 420])


%========================================================
% PART 3 data analysis with FIC approximation
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

% Here we conduct the same analysis as in part 1, but this time we 
% use FIC approximation

% Initialize the inducing inputs in a regular grid over the input space
[u1,u2]=meshgrid(linspace(-1.8,1.8,6),linspace(-1.8,1.8,6));
X_u = [u1(:) u2(:)];

% Create the FIC GP data structure
gp_fic = gp_init('init', 'FIC', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.001, 'X_u', X_u)

% -----------------------------
% --- Conduct the inference ---

% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg for more details)

% Now you can choose, if you want to optimize only hyperparameters or 
% optimize simultaneously hyperparameters and inducing inputs. Note that 
% the inducing inputs are not transformed through logarithm when packed

% param = 'hyper+inducing'; % optimize hyperparameters and inducing inputs
param = 'hyper';          % optimize only hyperparameters

% set the options
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

w = gp_pak(gp_fic, param);          % pack the hyperparameters into one vector
[w, opt, flog]=scg2(fe, w, opt, fg, gp_fic, x, y, param);       % do the optimization
gp_fic = gp_unpak(gp_fic,w, param);     % Set the optimized hyperparameter values back to the gp structure

% Make the prediction
[Ef_fic, Varf_fic] = gp_pred(gp_fic, x, y, p);

% Plot the solution of full GP and FIC
figure(1)
clf, subplot(1,2,1)
mesh(p1, p2, reshape(Ef_full,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title(['The predicted underlying function ';
       'and data points (full GP)         ']);
xlim([-2 2]), ylim([-2 2])
subplot(1,2,2)
mesh(p1, p2, reshape(Ef_fic,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
plot3(gp_fic.X_u(:,1), gp_fic.X_u(:,2), -3*ones(length(u1(:))), 'rx')
axis on;
title(['The predicted underlying function,   ';
       'data points and inducing inputs (FIC)']);
xlim([-2 2]), ylim([-2 2])
set(gcf,'pos',[93 511 1098 420])


% --- MCMC approach ---
%  (see gp_mc for details
% The hyperparameters are sampled with hybrid Monte Carlo 
% the Inducing inputs are kept fixed at the optimized locations

% The sampling options are set to 'opt' structure, which is given to
% 'gp_mc' sampler
opt=gp_mcopt;
opt.nsamples= 300;
opt.repeat=5;
opt.hmc_opt.steps=3;
opt.hmc_opt.stepadj=0.02;
opt.hmc_opt.persistence=0;
opt.hmc_opt.decay=0.6;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

% Do the sampling (this takes approximately 3-5 minutes)
[rfic,g2,rstate2] = gp_mc(opt, gp_fic, x, y);

% After sampling we delete the burn-in and thin the sample chain
rfic = thin(rfic, 10, 2);

% Make the predictions. 
Ef_sfic = gp_preds(rfic, x, y, p);
meanEf_fic = mean(squeeze(Ef_sfic)');

% Plot the results
figure(1)
clf, subplot(1,2,1)
mesh(p1, p2, reshape(Ef_fic,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title(['The predicted underlying function and';
       'the data points (MAP solutionm, FIC) ']);
subplot(1,2,2)
mesh(p1, p2, reshape(meanEf_fic,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title(['The predicted underlying function and';
       'the data points (MCMC solution, FIC) ']);
set(gcf,'pos',[93 511 1098 420])


% Here we copare the hyperparameter posteriors of FIC and full GP
figure(3)
clf, subplot(2,4,1)
hist(rfic.cf{1}.lengthScale(:,1),20)
hold on
plot(gp_fic.cf{1}.lengthScale(1), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 1 (FIC)')
subplot(2,4,2)
hist(rfic.cf{1}.lengthScale(:,2),20)
hold on
plot(gp_fic.cf{1}.lengthScale(2), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 2 (FIC)')
subplot(2,4,3)
hist(rfic.cf{1}.magnSigma2,20)
hold on
plot(gp_fic.cf{1}.magnSigma2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('magnitude (FIC)')
subplot(2,4,4)
hist(rfic.noise{1}.noiseSigmas2,20)
hold on
plot(gp_fic.noise{1}.noiseSigmas2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Noise variance (FIC)')
subplot(2,4,5)
hist(rfull.cf{1}.lengthScale(:,1),20)
hold on
plot(gp.cf{1}.lengthScale(1), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 1 (full GP)')
subplot(2,4,6)
hist(rfull.cf{1}.lengthScale(:,2),20)
hold on
plot(gp.cf{1}.lengthScale(2), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 2 (full GP)')
subplot(2,4,7)
hist(rfull.cf{1}.magnSigma2,20)
hold on
plot(gp.cf{1}.magnSigma2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('magnitude (full GP)')
subplot(2,4,8)
hist(rfull.noise{1}.noiseSigmas2,20)
hold on
plot(gp.noise{1}.noiseSigmas2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Noise variance (full GP)')
set(gcf,'pos',[93 511 1098 420])


%========================================================
% PART 4 data analysis with PIC approximation
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

% Here we conduct the same analysis as in part 1, but this time we 
% use FIC approximation

% Initialize the inducing inputs in a regular grid over the input space
[u1,u2]=meshgrid(linspace(-1.8,1.8,6),linspace(-1.8,1.8,6));
X_u = [u1(:) u2(:)];

[p1,p2]=meshgrid(-1.8:0.1:1.8,-1.8:0.1:1.8);
p=[p1(:) p2(:)];

% set the data points into clusters
b1 = [-1.7 -0.8 0.1 1 1.9];
mask = zeros(size(x,1),size(x,1));
trindex={}; tstindex={}; 
for i1=1:4
    for i2=1:4
        ind = 1:size(x,1);
        ind = ind(: , b1(i1)<=x(ind',1) & x(ind',1) < b1(i1+1));
        ind = ind(: , b1(i2)<=x(ind',2) & x(ind',2) < b1(i2+1));
        trindex{4*(i1-1)+i2} = ind';
        ind2 = 1:size(p,1);
        ind2 = ind2(: , b1(i1)<=p(ind2',1) & p(ind2',1) < b1(i1+1));
        ind2 = ind2(: , b1(i2)<=p(ind2',2) & p(ind2',2) < b1(i2+1));
        tstindex{4*(i1-1)+i2} = ind2';
    end
end

% Create the FIC GP data structure
gp_pic = gp_init('init', 'PIC_BLOCK', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.001, 'X_u', X_u)
gp_pic = gp_init('set', gp_pic, 'blocks', {'manual', x, trindex});

% -----------------------------
% --- Conduct the inference ---

% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg for more details)

% Now you can choose, if you want to optimize only hyperparameters or 
% optimize simultaneously hyperparameters and inducing inputs. Note that 
% the inducing inputs are not transformed through logarithm when packed

% param = 'hyper+inducing'; % optimize hyperparameters and inducing inputs
param = 'hyper';          % optimize only hyperparameters

% set the options
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

w = gp_pak(gp_pic, param);          % pack the hyperparameters into one vector
[w, opt, flog]=scg2(fe, w, opt, fg, gp_pic, x, y, param);       % do the optimization
gp_pic = gp_unpak(gp_pic,w, param);     % Set the optimized hyperparameter values back to the gp structure

% Make the prediction
[Ef_pic, Varf_pic] = gp_pred(gp_pic, x, y, p, tstindex);

% Plot the solution of full GP and FIC
figure(1)
clf, subplot(1,2,1)
mesh(p1, p2, reshape(Ef_full,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title(['The predicted underlying function ';
       'and data points (full GP)         ']);
xlim([-2 2]), ylim([-2 2])
subplot(1,2,2)
mesh(p1, p2, reshape(Ef_pic,37,37));
hold on
% plot the data points in each block with different colors and marks
col = {'b*','g*','r*','c*','m*','y*','k*','b*','b.','g.','r.','c.','m.','y.','k.','b.'};
hold on
for i=1:16
    plot3(x(trindex{i},1),x(trindex{i},2), y(trindex{i}),col{i})
end
plot3(gp_pic.X_u(:,1), gp_pic.X_u(:,2), -3*ones(length(u1(:))), 'rx')
axis on;
title(['The predicted underlying function, data points (colors ';
       'distinguish the blocks) and inducing inputs (PIC)      ']);
xlim([-2 2]), ylim([-2 2])
set(gcf,'pos',[93 511 1098 420])


% --- MCMC approach ---
%  (see gp_mc for details
% The hyperparameters are sampled with hybrid Monte Carlo 
% the Inducing inputs are kept fixed at the optimized locations

% The sampling options are set to 'opt' structure, which is given to
% 'gp_mc' sampler
opt=gp_mcopt;
opt.nsamples= 300;
opt.repeat=5;
opt.hmc_opt.steps=3;
opt.hmc_opt.stepadj=0.02;
opt.hmc_opt.persistence=0;
opt.hmc_opt.decay=0.6;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

% Do the sampling (this takes approximately 3-5 minutes)
[rpic,g2,rstate2] = gp_mc(opt, gp_pic, x, y);

% After sampling we delete the burn-in and thin the sample chain
rpic = rmfield(rpic, 'tr_index');
rpic = thin(rpic, 10, 2);
rpic.tr_index = trindex;

% Make the predictions. 
Ef_spic = gp_preds(rpic, x, y, p, tstindex);
meanEf_pic = mean(squeeze(Ef_spic)');

% Plot the results
figure(1)
clf, subplot(1,2,1)
mesh(p1, p2, reshape(Ef_pic,37,37));
hold on
% plot the data points in each block with different colors and marks
col = {'b*','g*','r*','c*','m*','y*','k*','b*','b.','g.','r.','c.','m.','y.','k.','b.'};
hold on
for i=1:16
    plot3(x(trindex{i},1),x(trindex{i},2), y(trindex{i}),col{i})
end
axis on;
title(['The predicted underlying function and';
       'the data points (MAP solutionm, PIC) ']);
subplot(1,2,2)
mesh(p1, p2, reshape(meanEf_pic,37,37));
hold on
% plot the data points in each block with different colors and marks
col = {'b*','g*','r*','c*','m*','y*','k*','b*','b.','g.','r.','c.','m.','y.','k.','b.'};
hold on
for i=1:16
    plot3(x(trindex{i},1),x(trindex{i},2), y(trindex{i}),col{i})
end
axis on;
title(['The predicted underlying function and';
       'the data points (MCMC solution, PIC) ']);
set(gcf,'pos',[93 511 1098 420])


% Here we copare the hyperparameter posteriors of FIC and full GP
figure(3)
clf, subplot(2,4,1)
hist(rpic.cf{1}.lengthScale(:,1),20)
hold on
plot(gp_pic.cf{1}.lengthScale(1), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 1 (PIC)')
subplot(2,4,2)
hist(rpic.cf{1}.lengthScale(:,2),20)
hold on
plot(gp_pic.cf{1}.lengthScale(2), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 2 (PIC)')
subplot(2,4,3)
hist(rpic.cf{1}.magnSigma2,20)
hold on
plot(gp_pic.cf{1}.magnSigma2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('magnitude (PIC)')
subplot(2,4,4)
hist(rpic.noise{1}.noiseSigmas2,20)
hold on
plot(gp_pic.noise{1}.noiseSigmas2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Noise variance (PIC)')
subplot(2,4,5)
hist(rfull.cf{1}.lengthScale(:,1),20)
hold on
plot(gp.cf{1}.lengthScale(1), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 1 (full GP)')
subplot(2,4,6)
hist(rfull.cf{1}.lengthScale(:,2),20)
hold on
plot(gp.cf{1}.lengthScale(2), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 2 (full GP)')
subplot(2,4,7)
hist(rfull.cf{1}.magnSigma2,20)
hold on
plot(gp.cf{1}.magnSigma2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('magnitude (full GP)')
subplot(2,4,8)
hist(rfull.noise{1}.noiseSigmas2,20)
hold on
plot(gp.noise{1}.noiseSigmas2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Noise variance (full GP)')
set(gcf,'pos',[93 511 1098 420])

