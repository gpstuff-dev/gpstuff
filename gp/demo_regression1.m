%DEMO_REGRESSION1    Regression problem demonstration for 2-input 
%                    function with Gaussian process
%
%    Description
%    The regression problem consist of a data with two input variables
%    and one output variable with Gaussian noise. The model
%    constructed is following:
%
%    The observations y are assumed to satisfy
%
%         y = f + e,    where e ~ N(0, s^2)
%
%    where f is an underlying function, which we are interested in. We
%    place a zero mean Gaussian process prior for f, which implies
%    that at the observed input locations latent values have prior
%
%         f ~ N(0, K),
%
%    where K is the covariance matrix, whose elements are given as
%    K_ij = k(x_i, x_j | th). The function k(x_i, x_j | th) is
%    covariance function and th its parameters, hyperparameters.
%
%    Since both likelihood and prior are Gaussian, we obtain a
%    Gaussian marginal likelihood
%
%        p(y|th) = N(0, K + I*s^2).
%    
%   By placing a hyperprior for hyperparameters, p(th), we can find
%   the maximum a posterior (MAP) estimate for them by maximizing
%
%       argmax   log p(y|th) + log p(th).
%         th
%   
%   An approximation for the posterior of the hyperparameters, can be
%   found using Markov chain Monte Carlo (MCMC) methods. We can
%   integrate over the hyperparameters also with other integration
%   approximations such as grid integration.
%
%   After finding MAP estimate or posterior samples of
%   hyperparameters, we can use them to make predictions for f_new:
%
%       p(f_new | y, th) = N(m, S),
%
%          m = K_nt*(K + I*s^2)^(-1)*y
%          S = K_new - K_nt*(K + I*s^2)^(-1)*K_tn
%   
%   where K_new is the covariance matrix of new f, and K_nt between
%   new f and training f.
%
%   For more detailed discussion of Gaussian process regression see,
%   for example, Rasmussen and Williams (2006) or Vanhatalo and
%   Vehtari (2008)
%
%   The demo is organised in three parts:
%     1) data analysis with MAP estimate for the hyperparameters
%     2) data analysis with grid integration over the hyperparameters
%     3) data analysis with MCMC integration over the hyperparameters
%
%   See also  DEMO_REGRESSION2
%
%   Refernces:
%    Rasmussen, C. E. and Williams, C. K. I. (2006). Gaussian
%    Processes for Machine Learning. The MIT Press.
%
%    Vanhatalo, J. and Vehtari, A. (2008). Modelling local and global
%    phenomena with sparse Gaussian processes. Proceedings of the 24th
%    Conference on Uncertainty in Artificial Intelligence,

% Copyright (c) 2008-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

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
gpcf1 = gpcf_sexp('init', 'lengthScale', [1.1 1.2], 'magnSigma2', 0.2^2)
gpcf2 = gpcf_noise('init', 'noiseSigma2', 0.2^2);

pl = prior_t('init');                          % a prior structure
pm = prior_t('init', 's2', 0.3);               % a prior structure
gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
gpcf2 = gpcf_noise('set', gpcf2, 'noiseSigma2_prior', pm);

gp = gp_init('init', 'FULL', 'gaussian', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001.^2);

% Demostrate how to evaluate covariance matrices. 
% K contains the covariance matrix without noise variance 
%  at the diagonal (the prior covariance)
% C contains the covariance matrix with noise variance at 
% the diagonal (the posterior covariance)
example_x = [-1 -1 ; 0 0 ; 1 1];
[K, C] = gp_trcov(gp, example_x)

% What has happend this far is the following
% - we created data structures 'gpcf1' and 'gpcf2', which describe 
%   the properties of the covariance function and Gaussian noise (see
%   gpcf_sexp and gpcf_noise for more details)
% - we created data structures that describe the prior of the length-scale 
%   and magnitude of the squared exponential covariance function and
%   the prior of the noise variance. These structures were set into
%   'gpcf1' and 'gpcf2' (see prior_t for more details)
% - we created a GP data structure 'gp', which has among others 'gpcf1' 
%   and 'gpcf2' data structures.  (see gp_init for more details)

% -----------------------------
% --- Conduct the inference ---
%
% We will make the inference first by finding a maximum a posterior
% estimate for the hyperparameters via gradient based
% optimization. After this we will use grid integration and Markov
% chain Monte Carlo sampling to integrate over the hyperparameters.
 

% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg for more details)

w=gp_pak(gp);  % pack the hyperparameters into one vector
fe=str2fun('gp_e');     % create a function handle to negative log posterior
fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior

% set the options for scg2
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do the optimization
w=scg2(fe, w, opt, fg, gp, x, y);

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w);

% NOTICE here that when the hyperparameters are packed into vector
% with 'gp_pak' they are also transformed through logarithm. The
% reason for this is that they are easier to optimize and sample with
% MCMC after log transformation.

% For last, make predictions of the underlying function on a dense
% grid and plot it. Below Ef_map is the predictive mean and Varf_map
% the predictive variance.
[p1,p2]=meshgrid(-1.8:0.1:1.8,-1.8:0.1:1.8);
p=[p1(:) p2(:)];
[Ef_map, Varf_map] = gp_pred(gp, x, y, p);

% Plot the prediction and data
figure(1)
mesh(p1, p2, reshape(Ef_map,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title('The predicted underlying function and the data points (MAP solution)');


% --- Grid integration ---
% Perform the grid integration and make predictions for p
[gp_array, P_TH, th, Ef_ia, Varf_ia, fx_ia, x_ia] = gp_ia(gp, x, y, p, 'int_method', 'grid');

% Plot the prediction for few input location
figure(2)
subplot(1,2,1)
plot(x_ia(100,:), fx_ia(100,:))
title('p(f|D) at input location (-1.6, 0.7)');
subplot(1,2,2)
plot(x_ia(400,:), fx_ia(400,:))
title('p(f|D) at input location (-0.8, 1.1)');


% --- MCMC ---
%  (see gp_mc for details)
% The hyperparameters are sampled with hybrid Monte Carlo 
% (see, for example, Neal (1996)). 

% The HMC sampling options are set to 'hmc_opt' structure, which is
% given to 'gp_mc' sampler
hmc_opt = hmc2_opt;
hmc_opt.steps=4;
hmc_opt.stepadj=0.05;
hmc_opt.persistence=0;
hmc_opt.decay=0.6;
hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

% Do the sampling (this takes approximately 5-10 minutes)
% 'rfull'   will contain a record structure with all the sampls
% 'g'       will contain a GP structure at the current state of the sampler
% 'rstate1' will contain a structure with information of the state of the sampler
[rfull,g,opt] = gp_mc(gp, x, y, 'nsamples', 400, 'repeat', 5, 'hmc_opt', hmc_opt);

% After sampling we delete the burn-in and thin the sample chain
rfull = thin(rfull, 10, 2);

% Now we make the predictions. 'mc_pred' is a function that returns 
% the predictive mean of the latent function with every sampled 
% hyperparameter value. Thus, the returned Ef_mc is a matrix of 
% size n x (number of samples). By taking the mean over the samples
% we do the Monte Carlo integration over the hyperparameters.
[Ef_mc, Varf_mc] = mc_pred(rfull, x, y, p);

figure(1)
clf, subplot(1,2,1)
mesh(p1, p2, reshape(Ef_map,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title(['The predicted underlying function ';
       'and the data points (MAP solution)']);
subplot(1,2,2)
mesh(p1, p2, reshape(mean(Ef_mc'),37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title(['The predicted underlying function  ';
       'and the data points (MCMC solution)']);
set(gcf,'pos',[93 511 1098 420])

% We can compare the posterior samples of the hyperparameters to the 
% MAP estimate that we got from optimization
figure(3)
clf, subplot(1,2,1)
plot(rfull.cf{1}.lengthScale)
title('The sample chain of length-scales')
subplot(1,2,2)
plot(rfull.cf{1}.magnSigma2)
title('The sample chain of magnitude')
set(gcf,'pos',[93 511 1098 420])

figure(4)
clf, subplot(1,4,1)
hist(rfull.cf{1}.lengthScale(:,1))
hold on
plot(gp.cf{1}.lengthScale(1), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 1')
subplot(1,4,2)
hist(rfull.cf{1}.lengthScale(:,2))
hold on
plot(gp.cf{1}.lengthScale(2), 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Length-scale 2')
subplot(1,4,3)
hist(rfull.cf{1}.magnSigma2)
hold on
plot(gp.cf{1}.magnSigma2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('magnitude')
subplot(1,4,4)
hist(rfull.noise{1}.noiseSigma2)
hold on
plot(gp.noise{1}.noiseSigma2, 0, 'rx', 'MarkerSize', 11, 'LineWidth', 2)
title('Noise variance')
legend('MCMC samples', 'MAP estimate')
set(gcf,'pos',[93 511 1098 420])


% Sample from two posterior marginals and plot them alongside 
% with the MAP and grid integration results
sf = normrnd(Ef_mc(100,:), sqrt(Varf_mc(100,:)));
sf2 = normrnd(Ef_mc(400,:), sqrt(Varf_mc(400,:)));

figure(2)
subplot(1,2,1)
[N,X] = hist(sf);
hist(sf)
hold on
plot(x_ia(100,:), max(N)/max(fx_ia(100,:))*fx_ia(100,:), 'k')
ff = normpdf(x_ia(100,:)', Ef_map(100), sqrt(Varf_map(100)));
plot(x_ia(100,:), max(N)/max(ff)*ff, 'r', 'lineWidth', 2)
set(gca, 'Ytick', [])
title('p(f|D) at input location (-1.6, 0.7)');
%xlim([0 1])

subplot(1,2,2)
[N,X] = hist(sf2);
hist(sf2)
hold on
plot(x_ia(400,:), max(N)/max(fx_ia(400,:))*fx_ia(400,:), 'k')
ff = normpdf(x_ia(400,:)', Ef_map(400), sqrt(Varf_map(400)));
plot(x_ia(400,:), max(N)/max(ff)*ff, 'r', 'lineWidth', 2)
set(gca, 'Ytick', [])
title('p(f|D) at input location (-0.8, 1.1)');
%xlim([-1.2 -0.5])




% ========================
% Print figures for manual
% ========================
% $$$ sf = normrnd(Ef_mc(100,:), sqrt(Varf_mc(100,:)));
% $$$ sf2 = normrnd(Ef_mc(400,:), sqrt(Varf_mc(400,:)));
% $$$ 
% $$$ figure
% $$$ subplot(1,2,1)
% $$$ [N,X] = hist(sf);
% $$$ hist(sf)
% $$$ h = findobj(gca,'Type','patch');
% $$$ set(h,'FaceColor','w','EdgeColor','k')
% $$$ hold on
% $$$ plot(x_ia(100,:), max(N)/max(fx_ia(100,:))*fx_ia(100,:), 'k')
% $$$ ff = normpdf(x_ia(100,:)', Ef_map(100), sqrt(Varf_map(100)));
% $$$ plot(x_ia(100,:), max(N)/max(ff)*ff, 'k', 'lineWidth', 2)
% $$$ set(gca, 'Ytick', [])
% $$$ xlim([0 1])
% $$$ ylim([0 110])
% $$$ 
% $$$ subplot(1,2,2)
% $$$ [N,X] = hist(sf2);
% $$$ hist(sf2)
% $$$ h = findobj(gca,'Type','patch');
% $$$ set(h,'FaceColor','w','EdgeColor','k')
% $$$ hold on
% $$$ plot(x_ia(400,:), max(N)/max(fx_ia(400,:))*fx_ia(400,:), 'k')
% $$$ ff = normpdf(x_ia(400,:)', Ef_map(400), sqrt(Varf_map(400)));
% $$$ plot(x_ia(400,:), max(N)/max(ff)*ff, 'k', 'lineWidth', 2)
% $$$ set(gca, 'Ytick', [])
% $$$ xlim([-1.2 -0.5])
% $$$ 
% $$$ 
% $$$ set(gcf,'units','centimeters');
% $$$ set(gcf,'pos',[15 14 7 5])
% $$$ set(gcf,'paperunits',get(gcf,'units'))
% $$$ set(gcf,'paperpos',get(gcf,'pos'))
% $$$ print -depsc2 /proj/bayes/jpvanhat/software/doc/GPstuffDoc/pics/demo_regression1_fig3.eps
% $$$ 
% $$$ 
% $$$ figure(4)
% $$$ clf, subplot(1,4,1)
% $$$ hist(rfull.cf{1}.lengthScale(:,1))
% $$$ h = findobj(gca,'Type','patch');
% $$$ set(h,'FaceColor','w','EdgeColor','k')
% $$$ hold on
% $$$ plot(gp.cf{1}.lengthScale(1), 0, 'kx', 'MarkerSize', 11, 'LineWidth', 2)
% $$$ xlabel('Length-s 1')
% $$$ xlim([0.3 1.6])
% $$$ 
% $$$ subplot(1,4,2)
% $$$ hist(rfull.cf{1}.lengthScale(:,2))
% $$$ h = findobj(gca,'Type','patch');
% $$$ set(h,'FaceColor','w','EdgeColor','k')
% $$$ hold on
% $$$ plot(gp.cf{1}.lengthScale(2), 0, 'kx', 'MarkerSize', 11, 'LineWidth', 2)
% $$$ xlabel('Length-s 2')
% $$$ xlim([0.4 1.4])
% $$$ 
% $$$ subplot(1,4,3)
% $$$ hist(rfull.cf{1}.magnSigma2)
% $$$ h = findobj(gca,'Type','patch');
% $$$ set(h,'FaceColor','w','EdgeColor','k')
% $$$ hold on
% $$$ plot(gp.cf{1}.magnSigma2, 0, 'kx', 'MarkerSize', 11, 'LineWidth', 2)
% $$$ xlabel('magnitude')
% $$$ xlim([0.5 6])
% $$$ 
% $$$ subplot(1,4,4)
% $$$ hist(rfull.noise{1}.noiseSigma2)
% $$$ h = findobj(gca,'Type','patch');
% $$$ set(h,'FaceColor','w','EdgeColor','k')
% $$$ hold on
% $$$ plot(gp.noise{1}.noiseSigma2, 0, 'kx', 'MarkerSize', 11, 'LineWidth', 2)
% $$$ xlabel('Noise variance')
% $$$ xlim([0.03 0.06])
% $$$ set(gca, 'Xtick', [0.03 0.06])
% $$$ 
% $$$ set(gcf,'units','centimeters');
% $$$ set(gcf,'pos',[15 14 11 5])
% $$$ set(gcf,'paperunits',get(gcf,'units'))
% $$$ set(gcf,'paperpos',get(gcf,'pos'))
% $$$ 
% $$$ print -depsc2 /proj/bayes/jpvanhat/software/doc/GPstuffDoc/pics/demo_regression1_fig2.eps
