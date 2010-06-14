%DEMO_PPCSCOV    Regression problem demonstration for 2-input 
%                function with Gaussian process using CS covariance
%
%    Description
%    We will analyze a US annual precipitation data from year 1995,
%    which contains XX data points. The GP constructed utilizes
%    compactly supported covariance function gpcf_ppcs2, for which
%    reason theinference is lot faster than with globally supported
%    covariance function (such as gpcf_sexp). The full data is
%    available at http://www.image.ucar.edu/Data/
% 
%    The regression problem consist of a data with two input variables
%    and output variable contaminated with Gaussian noise. The model
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


% Load the data and take only the stations that have all the measurements
% present. Sum up the monthly precipitation figures to get the annual
% precipitation

S = which('demo_ppcsCov');
L = strrep(S,'demo_ppcsCov.m','demos/USprec1');
prec = load(L);

S = which('demo_ppcsCov');
L = strrep(S,'demo_ppcsCov.m','demos/USprec2');
stats = load(L);

y = sum(prec(prec(:,14)==0,2:13),2);
y = y/100;
avgy = mean(y);
y = y-avgy;
x = stats(prec(:,14)==0,2:3);
clear prec
clear stats

[n,nin] = size(x);
x = x-repmat(min(x),n,1) + 1;   % Note! Here we just move the input space

% Construct a meshgrid for surface testing
[X1,X2]=meshgrid(0:0.5:58,0:0.5:26);
xx = [X1(:) X2(:)];
dist = sqrt(bsxfun(@minus,xx(:,1),x(:,1)').^2 + bsxfun(@minus,xx(:,2),x(:,2)').^2);
ind = find(min(dist,[],2)<=1);
xx = xx(ind,:);

% Plot the prediction inputs
figure(1)
plot(xx(:,1),xx(:,2),'k.')
title('Inputs where to predict')

% Create covariance function
pl2 = prior_gamma('init', 'sh', 5, 'is', 1);
pm2 = prior_t('init', 'nu', 1, 's2', 150);
gpcf2 = gpcf_ppcs2('init', 'nin', nin, 'lengthScale', [1 2], 'magnSigma2', 3);
gpcf2 = gpcf_ppcs2('set', gpcf2, 'lengthScale_prior', pl2, 'magnSigma2_prior', pm2);

pn = prior_t('init', 'nu', 4, 's2', 0.3);
gpcfn = gpcf_noise('init', 'noiseSigma2', 1, 'noiseSigma2_prior', pn);


% MAP ESTIMATE
% ============================================
gp = gp_init('init', 'FULL', 'regr', {gpcf2}, {gpcfn}, 'jitterSigma2', 0.001.^2);

% Optimize the hyperparameters
% ---------------------------------
w=gp_pak(gp);  
fe=str2fun('gp_e');     
fg=str2fun('gp_g');     

% set the options for scg2
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do the optimization
w=scg2(fe, w, opt, fg, gp, x, y);
gp = gp_unpak(gp,w);

% Evaluate the sparsity of the covariance function
K = gp_trcov(gp,x);
nnz(K) / prod(size(K))

figure(2)
p = amd(K);
spy(K(p,p), 'k')


% plot figure
% ------------------------------------
Ef = gp_pred(gp, x, y, xx);
figure(3)
G=repmat(NaN,size(X1));
G(ind)=(Ef + avgy)*100;
pcolor(X1,X2,G),shading flat
axis equal
xlim([0 60])
ylim([0 28])








% =========================
% Print the figures for manual
% =========================
% $$$ set(gca,'YTick', [])
% $$$ set(gca,'XTick', [])
% $$$ xlabel('')
% $$$ set(gcf,'units','centimeters');
% $$$ set(gcf,'pos',[15 14 4 4])
% $$$ set(gcf,'paperunits',get(gcf,'units'))
% $$$ set(gcf,'paperpos',get(gcf,'pos'))
% $$$ 
% $$$ print -depsc2 /proj/bayes/jpvanhat/software/doc/GPstuffDoc/pics/demo_ppcsCov1
% $$$ 
% $$$ set(gca,'YTick', [])
% $$$ set(gca,'XTick', [])
% $$$ colormap(mapcolor(G)), colorbar
% $$$ %title('FULL GP with gpcf_ppcs2')
% $$$ 
% $$$ set(gcf,'units','centimeters');
% $$$ set(gcf,'pos',[15 14 10 4])
% $$$ set(gcf,'paperunits',get(gcf,'units'))
% $$$ set(gcf,'paperpos',get(gcf,'pos'))
% $$$ print -depsc2 /proj/bayes/jpvanhat/software/doc/GPstuffDoc/pics/demo_ppcsCov2.eps








% Grid integration
% ===================================
% Perform the grid integration and make predictions for p
[gp_array, P_TH, th, Ef_ia, Varf_ia, fx_ia, x_ia] = gp_ia(gp, x, y, xx, 'int_method', 'CCD');

% Plot the prediction for few input location
figure(2)
subplot(1,2,1)
plot(x_ia(100,:), fx_ia(100,:))
title( sprintf('p(f|D) at input location (%.1f,%.1f)', xx(100,1), xx(100,2)) );
subplot(1,2,2)
plot(x_ia(400,:), fx_ia(400,:))
title( sprintf('p(f|D) at input location (%.1f,%.1f)', xx(400,1), xx(400,2)) );


% --- MCMC ---
%  (see gp_mc for details)
% The hyperparameters are sampled with hybrid Monte Carlo 
% (see, for example, Neal (1996)). 

% The HMC sampling options are set to 'hmc_opt' structure, which is
% given to 'gp_mc' sampler
hmc_opt = hmc2_opt;
hmc_opt.steps=2;
hmc_opt.stepadj=0.02;
hmc_opt.persistence=0;
hmc_opt.decay=0.6;
hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

% Do the sampling (this takes approximately 5-10 minutes)
% 'rfull'   will contain a record structure with all the sampls
% 'g'       will contain a GP structure at the current state of the sampler
% 'rstate1' will contain a structure with information of the state of the sampler
[rfull,g,opt] = gp_mc(gp, x, y, 'nsamples', 300, 'repeat', 2, 'hmc_opt', hmc_opt);

% After sampling we delete the burn-in and thin the sample chain
rfull = thin(rfull, 10, 2);

% Now we make the predictions. 'mc_pred' is a function that returns 
% the predictive mean of the latent function with every sampled 
% hyperparameter value. Thus, the returned Ef_mc is a matrix of 
% size n x (number of samples). By taking the mean over the samples
% we do the Monte Carlo integration over the hyperparameters.
Ef_mc = mc_pred(rfull, x, y, xx);

figure
G=repmat(NaN,size(X1));
G(ind)=(Ef + avgy)*100;
pcolor(X1,X2,G),shading flat
axis equal
xlim([0 60])
ylim([0 28])
title(['The predicted underlying function ';
       'and the data points (MAP solution)']);
figure
G=repmat(NaN,size(X1));
G(ind)=(Ef_mc + avgy)*100;
pcolor(X1,X2,G),shading flat
axis equal
xlim([0 60])
ylim([0 28])
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













