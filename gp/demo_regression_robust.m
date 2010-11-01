%DEMO_ROBUSTREGRESSION  A regression demo with Student-t distribution as a residual model.
%
%  Description
%    The synthetic data used here is the same used by Radford M. 
%    Neal in his regression problem with outliers example in
%    Software for Flexible Bayesian Modeling
%    (http://www.cs.toronto.edu/~radford/fbm.software.html). The
%    problem consist of one dimensional input and target variables. 
%    The input data, x, is sampled from standard Gaussian
%    distribution and the corresponding target values come from a
%    distribution with mean given by
%
%           f = 0.3 + 0.4x + 0.5sin(2.7x) + 1.1/(1+x^2).
%
%    For most of the cases the distribution about this mean is
%    Gaussian with standard deviation of 0.1, but with probability
%    0.05 a case is an outlier for wchich the standard deviation is
%    1.0. There are total 200 cases from which the first 100 are
%    used for training and the last 100 for testing.
%
%    We use Student-t distribution as an abservation model
%
%          y ~ St(f, nu, s^2),
%
%    where f is the mean, nu the degrees of freedom and s^2 the
%    scale. The mean is given a GP prior
%
%          f ~ N(0, K).
%
%    The model can be inferred with MCMC or Laplace approximation. 
%    The MCMC can be performed either by utilizing the scale
%    mixture representation of the Student-t distribution or the
%    actual distribution. The scale mixture representation is given
%    as in Gelman et.al (2004)
%
%          y_i ~ N (f_i, a^2*U_i)
%          U_i ~ Inv-Chi^2 (nu, t^2),
%
%    where nu represents the degrees of freedom and a*t = s in the
%    Student-t distribution.
%
%    The demo is organized as follows:
%
%     1) Optimization approach with Normal noise
%     2) MCMC approach with scale mixture noise model (~=Student-t)
%        All parameters sampled
%     3) Laplace approximation Student-t likelihood
%        All parameters optimized
%     4) MCMC approach with Student-t likelihood nu kept fixed to 4
%     5) Laplace approximation Student-t likelihood
%        nu kept fixed to 4
%
%  See Vanhatalo et.al. for discussion on the model and methods.
%
%  Refernces:
%    Vanhatalo, J., Jyl�nki P. and Vehtari, A. (2009). Gaussian
%    process regression with Student-t likelihood. Advances in
%    Neural Information Processing systems
%
%    Gelman, Carlin, Stern and Rubin (2004) Bayesian Data Analysis,
%    second edition. Chapman & Hall / CRC.
%

% Copyright (c) 2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% ========================================
% Optimization approach with Normal noise
% ========================================

% load the data. First 100 variables are for training
% and last 100 for test
S = which('demo_regression_robust');
L = strrep(S,'demo_regression_robust.m','demos/odata.txt');
x = load(L);
y = x(1:100,2);
x = x(1:100,1);
[n, nin] = size(x); 

% Test data
xt = [-2.7:0.01:2.7]';
yt = 0.3+0.4*xt+0.5*sin(2.7*xt)+1.1./(1+xt.^2);

% We create a Gaussian process and priors for GP parameters. Prior for GP
% parameters is Gaussian multivariate hierarchical. The residual is given at
% first Gaussian prior to find good starting value for noiseSigmas..

% Construct the priors for the parameters of covariance functions...
pl = prior_t();
pm = prior_sqrtt('s2', 0.3);
pn = prior_logunif();

% create the Gaussian process
gpcf1 = gpcf_sexp('lengthScale', 1, 'magnSigma2', 0.2^2, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
gpcf2 = gpcf_noise('noiseSigma2', 0.2^2, 'noiseSigma2_prior', pn);

% ... Finally create the GP data structure
gp = gp_set('cf', {gpcf1}, 'noisef', {gpcf2}, 'jitterSigma2', 1e-6)    

% --- MAP estimate using scaled conjugate gradient algorithm ---
% Set the options for the scaled conjugate optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter');
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'optimf',@fminscg,'opt',opt);

% Prediction
[Eft, Varft, Eyt, Varyt] = gp_pred(gp, x, y, xt);
std_ft = sqrt(Varft);

% Plot the prediction and data
% plot the training data with dots and the underlying 
% mean of it as a line
figure
hold on
h1=plot(xt,yt, 'k');
h2=plot(xt, Eft, xt, Eft-2*std_ft, 'r--', xt, Eft+2*std_ft, 'r--');
h3=plot(x,y,'b.');
%plot(xt,yt,'r.')
legend([h1 h2(1) h2(3) h3],'real f', 'Ef', 'Ef+std(f)','y',4)
axis on;
title('The predictions and the data points (MAP solution and normal noise)');
S1 = sprintf('length-scale: %.3f, magnSigma2: %.3f  \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

% ========================================
% MCMC approach with scale mixture noise model (~=Student-t)
% Here we sample all the variables 
%     (lenghtScale, magnSigma, sigma(noise-t) and nu)
% ========================================
gpcf1 = gpcf_sexp('lengthScale', 1, 'magnSigma2', 0.2^2);
gpcf1 = gpcf_sexp(gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
% Here, set own Sigma2 for every data point
gpcf2 = gpcf_noiset('ndata', n, 'noiseSigmas2', repmat(1^2,n,1));

% Free nu
gpcf2 = gpcf_noiset(gpcf2, 'nu_prior', []);

gp = gp_set('cf', {gpcf1}, 'noisef', {gpcf2}, 'jitterSigma2', 1e-3)

hmc_opt.steps=10;
hmc_opt.stepadj=0.08;
hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));
hmc_opt.persistence=1;
hmc_opt.decay=0.6;

gibbs_opt = sls1mm_opt;
gibbs_opt.maxiter = 50;
gibbs_opt.mmlimits = [0 40];
gibbs_opt.method = 'minmax';

% Sample 
[r,g,opt]=gp_mc(gp, x, y, 'nsamples', 300, 'hmc_opt', hmc_opt, 'gibbs_opt', gibbs_opt);

% thin the record
rr = thin(r,100,2);

figure 
subplot(2,2,1)
hist(rr.noisef{1}.nu,20)
title('Mixture model, \nu')
subplot(2,2,2)
hist(sqrt(rr.noisef{1}.tau2).*rr.noisef{1}.alpha,20)
title('Mixture model, \sigma')
subplot(2,2,3) 
hist(rr.cf{1}.lengthScale,20)
title('Mixture model, length-scale')
subplot(2,2,4) 
hist(rr.cf{1}.magnSigma2,20)
title('Mixture model, magnSigma2')

% make predictions for test set
[Eft_mc, Varft_mc] = mc_preds(rr,x,y,xt);
Eft = mean(Eft_mc,2);
std_ft = sqrt(mean(Varft_mc,2) + var(Eft_mc,0,2));

% Plot the network outputs as '.', and underlying mean with '--'
figure
hold on
h1=plot(xt,yt, 'k');
h2=plot(xt, Eft, xt, Eft-2*std_ft, 'r--', xt, Eft+2*std_ft, 'r--');
h3=plot(x,y,'b.');
%plot(xt,yt,'r.')
legend([h1 h2(1) h2(3) h3],'real f', 'Ef', 'Ef+std(f)','y',4)
axis on;
title('The predictions and the data points (MCMC solution and scale mixture noise)')
S2 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

% ========================================
% Laplace approximation Student-t likelihood
%  Here we optimize all the variables 
%  (lengthScale, magnSigma2, sigma(noise-t) and nu)
% ========================================

pl = prior_t();
pm = prior_sqrtt('s2', 0.3);
gpcf1 = gpcf_sexp('lengthScale', 1, 'magnSigma2', 0.2);
gpcf1 = gpcf_sexp(gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
pll = prior_logunif();
lik = lik_t('nu', 4, 'sigma2', 20, 'sigma2_prior', pll, 'nu_prior', []);

% ... Finally create the GP data structure
gp = gp_set('lik', lik, 'cf', {gpcf1}, 'jitterSigma2', 1e-3, ...
            'infer_params', 'covariance+likelihood', ...
            'latent_method', 'Laplace');

% Set the options for the scaled conjugate optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter','Maxiter',20);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'optimf',@fminscg,'opt',opt);

% Predictions to test points
[Eft, Varft] = gp_pred(gp, x, y, xt);
std_ft = sqrt(Varft);

% Plot the prediction and data
figure
h1=plot(xt,yt, 'k');
h2=plot(xt, Eft, xt, Eft-2*std_ft, 'r--', xt, Eft+2*std_ft, 'r--');
h3=plot(x,y,'b.');
%plot(xt,yt,'r.')
legend([h1 h2(1) h2(3) h3],'real f', 'Ef', 'Ef+std(f)','y',4)
axis on;
title(sprintf('The predictions and the data points (MAP solution, Student-t (nu=%.2f,sigma2=%.3f) noise)',gp.lik.nu, gp.lik.sigma2));
S3 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

% ========================================
% MCMC approach with Student-t likelihood
%  Here we analyse the model with fixed degrees of freedom
%   nu = 4 
%   Notice that the default value for freeze_nu = 1, 
%   which means that degrees of freedom is not sampled/optimized
% ========================================
pl = prior_t();
pm = prior_sqrtt('s2', 0.3);
gpcf1 = gpcf_sexp('lengthScale', 1, 'magnSigma2', 0.2^2);
gpcf1 = gpcf_sexp(gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
pll = prior_logunif();
lik = lik_t('nu', 4, 'sigma2', 0.5^2, 'sigma2_prior', pll, 'nu_prior', []);

% ... Finally create the GP data structure
gp = gp_set('lik', lik, 'cf', {gpcf1}, 'jitterSigma2', 1e-4, ...
             'latent_method', 'MCMC', ...
             'infer_params' , 'covariance+likelihood');

% Set the parameters for MCMC...
clear opt
opt.hmc_opt = hmc2_opt;
opt.hmc_opt.steps=5;
opt.hmc_opt.stepadj=0.02;
opt.hmc_opt.nsamples=1;

% Latent-options
opt.latent_opt = hmc2_opt;
opt.latent_opt.display=0;
opt.latent_opt.repeat = 10
opt.latent_opt.sample_latent_scale = 0.05

% Likelihood-option
opt.lik_hmc_opt = hmc2_opt;
opt.lik_hmc_opt.steps=10;
opt.lik_hmc_opt.stepadj=0.1;
opt.lik_hmc_opt.nsamples=1;

% Sample 
[rgp,g,opt]=gp_mc(gp, x, y, 'nsamples', 400, opt);
rr = thin(rgp,100,2);

% make predictions for test set
[Ef_mc, Varf_mc] = mc_preds(rr,x,y,xx');
Ef = mean(Ef_mc,2);
std_f = sqrt( var(Ef_mc,0,2) );

% Plot the network outputs as '.', and underlying mean with '--'
figure
plot(xx,yy,'k')
hold on
plot(xx,Ef)
plot(xx, Ef-2*std_f, 'r--')
plot(x,y,'.')
legend('real f', 'Ef', 'Ef+std(f)','y')
plot(xx, Ef+2*std_f, 'r--')
title('The predictions and the data points (MCMC solution Student-t noise, \nu fixed)')
S2 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

% ========================================
% Laplace approximation Student-t likelihood
%  Here we analyse the model with fixed degrees of freedom
%   nu = 4 
%   Notice that the default value for fix_nu = 1, 
%   which means that degrees of freedom is not sampled/optimized
% ========================================

gpcf1 = gpcf_sexp('lengthScale', 2, 'magnSigma2', 1);

% Create the likelihood structure
pll = prior_logunif();
lik = lik_t('nu', 4, 'sigma2', 1, 'sigma2_prior', pll);

% ... Finally create the GP data structure
gp = gp_set('lik', lik, 'cf', {gpcf1}, 'jitterSigma2', 1e-6, ...
            'latent_method', 'Laplace');

% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg for more details)

w=gp_pak(gp);  % pack the hyperparameters into one vector
fe=@gpla_e;     % create a function handle to negative log posterior
fg=@gpla_g;     % create a function handle to gradient of negative log posterior

n=length(y);
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do scaled conjugate gradient optimization 
w=gp_pak(gp);
w=scg2(fe, w, opt, fg, gp, x, y);
gp =gp_unpak(gp,w);

% Predictions to test points
[Ef, Varf] = la_pred(gp, x, y, xx');
std_f = sqrt(Varf);

% Plot the prediction and data
figure
plot(xx,yy,'k')
hold on
plot(xx,Ef)
plot(xx, Ef-2*std_f, 'r--')
plot(x,y,'.')
legend('real f', 'Ef', 'Ef+std(f)','y')
plot(xx, Ef+2*std_f, 'r--')
title(sprintf('The predictions and the data points (MAP solution, Student-t , \nu fixed (nu=%.2f,sigma=%.3f) noise)',gp.lik.nu, sqrt(gp.lik.sigma2)));
S4 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)











































%%

gpcf1 = gpcf_sexp('lengthScale', 2, 'magnSigma2', 1);

% Create the likelihood structure
pll = prior_logunif();
lik = lik_t('nu', 4, 'sigma2', 1, 'sigma2_prior', pll);

% ... Finally create the GP data structure
gp = gp_init('FULL', lik, {gpcf1}, {}, 'jitterSigma2', 0.001.^2);
gp = gp_init(gp, 'latent_method', {'EP', x, y});

% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg for more details)

gradcheck(gp_pak(gp), @gpep_e, @gpep_g, gp, x, y)
gpep_g(w,gp,x,y)

w = [1.5981    0.4444   -4.6361];

[e, edata, eprior, site_tau, site_nu, L]=gpep_e(w,gp,x,y);
Sigm = L'*L;
myy = Sigm*site_nu;
plot(x,myy,'b.',x,y,'ro')

%%

min(gp.site_tau)
gp = gp_unpak(gp,w);
[Ef, Varf] = ep_pred(gp, x, y, xx');
std_f = sqrt(Varf);

% Plot the prediction and data
figure
plot(xx,yy,'k')
hold on
plot(xx,Ef)
plot(xx, Ef-2*std_f, 'r--')
plot(x,y,'.')
legend('real f', 'Ef', 'Ef+std(f)','y')
plot(xx, Ef+2*std_f, 'r--')
title(sprintf('The predictions and the data points (MAP solution, Student-t (nu=%.2f,sigma=%.3f) noise)',gp.lik.nu, sqrt(gp.lik.sigma2)));
S4 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

%%
w=gp_pak(gp);  % pack the hyperparameters into one vector
fe=@gpep_e;     % create a function handle to negative log posterior
fg=@gpep_g;     % create a function handle to gradient of negative log posterior

n=length(y);
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do scaled conjugate gradient optimization 
w=gp_pak(gp);
w=scg2(fe, w, opt, fg, gp, x, y);
gp =gp_unpak(gp,w);

% Predictions to test points
[Ef, Varf] = ep_pred(gp, x, y, xx');
std_f = sqrt(Varf);

% Plot the prediction and data
figure
plot(xx,yy,'k')
hold on
plot(xx,Ef)
plot(xx, Ef-2*std_f, 'r--')
plot(x,y,'.')
legend('real f', 'Ef', 'Ef+std(f)','y')
plot(xx, Ef+2*std_f, 'r--')
title(sprintf('The predictions and the data points (MAP solution, Student-t (nu=%.2f,sigma=%.3f) noise)',gp.lik.nu, sqrt(gp.lik.sigma2)));
S4 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)
