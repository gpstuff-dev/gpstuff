%DEMO_BINOMIAL    Demonstration of Gaussian process model with binomial
%                 likelihood
%
%      Description
%      Demonstration of estimating the unknown population proportion
%      in binomial model from a sequence of success/failure
%      trials. Data consists of observations Y describing the number
%      of successes in a sequence of N iid trials, and of explanatory
%      variables X.  The binomial model is
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
%      The elements of the covariance matrix K are given as K_ij =
%      k(x_i, x_j | th). The function k(x_i, x_j | th) is covariance
%      function and th its hyperparameters.  We place a hyperprior for
%      hyperparameters, p(th). The inference is done with Laplace
%      approximation.
%
%      NOTE! In the prediction, the total number of trials Nt at the
%      test points Xt must be set additionally in the likelihood
%      structure when E(Yt), Var(Yt) or predictive densities p(Yt) are
%      computed.

% Copyright (c) 2010 Jaakko Riihimäki

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


%========================================================
% data analysis with full GP model
%========================================================


%- generate some toy data
x=rand(450,1)*2-1; f = -1.5.*x.^3+0.5*x.^2+0.75*x;
% number of trials
N=ones(size(x,1),1)*100;
% number of successes
y=binornd(N, 1./(1+exp(-f)));

% create test and training data
nt=400;
xt=x(1:nt,:); x(1:nt,:)=[];
yt=y(1:nt,:); y(1:nt,:)=[];
Nt=N(1:nt,:); N(1:nt,:)=[];

% equally spaced test points for visualisation
xgrid=linspace(min(x(:,1))-0.3,max(x(:,1))+0.3,100)';
Ntgrid=ones(size(xgrid))*100;
%-


[n, nin] = size(x);

% Create covariance functions
gpcf1 = gpcf_sexp('init', 'lengthScale', ones(1,nin), 'magnSigma2', 1);

% Set the prior for the parameters of covariance functions 
pn = prior_normal('init');
ps2 = prior_sinvchi2('init', 's2', 2.7^2, 'nu', 0.2);
ppn = prior_normal('init', 'mu', 6, 's2', 9, 'mu_prior', pn, 's2_prior', ps2);

gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', ppn, 'magnSigma2_prior', ps2);

% Create the likelihood structure
likelih = likelih_binomial('init');

% Create the GP data structure
gp = gp_init('init', 'FULL', likelih, {gpcf1}, [], 'jitterSigma2', 1e-3, 'infer_params', 'covariance');


% ------- Laplace approximation --------

% Set the approximate inference method
gp = gp_init('set', gp, 'latent_method', {'Laplace', x, y, 'z', N});

fe=str2fun('gpla_e');
fg=str2fun('gpla_g');

% set the options for scaled conjugate optimazation
opt_scg = scg2_opt;
opt_scg.tolfun = 1e-3;
opt_scg.tolx = 1e-3;
opt_scg.display = 1;

% do scaled conjugate gradient optimization 
w=gp_pak(gp);
[wopt, opt, flog]=scg2(fe, w, opt_scg, fg, gp, x, y, 'z', N);
gp=gp_unpak(gp, wopt);

% Make predictions at the grid points

% Set the total number of trials Nt at the grid points xgrid
[Ef_la, Varf_la, Ey_la, Vary_la] = la_pred(gp, x, y, xgrid, 'z', N, 'zt', Ntgrid);

% Visalise the predictions
figure, set(gcf, 'color', 'w'), hold on
color1=ones(1,3)*0.8; color2=ones(1,3)*0.5;

% GP 95% credible interval
h1=fill([xgrid' fliplr(xgrid')], [(Ey_la+1.96*sqrt(Vary_la))' fliplr((Ey_la-1.96*sqrt(Vary_la))')], color1, 'edgecolor', color1);
% GP mean
h2=plot(xgrid, Ey_la, 'color', color2, 'linewidth', 3);
% observations
h3=plot(x, y, 'xk', 'markersize', 10, 'linewidth', 2);
% true function
h4=plot(xgrid, 1./(1+exp(-(-1.5.*xgrid.^3+0.5*xgrid.^2+0.75*xgrid)))*100, 'color', 'r', 'linewidth', 2);
legend([h1 h2 h3 h4], 'GP 95% CI', 'GP mean', 'observations', 'true latent function')
title('Gaussian process prediction with a squared exponential covariance function')


% To compute predictive densities at the test points xt, the total number
% of trials Nt must be set additionally:
[Ef_la, Varf_la, Ey_la, Vary_la, py_la] = la_pred(gp, x, y, xt, 'z', N, 'yt', yt, 'zt', Nt);

figure, set(gcf, 'color', 'w'), hold on
hist(log(py_la), 20)
title('histogram of log-predictive densities at the test points')

figure, set(gcf, 'color', 'w'), hold on
plot([min(yt) max(yt)], [min(yt) max(yt)], 'r', 'linewidth', 2)
plot(yt, Ey_la, '.k', 'markersize', 15)
axis equal
xlabel('observed y')
ylabel('predicted E[y]')
title('Observations versus predictions E[y]')


