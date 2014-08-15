%DEMO_SVI_CLASSIFIC  Classification problem demonstration for SVI GP
%
%  Description
%    The demonstration program is based on synthetic two class data
%    used by B.D. Ripley (Pattern Recognition and Neural Networks,
%    1996}. The data consists of 2-dimensional vectors that are
%    divided into two classes, labeled 0 or 1. Each class has a
%    bimodal distribution generated from equal mixtures of Gaussian
%    distributions with identical covariance matrices. A Bayesian
%    approach is used to find the decision line and predict the
%    classes of new data points. The result can be compared to the 
%    ones from the DEMO_CLASSIFIC.
%
%    The probability of y being one is assumed to be 
%
%      p(y=1|f) = normcdf(f)
%
%    The latent values f are given a zero mean Gaussian process
%    prior. This implies that at the observed input locations
%    latent values have prior
%
%      f ~ N(0, K),
%
%    where K is the covariance matrix, whose elements are given as
%    K_ij = k(x_i, x_j | th). The function k(x_i, x_j | th) is
%    covariance function and th its parameters.
% 
%    Here we demonstarte use of stochastic variational inference
%    methods to find the posterior of the latent values and parameters.
%    With these we can make predictions on the class probability of
%    future observations. See Hensman et. al. (2013) for the
%    detailed treatment.
%
%  See also
%    DEMO_SVI_REGRESSION, DEMO_CLASSIFIC
%
%
%  References:
%    Hensman, J., Fusi, N. and Lawrence, N. D. (2013). Gaussian
%    processes for big data. arXiv preprint arXiv:1309.6835.

% Copyright (c) 2014 Tuomas Sivula

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% This demonstration is based on the dataset used in the book Pattern
% Recognition and Neural Networks by B.D. Ripley (1996), Cambridge
% University Press.

% Training data
S = which('demo_classific');
L = strrep(S,'demo_classific.m','demodata/synth.tr');
x=load(L);
y=x(:,end);
y = 2.*y-1;
x(:,end)=[];
[n, nin] = size(x);

% Test data
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xt=[xt1(:) xt2(:)];

% Create likelihood function
lik = lik_probit();
%lik = lik_logit();

% Create covariance functions
gpcf = gpcf_sexp('lengthScale', [0.9 0.9], 'magnSigma2', 10);
% Set the prior for the parameters of covariance functions 
pl = prior_t();
pm = prior_sqrtt('s2',0.5);
gpcf = gpcf_sexp(gpcf, 'lengthScale_prior', pl,'magnSigma2_prior', pm);

% Create the GP structure (type is by default FULL)
fprintf('SVI GP classification model with probit likelihood\n')
gp = gp_set('lik', lik, 'cf', gpcf, ...
  'latent_method', 'SVI', 'jitterSigma2', 1e-6);

% Select 20 inducing inputs by clustering 10 from both training class
% inputs
fprintf(['Select 20 inducing inputs by clustering 10 points ', ...
  'from both training classes ...'])
Sw = warning('off','stats:kmeans:EmptyCluster');
[~,X_u1] = kmeans(x(y==1,:), 10,'Start','uniform',...
    'EmptyAction','singleton');
[~,X_u2] = kmeans(x(y==-1,:), 10,'Start','uniform',...
    'EmptyAction','singleton');
warning(Sw);
X_u = [X_u1 ; X_u2];
fprintf(' done\n')

% Optimise
maxi = 1000; % The maximum number of iteration rounds
gp = svigp(gp,x,y,'X_u',X_u,'maxiter',maxi,'mu2',1e-7);
% Make predictions
[Eft, Varft, lpyt, Eyt, Varyt] = ...
    gpsvi_pred(gp, x, y, xt, 'yt', ones(size(xt,1),1) );

% Visualise predictive probability p(ystar = 1) with grayscale
figure, hold on;
n_pred=size(xt,1);
h1=pcolor(reshape(xt(:,1),20,20),reshape(xt(:,2),20,20),reshape(exp(lpyt),20,20));
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), %axis off
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
plot(gp.X_u(:,1),gp.X_u(:,2),'gs', 'markersize', 8, 'linewidth', 1);
set(gcf, 'color', 'w'), title('predictive probability, training cases and inducing inputs with SVIGP', 'fontsize', 14)


