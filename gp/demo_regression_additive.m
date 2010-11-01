%DEMO_REGRESSION_ADDITIVE  Regression demonstration with additive Gaussian
%                          process using linear, squared exponential and
%                          neural network covariance fucntions 
%
%  Description
%    Gaussian process solutions in 2D regression problem using constant, 
%    linear, squared exponential (sexp) and neural network covariance
%    functions, and with various additive combinations of these four
%    covariance functions. The noisy observations y are assumed to satisfy
%
%         y = f + e,    where e ~ N(0, s^2)
%
%    where f is an unknown underlying function. A zero mean Gaussian
%    process prior is assumed for f
%
%         f ~ N(0, K),
%
%    where K is the covariance matrix whose elements are given by one of
%    the following six covariance function:
%    
%    - constant + linear
%    - costant + sexp for 1. input + linear for 2. input
%    - sexp for 1. input + sexp for 2. input
%    - sexp
%    - neural network for 1. input + neural network for 2. input
%    - neural network
%
%    A hyperprior is assumed for hyperparameters of the covariance
%    functions, and the inference is done with a MAP estimate for
%    hyperparameter values.
%
%    For more detailed discussion of  covariance functions, see e.g.
%
%    Rasmussen, C. E. and Williams, C. K. I. (2006). Gaussian
%    Processes for Machine Learning. The MIT Press.
%
%
%  See also  DEMO_REGRESSION1
%
% Copyright (c) 2010 Jaakko Riihimäki

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


% REGRESSION TOY DATA
% Create an example regression data
x=rand(225,2)*2-1;
y=3*normcdf(2*x(:,2))+2*normcdf(4*x(:,1));
% add some noise
y=y+randn(size(y))*0.25;
y=y-mean(y);
[n, nin] = size(x);
% create equally spaced points to visualise the predictions:
[xt1,xt2]=meshgrid(-2:0.1:2,-2:0.1:2);
xt=[xt1(:) xt2(:)];
nxt=size(xt1,1);

% Assume a Student-t distribution for the GP hyperparameters
pt = prior_t('nu', 4, 's2', 10);	% a prior structure

% Create a Gaussian noise model
gpcf_n = gpcf_noise('noiseSigma2', 0.2^2, 'noiseSigma2', 0.2^2);

% Set a small amount of jitter 
jitter=1e-4;

% Set the options for the scaled conjugate optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter');

% CONSTANT + LINEAR COVARIANCE FUNCTION
% constant covariance function
gpcf_c = gpcf_constant('constSigma2', 1, 'constSigma2_prior', pt);
% linear covariance function
gpcf_l = gpcf_linear('coeffSigma2_prior', pt);
gp = gp_set('cf', {gpcf_c gpcf_l}, 'noisef', {gpcf_n}, 'jitterSigma2', jitter);

% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'optimf',@fminscg,'opt',opt);

% Compute predictions in a grid using the MAP estimate
Eft_map = gp_pred(gp, x, y, xt);

% Plot the prediction and data
figure, set(gcf, 'color', 'w')
mesh(xt1, xt2, reshape(Eft_map,nxt,nxt));
hold on
plot3(x(:,1), x(:,2), y, '*')
xlabel('x_1'), ylabel('x_2')
title('The predicted underlying function (constant + linear)');

% CONSTANT + SQUARED EXPONENTIAL COVARIANCE FUNCTION (W.R.T. THE
% FIRST INPUT DIMENSION) + LINEAR (W.R.T. THE SECOND INPUT
% DIMENSION)

% Metric function for the first input variable
metric1 = metric_euclidean('components', {[1]}, 'lengthScale',[0.5], 'lengthScale_prior', pt);
% Covariance function for the first input variable
gpcf_s1 = gpcf_sexp('magnSigma2', 0.15, 'magnSigma2_prior', pt, 'metric', metric1);
gpcf_l2 = gpcf_linear('selectedVariables', [2], 'coeffSigma2_prior', pt);
gp = gp_set('cf', {gpcf_c gpcf_s1 gpcf_l2}, 'noisef', {gpcf_n}, 'jitterSigma2', jitter);

% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'optimf',@fminscg,'opt',opt);

% Compute predictions in a grid using the MAP estimate
Eft_map = gp_pred(gp, x, y, xt);

% Plot the prediction and data
figure, set(gcf, 'color', 'w')
mesh(xt1, xt2, reshape(Eft_map,nxt,nxt));
hold on
plot3(x(:,1), x(:,2), y, '*')
xlabel('x_1'), ylabel('x_2')
title('The predicted underlying function (sexp for 1. input + linear for 2. input )');

% ADDITIVE SQUARED EXPONENTIAL COVARIANCE FUNCTION

% Metric function for the second input variable
metric2 = metric_euclidean('components', {[2]},'lengthScale',[0.5], 'lengthScale_prior', pt);
% Covariance function for the second input variable
gpcf_s2 = gpcf_sexp('magnSigma2', 0.15, 'magnSigma2_prior', pt, 'metric', metric2);
gp = gp_set('cf', {gpcf_s1,gpcf_s2}, 'noisef', {gpcf_n}, 'jitterSigma2', jitter);

% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'optimf',@fminscg,'opt',opt);

% Compute predictions in a grid using the MAP estimate
Eft_map = gp_pred(gp, x, y, xt);

% Plot the prediction and data
figure, set(gcf, 'color', 'w')
mesh(xt1, xt2, reshape(Eft_map,nxt,nxt));
hold on
plot3(x(:,1), x(:,2), y, '*')
xlabel('x_1'), ylabel('x_2')
title('The predicted underlying function (additive sexp)');

% SQUARED EXPONENTIAL COVARIANCE FUNCTION
gpcf_s = gpcf_sexp('lengthScale', ones(1,nin), 'magnSigma2', 0.2^2, ...
                   'lengthScale_prior', pt, 'magnSigma2_prior', pt);
gp = gp_set('cf', {gpcf_s}, 'noisef', {gpcf_n}, 'jitterSigma2', jitter);

% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'optimf',@fminscg,'opt',opt);

% Compute predictions in a grid using the MAP estimate
Eft_map = gp_pred(gp, x, y, xt);

% Plot the prediction and data
figure, set(gcf, 'color', 'w')
mesh(xt1, xt2, reshape(Eft_map,nxt,nxt));
hold on
plot3(x(:,1), x(:,2), y, '*')
xlabel('x_1'), ylabel('x_2')
title('The predicted underlying function (sexp)');

% ADDITIVE NEURAL NETWORK COVARIANCE FUNCTION
gpcf_nn1 = gpcf_neuralnetwork('weightSigma2', 1, 'biasSigma2', 1, 'selectedVariables', [1], ...
                              'weightSigma2_prior', pt, 'biasSigma2_prior', pt);
gpcf_nn2 = gpcf_neuralnetwork('weightSigma2', 1, 'biasSigma2', 1, 'selectedVariables', [2], ...
                              'weightSigma2_prior', pt, 'biasSigma2_prior', pt);
gp = gp_set('cf', {gpcf_nn1,gpcf_nn2}, 'noisef', {gpcf_n}, 'jitterSigma2', jitter);

% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'optimf',@fminscg,'opt',opt);

% Compute predictions in a grid using the MAP estimate
Eft_map = gp_pred(gp, x, y, xt);

% Plot the prediction and data
figure, set(gcf, 'color', 'w')
mesh(xt1, xt2, reshape(Eft_map,nxt,nxt));
hold on
plot3(x(:,1), x(:,2), y, '*')
xlabel('x_1'), ylabel('x_2')
title('The predicted underlying function (additive neural network)');

% NEURAL NETWORK COVARIANCE FUNCTION
gpcf_nn = gpcf_neuralnetwork('weightSigma2', ones(1,nin), 'biasSigma2', 1, ...
                             'weightSigma2_prior', pt, 'biasSigma2_prior', pt);
gp = gp_set('cf', {gpcf_nn}, 'noisef', {gpcf_n}, 'jitterSigma2', jitter);

% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'optimf',@fminscg,'opt',opt);

% Compute predictions in a grid using the MAP estimate
Eft_map = gp_pred(gp, x, y, xt);

% Plot the prediction and data
figure, set(gcf, 'color', 'w')
mesh(xt1, xt2, reshape(Eft_map,nxt,nxt));
hold on
plot3(x(:,1), x(:,2), y, '*')
xlabel('x_1'), ylabel('x_2')
title('The predicted underlying function (neural network)');
