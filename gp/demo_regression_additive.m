%DEMO_REGRESSION_ADDITIVE   Regression demonstration with additive Gaussian
%                           process using linear, squared exponential and
%                           neural network covariance fucntions 
%
%    Description
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
%    Rasmussen, C. E. and Williams, C. K. I. (2006). Gaussian Processes for 
%    Machine Learning. The MIT Press.
%
%
%   See also  DEMO_REGRESSION1
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
[p1,p2]=meshgrid(-2:0.1:2,-2:0.1:2);
p=[p1(:) p2(:)];
np=size(p1,1);


% Assume a Student-t distribution for the GP hyperparameters
pt = prior_t('init', 'nu', 4, 's2', 10);	% a prior structure

% Create a Gaussian noise model
gpcf_n = gpcf_noise('init', 'noiseSigma2', 0.2^2, 'noiseSigma2', 0.2^2);

% Set a small amount of jitter 
jitter=1e-3;

fe=str2fun('gp_e');     % create a function handle to negative log posterior
fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior

% set the options for scg2
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;



% CONSTANT + LINEAR COVARIANCE FUNCTION

% constant covariance function
gpcf_c = gpcf_constant('init', 'constSigma2', 1);
gpcf_c = gpcf_constant('set', gpcf_c, 'constSigma2_prior', pt);

% linear covariance function
gpcf_l = gpcf_linear('init');
gpcf_l = gpcf_linear('set', gpcf_l, 'coeffSigma2_prior', pt);

gp = gp_init('init', 'FULL', 'gaussian', {gpcf_c gpcf_l}, {gpcf_n}, 'jitterSigma2', jitter, 'infer_params', 'covariance');

w=gp_pak(gp);  % pack the hyperparameters into one vector

% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg for more details)

% do the optimization
w=scg2(fe, w, opt, fg, gp, x, y);

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w);

% Compute predictions in a grid using the MAP estimate
[Ef_map, Varf_map] = gp_pred(gp, x, y, p);

% Plot the prediction and data
figure, set(gcf, 'color', 'w')
mesh(p1, p2, reshape(Ef_map,np,np));
hold on
plot3(x(:,1), x(:,2), y, '*')
xlabel('x_1'), ylabel('x_2')
title('The predicted underlying function (constant + linear)');



% CONSTANT + SQUARED EXPONENTIAL COVARIANCE FUNCTION (W.R.T. THE FIRST
% INPUT DIMENSION) + LINEAR (W.R.T. THE SECOND INPUT DIMENSION) 

% Covariance function for the first input variable
gpcf_s1 = gpcf_sexp('init', 'magnSigma2', 0.15, 'magnSigma2_prior', pt);
% create metric structure:
metric1 = metric_euclidean('init', {[1]},'lengthScales',[0.5], 'lengthScales_prior', pt);
% set the metric to the covariance function structure:
gpcf_s1 = gpcf_sexp('set', gpcf_s1, 'metric', metric1);

gpcf_l2 = gpcf_linear('init', 'selectedVariables', [2]);
gpcf_l2 = gpcf_linear('set', gpcf_l2, 'coeffSigma2_prior', pt);

gp = gp_init('init', 'FULL', 'gaussian', {gpcf_c gpcf_s1 gpcf_l2}, {gpcf_n}, 'jitterSigma2', jitter, 'infer_params', 'covariance');

w=gp_pak(gp);  % pack the hyperparameters into one vector

% do the optimization
w=scg2(fe, w, opt, fg, gp, x, y);

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w);

% Compute predictions in a grid using the MAP estimate
[Ef_map, Varf_map] = gp_pred(gp, x, y, p);

% Plot the prediction and data
figure, set(gcf, 'color', 'w')
mesh(p1, p2, reshape(Ef_map,np,np));
hold on
plot3(x(:,1), x(:,2), y, '*')
xlabel('x_1'), ylabel('x_2')
title('The predicted underlying function (sexp for 1. input + linear for 2. input )');



% ADDITIVE SQUARED EXPONENTIAL COVARIANCE FUNCTION

% Covariance function for the second input variable
gpcf_s2 = gpcf_sexp('init', 'magnSigma2', 0.15, 'magnSigma2_prior', pt);
% create metric structure:
metric2 = metric_euclidean('init', {[2]},'lengthScales',[0.5], 'lengthScales_prior', pt);
% set the metric to the covariance function structure:
gpcf_s2 = gpcf_sexp('set', gpcf_s2, 'metric', metric2);

gp = gp_init('init', 'FULL', 'gaussian', {gpcf_s1,gpcf_s2}, {gpcf_n}, 'jitterSigma2', jitter, 'infer_params', 'covariance');

w=gp_pak(gp);  % pack the hyperparameters into one vector

% do the optimization
w=scg2(fe, w, opt, fg, gp, x, y);

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w);

% Compute predictions in a grid using the MAP estimate
[Ef_map, Varf_map] = gp_pred(gp, x, y, p);

% Plot the prediction and data
figure, set(gcf, 'color', 'w')
mesh(p1, p2, reshape(Ef_map,np,np));
hold on
plot3(x(:,1), x(:,2), y, '*')
xlabel('x_1'), ylabel('x_2')
title('The predicted underlying function (additive sexp)');



% SQUARED EXPONENTIAL COVARIANCE FUNCTION

gpcf_s = gpcf_sexp('init', 'lengthScale', ones(1,nin), 'magnSigma2', 0.2^2);
gpcf_s = gpcf_sexp('set', gpcf_s, 'lengthScale_prior', pt, 'magnSigma2_prior', pt);

gp = gp_init('init', 'FULL', 'gaussian', {gpcf_s}, {gpcf_n}, 'jitterSigma2', jitter, 'infer_params', 'covariance');

% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg for more details)

w=gp_pak(gp);  % pack the hyperparameters into one vector

% do the optimization
w=scg2(fe, w, opt, fg, gp, x, y);

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w);
[Ef_map, Varf_map] = gp_pred(gp, x, y, p);

% Plot the prediction and data
figure, set(gcf, 'color', 'w')
mesh(p1, p2, reshape(Ef_map,np,np));
hold on
plot3(x(:,1), x(:,2), y, '*')
xlabel('x_1'), ylabel('x_2')
title('The predicted underlying function (sexp)');



% ADDITIVE NEURAL NETWORK COVARIANCE FUNCTION

gpcf_nn1 = gpcf_neuralnetwork('init', 'weightSigma2', 1, 'biasSigma2', 1, 'selectedVariables', [1]);
gpcf_nn1 = gpcf_neuralnetwork('set', gpcf_nn1, 'weightSigma2_prior', pt, 'biasSigma2_prior', pt);

gpcf_nn2 = gpcf_neuralnetwork('init', 'weightSigma2', 1, 'biasSigma2', 1, 'selectedVariables', [2]);
gpcf_nn2 = gpcf_neuralnetwork('set', gpcf_nn2, 'weightSigma2_prior', pt, 'biasSigma2_prior', pt);

gp = gp_init('init', 'FULL', 'gaussian', {gpcf_nn1,gpcf_nn2}, {gpcf_n}, 'jitterSigma2', jitter, 'infer_params', 'covariance');

w=gp_pak(gp);   % pack the hyperparameters into one vector

% do the optimization
w=scg2(fe, w, opt, fg, gp, x, y);

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w);

% Compute predictions in a grid using the MAP estimate
[Ef_map, Varf_map] = gp_pred(gp, x, y, p);

% Plot the prediction and data
figure, set(gcf, 'color', 'w')
mesh(p1, p2, reshape(Ef_map,np,np));
hold on
plot3(x(:,1), x(:,2), y, '*')
xlabel('x_1'), ylabel('x_2')
title('The predicted underlying function (additive neural network)');



% NEURAL NETWORK COVARIANCE FUNCTION

gpcf_nn = gpcf_neuralnetwork('init', 'weightSigma2', ones(1,nin), 'biasSigma2', 1);
gpcf_nn = gpcf_neuralnetwork('set', gpcf_nn, 'weightSigma2_prior', pt, 'biasSigma2_prior', pt);

gp = gp_init('init', 'FULL', 'gaussian', {gpcf_nn}, {gpcf_n}, 'jitterSigma2', jitter, 'infer_params', 'covariance');

w=gp_pak(gp);   % pack the hyperparameters into one vector

% do the optimization
w=scg2(fe, w, opt, fg, gp, x, y);

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w);

% Compute predictions in a grid using the MAP estimate
[Ef_map, Varf_map] = gp_pred(gp, x, y, p);

% Plot the prediction and data
figure, set(gcf, 'color', 'w')
mesh(p1, p2, reshape(Ef_map,np,np));
hold on
plot3(x(:,1), x(:,2), y, '*')
xlabel('x_1'), ylabel('x_2')
title('The predicted underlying function (neural network)');
