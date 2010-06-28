%DEMO_INFNEURALNETWORK   Demonstration of Gaussian process with a neural
%                        network covariance function
%                    
%    Description
%    Infinite neural network solutions in 2D and 1D regression
%    problems with a comparison to Gaussian process solution given by
%    squared exponential covariance function. The noisy observations y
%    are assumed to satisfy
%
%         y = f + e,    where e ~ N(0, s^2)
%
%    where f is an unknown underlying function. A zero mean Gaussian
%    process prior is assumed for f
%
%         f ~ N(0, K),
%
%    where K is the covariance matrix whose elements are given by
%    neural network (or squared exponential) covariance function. A
%    hyperprior is assumed for hyperparameters of the covariance
%    functions, and the inference is done with a MAP estimate for
%    hyperparameter values.
%
%    For more detailed discussion of infinite neural networks, see
%    e.g. 
%
%    Neal, R. M. (1996). Bayesian Learning for Neural Networks.
%    Springer-Verlag.
%
%    Williams, C. K. I. (1996). Computing with infinite networks. In
%    Advances in Neural Information Processing Systems 9. MIT Press,
%    Cambridge, MA.
%
%
%  See also  DEMO_REGRESSION1
%
% Copyright (c) 2010 Jaakko Riihimäki

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.



% 2D REGRESSION DATA

% create 2D example data
x=rand(300,2)*2-1;
y=zeros(size(x,1),1); y(x(:,1)>0&x(:,2)>0)=1;
y=y+0.1*randn(size(y));

[n, nin] = size(x);


% --- Construct the model ---

% squared exponential covariance function
gpcf1 = gpcf_sexp('init', 'lengthScale', ones(1,nin), 'magnSigma2', 1);
% neural network covariance function
gpcf2 = gpcf_neuralnetwork('init', 'weightSigma2', ones(1,nin), 'biasSigma2', 1);
% Gaussian noise data structures
gpcf3 = gpcf_noise('init', 'noiseSigma2', 0.2^2);

% a prior structure for GP hyperparameters
pt = prior_t('init', 's2', 4);
gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pt, 'magnSigma2_prior', pt);
gpcf2 = gpcf_neuralnetwork('set', gpcf2, 'weightSigma2_prior', pt, 'biasSigma2_prior', pt);
gpcf3 = gpcf_noise('set', gpcf3, 'noiseSigma2_prior', pt);

gp = gp_init('init', 'FULL', 'gaussian', {gpcf1}, {gpcf3}, 'jitterSigma2', 1e-6, 'infer_params', 'covariance');
gp2 = gp_init('init', 'FULL', 'gaussian', {gpcf2}, {gpcf3}, 'jitterSigma2', 1e-6, 'infer_params', 'covariance');

% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg for more details)

w0=gp_pak(gp);  % pack the hyperparameters into one vector
fe=str2fun('gp_e');     % create a function handle to negative log posterior
fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior

% set the options for scg2
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do the optimization
wopt=scg2(fe, w0, opt, fg, gp, x, y);
wopt2=scg2(fe, w0, opt, fg, gp2, x, y);

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp, wopt);
gp2=gp_unpak(gp2, wopt2);

% create points where predictions are made
[p1,p2]=meshgrid(-1.5:0.05:1.5,-1.5:0.05:1.5);
p=[p1(:) p2(:)];
% compute the predictions
[Ef_map, Varf_map] = gp_pred(gp, x, y, p);
[Ef_map2, Varf_map2] = gp_pred(gp2, x, y, p);

% Plot the predictions and data
figure, set(gcf, 'color', 'w')
mesh(p1, p2, reshape(Ef_map,size(p1,1),size(p1,2)));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title('GP (squared exponential) predictions and the data points');

figure, set(gcf, 'color', 'w')
mesh(p1, p2, reshape(Ef_map2,size(p1,1),size(p1,2)));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title('GP (neural network) predictions and the data points');



% 1D REGRESSION DATA

% create a 1D toy data
x=rand(100,1)*4-2;
y=normpdf(4*x)+0.05*randn(size(x));
[n, nin] = size(x);

gpcf1 = gpcf_sexp('init', 'lengthScale', ones(1,nin), 'magnSigma2', 1);
gpcf2 = gpcf_neuralnetwork('init', 'weightSigma2', ones(1,nin), 'biasSigma2', 1);

gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pt, 'magnSigma2_prior', pt);
gpcf2 = gpcf_neuralnetwork('set', gpcf2, 'weightSigma2_prior', pt, 'biasSigma2_prior', pt);
gpcf3 = gpcf_noise('set', gpcf3, 'noiseSigma2_prior', pt);

gp = gp_init('init', 'FULL', 'gaussian', {gpcf1}, {gpcf3}, 'jitterSigma2', 1e-6, 'infer_params', 'covariance');
gp2 = gp_init('init', 'FULL', 'gaussian', {gpcf2}, {gpcf3}, 'jitterSigma2', 1e-6, 'infer_params', 'covariance');

% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg for more details)

w0=gp_pak(gp);  % pack the hyperparameters into one vector

% do the optimization
wopt=scg2(fe, w0, opt, fg, gp, x, y);
wopt2=scg2(fe, w0, opt, fg, gp2, x, y);

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp, wopt);
gp2=gp_unpak(gp2, wopt2);

% create points where predictions are made
xgrid=linspace(min(x)-1.5,max(x)+1.5,200)';
[Ef_map, Varf_map, Ey_map, Vary_map] = gp_pred(gp, x, y, xgrid);
[Ef_map2, Varf_map2, Ey_map2, Vary_map2] = gp_pred(gp2, x, y, xgrid);

% Plot the predictions and data
color1=ones(1,3)*0.8; color2=ones(1,3)*0.5;
figure, set(gcf, 'color', 'w'), hold on
h1=fill([xgrid' fliplr(xgrid')], [(Ey_map+1.96*sqrt(Vary_map))' fliplr((Ey_map-1.96*sqrt(Vary_map))')], color1, 'edgecolor', color1);
% GP mean
h2=plot(xgrid, Ey_map, 'color', color2, 'linewidth', 3);
% observations
h3=plot(x, y, 'xk', 'markersize', 10, 'linewidth', 2);
% true function
h4=plot(xgrid, normpdf(4*xgrid), 'color', 'r', 'linewidth', 2);
legend([h1 h2 h3 h4], 'GP 95% CI', 'GP mean', 'observations', 'true latent function')
title('GP (squared exponential) predictions and the data points');

figure, set(gcf, 'color', 'w'), hold on
h1=fill([xgrid' fliplr(xgrid')], [(Ey_map2+1.96*sqrt(Vary_map2))' fliplr((Ey_map2-1.96*sqrt(Vary_map2))')], color1, 'edgecolor', color1);
% GP mean
h2=plot(xgrid, Ey_map2, 'color', color2, 'linewidth', 3);
% observations
h3=plot(x, y, 'xk', 'markersize', 10, 'linewidth', 2);
% true function
h4=plot(xgrid, normpdf(4*xgrid), 'color', 'r', 'linewidth', 2);
legend([h1 h2 h3 h4], 'GP 95% CI', 'GP mean', 'observations', 'true latent function')
title('GP (neural network) predictions and the data points');

