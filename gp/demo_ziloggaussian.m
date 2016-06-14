% DEMO_ZILOGGAUSSIAN  Regression problem demonstration for 2-input 
%                     function with zero-inflated log-Gaussian likelihood
%
%  Description
%    The regression problem consist of a data with two input
%    variables and one output variable with zero-inflated log-Gaussian
%    likelihood. The inference is conducted with Laplace approximation for
%    the conditional posterior of latent function and (Laplace approximate)
%    MAP estimate for the hyperparameters.
%
% Copyright (c) 2016 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% Load the data
S = which('demo_regression1');
L = strrep(S,'demo_regression1.m','demodata/dat.1');
data=load(L);
x = [data(:,1) data(:,2)];
y = data(:,3);
[n, nin] = size(x);

% set zero observations and transfer the rest to exp scale
y(y<0) = 0;
y(y>0) = exp(y(y>0));

% Construct the model
lik = lik_ziloggaussian('sigma2_prior',prior_t('s2',0.1));
cfc = gpcf_constant;
gpcf = gpcf_sexp('lengthScale', [1.1 1.2], 'magnSigma2', 0.2^2)
gp = gp_set('lik', lik, 'cf', {cfc gpcf cfc gpcf},'comp_cf', {1:2 3:4});

disp(' MAP estimate for the parameters')

% Optimize with the scaled conjugate gradient method
opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter');
gp=gp_optim(gp,x,y,'opt',opt);

% Make predictions of the underlying function on a dense grid and plot it.
[xt1,xt2]=meshgrid(-1.8:0.1:1.8,-1.8:0.1:1.8);
xt=[xt1(:) xt2(:)];
nt = size(xt,1);
[Eft_map, Varft_map, Lpy_map, Ey_map, Vary_map] = gp_pred(gp, x, y, xt, 'yt', zeros(size(xt,1),1));

% Plot the prediction and data
figure(1)
clf
subplot(1,2,1)
mesh(xt1, xt2, reshape(1-exp(Lpy_map),37,37));
hold on
plot3(x(y>0,1), x(y>0,2), ones(sum(y>0),1), 'r*')
plot3(x(y==0,1), x(y==0,2), zeros(sum(y==0)), 'k*')
axis on;
title('The probability of being greater than zero');
subplot(1,2,2)
mesh(xt1, xt2, reshape(Ey_map,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title('The posterior mean of y');


% %% Test the likelihood function
% % This block is a general purpose code for testing new likelihood
% % functions
% f = 2*randn(size(y,1)*2,1);
% 
% lik.fh.ll(lik,y,f)
% 
% % check llg with respect to latent
% fe = @(x) lik.fh.ll(lik,y,x');
% fg = @(x) lik.fh.llg(lik,y,x','latent')';
% gradcheck(randn(size(f')),fe,fg);
% 
% % check llg with respect to param
% [w,s] = lik.fh.pak(lik)
% w=randn(size(w));
% fe = @(x) lik.fh.ll(lik.fh.unpak(lik,x),y,f);
% fg = @(x) lik.fh.llg(lik.fh.unpak(lik,x),y,f,'param');
% gradcheck(randn(size(w)),fe,fg);
% 
% % check llg2 with respect to latent
% ind = unique( ceil(length(y)*rand(20,1)));
% h1 = lik.fh.llg2(lik,y(ind),f([ind ; ind+n]), 'latent');
% ny=length(ind);
% h1 = [diag(h1(1:ny,1)) diag(h1(ny+1:end,1)) ; diag(h1(1:ny,2)) diag(h1(ny+1:end,2))];
% fe = @(x) lik.fh.ll(lik,y(ind),x);
% h2 = hessian(fe,f([ind ; ind+n]));
% [min(min(h1)) max(max(h1))]
% [min(min(h2)) max(max(h2))]
% [min(min(h1-h2)) max(max(h1-h2))]
% 
% gpla_e(gp_pak(gp),gp,x,y)
% gpla_g(gp_pak(gp),gp,x,y)
% 
% gradcheck(randn(size(gp_pak(gp))),@gpla_e,@gpla_g,gp,x,y);