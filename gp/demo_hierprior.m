%DEMO_HIERPRIOR  Demonstration of hierarchical prior structures for 
%                  Gaussian processes
%
%  Description
%    This demonstration shows how to set hierarchical prior structures for
%    parameters of covariance function and/or likelihood function. This
%    demo is intended for showing how to set hierarchical prior structures
%    and for other testing purposes, not as a demonstration of model 
%    selection procedures or comparisons. We test the hierarchical priors 
%    for both regression and classification with Laplace and EP.
%
%
%  See also DEMO_REGRESSION_ROBUST, DEMO_SPATIAL1, DEMO_*
%
% Copyright (c) 2014 Ville Tolvanen

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

%% Regression
% load the data. First 100 variables are for training
% and last 100 for test

S = which('demo_hierprior');
L = strrep(S,'demo_hierprior.m','demodata/odata.txt');
x = load(L);
y = x(1:100,2);
x = x(1:100,1);
[n, nin] = size(x); 

% Test data
xt = [-2.7:0.01:2.7]';
yt = 0.3+0.4*xt+0.5*sin(2.7*xt)+1.1./(1+xt.^2);

% Create hierarchial prior
pl=prior_t('mu_prior',prior_t);

% Create the model
gpcf = gpcf_sexp('magnSigma2_prior', pl, 'lengthScale_prior', pl);
gpcf2 = gpcf_linear('coeffSigma2_prior', pl);
% gpcf2 = gpcf_neuralnetwork('weightSigma2_prior', pl, 'biasSigma2_prior', pl);
lik = lik_gaussian('sigma2_prior', pl);
gp = gp_set('lik', lik, 'cf', {gpcf gpcf2}, 'jitterSigma2',1e-9);

% Parameters
[w,s]=gp_pak(gp)

% Optimize the parameters
opt=optimset('TolX',1e-5,'TolFun',1e-5,'Display','iter');
gp = gp_optim(gp,x,y,'opt',opt);

derivs={};
derivs{1}=derivativecheck(gp_pak(gp), @(ww) gp_eg(ww, gp, x, y));

[Ef,Varf,lpyt,Ey,Vary]=gp_pred(gp,x,y,xt,'yt',yt);

%% Regression with sparse approximations
% Here we optimize all the model parameters, including the inducing inputs

X_u=linspace(-2,3, 10)';
gp = gp_set('lik', lik, 'cf', {gpcf gpcf2}, 'jitterSigma2',1e-6, 'type', 'FIC', ...
  'X_u', X_u, 'infer_params', 'covariance+likelihood+inducing', ...
  'Xu_prior', prior_gaussian('mu_prior', prior_t()));
gp = gp_optim(gp,x,y,'opt',opt);
derivs{2}(:,1)=derivativecheck(gp_pak(gp), @(ww) gp_eg(ww, gp, x, y));
Ef2=gp_pred(gp,x,y,xt,'yt',yt);

gp = gp_set('lik', lik, 'cf', {gpcf gpcf2}, 'jitterSigma2',1e-6, 'type', 'VAR', ...
  'X_u', X_u, 'infer_params', 'covariance+likelihood+inducing', ...
  'Xu_prior', prior_gaussian('mu_prior', prior_t()));
gp = gp_optim(gp,x,y,'opt',opt);
derivs{2}(:,2)=derivativecheck(gp_pak(gp), @(ww) gp_eg(ww, gp, x, y));
Ef3=gp_pred(gp,x,y,xt,'yt',yt);

gp = gp_set('lik', lik, 'cf', {gpcf gpcf2}, 'jitterSigma2',1e-6, 'type', 'SOR', ...
  'X_u', X_u, 'infer_params', 'covariance+likelihood+inducing', ...
  'Xu_prior', prior_gaussian('mu_prior', prior_t()));
gp = gp_optim(gp,x,y,'opt',opt);
derivs{2}(:,3)=derivativecheck(gp_pak(gp), @(ww) gp_eg(ww, gp, x, y));
Ef4=gp_pred(gp,x,y,xt,'yt',yt);

%% Plot

plot(x,y,'.', xt, Ey, xt, Ef2, xt, Ef3, xt, Ef4);
legend('Data', 'Full-GP', 'FIC', 'VAR', 'SOR');

%% Classification with Laplace & EP (full and sparse approximation)
S = which('demo_spatial1');
data = load(strrep(S,'demo_spatial1.m','demodata/spatial1.txt'));

x = data(:,1:2);
ye = data(:,3);
y = data(:,4);
% Remove some of the data
inds=find(x(:,2)>30);
x=x(inds,:);
y=y(inds,:);
ye=ye(inds,:);

dims = [30    60     1    35];
[trindex, Xu] = set_PIC(x, dims, 7, 'corners', 0);
[n,nin] = size(x);


% Create the covariance functions
pl = prior_t('s2',10, 's2_prior', prior_t());
pm = prior_sqrtunif();
pxu={};
for i1=1:size(Xu,1)
  pxu={pxu{:} pl};
end
gpcf1 = gpcf_matern32('lengthScale', 1, 'magnSigma2', 0.03, ...
  'lengthScale_prior', pl, 'magnSigma2_prior', prior_fixed());
gpcf2 = gpcf_ppcs3('nin',nin,'lengthScale', 5, 'magnSigma2', 0.05,...
  'lengthScale_prior', pl, 'magnSigma2_prior', prior_fixed());

lik = lik_negbin();

% Create the GP structures (Laplace)
gp = gp_set('lik', lik, 'cf', gpcf1,'jitterSigma2', 1e-4, ...
            'infer_params', 'covariance+likelihood');
gp2 = gp_set('type', 'FIC', 'lik', lik, 'cf', gpcf1, 'X_u', Xu, 'Xu_prior', pxu, ...
            'jitterSigma2', 1e-4, 'infer_params', 'covariance+likelihood+inducing');
gp3 = gp_set('type', 'PIC', 'lik', lik, 'cf', gpcf1, 'X_u', Xu, 'Xu_prior', pxu,  ...
           'jitterSigma2', 1e-4, 'infer_params', 'covariance+likelihood+inducing',...
           'tr_index', trindex);
gp4 = gp_set('type', 'CS+FIC', 'lik', lik, 'cf', {gpcf1 gpcf2}, 'X_u', Xu, 'Xu_prior', pxu,  ...
           'jitterSigma2', 1e-4, 'infer_params', 'covariance+likelihood+inducing');
gp4.cf{1}.p.lengthScale=prior_fixed();
gp4.cf{1}.p.magnSigma2=prior_fixed();


opt=optimset('TolX',1e-3,'TolFun',1e-3,'Display','iter');
fprintf('Laplace \n \n');
fprintf('Full GP:\n \n')
derivs{3}(:,1)=derivativecheck(gp_pak(gp), @(ww) gp_eg(ww, gp, x, y, 'z', ye));
fprintf('FIC:\n \n')
derivs{4}(:,1)=derivativecheck(gp_pak(gp2), @(ww) gp_eg(ww, gp2, x, y, 'z', ye));
fprintf('PIC:\n \n')
derivs{4}(:,2)=derivativecheck(gp_pak(gp3), @(ww) gp_eg(ww, gp3, x, y, 'z', ye));
fprintf('CS+FIC:\n \n')
derivs{4}(:,3)=derivativecheck(gp_pak(gp4), @(ww) gp_eg(ww, gp4, x, y, 'z', ye));

% Create the GP structures (EP)
gp = gp_set('lik', lik, 'cf', gpcf1,'jitterSigma2', 1e-4, ...
            'infer_params', 'covariance+likelihood', ...
            'latent_method', 'EP');
gp2 = gp_set('type', 'FIC', 'lik', lik, 'cf', gpcf1, 'X_u', Xu,  'Xu_prior', pxu, ...
            'jitterSigma2', 1e-4, 'infer_params', 'covariance+likelihood+inducing', ...
            'latent_method', 'EP');
gp3 = gp_set('type', 'PIC', 'lik', lik, 'cf', gpcf1, 'X_u', Xu, 'Xu_prior', pxu,  ...
           'jitterSigma2', 1e-4, 'infer_params', 'covariance+likelihood+inducing',...
           'tr_index', trindex, 'latent_method', 'EP');
gp4 = gp_set('type', 'CS+FIC', 'lik', lik, 'cf', {gpcf1 gpcf2}, 'X_u', Xu,  'Xu_prior', pxu, ...
           'jitterSigma2', 1e-4, 'infer_params', 'covariance+likelihood+inducing', ...
            'latent_method', 'EP');
gp4.cf{1}.p.lengthScale=prior_fixed();
gp4.cf{1}.p.magnSigma2=prior_fixed();

fprintf('EP \n \n');
fprintf('Full GP:\n \n')
derivs{5}(:,1)=derivativecheck(gp_pak(gp), @(ww) gp_eg(ww, gp, x, y, 'z', ye));
fprintf('FIC:\n \n')
derivs{6}(:,1)=derivativecheck(gp_pak(gp2), @(ww) gp_eg(ww, gp2, x, y, 'z', ye));
fprintf('PIC:\n \n')
derivs{6}(:,2)=derivativecheck(gp_pak(gp3), @(ww) gp_eg(ww, gp3, x, y, 'z', ye));
fprintf('CS+FIC:\n \n')
derivs{6}(:,3)=derivativecheck(gp_pak(gp4), @(ww) gp_eg(ww, gp4, x, y, 'z', ye));
