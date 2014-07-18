%DEMO_EPINF Demonstration of input dependent noise and magnitude model
%
% Description
%       Uses toy data sets to demonstrate heteroscedastic noise with input
%       dependent noise and magnitude model. Compare with results from
%       DEMO_INPUTDEPENDENTNOISE.
%
% See also
%       DEMO_INPUTDEPENDENTNOISE, LIK_EPGAUSSIAN
%
%
% Copyright (c) Ville Tolvanen 2011-2014
% Copyright (c) Tuomas Sivula 2014
 
% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

% Initialize random stream
prevstream=setrandstream(0);

% Create toy data
% x = 100*rand([40 1]);
n = 501;
x=linspace(-100,200,n)';
f1 = [5.*sin(-3+0.2.*x(1:ceil(0.23*n))); ...
      0*sin(0.1*x(ceil(0.23*n)+1:ceil(0.85*n))); ...
      5.*sin(2.8+0.2.*x(ceil(0.85*n)+1:end))];
f2 = 100*norm_pdf(x,110,20) + 100*norm_pdf(x,-10,20);
sigma2 = 0.5;

x=x-mean(x); x=x./std(x);
f1 = f1-mean(f1); f1=f1./std(f1);

y = f1 + sqrt((sigma2.*exp(f2))).*randn(size(x));
yt = f1(1:2:end);
xt = x(1:2:end);
nt = size(xt,1);
x=x(:); y=y(:); xt=xt(:);

% Create the covariance functions
pl = prior_t('s2',1);
pm = prior_t('s2',2);
gpcf1 = gpcf_sexp('lengthScale', 0.08, 'magnSigma2', 1.2, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);
gpcf2 = gpcf_sexp('lengthScale', 0.25, 'magnSigma2', 0.7, ...
                 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
gpcf3 = gpcf_sexp('lengthScale', 0.8, 'magnSigma2', 2.8, ...
                 'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure. Don't set prior for sigma2 if covariance
% function magnitude for noise process has a prior.
lik = lik_epgaussian('sigma2', sigma2, 'sigma2_prior', prior_fixed(), ...
                     'int_likparam', true, 'inputparam', true, ...
                     'int_magnitude', true, 'inputmagnitude', true);

% Set latent options
latent_opt = struct('maxiter',10000, ...
                    'df',0.8, ...
                    'df2',0.6, ...
                    'tol',1e-6, ...
                    'parallel', 'on', ...
                    'init_prev','off', ...
                    'display','off');

% NOTE! if multiple covariance functions per latent is used, define
% gp.comp_cf as follows:
% gp = gp_set(..., 'comp_cf' {[1 2] [5 6]};
gp = gp_set('lik', lik, 'cf', {gpcf1 gpcf2 gpcf3}, ...
            'jitterSigma2', 1e-9, 'comp_cf', {1 2 3}, ...
            'latent_method', 'EP', 'latent_opt', latent_opt);

% Set the options for the optimization
opt=optimset('TolFun',1e-4,'TolX',1e-4,'Derivativecheck','off','display','iter');
% Optimize with the scaled conjugate gradient method
fprintf('Optimise the input-dependent model\n');
gp = gp_optim(gp,x,y,'opt',opt, 'optimf', @fminscg);

% make prediction to the data points
[Ef, Varf, lpyt] = gp_pred(gp, x, y, xt, 'yt', yt);
%prctmus = gp_predprctmu(gp, x, y, xt);
prctmus=[Ef(:,1)-1.645*sqrt(Varf(:,1)) ...
         Ef(:,1) ...
         Ef(:,1)+1.645*sqrt(Varf(:,1))];
fprintf('mlpd input-dependent: %.2f\n', mean(lpyt));

% Gaussian for comparison
lik2 = lik_gaussian();
gp2 = gp_set('lik', lik2, 'cf', gpcf1, 'jitterSigma2', 1e-9);
fprintf('Optimise the normal gaussian model\n');
gp2 = gp_optim(gp2,x,y,'opt',opt);
[Ef2, Varf2, lpyt2] = gp_pred(gp2, x, y, xt,'yt',yt);
prctmus2 = gp_predprctmu(gp2, x, y, xt);
fprintf('mlpd gaussian: %.2f\n', mean(lpyt2));

figure();
% plot mean and 5% and 95% quantiles
subplot(2,1,1)
plot(xt,Ef(:,1),'b', ...
     xt,prctmus(:,1),'r', ...
     xt,prctmus(:,3),'r', ...
     x, f1, 'k:', ...
     x, y, 'k.')
ylim([-3 3]), title('Input dependent noise model');
legend('mean','5%','95%','true mean','samples',2)

% Compare to Gaussian with homoscedastic scale
subplot(2,1,2),
plot(xt, Ef2,'b', ...
     xt,prctmus2(:,1),'r', ...
     xt,prctmus2(:,3),'r', ...
     x, f1, 'k:', ...
     x, y, 'k.')
ylim([-3 3]), title('Gaussian noise model')

figure()
subplot(2,1,1)
s2=gp.lik.sigma2;
plot(xt, s2.*exp(Ef(:,2)), '-b', ...
     x, sigma2.*exp(f2), '-k', ...
     xt, s2.*exp(Ef(:,2) + 1.96.*sqrt(Varf(:,2))), '-r', ...
     xt, s2.*exp(Ef(:,2) - 1.96.*sqrt(Varf(:,2))), '-r')
legend('Predicted', 'Real','95% CI',2);
title('Noise variance')

subplot(2,1,2)
s2=gp.lik.sigma2;
plot(xt, exp(Ef(:,3)), '-b', ...
     xt, exp(Ef(:,3) + 1.96.*sqrt(Varf(:,3))), '-r', ...
     xt, exp(Ef(:,3) - 1.96.*sqrt(Varf(:,3))), '-r')
title('Predicted signal variance')
