%DEMO_MONOTONIC  Demonstration of monotonic GP regression
%
%  Description
%    This demonstration consists of comparison between monotonic GP and
%    standard GP regression. The monotonocity of the latent functions is
%    forced by the GP prior where we can compute analytically the
%    covariance between derivative of the latent function and the latent
%    functions and the cross-covariance terms.
%
%  Reference
%     
%    Riihimäki and Vehtari (2010). Gaussian processes with
%    monotonicity information.  Journal of Machine Learning Research:
%    Workshop and Conference Proceedings, 9:645-652.
%
%  See also DEMO_*, GP_MONOTONIC
%
% Copyright (c) 2014 Ville Tolvanen

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% Toy data
setrandstream(0);
x=rand(300,2);
[xt1,xt2]=meshgrid(linspace(0,1,25), linspace(0,1,25));
xt=[xt1(:) xt2(:)];
y=x(:,1)+x(:,2) - 1.5.*x(:,1).^2 - 1.5.*x(:,2).^2;
y=y+0.1.*randn(300,1);

gpcf=gpcf_sexp();
lik=lik_gaussian();

gp=gp_set('cf', gpcf, 'lik', lik, 'jitterSigma2', 1e-9);

gp2=gp;

% Create monotonic GP with 25 virtual observations and monotonically
% decreasing latent function for both input dimensions and optimize the
% parameters
opt=optimset('TolX',1e-4,'TolFun',1e-4,'Display','iter');
gp=gp_monotonic(gp,x,y,'nv', 35, 'nvd', [-1 -2], 'optimize', 'on', ...
  'opt', opt, 'optimf', @fminlbfgs);
% Optimize the normal GP for comparison
gp2=gp_optim(gp2,x,y,'opt',opt, 'optimf', @fminlbfgs);

% Do predictions
Ef=gp_pred(gp,x,y,xt);
Ef2=gp_pred(gp2,x,y,xt);

figure(1); 
mesh(xt1,xt2,reshape(Ef(1:25^2),25,25));
hold all;
plot3(x(:,1),x(:,2), y, '*r')
title('Predicted latent function with assumed monotonicity for both input dimensions.');
figure(2);
mesh(xt1,xt2,reshape(Ef2,25,25))
hold all;
plot3(x(:,1),x(:,2), y, '*r')
title('Predicted latent function with no monotonicity.');