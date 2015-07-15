%DEMO_ADDITIVE  A toy data regression example for additive covariance
%               function
%
%  Description
%    A simple toy data demonstration of additive covariance function.
%
%  See also
%    GPCF_ADDITIVE
%
%  References:
%    Duvenaud, D. K., Nickisch, H., & Rasmussen, C. E. (2011). Additive
%    gaussian processes. In Advances in neural information processing
%    systems (pp. 226-234).

% Copyright (c) 2014 Tuomas Sivula

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% setrandstream(0);

% The true function
fun = @(x) 0.5*sum(sin(2*(x+0.4)),2);

% Generate toy data (similar to the reference)
n1 = 11;
n2 = 11;
x = [[ 0.5*rand(n1,1)-2  ,    4*rand(n1,1)-2 ] ; ...
     [   4*rand(n2,1)-2  ,  0.5*rand(n2,1)-2 ]];
y = fun(x);
[X1,X2] = meshgrid(linspace(-2,2,151),linspace(-2,2,151));
xt = [X1(:), X2(:)];
yt = fun(xt);

% Optimisation options
opt = optimset('TolFun', 1e-4, ...
               'TolX',   1e-4, ...
               'Derivativecheck','off', ...
               'display','final');

% Fit 'basic' squared-exp GP model
fprintf('Fit squared-exp GP model\n')
gp2 = gp_set('cf', gpcf_sexp);
gp2 = gp_optim(gp2, x, y, 'opt', opt);
Ef2 = gp_pred(gp2, x, y, xt);

% Fit additive GP model
fprintf('Fit additive GP model\n')
cf = gpcf_additive('cf', {gpcf_sexp, gpcf_sexp}, 'max_deg', 2);
gp = gp_set('cf', cf);
gp = gp_optim(gp, x, y, 'opt', opt);
Ef = gp_pred(gp, x, y, xt);
params = exp(gp_pak(gp));
fprintf('Inferred order variances:\n1st: %f\n2nd: %f\n', params(1), params(2))


% ----------------
% Plot the figures
% ----------------

% True function
figure()
pcolor(X1, X2, reshape(yt,size(X1,1),size(X1,2)));
hold on
plot(x(:,1), x(:,2), 'k.')
shading interp;
title('True function and data locations')

% Squared-exp GP posterior mean
figure()
pcolor(X1, X2, reshape(Ef2,size(X1,1),size(X1,2)));
hold on
plot(x(:,1), x(:,2), 'k.')
shading interp;
title('Squared-exp GP')

% Additive GP posterior mean
figure()
pcolor(X1, X2, reshape(Ef,size(X1,1),size(X1,2)));
hold on
plot(x(:,1), x(:,2), 'k.')
shading interp;
title('Additive GP')


