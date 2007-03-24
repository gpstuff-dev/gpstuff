function y = mnorm_pdf(x,mu,S)
%MNORM_PDF Multivariate-Normal probability density function (pdf).
%
%   Description
%   Y = MNORM_PDF(X,MU,SIGMA) Returns the multivariate-normal
%   pdf with mean, MU, and covariance matrix, SIGMA, at the values in X.
%
%   Default values for MU and SIGMA are 0 and 1 respectively.

% Copyright (c) 1998-2005 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin < 3, 
  sigma = 1;
end

if nargin < 2;
  mu = 0;
end

if nargin < 1, 
  error('Requires at least one input argument.');
end

[n,m]=size(x);
xmu=x-repmat(mu,n,1);  
% Use Cholesky decomposition, since it is faster and
% numerically more stable.
L=chol(S)';
y=zeros(n,1);
for i1=1:n
  b=L\xmu(i1,:)';
  y(i1)=-.5*b'*b;
end
y=exp(y-sum(log(diag(L)))-.5*m*log(2*pi));
