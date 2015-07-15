function [k,sigma] = gpdfitnew(x)
%GPDFITNEW Estimate the paramaters for the Generalized Pareto Distribution
%   
%  Description
%    [K,SIGMA] = GPDFITNEW(X) returns empirical Bayes estimate for the
%    parameters of the two-parameter generalized Parato distribution
%    (GPD) given the data in X.
%    
%  Reference
%    Jin Zhang & Michael A. Stephens (2009) A New and Efficient
%    Estimation Method for the Generalized Pareto Distribution,
%    Technometrics, 51:3, 316-325, DOI: 10.1198/tech.2009.08017
%
%  Note
%    This function returns a negative of Zhang and Stephens's k,
%    because it is more common parameterisation
%
% Copyright (c) 2015 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

n=numel(x);
prior=3;
x=sort(x);
m=80+floor(sqrt(n));
bs=1/x(n)+(1-sqrt(m./([1:m]'-.5)))./prior./x(floor(n./4+0.5));
% loop version matching Zhang and Stephens paper
% w=zeros(m,1);
% L=zeros(m,1);
% ks=zeros(m,1);
% for i=1:m
%     k=-mean(log1p(-bs(i).*x));
%     ks(i)=-k;
%     L(i)=n*(log(bs(i)/k)+k-1);
% end
% for i=1:m
%     w(i)=1/sum(exp(L-L(i)));
% end
% faster vectorized version
% we use negative of Zhang and Stephens's k, because it
% is more common parameterisation
ks=mean(log1p(bsxfun(@times,-bs,x')),2);
L=n*(log(bs./-ks)-ks-1);
w=1./sum(exp(bsxfun(@minus,L,L')))';
sigmas=-ks./bs;

% remove negligible weights
dii=w<eps*10;
w(dii)=[];w=w./sum(w);
bs(dii)=[];
ks(dii)=[];

% posterior mean for b
b=sum(bs.*w);
% estimate for k, note that we return a negative of Zhang and
% Stephens's k, because it is more common parameterisation
k=mean(log1p(-b*x));
% estimate for sigma
sigma=-k/b;
