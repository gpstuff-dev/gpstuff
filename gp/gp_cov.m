function [C, Cinv] = gp_cov(gp, x1, x2, varargin)
% GPCOV     Evaluate covariance matrix between two input vectors. 
%
%         Description
%         C = GPCOV(GP, TX, X) takes in Gaussian process GP and two
%         matrixes TX and X that contain input vectors to GP. Returns 
%         covariance matrix C. Every element ij of C contains covariance 
%         between inputs i in TX and j in X.
%
%         [C, Cinv] = GPCOV(GP, TX, X, VARARGIN) returns also inverse of covariance.
%
%         For covariance function definition see manual or 
%         Neal R. M. Regression and Classification Using Gaussian 
%         Process Priors, Bayesian Statistics 6.

% Copyright (c) 2006 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

ncf = length(gp.cf);

C = sparse(0);
for i=1:ncf
  gpcf = gp.cf{i};
  C = C + feval(gpcf.fh_cov, gpcf, x1, x2);
end

if ~isempty(gp.jitterSigmas) & size(x1,1)==size(x2,1) & x1==x2
  [n,m]=size(x1);
  n1 = n+1;
  C(1:n1:end)=C(1:n1:end)+gp.jitterSigmas.^2;
end

%C(C<eps)=0;
