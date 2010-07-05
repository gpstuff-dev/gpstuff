function [C, Cinv] = gp_cov(gp, x1, x2, predcf)
% GPCOV     Evaluate covariance matrix between two input vectors. 
%
%         Description
%         C = GPCOV(GP, TX, X, PREDCF) takes in Gaussian process GP
%         and two matrixes TX and X that contain input vectors to
%         GP. Returns covariance matrix C. Every element ij of C
%         contains covariance between inputs i in TX and j in
%         X. PREDCF is an optional array specifying the indexes of
%         covariance functions, which are used for forming the
%         matrix. If empty or not given, the matrix is formed with all
%         functions.

% Copyright (c) 2007-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

ncf = length(gp.cf);

C = sparse(0);
if nargin < 4 || isempty(predcf)
    predcf = 1:ncf;
end      
for i=1:length(predcf)
  gpcf = gp.cf{predcf(i)};
  C = C + feval(gpcf.fh_cov, gpcf, x1, x2);
end

% Add jitter if the inputs are equal
n1 = size(x1,1);
n2 = size(x2,1);
if n1 == n2  && max(max(abs(x1-x2))) == 0
    if isfield(gp, 'jitterSigma2') && ~isempty(gp.jitterSigma2)
        n = size(C,1);
        Inn = sparse(1:n,1:n,1,n,n);
        C = C + Inn.*gp.jitterSigma2;
    end
end