function [C, Cinv] = gp_cov(gp, x1, x2, predcf)
% GPCOV     Evaluate covariance matrix between two input vectors. 
%
%         Description
%         C = GPCOV(GP, TX, X, PREDCF) takes in Gaussian process GP and two
%         matrixes TX and X that contain input vectors to GP. Returns 
%         covariance matrix C. Every element ij of C contains covariance 
%         between inputs i in TX and j in X. PREDCF is an array specifying
%         the indexes of covariance functions, which are used for forming the
%         matrix. If empty or not given, the matrix is formed with all functions.
%
%         [C, Cinv] = GPCOV(GP, TX, X, PREDCF) returns also inverse of covariance.

% Copyright (c) 2006 Jarno Vanhatalo

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

% $$$ if ~isempty(gp.jitterSigma2) & size(x1,1)==size(x2,1) & x1==x2
% $$$   [n,m]=size(x1);
% $$$   n1 = n+1;
% $$$   C(1:n1:end)=C(1:n1:end)+gp.jitterSigma2.^2;
% $$$ end

% $$$ if issparse(C)
% $$$     [I,J,c] = find(C);
% $$$     c(c<eps) = 0;      
% $$$     C = sparse(I,J,c);
% $$$ else
% $$$     C(C<eps)=0;
% $$$ end
