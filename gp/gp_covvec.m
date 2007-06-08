function C = gp_covvec(gp, x1, x2, varargin)
% GPCOV     Evaluate covariance vector between two input vectors. 
%
%         Description
%         C = GPCOV(GP, TX, X) takes in Gaussian process GP and two
%         matrixes TX and X that contain input vectors to GP. Returns 
%         covariance vector C, where every element i of C contains covariance
%         between input i in TX and i in X.
%

% Copyright (c) 2006 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

ncf = length(gp.cf);

C = 0;
for i=1:ncf
  gpcf = gp.cf{i};
  C = C + feval(gpcf.fh_covvec, gpcf, x1, x2, varargin);
end

if ~isempty(gp.jitterSigmas) & size(x1,1)==size(x2,1) 
    C(sum(x1==x2,2)==gp.nin) = C(sum(x1==x2,2)==gp.nin) + gp.jitterSigmas.^2;
end

C(C<eps)=0;
C = reshape(C,length(x1),1);