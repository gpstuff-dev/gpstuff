function [K, C] = gp_trvar(gp, x1)
% GP_TRVAR     Evaluate training variance vector. 
%
%         Description
%         K = GP_TRVAR(GP, TX) takes in Gaussian process GP and matrix
%         TX that contains training input vectors to GP. Returns 
%         noiseless variance vector K. Every element i of K contains  
%         variance of input i in TX.
%
%         [K, C] = GP_TRVAR(GP, TX) returns also the noisy
%         variance vector C.

% Copyright (c) 2006 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

[n,m]=size(x1);
n1 = n+1;
ncf = length(gp.cf);

% Evaluate the covariance without noise
K = 0;
for i=1:ncf
    gpcf = gp.cf{i};
    K = K + feval(gpcf.fh_trvar, gpcf, x1);
end

if ~isempty(gp.jitterSigmas)
    K = K + gp.jitterSigmas.^2;
end

if nargout >1
  C=K;
  
  % Add noise to the covariance
  if isfield(gp, 'noise')
    nn = length(gp.noise);
    for i=1:nn
      noise = gp.noise{i};
      C = C + feval(noise.fh_trvar, noise, x1);
    end
  end
  C(C<eps)=0;
end
K(K<eps)=0;