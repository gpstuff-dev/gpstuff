function [K, C] = gp_trcov(gp, x1)
% GP_TRCOV     Evaluate training covariance matrix. 
%
%         Description
%         K = GP_TRCOV(GP, TX) takes in Gaussian process GP and matrix
%         TX that contains training input vectors to GP. Returns noiseless
%         covariance matrix K. Every element ij of K contains covariance 
%         between inputs i and j in TX.
%
%         [K, C] = GP_TRCOV(GP, TX) returns also the noisy
%         covariance matrix C.
%
%         For covariance function definition see manual or 
%         Neal R. M. Regression and Classification Using Gaussian 
%         Process Priors, Bayesian Statistics 6.

% Copyright (c) 2006 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

[n,m]=size(x1);
n1 = n+1;
ncf = length(gp.cf);

% Evaluate the covariance without noise
K = sparse(0);
is_sparse = 0;
for i=1:ncf
  gpcf = gp.cf{i};
  K = K + feval(gpcf.fh_trcov, gpcf, x1);
  if issparse(K) & is_sparse == 0
      is_sp = 1;
  end
end

if ~isempty(gp.jitterSigmas)
  K(1:n1:end)=K(1:n1:end) + gp.jitterSigmas.^2;
  if issparse(K) & is_sparse == 0
      is_sp = 1;
  end
end

if nargout > 1 
  C=K;
  % Add noise to the covariance
  if isfield(gp, 'noise')
    nn = length(gp.noise);
    for i=1:nn
      noise = gp.noise{i};
      C = C + feval(noise.fh_trcov, noise, x1);
    end
  end

  if is_sparse
      [I,J,c] = find(C);
      c(c<0) = 0;      
      C = sparse(I,J,c);
  end
end

if is_sparse
    [I,J,k] = find(K);
    k(k<0) = 0;
    K = sparse(I,J,k);
end

