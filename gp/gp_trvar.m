function [K, C] = gp_trvar(gp, x1, predcf)
% GP_TRVAR     Evaluate training variance vector. 
%
%         Description
%          K = GP_TRVAR(GP, TX, PREDCF) takes in Gaussian process GP
%          and matrix TX that contains training input vectors to
%          GP. Returns noiseless variance vector K (diagonal of the
%          covariance matrix returned by gp_trcov), which is formed as
%          a sum of the variances from covariance functions in GP.cf
%          array. Every element ij of K contains covariance between
%          inputs i and j in TX. PREDCF is an array specifying the
%          indexes of covariance functions, which are used for forming
%          the matrix. If not given, the matrix is formed with all
%          functions.
%
%          [K, C] = GP_TRCOV(GP, TX, PREDCF) returns also the noisy
%          variance vector C, which is sum of K and the variances from
%          the covariance functions is gp.noise array.

% Copyright (c) 2006 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

[n,m]=size(x1);
n1 = n+1;
ncf = length(gp.cf);

if isfield(gp,'derivobs') && gp.derivobs
  % Derivative observations
    
    % Evaluate the covariance without noise
    K = 0;
    gpcf = gp.cf{1};
    %right sized vector for the tr_var
    x2=zeros(m*n+n,1);
    K = K + feval(gpcf.fh_trvar, gpcf, x2);

    if ~isempty(gp.jitterSigma2)
        K = K + gp.jitterSigma2;
    end

    if nargout >1
       C=K;
     % Add noise to the covariance
       if isfield(gp, 'noisef')
          nn = length(gp.noisef);
          for i=1:nn
              noisef = gp.noisef{i};
              C = C + feval(noisef.fh_trvar, noisef, x2);
          end
       end
       C(C<eps)=0;
    end
    K(K<eps)=0;
        
else

    % Evaluate the covariance without noise
    K = 0;
    if nargin < 3 || isempty(predcf)
        predcf = 1:ncf;
    end      
    for i=1:length(predcf)
        gpcf = gp.cf{predcf(i)};
        K = K + feval(gpcf.fh_trvar, gpcf, x1);
    end

    if ~isempty(gp.jitterSigma2)
        K = K + gp.jitterSigma2;
    end

    if nargout >1
      C=K;

      % Add noise to the covariance
      if isfield(gp, 'noisef')
        nn = length(gp.noisef);
        for i=1:nn
          noisef = gp.noisef{i};
          C = C + feval(noisef.fh_trvar, noisef, x1);
        end
      end
    end
end