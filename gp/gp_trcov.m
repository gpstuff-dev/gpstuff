function [K, C] = gp_trcov(gp, x1, predcf)
% GP_TRCOV     Evaluate training covariance matrix (gp_cov + noise covariance). 
%
%         Description
%          K = GP_TRCOV(GP, TX, PREDCF) takes in Gaussian process GP
%          and matrix TX that contains training input vectors to
%          GP. Returns noiseless covariance matrix K, which is formed
%          as a sum of the covarance matrices from covariance
%          functions in GP.cf array. Every element ij of K contains
%          covariance between inputs i and j in TX. PREDCF is an array
%          specifying the indexes of covariance functions, which are
%          used for forming the matrix. If not given, the matrix is
%          formed with all functions.
%
%          [K, C] = GP_TRCOV(GP, TX, PREDCF) returns also the noisy
%          covariance matrix C, which is sum of K and the covariance
%          functions is gp.noise array.

% Copyright (c) 2006-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

switch gp.type
  case {'FULL' 'FIC' 'PIC' 'PIC_BLOCK' 'CS+FIC' 'VAR' 'DTC' 'SOR'}
    [n,m]=size(x1);
    n1 = n+1;
    ncf = length(gp.cf);
    
    % Evaluate the covariance without noise
    K = sparse(0);
    if nargin < 3 || isempty(predcf)
        predcf = 1:ncf;
    end      
    for i=1:length(predcf)
        gpcf = gp.cf{predcf(i)};
        K = K + feval(gpcf.fh_trcov, gpcf, x1);
    end
    
    if ~isempty(gp.jitterSigma2)
        if issparse(K)
            K = K + sparse(1:n,1:n,gp.jitterSigma2,n,n);
        else
            K(1:n1:end)=K(1:n1:end) + gp.jitterSigma2;
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
    end

  case 'SSGP'
    [n,m]=size(x1);
    n1 = n+1;
    ncf = length(gp.cf);
    
    % Evaluate the covariance without noise
    gpcf = gp.cf{1};
    K = feval(gpcf.fh_trcov, gpcf, x1);

    if nargout > 1 
        C=sparse(0);
        % Add noise to the covariance
        if isfield(gp, 'noise')
            nn = length(gp.noise);
            for i=1:nn
                noise = gp.noise{i};
                C = C + feval(noise.fh_trcov, noise, x1);
            end
        end
    end   
end
