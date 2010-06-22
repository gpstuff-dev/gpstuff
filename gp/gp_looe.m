function [ecv] = gp_looe(w, gp, x, y, varargin)
%GP_LOOE   Evaluate the leave one out predictive density in case of
%          Gaussian observation model
%
%	Description
%	ECV = GP_CVE(W, GP, X, Y, PARAM) takes a Gaussian process data
%       structure GP together with a matrix X of input vectors and a
%       matrix Y of targets, and evaluates the leave one out
%       predictive density E.  Each row of X corresponds to one input
%       vector and each row of Y corresponds to one target vector.
%
%       The energy is minus log LOO-CV cost function:
%            ECV  = - sum log p(Y_i | X, Y_{\i}, th),
%       where th represents the hyperparameters (lengthScale, magnSigma2...), 
%       X is inputs and Y is observations (regression) or latent values 
%       (non-Gaussian likelihood).
%
%	See also
%	GP_G, GPCF_*, GP_INIT, GP_PAK, GP_UNPAK, GP_FWD
%

% Copyright (c) 2008-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.


gp=gp_unpak(gp, w);
ncf = length(gp.cf);
n=length(x);

% First Evaluate the data contribution to the error
switch gp.type
    % ============================================================
    % FULL GP (and compact support GP)
    % ============================================================
  case 'FULL'   % A full GP
    [K, C] = gp_trcov(gp, x);

    if issparse(C)
        LD = ldlchol(C);
        edata = 0.5*(n.*log(2*pi) + sum(log(diag(LD))) + t'*ldlsolve(LD,t));
    else
        b=C\y;
        iC= inv(C);        
        myy = y - b./diag(iC);
        sigma2 = 1./diag(iC);
        
        ecv = 0.5 * (log(2*pi) + sum(log(sigma2) + (y-myy).^2./sigma2)./n );
        
    end
   
    % ============================================================
    % FIC
    % ============================================================
  case 'FIC'
    error('GP_CVE is not implemented for FIC!')
    
    % ============================================================
    % PIC
    % ============================================================
  case {'PIC' 'PIC_BLOCK'}
    error('GP_CVE is not implemented for PIC!')
    
    % ============================================================
    % CS+FIC
    % ============================================================
  case 'CS+FIC'
    error('GP_CVE is not implemented for CS+FIC!')
        
    % ============================================================
    % SSGP
    % ============================================================    
  case 'SSGP'
    error('GP_CVE is not implemented for SSGP!')
    
  otherwise
    error('Unknown type of Gaussian process!')
end

% ============================================================
% Evaluate the prior contribution to the error from covariance functions
% ============================================================

end
