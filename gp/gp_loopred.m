function [Eft, Varft, Eyt, Varyt, pyt] = gp_loopred(gp, x, y)
%GP_LOOPRED  Leave-one-out predictions assuming Gaussian observation model
%
%  Description
%    [EFT, VARFT, EYT, VARYT, PYT] = GP_LOOPRED(GP, X, Y) takes a
%    Gaussian process structure GP, a matrix X of input vectors and
%    a matrix Y of targets, and evaluates the leave-one-out
%    predictive distribution at inputs X. Returns a posterior mean
%    EFT and variance VARFT of latent variables, the posterior
%    predictive mean EYT and variance VARYT of observations, and
%    posterior predictive density PYT at input locations X.
%
%  References:
%    S. Sundararajan and S. S. Keerthi (2001). Predictive
%    Approaches for Choosing Hyperparameters in Gaussian Processes. 
%    Neural Computation 13:1103-1118.
%
%  See also
%   GP_G, GPCF_*, GP_SET, GP_PAK, GP_UNPAK
%

% Copyright (c) 2008-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

% Nothing to parse, but check the arguments anyway
ip=inputParser;
ip.FunctionName = 'GP_LOOPRED';
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.parse(gp, x, y);

if isfield(gp,'mean') & ~isempty(gp.mean.meanFuncs)
  error('GP_LOOPRED: Mean functions not yet supported');
end

% First Evaluate the data contribution to the error
switch gp.type
    % ============================================================
    % FULL GP (and compact support GP)
    % ============================================================
  case 'FULL'   % A full GP
    [K, C] = gp_trcov(gp, x);

    if issparse(C)
      iC = spinv(C); % evaluate the sparse inverse
      LD = ldlchol(C);
      b = ldlsolve(LD,y);
      myy = y - b./full(diag(iC));
      sigma2 = 1./full(diag(iC));
    else
      iC= inv(C);    % evaluate the full inverse
      b=C\y;
      myy = y - b./diag(iC);
      sigma2 = 1./diag(iC);
    end
    Eft = myy;
    Varft = sigma2-gp.lik.sigma2;
    Eyt = myy;
    Varyt = sigma2;
    pyt = exp(-0.5 * (log(2*pi) + log(sigma2) + (y-myy).^2./sigma2));
    
    % ============================================================
    % FIC
    % ============================================================
  case 'FIC'
    error('GP_LOOPRED is not implemented for FIC!')
    
    % ============================================================
    % PIC
    % ============================================================
  case {'PIC' 'PIC_BLOCK'}
    error('GP_LOOPRED is not implemented for PIC!')
    
    % ============================================================
    % CS+FIC
    % ============================================================
  case 'CS+FIC'
    error('GP_LOOPRED is not implemented for CS+FIC!')
        
    % ============================================================
    % SSGP
    % ============================================================    
  case 'SSGP'
    error('GP_LOOPRED is not implemented for SSGP!')
    
  otherwise
    error('Unknown type of Gaussian process!')
end

end
