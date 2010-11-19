function eloo = gp_looe(w, gp, x, y, varargin)
%GP_LOOE  Evaluate the mean negative log leave-one-out predictive 
%         density, assuming Gaussian observation model.
%
%  Description
%    LOOE = GP_LOOE(W, GP, X, Y, PARAM) takes a parameter vector W,
%    Gaussian process structure GP, a matrix X of input vectors and
%    a matrix Y of targets, and evaluates the mean negative log
%    leave-one-out predictive density
%       LOOE  = - 1/n sum log p(Y_i | X, Y_{\i}, th)
%    where th represents the parameters (lengthScale,
%    magnSigma2...), X is inputs and Y is observations.
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

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.


gp=gp_unpak(gp, w);
ncf = length(gp.cf);
n=length(x);

if isfield(gp,'mean') & ~isempty(gp.mean.meanFuncs)
  error('GP_LOOE: Mean functions not yet supported');
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
    eloo = 0.5 * (log(2*pi) + sum(log(sigma2) + (y-myy).^2./sigma2)./n );
    
    % ============================================================
    % FIC
    % ============================================================
  case 'FIC'
    error('GP_LOOE is not implemented for FIC!')
    
    % ============================================================
    % PIC
    % ============================================================
  case {'PIC' 'PIC_BLOCK'}
    error('GP_LOOE is not implemented for PIC!')
    
    % ============================================================
    % CS+FIC
    % ============================================================
  case 'CS+FIC'
    error('GP_LOOE is not implemented for CS+FIC!')
        
    % ============================================================
    % SSGP
    % ============================================================    
  case 'SSGP'
    error('GP_LOOE is not implemented for SSGP!')
    
  otherwise
    error('Unknown type of Gaussian process!')
end

end
