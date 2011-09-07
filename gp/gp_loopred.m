function [Eft, Varft, lpyt, Eyt, Varyt] = gp_loopred(gp, x, y, varargin)
%GP_LOOPRED  Leave-one-out predictions assuming Gaussian observation model
%
%  Description
%    [EFT, VARFT, LPYT, EYT, VARYT] = GP_LOOPRED(GP, X, Y) takes a
%    Gaussian process structure GP, a matrix X of input vectors and
%    a matrix Y of targets, and evaluates the leave-one-out
%    predictive distribution at inputs X. Returns a posterior mean
%    EFT and variance VARFT of latent variables, the posterior
%    predictive mean EYT and variance VARYT of observations, and logarithm
%    of the posterior predictive density PYT at input locations X.
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
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

if iscell(gp) || numel(gp.jitterSigma2)>1 || isfield(gp,'latent_method')
  % use inference specific methods
  if numel(gp.jitterSigma2)>1
    error('Leave-one-out not yet supported for MCMC')
  elseif iscell(gp) && ~isfield(gp{1},'latent_method')
    error('Leave-one-out not yet supported for GP_IA and Gaussian likelihood')
  elseif isfield(gp,'latent_method') || ...
      (iscell(gp) && isfield(gp{1},'latent_method'))
    if iscell(gp)
      latent_method=gp{1}.latent_method;
      lik_type=gp{1}.lik.type;
    else
      latent_method=gp.latent_method;
      lik_type=gp.lik.type;
    end
    switch latent_method
      case 'Laplace'
        switch lik_type
          case {'Multinom' 'Softmax' 'Zinegbin' 'Coxph' 'Logitgp'}
            error('Laplace leave-one-out not yet supported for likelihoods with non-diagonal W')
          otherwise
            fh_pred=@gpla_loopred;
        end
      case 'EP'
        fh_pred=@gpep_loopred;
      case 'MCMC'
        error('Leave-one-out not yet supported for MCMC')
    end
  else
    error('Logical error by coder of this function!')
  end
  switch nargout
    case 1
      [Eft] = fh_pred(gp, x, y, varargin{:});
    case 2
      [Eft, Varft] = fh_pred(gp, x, y, varargin{:});
    case 3
      [Eft, Varft, lpyt] = fh_pred(gp, x, y, varargin{:});
    case 4
      [Eft, Varft, lpyt, Eyt] = fh_pred(gp, x, y, varargin{:});
    case 5
      [Eft, Varft, lpyt, Eyt, Varyt] = fh_pred(gp, x, y, varargin{:});
  end
  return
end

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
    lpyt = (-0.5 * (log(2*pi) + log(sigma2) + (y-myy).^2./sigma2));
    
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
