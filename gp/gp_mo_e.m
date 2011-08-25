function [e, edata, eprior] = gp_mo_e(w, gp, x, y, varargin)
%GP_E  Evaluate the energy function (un-normalized negative marginal
%      log posterior)
%
%  Description
%    E = GP_E(W, GP, X, Y, OPTIONS) takes a Gaussian process
%    structure GP together with a matrix X of input vectors and a
%    matrix Y of targets, and evaluates the energy function E. Each
%    row of X corresponds to one input vector and each row of Y
%    corresponds to one target vector.
%
%    [E, EDATA, EPRIOR] = GP_E(W, GP, X, Y, OPTIONS) also returns
%    the data and prior components of the total energy.
%
%    The energy is minus log posterior cost function:
%        E = EDATA + EPRIOR 
%          = - log p(Y|X, th) - log p(th),
%    where th represents the parameters (lengthScale,
%    magnSigma2...), X is inputs and Y is observations (regression)
%    or latent values (non-Gaussian likelihood).
%
%    OPTIONS is optional parameter-value pair
%      z - optional observed quantity in triplet (x_i,y_i,z_i)
%          Some likelihoods may use this. For example, in case of
%          Poisson likelihood we have z_i=E_i, that is, expected
%          value for ith case.
%
%  See also
%    GP_G, GPCF_*, GP_SET, GP_PAK, GP_UNPAK
%

% Copyright (c) 2006-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

if isfield(gp,'latent_method') && ~strcmp(gp.latent_method,'MCMC')
  % use inference specific methods
  % not the nicest way of doing this, but quick solution
  switch gp.latent_method
    case 'Laplace'
      switch gp.lik.type
%         case 'Softmax'
%           fh_e=@gpla_softmax_e;
        case {'Softmax' 'Multinom'}
          fh_e=@gpla_mo_e;
        otherwise
          fh_e=@gpla_e;
      end
    case 'EP'
      fh_e=@gpep_e;
  end
  switch nargout 
    case {0 1}
      [e] = fh_e(w, gp, x, y, varargin{:});
    case 2
      [e, edata] = fh_e(w, gp, x, y, varargin{:});
    case 3
      [e, edata, eprior] = fh_e(w, gp, x, y, varargin{:});
  end
  return
end

ip=inputParser;
ip.FunctionName = 'GP_E';
ip.addRequired('w', @(x) isempty(x) || ...
               isvector(x) && isreal(x) && all(isfinite(x)));
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.parse(w, gp, x, y, varargin{:});
z=ip.Results.z;

gp=gp_unpak(gp, w);
ncf = length(gp.cf);
[n,nout]=size(y);

% First Evaluate the data contribution to the error
switch gp.type
  % ============================================================
  % FULL GP (and compact support GP)
  % ============================================================
  case 'FULL'   % A full GP
      
    if isfield(gp, 'comp_cf')  % own covariance for each ouput component
        multicf = true;
        if length(gp.comp_cf) ~= nout
            error('GPLA_MO_E: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
        end
    else
        multicf = false;
    end
    
    b = zeros(n,nout);
    zc=0;
    if multicf
        for i1=1:nout
            [~, C] = gp_trcov(gp, x, gp.comp_cf{i1});
            [L,notpositivedefinite]=chol(C,'lower');
            if notpositivedefinite
              [e, edata, eprior] = set_output_for_notpositivedefinite();
              return
            end
            b(:,i1) = L\y(:,i1);
            zc = zc + sum(log(diag(L)));
        end
    else
        [~, C] = gp_trcov(gp, x);
        [L,notpositivedefinite]=chol(C,'lower');
        if notpositivedefinite
          [e, edata, eprior] = set_output_for_notpositivedefinite();
          return
        end
        for i1=1:nout
          b(:,i1) = L\y(:,i1);
          zc = zc + sum(log(diag(L)));
        end
    end
        
    % Are there specified mean functions
    edata = 0.5*n.*log(2*pi) + zc + 0.5*sum(sum(b.*b));
    
    % ============================================================
    % FIC
    % ============================================================
  case 'FIC'
    % ============================================================
    % PIC
    % ============================================================
  case {'PIC' 'PIC_BLOCK'}
    % ============================================================
    % CS+FIC
    % ============================================================
  case 'CS+FIC'
    % ============================================================
    % DTC/VAR
    % ============================================================
  case {'DTC' 'VAR' 'SOR'}
    
  otherwise
    error('Unknown type of Gaussian process!')
end

% ============================================================
% Evaluate the prior contribution to the error from covariance functions
% ============================================================
eprior = 0;
if ~isempty(strfind(gp.infer_params, 'covariance'))
  for i=1:ncf
    gpcf = gp.cf{i};
    eprior = eprior -gpcf.fh.lp(gpcf);
  end
end

% ============================================================
% Evaluate the prior contribution to the error from Gaussian likelihood
% ============================================================
if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh,'trcov') && isfield(gp.lik, 'p')
  % a Gaussian likelihood
  lik = gp.lik;
  eprior = eprior -lik.fh.lp(lik);
end

% ============================================================
% Evaluate the prior contribution to the error from the inducing inputs
% ============================================================
if ~isempty(strfind(gp.infer_params, 'inducing'))
  if isfield(gp, 'p') && isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
    for i = 1:size(gp.X_u,1)
      if iscell(gp.p.X_u) % Own prior for each inducing input
        pr = gp.p.X_u{i};
        eprior = eprior -pr.fh.lp(gp.X_u(i,:), pr);
      else
        eprior = eprior -gp.p.X_u.fh.lp(gp.X_u(i,:), gp.p.X_u);
      end
    end
  end
end

e = edata + eprior;

end

function [e, edata, eprior] = set_output_for_notpositivedefinite()
  % Instead of stopping to chol error, return NaN
  e = NaN;
  edata = NaN;
  eprior = NaN;
end

