function gloo = gp_loog(w, gp, x, y, varargin)
% GP_LOOG Evaluate the gradient of the mean negative log 
%         leave-one-out predictive density, assuming Gaussian 
%         observation model
%
%   Description
%     LOOG = GP_LOOG(W, GP, X, Y, PARAM) takes a parameter vector
%     W, Gaussian process structure GP, a matrix X of input vectors and
%     a matrix Y of targets, and evaluates the gradient of the mean 
%     negative log leave-one-out predictive density (see GP_LOOE).
%
%   References:
%     S. Sundararajan and S. S. Keerthi (2001). Predictive
%     Approaches for Choosing Hyperparameters in Gaussian Processes. 
%     Neural Computation 13:1103-1118.
%
%	See also
%       GP_LOOE, GP_PAK, GP_UNPAK, GPCF_*

% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GP_LOOE';
ip.addRequired('w', @(x) isvector(x) && isreal(x) && all(isfinite(x)));
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.parse(w, gp, x, y, varargin{:});

gp=gp_unpak(gp, w);       % unpak the parameters
ncf = length(gp.cf);
n=size(x,1);

if isfield(gp,'mean') & ~isempty(gp.mean.meanFuncs)
  error('GP_LOOE: Mean functions not yet supported');
end

g = [];
gloo = [];

switch gp.type
  case 'FULL'
    % ============================================================
    % FULL
    % ============================================================
    % Evaluate covariance
    [K, C] = gp_trcov(gp,x);
    
    if issparse(C)
      % evaluate the sparse inverse
      invC = spinv(C);
      LD = ldlchol(C);
      b = ldlsolve(LD,y);
    else
      % evaluate the full inverse
      invC = inv(C);        
      b = C\y;
    end

    % Get the gradients of the covariance matrices and gprior
    % from gpcf_* structures and evaluate the gradients
    for i=1:ncf
      i1=0;
      if ~isempty(gprior)
        i1 = length(gprior);
      end
      
      gpcf = gp.cf{i};
      gpcf.GPtype = gp.type;
      DKff = feval(gpcf.fh.cfg, gpcf, x);
      
      % Evaluate the gradient with respect to covariance function parameters
      for i2 = 1:length(DKff)
        i1 = i1+1;  
        Z = invC*DKff{i2};
        Zb = Z*b;            
        gloo(i1) = - sum( (b.*Zb - 0.5*(1 + b.^2./diag(invC)).*diag(Z*invC))./diag(invC) )./n;
      end
      
    end

    % Evaluate the gradient from Gaussian likelihood function
    if isfield(gp.lik.fh,'trcov')
      DCff = feval(lik.fh.cfg, lik, x);
      for i2 = 1:length(DCff)
        i1 = i1+1;
        Z = invC*eye(n,n).*DCff{i2};
        Zb = Z*b;            
        gloo(i1) = - sum( (b.*Zb - 0.5*(1 + b.^2./diag(invC)).*diag(Z*invC))./diag(invC) )./n;
      end
    end

  case 'FIC'
    % ============================================================
    % FIC
    % ============================================================
    
  case {'PIC' 'PIC_BLOCK'}
    % ============================================================
    % PIC
    % ============================================================
    
  case 'CS+FIC'
    % ============================================================
    % CS+FIC
    % ============================================================
    
  case 'SSGP'
    % ============================================================
    % SSGP
    % ============================================================

end
