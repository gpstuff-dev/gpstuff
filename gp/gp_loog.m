function gcv = gp_loog(w, gp, x, y, varargin)
%   GP_LOOE       Evaluate the gradient of the leave one out predictive 
%                 density (GP_LOOE) in case of Gaussian observation model 
%
%	Description
%	G = GP_CVG(W, GP, X, Y) takes a full GP hyper-parameter vector
%       W, data structure GP a matrix X of input vectors and a matrix
%       Y of target vectors, and evaluates the gradient G of the leave
%       one out predictive density. Each row of X corresponds to one
%       input vector and each row of Y corresponds to one target
%       vector. NOTE! This parametrization works only for full GP!
%
%	See also
%       GP_CVE, GP_PAK, GP_UNPAK, GPCF_*

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

g = [];
gdata = [];
gprior = [];

% ============================================================
% FULL
% ============================================================
switch gp.type
  case 'FULL'   % A full GP
                % Evaluate covariance
    [K, C] = gp_trcov(gp,x);
    
    if issparse(C)
        invC = spinv(C);       % evaluate the sparse inverse
        LD = ldlchol(C);
        b = ldlsolve(LD,y);
    else
        invC = inv(C);        % evaluate the full inverse
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
        [DKff,gprior] = feval(gpcf.fh_ghyper, gpcf, x);
        
        % Evaluate the gradient with respect to covariance function parameters
        for i2 = 1:length(DKff)
            i1 = i1+1;  
            Z = invC*DKff{i2};
            Zb = Z*b;            
            gcv(i1) = - sum( (b.*Zb - 0.5*(1 + b.^2./diag(invC)).*diag(Z*invC))./diag(invC) )./n;
        end
        
    end

    % Evaluate the gradient from noise functions
    if isfield(gp, 'noise')
        nn = length(gp.noise);
        for i=1:nn
            noise = gp.noise{i};
            noise.type = gp.type;
            [DCff,gprior] = feval(noise.fh_ghyper, noise, x);
            
            for i2 = 1:length(DCff)
                i1 = i1+1;
                Z = invC*eye(n,n).*DCff{i2};
                Zb = Z*b;            
                gcv(i1) = - sum( (b.*Zb - 0.5*(1 + b.^2./diag(invC)).*diag(Z*invC))./diag(invC) )./n;
            end
        end
    end

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
    % SSGP
    % ============================================================
  case 'SSGP'

end
