function gloo = gp_loog(w, gp, x, y, varargin)
% GP_LOOG Evaluate the gradient of the mean negative log 
%         leave-one-out predictive density, assuming Gaussian 
%         observation model
%
%   Description
%     LOOG = GP_LOOG(W, GP, X, Y, PARAM) takes a hyper-parameter vector
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
        [DKff,gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x);
        %gprior=[gprior gprior_cf];
        
        % Evaluate the gradient with respect to covariance function parameters
        for i2 = 1:length(DKff)
            i1 = i1+1;  
            Z = invC*DKff{i2};
            Zb = Z*b;            
            gloo(i1) = - sum( (b.*Zb - 0.5*(1 + b.^2./diag(invC)).*diag(Z*invC))./diag(invC) )./n;
        end
        
    end

    % Evaluate the gradient from noise functions
    if isfield(gp, 'noise')
        nn = length(gp.noise);
        for i=1:nn
            noise = gp.noise{i};
            noise.type = gp.type;
            [DCff,gprior_ncf] = feval(noise.fh_ghyper, noise, x);
            %gprior=[gprior gprior_ncf];
            
            for i2 = 1:length(DCff)
                i1 = i1+1;
                Z = invC*eye(n,n).*DCff{i2};
                Zb = Z*b;            
                gloo(i1) = - sum( (b.*Zb - 0.5*(1 + b.^2./diag(invC)).*diag(Z*invC))./diag(invC) )./n;
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

end
