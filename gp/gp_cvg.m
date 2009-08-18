function gcv = gp_cvg(w, gp, x, t, param, varargin)
%GP_CVG   Evaluate gradient of energy for Gaussian Process
%
%	Description
%	G = GP_CVG(W, GP, X, Y) takes a full GP hyper-parameter vector W,
%       data structure GP a matrix X of input vectors and a matrix Y
%       of target vectors, and evaluates the gradient G of the energy function. 
%	Each row of X corresponds to one input vector and each row of Y 
%       corresponds to one target vector. NOTE! This parametrization works 
%       only for full GP!
%
%	G = GP_G(W, GP, P, Y, PARAM) in case of sparse model takes also
%       string PARAM defining the parameters to take the gradients with
%       respect to. Possible parameters are 
%       'hyper'          = hyperparameters
%       'inducing'       = inducing inputs 
%       'hyper+inducing' = hyperparameters and inducing inputs
%
%	See also
%       GP_CVE, GP_PAK, GP_UNPAK, GPCF_*

% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

gp=gp_unpak(gp, w, param);       % unpak the parameters
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
            b = ldlsolve(LD,t);
        else
            invC = inv(C);        % evaluate the full inverse
            b = C\t;
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
            [DKff,gprior] = feval(gpcf.fh_ghyper, gpcf, x, t);
            i1 = i1+1;
            i2 = 1;
            
            % Evaluate the gradient with respect to magnSigma
            Z = invC*DKff{i2};
            Zb = Z*b;            
            gcv(i1) = - sum( (b.*Zb - 0.5*(1 + b.^2./diag(invC)).*diag(Z*invC))./diag(invC) )./n;
                         
            if isfield(gpcf.p.lengthScale, 'p') && ~isempty(gpcf.p.lengthScale.p)
                i1 = i1+1;
                if any(strcmp(fieldnames(gpcf.p.lengthScale.p),'nu'))
                    i1 = i1+1;
                end
            end

            % Evaluate the gradient with respect to lengthScale
            for i2 = 2:length(DKff)
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
                i1 = i1+1;
                
                noise = gp.noise{i};
                noise.type = gp.type;
                [DCff,gprior] = feval(noise.fh_ghyper, noise, x, t);
                
                Z = invC*eye(n,n).*DCff;
                Zb = Z*b;            
                gcv(i1) = - sum( (b.*Zb - 0.5*(1 + b.^2./diag(invC)).*diag(Z*invC))./diag(invC) )./n;
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

switch param
    case 'inducing'
        % Evaluate here the gradient from prior
        g = gdata_ind;
    case 'hyper+inducing'
        % Evaluate here the gradient from prior
        g = [g gdata_ind];
end
end
