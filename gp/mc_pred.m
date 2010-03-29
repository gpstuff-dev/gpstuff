function [Ef, Varf, Ey, Vary, py] = mc_pred(gp, x, y, xt, varargin)
%MC_PRED    Predictions with Gaussian Process MCMC solution.
%
%	Description
%	[Ef, Varf] = MC_PRED(RECGP, X, Y, XT, PREDCF, TSTIND) takes a Gaussian 
%       processes record structure RECGP (returned by gp_mc) together with a matrix XT 
%       of input vectors, matrix X of training inputs and vector Y of training targets. 
%       Returns matrices Ef and Varf that contain the predictive means and variances for 
%       Gaussian processes stored in RECGP. The i'th column of Ef and Varf contain the 
%       conditional predictive mean and variance for the latent variables given the i'th
%       hyperparameter sample th_i in RECGP. That is:
%       
%                    Ef(:,i) = E[f | x, y, th_i]
%                  Varf(:,i) = Var[f | x, y, th_i]
%    
%       The marginal posterior mean and variance can be evaluated from these as follows:
%
%                    E[f | xt, y] = E[ E[f | x, y, th] ]
%                                = mean(Ef, 2)
%                  Var[f | xt, y] = E[ Var[f | x, y, th] ] + Var[ E[f | x, y, th] ]
%                                = mean(Varf,2) + var(Ef,0,2)
%   
%       Each row of XT corresponds to one input vector and each row of Y corresponds to one 
%       output. PREDCF is an array specifying the indexes of covariance functions, which 
%       are used for making the prediction (others are considered noise). TSTIND is, in 
%       case of PIC, a cell array containing index vectors specifying the blocking 
%       structure for test data, or in FIC and CS+FI a vector of length n that points out 
%       the test inputs that are also in the training set (if none, set TSTIND = []).
%       
%       [Ef, Varf, Ey, Vary] = GP_PREDS(GP, X, Y, XT, PREDCF, TSTIND) returns also the 
%       predictive means and variances for observations at input locations XT. That is,
%
%                    Ey(:,i) = E[y | xt, x, y, th_i]
%                  Vary(:,i) = Var[y | xt, x, y, th_i]
%
%       where the latent variables have been marginalized out.
%
%	[Ef, Varf, Ey, Vary, PY] = GP_PRED(GP, X, Y, XT, PREDCF, TSTIND, Y) returns also the 
%       predictive density PY of the observations Y at input locations XT. This can be used for
%       example in the cross-validation. Here Y has to be vector.
%
%
%	See also
%	GP, GP_PAK, GP_UNPAK, GP_PRED

% Copyright (c) 2007-2010 Jarno Vanhatalo
    
% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    
    ip=inputParser;
    ip.FunctionName = 'MC_PRED';
    ip.addRequired('gp',@isstruct);
    ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addRequired('xt',  @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('yt', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('predcf', [], @(x) isempty(x) || ...
                     isvector(x) && isreal(x) && all(isfinite(x)&x>0))
    ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                     (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
    ip.parse(gp, x, y, xt, varargin{:});
    yt=ip.Results.yt;
    zt=ip.Results.zt;
    z=ip.Results.z;
    predcf=ip.Results.predcf;
    tstind=ip.Results.tstind;
    
    tn = size(x,1);
    if nargin < 4
        error('Requires at least 4 arguments');
    end

    if nargout > 4 && isempty(yt)
        error('mc_pred -> If py is wanted you must provide the vector y as 7''th input.')
    end
            
    nin  = size(x,2);
    nout = 1;
    nmc=size(gp.etr,1);
    
    % Non-Gaussian likelihood. Thus latent variables should be used in place of observations
    if isfield(gp, 'latentValues')
        y = gp.latentValues';
    else 
        y = repmat(y,1,nmc);
    end

    if strcmp(gp.type, 'PIC_BLOCK') || strcmp(gp.type, 'PIC')
        ind = gp.tr_index;           % block indeces for training points
        gp = rmfield(gp,'tr_index');
    end
    
    % loop over all samples
    for i1=1:nmc
        Gp = take_nth(gp,i1);
        
        switch gp.type            
          case 'FULL' 

          case {'FIC' 'CS+FIC'} 
            % Reformat the inducing inputs 
            u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
            Gp.X_u = u;

          case {'PIC' 'PIC_BLOCK'}
            % Reformat the inducing inputs 
            u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
            Gp.X_u = u;
            Gp.tr_index = ind;
        end
        
        if nargout < 3
            [Ef(:,i1), Varf(:,i1)] = gp_pred(Gp, x, y(:,i1), xt, 'predcf', predcf, 'tstind', tstind);
        else 
            if isfield(gp, 'latentValues')
                [Ef(:,i1), Varf(:,i1)] = gp_pred(Gp, x, y(:,i1), xt, 'predcf', predcf, 'tstind', tstind);
                Varf(Varf<0) = min(abs(Varf)); % Ensure positiviness, which may be a problem with FIC
                if isempty(yt)
                    [Ey(:,i1), Vary(:,i1)] = feval(Gp.likelih.fh_predy, Gp.likelih, Ef(:,i1), Varf(:,i1), [], zt);
                else
                    [Ey(:,i1), Vary(:,i1), py(:,i1)] = feval(Gp.likelih.fh_predy, Gp.likelih, Ef(:,i1), Varf(:,i1), yt, zt);
                end
            else
                if nargout < 5
                    [Ef(:,i1), Varf(:,i1), Ey(:,i1), Vary(:,i1)] = gp_pred(Gp, x, y(:,i1), xt, 'predcf', predcf, 'tstind', tstind);
                else
                    [Ef(:,i1), Varf(:,i1), Ey(:,i1), Vary(:,i1), py(:,i1)] = gp_pred(Gp, x, y(:,i1), xt, 'predcf', predcf, 'tstind', tstind, 'yt', yt); 
                end
            end            
        end
    end
end

function x = take_nth(x,nth)
%TAKE_NTH    Take n'th parameters from MCMC-chains
%
%   x = take_nth(x,n) returns chain containing only
%   n'th simulation sample 
%
%   See also
%     THIN, JOIN
    
% Copyright (c) 1999 Simo Sï¿½rkkï¿½
% Copyright (c) 2000 Aki Vehtari
% Copyright (c) 2006 Jarno Vanhatalo
    
% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.
    
    if nargin < 2
        n = 1;
    end
    
    [m,n]=size(x);

    if isstruct(x)
        if (m>1 | n>1)
            % array of structures
            for i=1:(m*n)
                x(i) = take_nth(x(i),n);
            end
        else
            % single structure
            names = fieldnames(x);
            for i=1:size(names,1)
                value = getfield(x,names{i});
                if length(value) > 1
                    x = setfield(x,names{i},take_nth(value,nth));
                elseif iscell(value)
                    x = setfield(x,names{i},{take_nth(value{1},nth)});
                end
            end
        end
    elseif iscell(x)
        % cell array
        for i=1:(m*n)
            x{i} = take_nth(x{i},nth);
        end
    elseif m > 1
        x = x(nth,:);
    end
end