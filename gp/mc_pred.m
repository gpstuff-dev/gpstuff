function [Ef, Varf, Ey, Vary, py] = mc_pred(gp, x, y, xt, varargin)
%MC_PRED    Predictions with Gaussian Process MCMC approximation.
%
%	Description
%	[EF, VARF] = MC_PRED(RECGP, X, Y, XT, OPTIONS) takes     
%        a Gaussian processes record structure RECGP (returned by
%        gp_mc) together with a matrix XT of input vectors, matrix X
%        of training inputs and vector Y of training targets. Returns
%        matrices EF and VARF that contain the posterior predictive
%        means and variances of latent variables for Gaussian
%        processes stored in RECGP. The i'th column contains mean and
%        variance for i'th sample.
%
%        If likelihood is other than Gaussian also the latent
%        variables are sampled and EF and VARF contain the marginal
%        mean and variance, E[f|D] and Var[f|D]. With Gaussian
%        regression the case is different since we have sampled from
%        the marginal posterior of hyperparameters p(th|D). Then the
%        i'th column of Ef and Varf contain the conditional predictive
%        mean and variance for the latent variables given the i'th
%        hyperparameter sample th_i in RECGP. That is:
%       
%                  Ef(:,i) = E[f | x, y, th_i]
%                Varf(:,i) = Var[f | x, y, th_i]
%    
%        The marginal posterior mean and variance can be evaluated from
%        these as follows:
%
%             E[f | xt, y] = E[ E[f | x, y, th] ]
%                          = mean(Ef, 2)
%           Var[f | xt, y] = E[ Var[f | x, y, th] ] + Var[ E[f | x, y, th] ]
%                          = mean(Varf,2) + var(Ef,0,2)
%   
%        OPTIONS is an optional parameter-value pair
%         'predcf' is index vector telling which covariance functions are 
%                  used for prediction. Default is all (1:gpcfn). See
%                  additional information below.
%         'tstind' is a vector/cell array defining, which rows of X belong 
%                  to which training block in *IC type sparse
%                  models. Deafult is []. In case of PIC, a cell array
%                  containing index vectors specifying the blocking
%                  structure for test data.  IN FIC and CS+FIC a
%                  vector of length n that points out the test inputs
%                  that are also in the training set (if none, set
%                  TSTIND = [])
%         'yt'     is optional observed yt in test points (see below)
%         'z'      is optional observed quantity in triplet (x_i,y_i,z_i)
%                  Some likelihoods may use this. For example, in case
%                  of Poisson likelihood we have z_i=E_i, that is,
%                  expected value for ith case.
%         'zt'     is optional observed quantity in triplet (xt_i,yt_i,zt_i)
%                  Some likelihoods may use this. For example, in case
%                  of Poisson likelihood we have z_i=E_i, that is, the
%                  expected value for the ith case.
%       
%       [EF, VARF, EY, VARY] = GP_PREDS(GP, X, Y, XT, OPTIONS) 
%        returns also the predictive means and variances for test observations
%        at input locations XT. That is,
%
%                    Ey(:,i) = E[y | xt, x, y, th_i]
%                  Vary(:,i) = Var[y | xt, x, y, th_i]
%
%       where the latent variables have been marginalized out.
%
%	[EF, VARF, EY, VARY, PYT] = GP_PRED(GP, X, Y, XT, 'yt', YT, OPTIONS) 
%       returns also the predictive density PY of the observations Y
%       at input locations XT. This can be used for example in the
%       cross-validation. Here Y has to be vector.
%
%       NOTE! In case of FIC and PIC sparse approximation the
%       prediction for only some PREDCF covariance functions is
%       just an approximation since the covariance functions are
%       coupled in the approximation and are not strictly speaking
%       additive anymore.
%
%       For example, if you use covariance such as K = K1 + K2 your
%       predictions Ef1 = mc_pred(GP, X, Y, X, 'predcf', 1) and 
%       Ef2 = mc_pred(gp, x, y, x, 'predcf', 2) should sum up to 
%       Ef = mc_pred(gp, x, y, x). That is Ef = Ef1 + Ef2. With 
%       FULL model this is true but with FIC and PIC this is true only 
%       approximately. That is Ef \approx Ef1 + Ef2.
%
%       With CS+FIC the predictions are exact if the PREDCF
%       covariance functions are all in the FIC part or if they are
%       CS covariances.
%
%       NOTE! When making predictions with a subset of covariance
%       functions with FIC approximation the predictive variance
%       can in some cases be ill-behaved i.e. negative or
%       unrealistically small. This may happen because of the
%       approximative nature of the prediction.
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
                Varf(Varf<0) = min(min(abs(Varf))); % Ensure positiviness, which may be a problem with FIC
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