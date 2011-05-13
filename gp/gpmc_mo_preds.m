function [Ef, Varf, lpy, Ey, Vary] = gpmc_mo_preds(gp, x, y, xt, varargin)
%GPMC_PREDS  Predictions with Gaussian Process MCMC approximation.
%
%  Description
%    [EFS, VARFS] = GPMC_PREDS(RECGP, X, Y, XT, OPTIONS) takes a
%    Gaussian processes record structure RECGP (returned by gp_mc)
%    together with a matrix XT of input vectors, matrix X of
%    training inputs and vector Y of training targets. Returns
%    matrices EFS and VARFS that contain means and variances of the
%    conditional posterior predictive distributions given RECGP.
%    In case of non-Gaussian likelihood   
%  
%        Efs(:,i) = E[ f(xt) | f_i, th_i, x, y ]
%      Varfs(:,i) = Var[ f(xt) | f_i, th_i, x, y ]
%
%    and in case of Gaussian likelihood (f integrated analytically)
%  
%        Efs(:,i) = E[ f(xt) | th_i, x, y ]
%      Varfs(:,i) = Var[ f(xt) | th_i, x, y ]
%  
%    The marginal posterior mean and variance can be evaluated from
%    these as follows (See also MC_PRED):
%  
%        E[f | xt, y] = E[ E[f | x, y, th] ]
%                     = mean(Efs,2)
%      Var[f | xt, y] = E[ Var[f | x, y, th] ] + Var[ E[f | x, y, th] ]
%                     = mean(Varfs,2) + var(Efs,0,2)
%   
%    OPTIONS is an optional parameter-value pair
%      predcf - index vector telling which covariance functions are 
%               used for prediction. Default is all (1:gpcfn). See
%               additional information below.
%      tstind - a vector/cell array defining, which rows of X belong 
%               to which training block in *IC type sparse models. 
%               Deafult is []. In case of PIC, a cell array
%               containing index vectors specifying the blocking
%               structure for test data. IN FIC and CS+FIC a vector
%               of length n that points out the test inputs that
%               are also in the training set (if none, set TSTIND=[])
%      yt     - optional observed yt in test points (see below)
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case
%               of Poisson likelihood we have z_i=E_i, that is,
%               expected value for ith case.
%      zt     - optional observed quantity in triplet (xt_i,yt_i,zt_i)
%               Some likelihoods may use this. For example, in case
%               of Poisson likelihood we have z_i=E_i, that is, the
%               expected value for the ith case.
%       
%    [EFS, VARFS, LPYS] = GP_PREDS(RECGP, X, Y, XT, 'yt', YT, OPTIONS) 
%    returns also the predictive density PYS of the observations YT
%    at input locations XT given RECGP
%
%        Pys(:,i) = p(yt | xt, x, y, th_i)
%
%    [EFS, VARFS, LPYS, EYS, VARYS] = GP_PREDS(RECGP, X, Y, XT, OPTIONS) 
%    returns also the predictive means and variances for test
%    observations at input locations XT given RECGP
%
%        Eys(:,i) = E[y | xt, x, y, th_i]
%      Varys(:,i) = Var[y | xt, x, y, th_i]
%
%    where the latent variables have been marginalized out.
%
%
%     NOTE! In case of FIC and PIC sparse approximation the
%     prediction for only some PREDCF covariance functions is just
%     an approximation since the covariance functions are coupled
%     in the approximation and are not strictly speaking additive
%     anymore.
%
%     For example, if you use covariance such as K = K1 + K2 your
%     predictions Ef1 = mc_pred(GP, X, Y, X, 'predcf', 1) and Ef2 =
%     mc_pred(gp, x, y, x, 'predcf', 2) should sum up to Ef =
%     mc_pred(gp, x, y, x). That is Ef = Ef1 + Ef2. With FULL model
%     this is true but with FIC and PIC this is true only
%     approximately. That is Ef \approx Ef1 + Ef2.
%
%     With CS+FIC the predictions are exact if the PREDCF
%     covariance functions are all in the FIC part or if they are
%     CS covariances.
%
%     NOTE! When making predictions with a subset of covariance
%     functions with FIC approximation the predictive variance can
%     in some cases be ill-behaved i.e. negative or unrealistically
%     small. This may happen because of the approximative nature of
%     the prediction.
%
%  See also
%    MC_PRED, GP_PRED, GP_SET, GP_MC

% Copyright (c) 2007-2010 Jarno Vanhatalo
    
% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    
    ip=inputParser;
    ip.FunctionName = 'GPMC_PREDS';
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

    if nargout > 2 && isempty(yt)
        error('mc_pred -> If lpy is wanted you must provide the vector y as 7''th input.')
    end
            
    [n,nout] = size(y);
    nmc=size(gp.jitterSigma2,1);
    
    % Non-Gaussian likelihood. Thus latent variables should be used in
    % place of observations
    if isfield(gp, 'latentValues') && ~isempty(gp.latentValues)
        y = gp.latentValues';
    else 
        y = repmat(y,1,nmc);
    end
    
    % loop over all samples
    for i1=1:nmc
        Gp = take_nth(gp,i1);
        if isfield(Gp,'latent_method') && isequal(Gp.latent_method,'MCMC')
          Gp = rmfield(Gp,'latent_method');
        end
                
        if nargout < 3
            [ef, vf] = gp_mo_pred(Gp, x, reshape(y(:,i1),n,nout), xt, 'predcf', predcf, 'tstind', tstind);
            Ef(:,i1) = ef(:);
            Varf(:,i1) = vf(:);
        else 
            if isfield(gp, 'latentValues')
                [ef, vf] = gp_mo_pred(Gp, x, reshape(y(:,i1),n,nout), xt, 'predcf', predcf, 'tstind', tstind);
                Ef(:,i1) = ef(:);
                Varf(:,i1) = vf(:);
%                 if isempty(yt)
%                     [Ey(:,i1), Vary(:,i1)] = feval(Gp.lik.fh.predy, Gp.lik, ef, vf, [], zt);
                if nargout > 3
                    [ey,vy,ppy] = feval(Gp.lik.fh.predy, Gp.lik, ef, vf, yt, zt);
                    Ey(:,i1) = ey(:);
                    Vary(:,i1) = vy(:);
                    lpy(:,i1) = ppy(:);
                else
                    ppy = feval(Gp.lik.fh.predy, Gp.lik, ef, vf, yt, zt);
                    lpy(:,i1) = ppy(:);
                end
            else
                if nargout < 4
                    [Ef(:,i1), Varf(:,i1), lpy(:,i1)] = gp_mo_pred(Gp, x, reshape(y(:,i1),n,nout), xt, 'predcf', predcf, 'tstind', tstind, 'yt', yt);
                else
                    [Ef(:,i1), Varf(:,i1), lpy(:,i1), Ey(:,i1), Vary(:,i1)] = gp_mo_pred(Gp, x, reshape(y(:,i1),n,nout), xt, 'predcf', predcf, 'tstind', tstind, 'yt', yt); 
                end
            end            
        end
    end    
end