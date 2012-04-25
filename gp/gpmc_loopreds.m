function [Ef, Varf, lpy, Ey, Vary] = gpmc_loopreds(gp, x, y, varargin)
%GPMC_LOOPREDS  Leave-one-out predictions with Gaussian Process MCMC approximation.
%
%  Description
%    [EFT, VARFT, LPYT, EYT, VARYT] = GPMC_LOOPREDS(RECGP, X, Y) takes a
%    Gaussian processes record structure RECGP (returned by gp_mc)
%    together with a matrix XT of input vectors, matrix X of
%    training inputs and vector Y of training targets. Evaluates the
%    leave-one-out predictive distribution at inputs X with respect to
%    latent variables and returns posterior predictive means EFT and
%    variances VARFT of latent variables, the posterior predictive means
%    EYT and variances VARYT of observations, and logarithm of the
%    posterior predictive densities PYT at input locations X. That is: 
%      - The hyperparameters, hp, are sampled from the full posterior p(S|x,y)
%      by gp_mc 
%      - With each hyperparameter sample, hp_s, we evaluate the LOO-CV
%      distributions p(f_i | x_\i, y_\i, hp_s)
%
%  References:
%    S. Sundararajan and S. S. Keerthi (2001). Predictive
%    Approaches for Choosing Hyperparameters in Gaussian Processes. 
%    Neural Computation 13:1103-1118.
%
%  See also
%   GP_G, GPCF_*, GP_SET, GP_PAK, GP_UNPAK
%

% Copyright (c) 2008-2010, 2012 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

% Nothing to parse, but check the arguments anyway
ip=inputParser;
ip.FunctionName = 'GPMC_LOOPREDS';
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.parse(gp, x, y);

if isfield(gp,'meanf') & ~isempty(gp.meanf)
  error('GPMC_LOOPREDS: Mean functions not yet supported');
end

nmc=size(gp.jitterSigma2,1);

for i1=1:nmc
    Gp = take_nth(gp,i1);
    
    if nargout < 3
        [Ef(:,i1), Varf(:,i1)] = gp_loopred(Gp, x, y);
    else
        if nargout < 4
            [Ef(:,i1), Varf(:,i1), lpy(:,i1)] = gp_loopred(Gp, x, y);
        else
            [Ef(:,i1), Varf(:,i1), lpy(:,i1), Ey(:,i1), Vary(:,i1)] = gp_loopred(Gp, x, y);
        end
    end
end