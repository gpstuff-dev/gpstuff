function [Eft, Varft, lpyt, Eyt, Varyt] = gpmc_loopred(gp, x, y, varargin)
%GPMC_LOOPRED  Leave-one-out predictions with Gaussian Process MCMC approximation.
%
%  Description
%    [EFT, VARFT, LPYT, EYT, VARYT] = GPMC_LOOPRED(RECGP, X, Y, OPTIONS)
%    takes a Gaussian processes record structure RECGP (returned by
%    gp_mc) together with a matrix X of training inputs and vector
%    Y of training targets, and evaluates the leave-one-out
%    predictive distribution at inputs X and returns means EFT and
%    variances VARFT of latent variables, the logarithm of the
%    predictive densities PYT, and the predictive means EYT and
%    variances VARYT of observations at input locations X.
%
%    OPTIONS is optional parameter-value pair
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case. 
%      is     - defines if importance sampling weighting is used 'on'
%               (default). If set to 'off', MCMC samples from the
%               full data posterior are used.
%
%    Given Gaussian likelihood or non-Gaussian likelihood and
%    latent method Laplace or EP, LOO-posterior given
%    hyperparameters is computed analytically or with analytic
%    approximation and LOO-posterior of the hyperparameters is
%    approximated using importance sampling. Optionally full data
%    posterior for hyperparameters can be used (by setting option
%    'is' to 'off'). Given non-Gaussian likelihood and latent
%    method MCMC, LOO-posterior of the hyperparameters and latent
%    values is approximated using importance sampling. 
%
%    To stabilise the importance sampling, Pareto smoothed importance
%    sampling (PSIS) is used.  Unlike basic IS, PSIS estimate has
%    always finite variance and mean, but if raw importance ratios
%    have infinite variance (Pareto k between 0.5 and 1) the
%    convergence to true value is slower and if raw importance ratios
%    have non-existing mean (Pareto k greater than 1) the the estimate
%    can't converge to true value. Warning is shown if any of the
%    Pareto k's is greater than 0.5. Even if the k's are larger than
%    0.5 or 1, the average (or sum) of the LOO predictive quantities
%    can be quite accurate. See Vehtari & Gelman (2015) and Vehtari,
%    Gelman & Gabry (2015) for more information.
%
%  References:
%    Aki Vehtari and Jouko Lampinen (2002). Bayesian model
%    assessment and comparison using cross-validation predictive
%    densities. Neural Computation, 14(10):2439-2468.
%
%    Aki Vehtari and Andrew Gelman (2015). Pareto smoothed importance
%    sampling. arXiv preprint arXiv:1507.02646.
%
%    Aki Vehtari, Andrew Gelman and Jonah Gabry (2015). Efficient
%    implementation of leave-one-out cross-validation and WAIC for
%    evaluating fitted Bayesian models. arXiv preprint arXiv:1507.04544.
%
%  See also
%   GP_LOOPRED, GP_MC, GP_PRED
%
% Copyright (c) 2012-2015 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GPMC_LOOPRED';
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('is', 'on', @(x) ismember(x,{'on' 'off'}))
ip.parse(gp, x, y, varargin{:});
z=ip.Results.z;
is=ip.Results.is;

if isfield(gp,'meanf') && ~isempty(gp.meanf)
  error('GPMC_LOOPRED: Mean functions not yet supported');
end

nmc=size(gp.jitterSigma2,1);
[n,nin]=size(x);

if strcmp(gp.type, 'PIC_BLOCK') || strcmp(gp.type, 'PIC')
  ind = gp.tr_index;           % block indeces for training points
  gp = rmfield(gp,'tr_index');
end

eyts_ok = true;
for i1=1:nmc
  % compute leave-one-out predictions for each hyperparameter sample
  % if latent method is MCMC, then these samples are for latent values, too
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
  if nargout <= 3
    [Efts(:,i1), Varfts(:,i1), lpyts(:,i1)] = gp_loopred(Gp, x, y, 'z', z);
  else
    [Efts_i, Varfts_i, lpyts_i, Eyts_i, Varyts_i] = ...
        gp_loopred(Gp, x, y, 'z', z);
    Efts(:,i1) = Efts_i;
    Varfts(:,i1) = Varfts_i;
    lpyts(:,i1) = lpyts_i;
    if isempty(Eyts_i)
      eyts_ok = false;
    else
      Eyts(:,i1) = Eyts_i;
      Varyts(:,i1) = Varyts_i;
    end
  end
end

if isequal(is,'off')
  w=1/nmc;
  lw=-log(nmc);
else
  % log importance sampling weights
  lw=-lpyts;
  % compute Pareto smoothed log weights given raw log weights
  [lw,pk]=psislw(lw');lw=lw';
  % importance sampling weights
  w=exp(lw);
  % check whether the variance and mean of the raw importance ratios is finite
  % PSIS weights have always finite variance and mean, but if raw importance
  % ratios have infinite variance the convergence to true value is
  % slower and if raw importance ratios have non-existing mean the the
  % estimate can't converge to true value
  vkn1=sum(pk>=0.5&pk<1);
  vkn2=sum(pk>=1);
  n=numel(pk);
  if vkn1>0&vkn2==0
      warning('%d (%.0f%%) PSIS Pareto k estimates between 0.5 and 1',vkn1,vkn1/n*100)
  elseif vkn1==0&vkn2>0
      warning('%d (%.0f%%) PSIS Pareto k estimates greater than 1',vkn2,vkn2/n*100)
  elseif vkn1>0&vkn2>0
      warning('%d (%.0f%%) PSIS Pareto k estimates between 0.5 and 1\n     and %d (%.0f%%) PSIS Pareto k estimates greater than 1',vkn1,vkn1/n*100,vkn2,vkn2/n*100)
  end
end

% IS-LOO predictions
Eft = sum(Efts.*w, 2);
Varft = sum(Varfts.*w,2) + sum(bsxfun(@minus,Efts,Eft).^2.*w,2);
if nargout > 2
  lpyt = sumlogs(lpyts+lw,2); % same as lpyt=log(sum(exp(lpyts).*w,2));
  if nargout > 3
    if eyts_ok
      Eyt = sum(Eyts.*w,2);
      Varyt = sum(Varyts.*w,2) + sum(bsxfun(@minus,Eyts,Eyt).^2.*w,2);
    else
      Eyt = [];
      Varyt = [];
    end
  end
end
