function [Eft, Varft, lpyt, Eyt, Varyt] = gpia_loopred(gp_array, x, y, varargin)
%GPIA_LOOPRED  Leave-one-out predictions with Gaussian Process IA approximation
%
%  Description
%    [EFT, VARFT, LPYT, EYT, VARYT] = GPIA_LOOPRED(RECGP, X, Y)
%    takes a cell array of GP structures together with matrix X of
%    input vectors, matrix X of training inputs and vector Y of
%    training targets, and evaluates the leave-one-out predictive
%    distribution at inputs X and returns means EFT and variances
%    VARFT of latent variables, the logarithm of the predictive
%    densities PYT, and the predictive means EYT and variances
%    VARYT of observations at input locations X.
%
%    OPTIONS is optional parameter-value pair
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case. 
%      is     - defines if importance sampling weighting is used 'on'
%               (default). If set to 'off', integration points and
%               weights for the full posterior are used.
%  
%    Given Gaussian likelihood or non-Gaussian likelihood and
%    latent method Laplace or EP, LOO-posterior given
%    hyperparameters is computed analytically or with analytic
%    approximation and LOO-posterior of the hyperparameters is
%    approximated using importance sampling reweighted integration
%    points from gp_ia. Optionally integration weights for full data
%    posterior of hyperparameters can be used (by setting option
%    'is' to 'off').
%
%  References:
%    Aki Vehtari and Jouko Lampinen (2002). Bayesian model
%    assessment and comparison using cross-validation predictive
%    densities. Neural Computation, 14(10):2439-2468.
%
%  See also
%   GP_LOOPRED, GP_IA, GP_PRED
%
% Copyright (c) 2009 Ville Pietil�inen
% Copyright (c) 2009-2010 Jarno Vanhatalo
% Copyright (c) 2010,2012 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPIA_LOOPRED';
  ip.addRequired('gp_array',@iscell);
  ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('is', 'on', @(x) ismember(x,{'on' 'off'}))
  ip.parse(gp_array, x, y, varargin{:});
  z=ip.Results.z;
  is=ip.Results.is;

  nGP = numel(gp_array);
  n=size(x,1);
  
  P_TH = zeros(1,nGP);
  Efts = zeros(n,nGP);
  Varfts = zeros(n,nGP);
  lpyts = zeros(n,nGP);
  if nargout > 3
    Eyts = zeros(n,nGP);
    Varyts = zeros(n,nGP);
  end
  
  for i1=1:nGP
    Gp=gp_array{i1};
    P_TH(1,i1) = Gp.ia_weight;
    % compute leave-one-out predictions for each hyperparameter sample
    % if latent method is MCMC, then these samples are for latent values, too
    if nargout <= 3
      [Efts(:,i1), Varfts(:,i1), lpyts(:,i1)] = gp_loopred(Gp, x, y, 'z', z);
    else
      [Efts(:,i1), Varfts(:,i1), lpyts(:,i1), Eyts_temp, Varyts_temp] = gp_loopred(Gp, x, y, 'z', z);
      if ~isempty(Eyts_temp)
        Eyts(:,i1) = Eyts_temp;
      end
      if ~isempty(Varyts_temp)
        Varyts(:,i1) = Varyts_temp;
      end
    end
  end

  if isequal(is,'off')
    P_TH=repmat(P_TH,n,1);
  else
    % log importance sampling weights
    lw=-lpyts;
    % normalize weights
    for i2=1:n
      % this works even when lw have large magnitudes
      lw(i2,:)=lw(i2,:)-sumlogs(lw(i2,:));
    end
    % importance sampling weights
    w=exp(lw);
    % reweight ia weights
    P_TH=bsxfun(@times,P_TH,w);
    P_TH=bsxfun(@rdivide,P_TH,sum(P_TH,2));
    % check the effective sample size
    m_eff=1./sum(P_TH.^2,2);
    if min(m_eff)<nGP/5
      warning(sprintf('For %d folds the effective sample size in IS is less than m/5',sum(m_eff<(nGP/5))))
    end
    %fprintf('nGP=%d, min(neff)=%.0f, min(neff)/nGP=%.2f\n',nGP, min(1./sum(P_TH.^2,2)), min(1./sum(P_TH.^2,2))./nGP)
    % PSIS
    if nGP>=200
        % (new) default sample size for is_normal and is_t is 200
        % CCD with nParam>=12 has at least 281 points
        [lw,pk] = psislw(log(P_TH'),10);
        P_TH=exp(lw');
        % check whether the variance and mean of the raw importance ratios is finite
        % PSIS weights have always finite variance and mean, but if raw importance
        % ratios have infinite variance the convergence to true value is
        % slower and if raw importance ratios have non-existing mean the the
        % estimate can't converge to true value
        vkn1=sum(pk>=0.5&pk<0.7);
        vkn2=sum(pk>=0.7&pk<1);
        vkn3=sum(pk>=1);
        n=numel(pk);
        if vkn1>0
            warning('%d (%.0f%%) PSIS Pareto k estimates between 0.5 and .7',vkn1,vkn1/n*100)
        end
        if vkn2>0
            warning('%d (%.0f%%) PSIS Pareto k estimates between 0.7 and 1',vkn2,vkn2/n*100)
        end
        if vkn3>0
            warning('%d (%.0f%%) PSIS Pareto k estimates greater than 1',vkn3,vkn3/n*100)
        end
    end
  end

  % compute combined predictions
  Eft = sum(Efts.*P_TH, 2);
  Varft = sum(Varfts.*P_TH,2) + sum(bsxfun(@minus,Efts,Eft).^2.*P_TH,2);
  if nargout > 2
    lpyt = log(sum(exp(lpyts+log(P_TH)),2));
    if nargout > 3
      Eyt = sum(Eyts.*P_TH,2);
      Varyt = sum(Varyts.*P_TH,2) + sum(bsxfun(@minus,Eyts,Eyt).^2.*P_TH, 2);
    end
  end
