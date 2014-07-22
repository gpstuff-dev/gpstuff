function [Eft, Varft, lpyt, Eyt, Varyt] = gpsvi_pred(gp, x, y, varargin)
%GPSVI_PRED  Make predictions with SVI GP
%
%  Description
%    [EFT, VARFT] = GPSVI_PRED(GP, X, Y, XT, OPTIONS)
%    takes a GP structure together with matrix X of training inputs and
%    vector Y of training targets, and evaluates the predictive
%    distribution at test inputs XT. Returns a posterior mean EFT and
%    variance VARFT of latent variables. Each row of X corresponds to one
%    input vector and each row of Y corresponds to one output vector.
%
%    [EFT, VARFT, LPYT] = GPSVI_PRED(GP, X, Y, XT, 'yt', YT, OPTIONS)
%    returns also logarithm of the predictive density LPYT of the
%    observations YT at test input locations XT. This can be used
%    for example in the cross-validation. Here Y has to be a vector.
% 
%    [EFT, VARFT, LPYT, EYT, VARYT] = GPSVI_PRED(GP, X, Y, XT, OPTIONS)
%    returns also the posterior predictive mean EYT and variance VARYT.
%
%    [EF, VARF, LPY, EY, VARY] = GPSVI_PRED(GP, X, Y, OPTIONS)
%    evaluates the predictive distribution at training inputs X
%    and logarithm of the predictive density LPY of the training
%    observations Y.
%
%    OPTIONS is optional parameter-value pair
%      predcf - an index vector telling which covariance functions are 
%               used for prediction. Default is all (1:gpcfn). 
%               See additional information below.
%      yt     - optional observed yt in test points (see below)
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case. 
%      zt     - optional observed quantity in triplet (xt_i,yt_i,zt_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, the expected 
%               value for the ith case. 
%
%  See also
%    GP_PRED, GP_SET, SVIGP, DEMO_SVI*
%

% Copyright (c) 2014 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GP_PRED';
ip.addRequired('gp',@(x) isstruct(x) || iscell(x));
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('xt', [], @(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))))
ip.addParamValue('yt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('predcf', [], @(x) isempty(x) || ...
                 isvector(x) && isreal(x) && all(isfinite(x)&x>=0))
if numel(varargin)==0 || isnumeric(varargin{1})
  % inputParser should handle this, but it doesn't
  ip.parse(gp, x, y, varargin{:});
else
  ip.parse(gp, x, y, [], varargin{:});
end
xt=ip.Results.xt;
yt=ip.Results.yt;
zt=ip.Results.zt;
z=ip.Results.z;
predcf=ip.Results.predcf;
if isempty(xt)
  xt=x;
  if isempty(yt)
    yt=y;
  end
  if isempty(zt)
    zt=z;
  end
end

tn = size(x,1);
if nargout > 2 && isempty(yt)
  lpyt=[];
end

%     [tmp,tmp,tmp,param]=gpsvi_e(gp_pak(gp),gp,x,y);

% Check if the variational parameters has been set
if ~isfield(gp, 'm') || ~isfield(gp, 'S')
  error('Variational parameters has not been set. Call SVIGP first.')
end

u = gp.X_u;
m=gp.m;
S=gp.S;
if size(u,2) ~= size(x,2)
  % Turn the inducing vector on right direction
  u=u';
end
% Calculate some help matrices
K_uu = gp_trcov(gp, u);     % u x u, noiseles covariance K_uu
K_nu = gp_cov(gp,xt,u);       % n x u
[Luu, notpositivedefinite] = chol(K_uu,'lower');
if notpositivedefinite
  Eft=NaN; Varft=NaN; lpyt=NaN; Eyt=NaN; Varyt=NaN;
  return
end

Eft = K_nu*(Luu'\(Luu\m));

if nargout > 1
  [Knn_v, Cnn_v] = gp_trvar(gp,xt,predcf);
  B2=Luu\(K_nu');
  B3=K_uu\(K_nu');

  Varft = Knn_v - sum(B2.*B2)' + sum(B3.*(S*B3))';
end
if ~isequal(gp.lik.type, 'Gaussian') % isequal(gp.lik.type,'Probit')
  Varft = Varft + gp.lik.sigma2;
end


if nargout > 2
  if isequal(gp.lik.type, 'Gaussian')
    Eyt = Eft;
    Varyt = Varft + Cnn_v - Knn_v;
    if ~isempty(yt)
      lpyt = norm_lpdf(yt, Eyt, sqrt(Varyt));
    end
  else
    if nargout>3
      [lpyt, Eyt, Varyt] = gp.lik.fh.predy(gp.lik, Eft, Varft, yt, zt);
    else
      lpyt = gp.lik.fh.predy(gp.lik, Eft, Varft, yt, zt);
    end
  end
end

