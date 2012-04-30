function [Eft, Varft, lpyt, Eyt, Varyt] = gp_mo_pred(gp, x, y, varargin)
%GP_MO_PRED  Make predictions with Gaussian process 
%
%  Description
%    [EFT, VARFT] = GP_MO_PRED(GP, X, Y, XT, OPTIONS)
%    takes a GP structure together with matrix X of training
%    inputs and vector Y of training targets, and evaluates the
%    predictive distribution at test inputs XT. Returns a posterior
%    mean EFT and variance VARFT of latent variables.
%
%        Eft =  E[f | xt,x,y,th]  = K_fy*(Kyy+s^2I)^(-1)*y
%      Varft = Var[f | xt,x,y,th] = diag(K_fy - K_fy*(Kyy+s^2I)^(-1)*K_yf). 
%
%    Each row of X corresponds to one input vector and each row of
%    Y corresponds to one output vector.
%
%    [EFT, VARFT, LPYT] = GP_MO_PRED(GP, X, Y, XT, 'yt', YT, ...)
%    returns also the log predictive density LPYT of the
%    observations YT at test input locations XT. This can be used
%    for example in the cross-validation. Here Y has to be vector.
% 
%    [EFT, VARFT, LPYT, EYT, VARYT] = GP_MO_PRED(GP, X, Y, XT, OPTIONS)
%    Returns also posterior predictive means EYT and variance VARYT.
%
%    [EF, VARF, LPY, EY, VARY] = GP_PRED(GP, X, Y, OPTIONS)
%    evaluates the predictive distribution at training inputs X
%    and logarithm of the predictive density LPY of the training
%    observations Y.
%
%    OPTIONS is optional parameter-value pair
%      predcf - an index vector telling which covariance functions are 
%                 used for prediction. Default is all (1:gpcfn). 
%                 See additional information below.
%      tstind - a vector/cell array defining, which rows of X belong 
%               to which training block in *IC type sparse models. 
%               Default is []. In case of PIC, a cell array
%               containing index vectors specifying the blocking
%               structure for test data. IN FIC and CS+FIC a
%               vector of length n that points out the test inputs
%               that are also in the training set (if none, set
%               TSTIND = [])
%      yt     - optional observed yt in test points (see below)
%
%    NOTE! In case of FIC and PIC sparse approximation the
%    prediction for only some PREDCF covariance functions is just
%    an approximation since the covariance functions are coupled in
%    the approximation and are not strictly speaking additive
%    anymore.
%
%    For example, if you use covariance such as K = K1 + K2 your
%    predictions Ef1 = gp_pred(GP, X, Y, X, 'predcf', 1) and Ef2 =
%    gp_pred(gp, x, y, x, 'predcf', 2) should sum up to Ef =
%    gp_pred(gp, x, y, x). That is Ef = Ef1 + Ef2. With FULL model
%    this is true but with FIC and PIC this is true only
%    approximately. That is Ef \approx Ef1 + Ef2.
%
%    With CS+FIC the predictions are exact if the PREDCF covariance
%    functions are all in the FIC part or if they are CS
%    covariances.
%
%    NOTE! When making predictions with a subset of covariance
%    functions with FIC approximation the predictive variance can
%    in some cases be ill-behaved i.e. negative or
%    unrealistically small. This may happen because of the
%    approximative nature of the prediction.
%
%  See also
%    GP_SET, GP_OPTIM, DEMO_REGRESSION*
%

% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2008 Jouni Hartikainen
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

if iscell(gp) || numel(gp.jitterSigma2)>1 || isfield(gp,'latent_method')
  % use inference specific methods
  if iscell(gp)
    fh_pred=@gpia_pred;
  elseif numel(gp.jitterSigma2)>1
    fh_pred=@gpmc_pred;
  elseif isfield(gp,'latent_method')
    switch gp.latent_method
      case 'Laplace'
        switch gp.lik.type
          %           case 'Softmax'
          %             fh_pred=@gpla_softmax_pred;
          case {'Multinom' 'Softmax'}
            fh_pred=@gpla_mo_pred;
          otherwise
            fh_pred=@gpla_pred;
        end
      case 'EP'
        fh_pred=@gpep_pred;
      case 'MCMC'
        switch gp.lik.type
          case {'Multinom' 'Softmax'}
            fh_pred=@gpmc_mo_pred;
          otherwise
            fh_pred=@gpmc_pred;
        end
    end
  else
    error('Logical error by coder of this function!')
  end
  switch nargout
    case 1
      [Eft] = fh_pred(gp, x, y, varargin{:});
    case 2
      [Eft, Varft] = fh_pred(gp, x, y, varargin{:});
    case 3
      [Eft, Varft, lpyt] = fh_pred(gp, x, y, varargin{:});
    case 4
      [Eft, Varft, lpyt, Eyt] = fh_pred(gp, x, y, varargin{:});
    case 5
      [Eft, Varft, lpyt, Eyt, Varyt] = fh_pred(gp, x, y, varargin{:});
  end
  return
end

ip=inputParser;
ip.FunctionName = 'GP_MO_PRED';
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('xt', [], @(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))))
ip.addParamValue('yt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('predcf', [], @(x) isempty(x) || ...
                 isvector(x) && isreal(x) && all(isfinite(x)&x>0))
ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                 (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
if numel(varargin)==0 || isnumeric(varargin{1})
  % inputParser should handle this, but it doesn't
  ip.parse(gp, x, y, varargin{:});
else
  ip.parse(gp, x, y, [], varargin{:});
end
xt=ip.Results.xt;
yt=ip.Results.yt;
predcf=ip.Results.predcf;
tstind=ip.Results.tstind;
if isempty(xt)
  xt=x;
    if isempty(tstind)
      if iscell(gp)
        gptype=gp{1}.type;
      else
        gptype=gp.type;
      end
      switch gptype
        case {'FULL' 'VAR' 'DTC' 'SOR'}
          tstind = [];
        case {'FIC' 'CS+FIC'}
          tstind = 1:size(x,1);
        case 'PIC'
          if iscell(gp)
            tstind = gp{1}.tr_index;
          else
            tstind = gp.tr_index;
          end
      end
    end
  if isempty(yt)
    yt=y;
  end
  if isempty(zt)
    zt=z;
  end
end

[tn, nout] = size(y);

if nargout > 2 && isempty(yt)
  error('GP_MO_PRED -> To compute LPYT, the YT has to be provided.')
end



if isfield(gp, 'comp_cf')  % own covariance for each ouput component
  multicf = true;
  if length(gp.comp_cf) ~= nout
    error('GP_MO_E: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
  end
  if ~isempty(predcf)
    if ~iscell(predcf) || length(predcf)~=nout
      error(['GP_MO_PRED: if own covariance for each output component is used,'...
             'predcf has to be cell array and contain nout (vector) elements.   '])
    end
  else
    predcf = gp.comp_cf;
  end
else
  multicf = false;
  for i1=1:nout
    predcf2{i1} = predcf;
  end
  predcf=predcf2;
end

L = zeros(tn,tn,nout);
ntest=size(xt,1);
K_nf = zeros(ntest,tn,nout);
if multicf
  for i1=1:nout
    [tmp,C] = gp_trcov(gp, x, gp.comp_cf{i1});
    L(:,:,i1) = chol(C)';
    K_nf(:,:,i1) = gp_cov(gp,xt,x,predcf{i1});
  end
else
  for i1=1:nout
    [tmp,C] = gp_trcov(gp, x);
    L(:,:,i1) = chol(C)';
    K_nf(:,:,i1) = gp_cov(gp,xt,x,predcf{i1});        
  end
end


Eft = zeros(ntest,nout);
for i1=1:nout
  Eft(:,i1) = K_nf(:,:,i1)*(L(:,:,i1)'\(L(:,:,i1)\y(:,i1))); 
end

% NOTE! THIS FUNCTION RETURNS THE VARIANCE IN DIFFERENT FORMAT THAN
% GPLA_MO_PRED
Varft = zeros(ntest,nout);
if nargout > 1
  for i1=1:nout
    v = L(:,:,i1)\K_nf(:,:,i1)';
    V = gp_trvar(gp,xt,predcf{i1});
    Varft(:,i1) = V - sum(v'.*v',2);
  end
end
if nargout > 2
  % normal case
  [V, Cv] = gp_trvar(gp,xt,predcf);
  Eyt = Eft;
  Varyt = Varft + Cv - V;
  lpyt = norm_lpdf(yt, Eyt, sqrt(Varyt));
  
end
