function h = gp_plot(gp, x, y, varargin)
%GP_PLOT  Make plot with Gaussian process 
%
%  Description
%    GP_PLOT(GP, X, Y, XT, OPTIONS)
%    takes a GP structure together with matrix X of training
%    inputs and vector Y of training targets and plots predictions
%    evaluated at test inputs XT.
%
%    Form of the plot depends on the dimensionality of X and options.
%
%    [EF, VARF, LPY, EY, VARY] = GP_PRED(GP, X, Y, OPTIONS)
%    plots predictions evaluated at training inputs X.
%
%    OPTIONS is optional parameter-value pair
%      predcf - an index vector telling which covariance functions are 
%               used for prediction. Default is all (1:gpcfn). 
%               See additional information below.
%      tstind - a vector/cell array defining, which rows of X belong 
%               to which training block in *IC type sparse models. 
%               Default is []. In case of PIC, a cell array
%               containing index vectors specifying the blocking
%               structure for test data. In FIC and CS+FIC a
%               vector of length n that points out the test inputs
%               that are also in the training set (if none, set
%               TSTIND = []).
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case. 
%      zt     - optional observed quantity in triplet (xt_i,yt_i,zt_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, the expected 
%               value for the ith case. 
%      fcorr  - Method used for latent marginal posterior corrections. 
%               Default is 'off'. Possible methods are 'fact' for EP
%               and either 'fact' or 'cm2' for Laplace. If method is
%               'on', 'fact' is used for EP and 'cm2' for Laplace.
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

ip=inputParser;
ip.FunctionName = 'GP_PRED';
ip.addRequired('gp',@(x) isstruct(x) || iscell(x));
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('xt', [], @(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))))
ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('predcf', [], @(x) isempty(x) || ...
                 isvector(x) && isreal(x) && all(isfinite(x)&x>=0))
ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                 (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
ip.addParamValue('fcorr', 'off', @(x) ismember(x, {'off', 'fact', 'cm2', 'on'}));
ip.addParamValue('tr', 0.25, @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('target', 'f', @(x) ismember(x,{'f','mu','cdf'}))
ip.addParamValue('normdata', [], @(x) isreal(x) && all(isfinite(x(:))))
if numel(varargin)==0 || isnumeric(varargin{1})
  % inputParser should handle this, but it doesn't
  ip.parse(gp, x, y, varargin{:});
else
  ip.parse(gp, x, y, [], varargin{:});
end
xt=ip.Results.xt;
zt=ip.Results.zt;
options=struct();
options.predcf=ip.Results.predcf;
options.tstind=ip.Results.tstind;
z=ip.Results.z;
if ~isempty(z)
  options.z=z;
end
if ~isempty(zt)
  options.zt=zt;
end
if isempty(zt)
  options.zt=z;
end
if isempty(xt)
  xt=x;
  zt=z;
end
target = ip.Results.target;
tr = ip.Results.tr;
normdata = ip.Results.normdata;
[xmean,xstd,ymean,ystd]=deal(normdata(1),normdata(2),normdata(3),normdata(4));

if iscell(gp)
  liktype=gp{1}.lik.type;
else
  liktype=gp.lik.type;
end

[n,m]=size(x);

if m==1
  if isequal(liktype, 'Gaussian')
    [Ef, Varf,~,Ey,Vary] = gp_pred(gp, x, y, xt, options);
    xtd=denormdata(xt,xmean,xstd);
    Stdf=sqrt(Varf);
    Stdy=sqrt(Vary);
    if ~isempty(normdata)
      xt=denormdata(xt,xmean,xstd);
      Ef=denormdata(Ef,ymean,ystd);
      Stdf=Stdf*ystd;
      Ey=denormdata(Ey,ymean,ystd);
      Stdy=Stdy*ystd;
    end
    plot(xt, Ef, '-r', xt, Ef-1.64*Stdf, '-.r', xt, Ef+1.64*Stdf, '-.r',xt, Ey-1.64*Stdy, '--r', xt, Ey+1.64*Stdy, '--r')
  else
    switch target
      case 'f'
        [Ef, Varf] = gp_pred(gp, x, y, xt, options);
        plot(xt, Ef, 'ob', xt, Ef, '-k', xt, Ef-1.64*sqrt(Varf), '--b', xt, Ef+1.64*sqrt(Varf), '--b')
      case 'mu'
      prctmu = gp_predprctmu(gp, x, y, xt, options);
      Ef = prctmu; Varf = [];
    end
  end
end