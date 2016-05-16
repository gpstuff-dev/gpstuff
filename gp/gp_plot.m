function h = gp_plot(gp, x, y, varargin)
%GP_PLOT  Make a plot with Gaussian process 
%
%  Description
%    GP_PLOT(GP, X, Y, XT, OPTIONS)
%    takes a GP structure together with matrix X of training
%    inputs and vector Y of training targets and plots predictions
%    evaluated at test inputs XT.
%
%    [EF, VARF, LPY, EY, VARY] = GP_PLOT(GP, X, Y, OPTIONS)
%    plots predictions evaluated at training inputs X.
%
%    The form of the plot depends on the dimensionality of X and options.
%      - 1D with Gaussian: mean and 5% and 95% quantiles of f and y
%      - 1D with non-Gaussian: median and 5% and 95% quantiles of mu or f
%      - 2D: Conditional predictions, see GP_CPRED + 2D plot of mu or f
%      - ND with N>2: Conditional predictions, see GP_CPRED
%
%    OPTIONS is optional parameter-value pair
%      target - option for choosing what is computed 'mu' (default)
%               or 'f'
%      normdata - a structure with fields xmean, xstd, ymean, and ystd
%               to allow plotting in the original data scale (see
%               functions normdata and denormdata)
%      xlabels - a cell array of covariate label strings
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
%    predictions Ef1 = gp_plot(GP, X, Y, X, 'predcf', 1) and Ef2 =
%    gp_plot(gp, x, y, x, 'predcf', 2) should sum up to Ef =
%    gp_plot(gp, x, y, x). That is Ef = Ef1 + Ef2. With FULL model
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
% Copyright (c) 2014 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GP_PLOT';
ip.addRequired('gp',@(x) isstruct(x) || iscell(x));
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('xt', [], @(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))))
ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('xlabels', [], @(x) isempty(x) || iscell(x));
ip.addParamValue('predcf', [], @(x) isempty(x) || ...
                 isvector(x) && isreal(x) && all(isfinite(x)&x>=0))
ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                 (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
ip.addParamValue('fcorr', 'off', @(x) ismember(x, {'off', 'fact', 'cm2', 'on'}));
ip.addParamValue('tr', 0.25, @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('target', 'mu', @(x) ismember(x,{'f','mu'}))
ip.addParamValue('normdata', struct(), @(x) isempty(x) || isstruct(x))
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
  if size(x,2)==1
    [xt,xi]=sort(x);
    if ~isempty(z)
      zt=z(xi);
    else
      zt=[];
    end
  else
    xt=x;
    zt=z;
  end
end
target = ip.Results.target;
if iscell(gp)
  liktype=gp{1}.lik.type;
else
  liktype=gp.lik.type;
end
if isequal(liktype, 'Coxph') && isequal(target,'mu')
    target='f';
    warning('GP_CPRED: Target ''mu'' not applicable for a Cox-PH model. Switching to target ''f''')
end
tr = ip.Results.tr;
xlabels=ip.Results.xlabels;
% normdata
nd=ip.Results.normdata;
ipnd=inputParser;
ipnd.FunctionName = 'normdata';
ipnd.addParamValue('xmean',0,@(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))));
ipnd.addParamValue('xstd',1,@(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))));
ipnd.addParamValue('xlog',0,@(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))));
ipnd.addParamValue('ymean',0,@(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))));
ipnd.addParamValue('ystd',1,@(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))));
ipnd.addParamValue('ylog',0,@(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))));
ipnd.parse(nd);
nd=ipnd.Results;

if iscell(gp)
  liktype=gp{1}.lik.type;
else
  liktype=gp.lik.type;
end

[n,m]=size(x);

switch m
  case 1
    if isequal(liktype, 'Gaussian')
      [Ef, Varf,~,Ey,Vary] = gp_pred(gp, x, y, xt, options);
      Stdf=sqrt(Varf);
      Stdy=sqrt(Vary);
      xt=denormdata(xt,nd.xmean,nd.xstd);
      Ef=denormdata(Ef,nd.ymean,nd.ystd);
      Stdf=Stdf*nd.ystd;
      Ey=denormdata(Ey,nd.ymean,nd.ystd);
      Stdy=Stdy*nd.ystd;
      hh=plot(xt, Ef, '-b', xt, Ef-1.64*Stdf, '--b', xt, Ef+1.64*Stdf, '--b',xt, Ey-1.64*Stdy, ':b', xt, Ey+1.64*Stdy, ':b');
    else
      switch target
        case 'f'
          [Ef, Varf] = gp_pred(gp, x, y, xt, options);
          Stdf=sqrt(Varf);
          xt=denormdata(xt,nd.xmean,nd.xstd);
          Ef=denormdata(Ef,nd.ymean,nd.ystd);
          Stdf=Stdf*nd.ystd;
          hh=plot(xt, Ef, '-b', xt, Ef-1.64*Stdf, '--b', xt, Ef+1.64*Stdf, '--b');
        case 'mu'
          prctmu = gp_predprctmu(gp, x, y, xt, options);
          xt=denormdata(xt,nd.xmean,nd.xstd);
          prctmu=denormdata(prctmu,nd.ymean,nd.ystd);
          Ef = prctmu; Varf = [];
          hh=plot(xt, prctmu(:,2), '-b', xt, prctmu(:,1), '--b', xt, prctmu(:,3), '--b');
      end
    end
    if ~isempty(xlabels)
        xlabel(xlabels{1})
    else
        xlabel('x')
    end
  case 2
    subplot(2,2,1);
    gp_cpred(gp,x,y,xt,1,'z',z,'zt',zt,'target',target,'plot','on',varargin{:});
    if ~isempty(xlabels)
        xlabel(xlabels{1})
    else
        xlabel('x1')
    end
    subplot(2,2,2);
    gp_cpred(gp,x,y,xt,2,'z',z,'zt',zt,'target',target,'plot','on',varargin{:});
    if ~isempty(xlabels)
        xlabel(xlabels{2})
    else
        xlabel('x2')
    end
    subplot(2,1,2);
    gp_cpred(gp,x,y,xt,[1 2],'z',z,'zt',zt,'target',target,'plot','on','tr',1e9,varargin{:});
    axis square
    if ~isempty(xlabels)
        xlabel(xlabels{1})
        ylabel(xlabels{2})
    else
        xlabel('x1')
        ylabel('x2')
    end
    view(3)
    shading faceted
    hhh=get(gca,'children');
    hh=hhh(1);
  otherwise
    sn1=ceil(sqrt(m));
    sn2=ceil(m/sn1);
    for xi=1:m
      subplot(sn1,sn2,xi);
      gp_cpred(gp,x,y,xt,xi,'z',z,'zt',zt,'target',target,'plot','on','normdata',nd,options);
      if ~isempty(xlabels)
          xlabel(xlabels{xi})
      else
          xlabel(sprintf('x%d',xi))
      end
      drawnow
      hhh=get(gca,'children');
      hh(xi)=hhh(1);
    end
end
if nargout>0
  h=hh;
end
