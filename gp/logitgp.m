function [l,lq,xx] = logitgp(x,varargin)
% LOGITGP - Logistic-Gaussian Process density estimate for 1D and 2D data
%   
%   LOGITGP(X)
%   [P,PQ,XT] = LOGITGP(X,XT,OPTIONS)
% 
%     X is 1D or 2D point data
%     XT is optional test points
%     OPTIONS are optional parameter-value pairs
%       'gridn' is optional number of grid points used in each axis direction
%         default is 100 for 1D, 15 for grid 2D, and 7 for Voronoi 2D
%       'range' tells the estimation range, default is data range
%         for 1D [XMIN XMAX]
%         for 2D [XMIN XMAX YMIN YMAX]
%       'gpcf' is optional function handle of a GPstuff covariance function 
%         (default is @gpcf_sexp)
%       'latent_method' is optional 'Laplace' (default) or 'MCMC'
%       'int_method' is optional 'mode' (default), 'CCD' or 'grid'
% 
%     L is the estimated density
%     LQ is the 5% and 95% percentiles of the density estimate
%     XX contains the used test points
  
% Copyright (c) 2011 Jaakko Riihimäki and Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LOGITGP';
  ip.addRequired('x', @(x) isnumeric(x) && size(x,2)==1 || size(x,2)==2);
  ip.addOptional('xt',NaN, @(x) isnumeric(x) && size(x,2)==1 || size(x,2)==2);
  ip.addParamValue('gridn',[], @(x) isnumeric(x));
  ip.addParamValue('range',[], @(x) isreal(x)&&(length(x)==2||length(x)==4));
  ip.addParamValue('gpcf',@gpcf_sexp,@(x) ischar(x) || isa(x,'function_handle'));
  ip.addParamValue('latent_method','Laplace', @(x) ismember(x,{'EP' 'Laplace' 'MCMC'}))
  %ip.addParamValue('latent_method','Laplace', @(x) ismember(x,{'EP' 'Laplace'}))
  ip.addParamValue('int_method','mode', @(x) ismember(x,{'mode' 'CCD', 'grid'}))
  ip.addParamValue('normalize',false, @islogical);
  
  ip.parse(x,varargin{:});
  x=ip.Results.x;
  xt=ip.Results.xt;
  gridn=ip.Results.gridn;
  xrange=ip.Results.range;
  gpcf=ip.Results.gpcf;
  latent_method=ip.Results.latent_method;
  int_method=ip.Results.int_method;
  normalize=ip.Results.normalize;
  
  [n,m]=size(x);
  
  switch m
    case 1 % 1D
      % Parameters for a grid
      if isempty(gridn)
        % number of points
        gridn=400;
      end
      xmin=min(x);xmax=max(x);
      if ~isempty(xrange)
        xmin=min(xmin,xrange(1));
        xmax=max(xmax,xrange(2));
      else
        xmin=xmin-.5*std(x);
        xmax=xmax+.5*std(x);
      end
      % Discretize the data
      if isnan(xt)
        xx=linspace(xmin,xmax,gridn)';
      else
        xx=xt;
        gridn=numel(xt);
      end
      
      xd=xx(2)-xx(1);
      yy=hist(x,xx)';
      
      % weight for normalization
      % w=1/gridn;
      % ye=w.*ones(gridn,1);
      
      % normalise, so that same prior is ok for different scales
      xxn=(xx-mean(xx))./std(xx);
      
      %[Ef,Covf]=gpsmooth(xxn,yy,[xxn; xtn],gpcf,latent_method,int_method);
      [Ef,Covf]=gpsmooth(xxn,yy,xxn,gpcf,latent_method,int_method);
      
      if strcmpi(latent_method,'MCMC')
        PJR=zeros(size(Ef,1),size(Covf,3));
        for i1=1:size(Covf,3)
          qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Covf(:,:,i1),'upper'),Ef(:,i1)');
          qjr=exp(qr)';
          pjr=bsxfun(@rdivide,qjr,sum(qjr));
          pjr=pjr./xd;
          PJR(:,i1)=mean(pjr,2);
        end
        pjr=PJR;
      else
        qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Covf,'upper'),Ef');
        qjr=exp(qr)';
        pjr=bsxfun(@rdivide,qjr,sum(qjr(1:gridn,:)));
        pjr=pjr./xd;
      end
      %l=prctile(pjr(gridn+1:end,:)',[50])';
      %lq=prctile(pjr(gridn+1:end,:)',[2.5 97.5])';
      l=mean(pjr')';
      lq=prctile(pjr',[2.5 97.5])';
      
      if nargout<1
        % no output, do the plot thing
        newplot
        hp=patch([xx; xx(end:-1:1)],[lq(:,1); lq(end:-1:1,2)],[.9 .9 .9]);
        set(hp,'edgecolor',[.9 .9 .9])
        xlim([xmin xmax])
        line(xx,l,'linewidth',2);
      end
      
    case 2 % 2D
      
      % Find unique points
      [xu,I,J]=unique(x,'rows');
      % and count number of repeated x's
      counts=crosstab(J); 
      nu=length(xu);
  
      % Parameters for a grid
      if isempty(gridn)
        % number of points in direction
        gridn=15;
      end
      if numel(gridn)==1
        gridn(2)=gridn(1);
      end
      x1min=min(x(:,1));x1max=max(x(:,1));
      x2min=min(x(:,2));x2max=max(x(:,2));
      if ~isempty(xrange)
        % range extension
        x1min=min(x1min,xrange(1));
        x1max=max(x1max,xrange(2));
        x2min=min(x2min,xrange(3));
        x2max=max(x2max,xrange(4));
      else
        x1min=x1min-0.5*std(x(:,1));
        x1max=x1max+0.5*std(x(:,1));
        x2min=x2min-0.5*std(x(:,2));
        x2max=x2max+0.5*std(x(:,2));
      end
      
      % Discretize the data
      if isnan(xt)
        % Form regular grid to discretize the data
        zz1=linspace(x1min,x1max,gridn(1))';
        zz2=linspace(x2min,x2max,gridn(2))';
        [z1,z2]=meshgrid(zz1,zz2);
        z=[z1(:),z2(:)];
        nz=length(z);
        % form data for GP (xx,yy,ye)
        xx=z;
        xt=z;
      else
        xx=xt;
        gridn=[length(unique(xx(:,1))) length(unique(xx(:,2)))];
      end
      yy=zeros(nz,1);
      zi=interp2(z1,z2,reshape(1:nz,gridn(2),gridn(1)),xu(:,1),xu(:,2),'nearest');
      for i1=1:nu
        yy(zi(i1),1)=yy(zi(i1),1)+counts(i1);
      end
      %ye=ones(nz,1)./nz.*n;
      
      unx1=unique(xx(:,1));
      unx2=unique(xx(:,2));
      xd=(unx1(2)-unx1(1))*(unx2(2)-unx2(1));
      
      % normalise, so that same prior is ok for different scales
      xxn=bsxfun(@rdivide,bsxfun(@minus,xx,mean(xx,1)),std(xx,1));
      
      % [Ef,Covf]=gpsmooth(xxn,yy,[xxn; xtn],gpcf,latent_method,int_method);
      [Ef,Covf]=gpsmooth(xxn,yy,xxn,gpcf,latent_method,int_method);
      
      if strcmpi(latent_method,'MCMC')
        PJR=zeros(size(Ef,1),size(Covf,3));
        for i1=1:size(Covf,3)
          qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Covf(:,:,i1),'upper'),Ef(:,i1)');
          qjr=exp(qr)';
          pjr=bsxfun(@rdivide,qjr,sum(qjr));
          pjr=pjr./xd;
          PJR(:,i1)=mean(pjr,2);
        end
        pjr=mean(PJR,2);
      else
        qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Covf,'upper'),Ef');
        qjr=exp(qr)';
        pjr=bsxfun(@rdivide,qjr,sum(qjr));
        pjr=pjr./xd;
      end
      
      l=mean(pjr')';
      lq=prctile(pjr',[2.5 97.5])';
      
      if nargout<1
        G=zeros(size(z1));
        G(:)=prctile(pjr',50);
        %contour(z1,z2,G);
        p=G(:);
        p1=p./sum(p);
        pu=unique(p1);pu=pu(end:-1:1);
        pc=cumsum(pu);
        PL=[.001 .01 .05 .1 .2 .5 .8 .9 .95];
        qi=[];
        for pli=1:numel(PL)
          qi(pli)=find(pc>PL(pli),1);
        end
        pl=pu(qi).*sum(p);
        contour(z1,z2,G,pl);
        %hold on, plot(x(:,1),x(:,2),'kx')
        colorbar
      end
    otherwise
      error('X has to be Nx1 or Nx2')
  end
end

function [Ef,Covf] = gpsmooth(xx,yy,xxt,gpcf,latent_method,int_method)
% Make inference with log Gaussian process and EP or Laplace approximation

  nin = size(xx,2);
  % init gp
  if strfind(func2str(gpcf),'ppcs')
    % ppcs still have nin parameter...
    gpcf1 = gpcf('nin',nin);
  else
    gpcf1 = gpcf();
  end
  % default vague prior
  pm = prior_t('s2', 1^2, 'nu', 4);
  pl = prior_t('s2', 1^2, 'nu', 4);
  pa = prior_t('s2', 10^2, 'nu', 4);
  % different covariance functions have different parameters
  if isfield(gpcf1,'magnSigma2')
     gpcf1 = gpcf(gpcf1, 'magnSigma2', .5, 'magnSigma2_prior', pm);
     %gpcf1 = gpcf(gpcf1, 'magnSigma2', .00001, 'magnSigma2_prior', []);
  end
  if isfield(gpcf1,'lengthScale')
     gpcf1 = gpcf(gpcf1, 'lengthScale', .5, 'lengthScale_prior', pl);
     %gpcf1 = gpcf(gpcf1, 'lengthScale', 10, 'lengthScale_prior', []);
     %gpcf1 = gpcf(gpcf1, 'lengthScale', 1, 'lengthScale_prior', pl);
  end
  if isfield(gpcf1,'alpha')
    gpcf1 = gpcf(gpcf1, 'alpha', 20, 'alpha_prior', pa);
  end
  %gpcf1=gpcf_matern52('lengthscale',.1,'lengthScale_prior', pl, 'magnSigma2', 1, 'magnSigma2_prior', pm);
  %gpcf1=gpcf_sexp('lengthscale',[.1 .1],'lengthScale_prior', pl, 'magnSigma2', 1, 'magnSigma2_prior', pm);
  %gpcf1=gpcf_exp('lengthscale',.1,'lengthScale_prior', pl, 'magnSigma2', 1, 'magnSigma2_prior', pm);
  
  % Create the GP structure
  gpmfco = gpmf_constant('prior_mean',0,'prior_cov',100);
  gpmflin = gpmf_linear('prior_mean',0,'prior_cov',100);
  gpmfsq = gpmf_squared('prior_mean',0,'prior_cov',100);
  %gp = gp_set('lik', lik_logitgp, 'cf', {gpcf1}, 'jitterSigma2', 1e-4);
  gp = gp_set('lik', lik_logitgp, 'cf', {gpcf1}, 'jitterSigma2', 1e-4, 'meanf', {gpmfco,gpmflin,gpmfsq});
  
  % First optimise hyperparameters using Laplace approximation
  gp = gp_set(gp, 'latent_method', 'Laplace');
  opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','off','Derivativecheck','off');
  gp=gp_optim(gp,xx,yy,'opt',opt);
  
  if strcmpi(latent_method,'Laplace')
    % Just make prediction for the test points
    [Ef,Covf] = gp_pred(gp, xx, yy, xxt);
  elseif strcmpi(latent_method,'MCMC')
    gp = gp_set(gp, 'latent_method', 'MCMC');

    % Here we use two stage sampling to get faster convergence
    hmc_opt=hmc2_opt;
    hmc_opt.steps=10;
    hmc_opt.stepadj=0.05;
    hmc_opt.nsamples=1;
    latent_opt.display=0;
    latent_opt.repeat = 20;
    latent_opt.sample_latent_scale = 0.5;
    hmc2('state', sum(100*clock))
    
    % The first stage sampling
    [r,g,opt]=gp_mc(gp, xx, yy, 'hmc_opt', hmc_opt, 'latent_opt', latent_opt, 'nsamples', 1, 'repeat', 15, 'display', 0);
    %[r,g,opt]=gp_mc(gp, xx, yy, 'latent_opt', latent_opt, 'nsamples', 1, 'repeat', 15);
    
    % re-set some of the sampling options
    hmc_opt.steps=4;
    hmc_opt.stepadj=0.05;
    %latent_opt.repeat = 5;
    hmc2('state', sum(100*clock));
    
    % The second stage sampling
    % Notice that previous record r is given as an argument
    [rgp,g,opt]=gp_mc(gp, xx, yy, 'hmc_opt', hmc_opt, 'nsamples', 500,'latent_opt', latent_opt, 'display', 0, 'record', r);
    %[rgp,g,opt]=gp_mc(gp, xx, yy, 'nsamples', 500, 'hmc_opt', hmc_opt, 'latent_opt', latent_opt, 'record', r);
    % Remove burn-in
    %rgp=thin(rgp,102);
    rgp=thin(rgp,102,4);
    
    [Ef, Covf] = gpmc_jpreds(rgp, xx, yy, xxt);
     
  else
    % integrate over the hyperparameters
    %[~, ~, ~, Ef, Covf] = gp_ia(opt, gp, xx, yy, xt, param);
    [notused, notused, notused, Ef, Covf]=...
        gp_ia(gp, xx, yy, xt, 'z', ye, 'int_method', int_method);
  end
  
end
