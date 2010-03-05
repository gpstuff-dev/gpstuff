function [p,pq,xt] = lgcpdens(x,varargin)
% LGCPDENS - Log Gaussian Cox Process density estimate for 1D and 2D data
%   
%   LGCPDENS(X)
%   [P,PQ,XT] = LGCPDENS(X,XT,OPTIONS)
% 
%     X is 1D or 2D point data
%     XT is optional test points
%     OPTIONS are optional parameter-value pairs
%       'gridn' is optional number of grid points used in each axis direction
%          default is 100 for 1D, 15 for grid 2D, and 7 for Voronoi 2D
%       'expansion' tells how much grid range is expanded compare to the data 
%          range 
%          for 1D left and right expansion can be defined separately as [A1 A2]
%          for 2D the expansion can be defined separately as [XA1 XA2 YA1 YA2]
%       'type' defines whether grid (default) or augmented 'Voronoi' tessalation 
%         is used
%       'gpcf' is optional function handle of a GPstuff covariance function 
%         (default is @gpcf_sexp)
%       'latent_method' is optional 'EP' (default) or 'Laplace'
%       'hyperint' is optional 'mode' (default), 'CCD' or 'grid_based'
% 
%     P is the estimated density  
%     PQ is the 5% and 95% percentiles of the density estimate
%     XT contains the used test points
  
% Copyright (c) 2009-2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LGCPDENS';
  ip.addRequired('x', @(x) size(x,2)==1 || size(x,2)==2);
  ip.addOptional('xt',NaN, @(x) size(x,2)==1 || size(x,2)==2);
  ip.addParamValue('gridn',[], @(x) isscalar(x) && x>0 && mod(x,1)==0);
  ip.addParamValue('expansion',[], @(x) isscalar(x) && x>=0);
  ip.addParamValue('type','grid',...
                   @(x) ischar(x) && (strcmpi(x,'Grid') || strcmpi(x,'Voronoi')));
  ip.addParamValue('gpcf',@gpcf_sexp,@(x) ischar(x) || isa(x,'function_handle'));
  ip.addParamValue('latent_method','EP',...
                   @(x) ischar(x) && (strcmpi(x,'EP') || strcmpi(x,'Laplace')));
  ip.addParamValue('base','gaussian',...
                   @(x) ischar(x) && (strcmpi(x,'unif') ||...
                                      strcmpi(x,'exp')||...
                                      strcmpi(x,'gaussian')));
  ip.addParamValue('hyperint','mode', ...
                   @(x) ischar(x) && (strcmpi(x,'mode') ||...
                                      strcmpi(x,'CCD') ||...
                                      strcmpi(x,'grid_based')));
  
  ip.parse(x,varargin{:});
  x=ip.Results.x;
  xt=ip.Results.xt;
  gridn=ip.Results.gridn;
  expansion=ip.Results.expansion;
  type=ip.Results.type;
  gpcf=ip.Results.gpcf;
  latent_method=ip.Results.latent_method;
  base=ip.Results.base;
  hyperint=ip.Results.hyperint;
  
  [n,m]=size(x);
  
  switch m
    case 1 % 1D
      % Parameters for a grid
      if isempty(gridn)
        % number of points
        gridn=100;
      end
      if isempty(expansion)
        % range extension
        expansion=min(5/n,.5);
      end
      if numel(expansion)==1
        % symmetric range extension
        expansion(2)=expansion(1);
      end
      xmin=min(x);xmax=max(x);xr=xmax-xmin;
      % Form regular grid to discretize the data
      zmin=min(xmin,min(xt))-expansion(1)*xr;
      zmax=max(xmax,max(xt))+expansion(2)*xr;
      z=linspace(zmin,zmax,gridn)';
      xx=z;
      % In fact, we are doing histogram smoothing...
      yy=hist(x,z)';
      ye=ones(gridn,1)./gridn.*n;
      % Test points
      if isnan(xt)
        xt=linspace(xmin-expansion(1)*xr,xmax+expansion(2)*xr,max(gridn,200))';
      end
      % normalise, so that same prior is ok for different scales
      xxn=(xx-mean(xx))./std(xx);
      xtn=(xt-mean(xx))./std(xx);
      % smooth...
      [Ef,Varf]=gpsmooth(xxn,yy,ye,xtn,gpcf,latent_method,base,hyperint);
      
      % compute mean and quantiles
      A=range(xx);
      pm=exp(Ef+Varf/2)./A;
      pq5=exp(Ef-sqrt(Varf)*1.96)./A;
      pq95=exp(Ef+sqrt(Varf)*1.96)./A;
      pq=[pq5 pq95];

      if nargout<1
        % no output, do the plot thing
        plot(xt,pm)
      else
        p=pm;
      end
      
    case 2 % 2D
      
      % Find unique points
      [xu,I,J]=unique(x,'rows');
      % and count number of repeated x's
      counts=crosstab(J); 
      nu=length(xu);
  
      switch lower(type)
        case 'voronoi'
          % Parameters for a grid
          if isempty(gridn)
            % number of points in direction
            gridn=7;
          end
          if isempty(expansion)
            % range extension
            expansion=min(1/ceil(sqrt(nu)),.25);
          end
          if numel(expansion)==1
            % symmetric range extension
            expansion(2:4)=expansion(1);
          end
          % Form regular grid to augment Voronoi tessalation
          x1min=min(x(:,1),min(xt(:,1)));x1max=max(x(:,1),max(xt(:,1)));x1r=x1max-x1min;
          zz1=linspace(x1min-expansion(1)*x1r,x1max+expansion(2)*x1r,gridn)';
          x2min=min(x(:,2),min(xt(:,2)));x2max=max(x(:,2),max(xt(:,2)));x2r=x2max-x2min;
          zz2=linspace(x2min-expansion(3)*x2r,x2max+expansion(4)*x2r,gridn)';
          zd1=diff(zz1(1:2));
          zd2=diff(zz2(1:2));
          [z1,z2]=meshgrid(zz1,zz2);
          z=[z1(:),z2(:)];
          nz=length(z);
          % using Voronoi tessalation makes the cell size adpative
          rii=any(abs(gminus(xu(:,1),z(:,1)'))/zd1<.5 & abs(gminus(xu(:,2),z(:,2)'))/zd2<.5);
          z(rii,:)=[];
          % combine data and surrounding grid points
          xz=[xu;z];
          nxz=length(xz);
          % compute Voronoi cell areas
          [v,c] = voronoin(xz);
          tess_area=zeros(nxz,1);
          for i=1:nxz
            ind = c{i}';
            tess_area(i,1)=polyarea(v(ind,1) , v(ind,2));
          end

          % remove unbounded cells
          dii=isnan(tess_area);
          tess_area(dii)=[];
          % form data for GP (xx,yy,ye)
          ye=tess_area./sum(tess_area)*n;
          xx=xz;xx(dii,:)=[];
          yy=zeros(nxz,1);yy(1:nu)=counts;yy(dii,:)=[];
          
        case 'grid'
          % Parameters for a grid
          if isempty(gridn)
            % number of points in direction
            gridn=15;
          end
          if isempty(expansion)
            % range extension
            expansion=min(.5/ceil(sqrt(nu)),.25);
          end
          if numel(expansion)==1
            % symmetric range extension
            expansion(2:4)=expansion(1);
          end
          % Form regular grid to discretize the data
          x1min=min(x(:,1));x1max=max(x(:,1));x1r=x1max-x1min;
          zz1=linspace(x1min-expansion(1)*x1r,x1max+expansion(2)*x1r,gridn)';
          x2min=min(x(:,2));x2max=max(x(:,2));x2r=x2max-x2min;
          zz2=linspace(x2min-expansion(3)*x2r,x2max+expansion(4)*x2r,gridn)';
          zd1=diff(zz1(1:2));
          zd2=diff(zz2(1:2));
          [z1,z2]=meshgrid(zz1,zz2);
          z=[z1(:),z2(:)];
          nz=length(z);
          % form data for GP (xx,yy,ye)
          xx=z;
          yy=zeros(nz,1);
          zi=interp2(z1,z2,reshape(1:nz,gridn,gridn),xu(:,1),xu(:,2),'nearest');
          for i1=1:nu
            yy(zi(i1),1)=yy(zi(i1),1)+counts(i1);
          end
          ye=ones(nz,1)./nz.*n;
      end

      % Default test points
      if isnan(xt)
        [xt1,xt2]=meshgrid(linspace(x1min-expansion(1)*x1r,x1max+expansion(2)*x1r,max(100,gridn)),...
                           linspace(x2min-expansion(3)*x2r,x2max+expansion(4)*x2r,max(100,gridn)));
        xt=[xt1(:) xt2(:)];
      end
      % normalise, so that same prior is ok for different scales
      xxn=grdivide(gminus(xx,mean(xx,1)),std(xx,1));
      xtn=grdivide(gminus(xt,mean(xx,1)),std(xx,1));
      % smooth...
      [Ef,Varf]=gpsmooth(xxn,yy,ye,xtn,gpcf,latent_method,base,hyperint);
      % compute mean
      A = range(xx(:,1)).*range(xx(:,2));
      pm=exp(Ef+Varf/2)./A;

      if nargout<1
        % no output, do the plot thing
        G=zeros(size(xt1));
        G(:)=pm;
        p1=pm./sum(pm);
        pu=unique(p1);pu=pu(end:-1:1);
        pc=cumsum(pu);
        PL=[.1 .5 .8 .9];
        qi=zeros(size(PL));
        for pli=1:numel(PL)
          qi(pli)=find(pc>PL(pli),1);
        end
        pl=pu(qi).*sum(pm);
        cs=contour(xt1,xt2,G,pl);
        hl=clabel(cs);
        for hli=1:numel(hl)
          plh=get(hl(hli),'Userdata');
          if ~isempty(plh)
            set(hl(hli),'String',sprintf('%.0f%%',PL(find(plh==pl))*100));
          end
        end
      else
        p=pm;
      end
      
    otherwise
      error('X has to be Nx1 or Nx2')
  end


function [Ef,Varf] = gpsmooth(xx,yy,ye,xt,gpcf,latent_method,base,hyperint)
% Make inference with log Gaussian process and EP or Laplace approximation

  nin = size(xx,2);
  % init gp
  if strfind(func2str(gpcf),'ppcs')
    % ppcs still have nin parameter...
    gpcf1 = gpcf('init',nin);
  else
    gpcf1 = gpcf('init');
  end
  % default vague prior
  pt = prior_t('init', 's2', 10^2);
  % different covariance functions have different parameters
  if isfield(gpcf1,'magnSigma2')
     gpcf1 = gpcf('set', gpcf1, 'magnSigma2', 5, 'magnSigma2_prior', pt);
  end
  if isfield(gpcf1,'lengthScale')
     gpcf1 = gpcf('set', gpcf1, 'lengthScale', 1, 'lengthScale_prior', pt);
  end
  if isfield(gpcf1,'alpha')
    gpcf1 = gpcf('set', gpcf1, 'alpha', 20, 'alpha_prior', pt);
  end
  
  % constant term
  gpcf2 = gpcf_constant('init', 'constSigma2', 1);
  gpcf2.p.constSigma2 = [];

  % possible basis function
  switch lower(base)
    case 'unif'
      gpcf3=[];
    case 'exp'
      % add linear term in log-space
      gpcf3 = gpcf_linear('init', 'coeffSigma2', 10);
      gpcf3.p.coeffSigma2 = [];
    case 'gaussian'
      % need to expand input data
      xx=[xx xx.^2];
      xt=[xt xt.^2];
      % but use only non-expanded for the gpcf1
      if nin==1
        metric1 = metric_euclidean('init', {[1]},'lengthScales',[1],'lengthScales_prior',pt);
      else
        metric1 = metric_euclidean('init', {[1] [2]},'lengthScales',[1 1],'lengthScales_prior',pt);
      end
      gpcf1 = gpcf('set', gpcf1, 'metric', metric1);
      % add linear and quadratic term in log-space
      gpcf3 = gpcf_linear('init', 'coeffSigma2', 1);
      gpcf3.p.coeffSigma2 = [];
  end

  % Create the likelihood structure
  likelih = likelih_poisson('init', yy, ye);
  
  % Create the GP data structure
  if isempty(gpcf3)
    gp = gp_init('init', 'FULL', likelih, {gpcf1 gpcf2}, [], 'jitterSigma2', 1e-4);
  else
    gp = gp_init('init', 'FULL', likelih, {gpcf1 gpcf2 gpcf3}, [], 'jitterSigma2', 1e-4);
  end

  % prepare to optimize covariance parameters
  param = 'covariance';
  opt=optimset('GradObj','on');
  opt=optimset(opt,'TolX', 1e-3);
  opt=optimset(opt,'LargeScale', 'off');
  opt=optimset(opt,'Display', 'off');

  w0 = gp_pak(gp, param);
  mydeal = @(varargin)varargin{1:nargout};
  switch latent_method
    case 'EP'
      % Set the approximate inference method
      gp = gp_init('set', gp, 'latent_method', {'EP', xx, yy, 'covariance'});
     
      % Optimize hyperparameters
      w = fminunc(@(ww) mydeal(gpep_e(ww, gp, xx, yy, param), ...
                               gpep_g(ww, gp, xx, yy, param)), w0, opt);
      gp = gp_unpak(gp,w,param);

      % Make prediction for the test points
      if strcmpi(hyperint,'mode')
        % point estimate for the hyperparameters
        [Ef,Varf] = ep_pred(gp, xx, yy, xt, param);
      else
        % integrate over the hyperparameters
        opt = gp_iaopt([], hyperint);
        opt.validate=0;
        %[~, ~, ~, Ef, Varf] = gp_ia(opt, gp, xx, yy, xt, param);
        [notused, notused, notused, Ef, Varf]=gp_ia(opt, gp, xx, yy, xt, param);
      end
      
    case 'Laplace'
      % Set the approximate inference method
      gp = gp_init('set', gp, 'latent_method', {'Laplace', xx, yy, 'covariance'});
      if nin>1
        % default Newton explodes too often (investigations ongoing)
        gp.laplace_opt.optim_method='fminunc_large';
      end
      
      % Optimize hyperparameters
      w = fminunc(@(ww) mydeal(gpla_e(ww, gp, xx, yy, param), gpla_g(ww, gp, xx, yy, param)), w0, opt);
      gp = gp_unpak(gp,w,param);
      
      % Make prediction for the test points
      if strcmpi(hyperint,'mode')
        % point estimate for the hyperparameters
        [Ef,Varf] = la_pred(gp, xx, yy, xt, param);
      else
        % integrate over the hyperparameters
        opt = gp_iaopt([], hyperint);
        opt.validate=0;
        %[~, ~, ~, Ef, Varf] = gp_ia(opt, gp, xx, yy, xt, param);
        [notused, notused, notused, Ef, Varf] = gp_ia(opt, gp, xx, yy, xt, param);
      end
      
  end
