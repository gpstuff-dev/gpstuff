function [p,pq,xx] = lgpdens(x,varargin)
%LGPDENS Logistic-Gaussian Process density estimate for 1D and 2D data
% 
%  Description  
%    LGPDENS(X,OPTIONS) Compute and plot LGP density estimate. X is
%    1D or 2D point data. For 1D data plot the mean and 95% region. 
%    For 2D data plot the density contours.
%  
%    [P,PQ,XT] = LGPDENS(X,OPTIONS) Compute LGP density estimate
%    and return mean density P, 5% and 95% percentiles PQ, and
%    grid locations.
%  
%    [P,PQ,XT] = LGPDENS(X,XT,OPTIONS) Compute LGP density estimate
%    in the given grid locations XT.
%  
%    OPTIONS is optional parameter-value pair
%      gridn     - optional number of grid points used in each axis direction
%                  default is 400 for 1D, 20 for 2D.
%      range     - tells the estimation range, default is 
%                  [min(min(x),mean(x)-3*std(x)), max(max(x),mean(x)+3*std(x))]
%                  for 1D [XMIN XMAX]
%                  for 2D [X1MIN X1MAX X2MIN X2MAX]
%      gpcf      - optional function handle of a GPstuff covariance function 
%                      (default is @gpcf_sexp)
%      latent_method - optional 'Laplace' (default) or 'MCMC'
%      int_method    - optional 'mode' (default), 'CCD' or 'grid'
%                      if latent_method is 'MCMC' then int_method is 'MCMC'
%      display   - defines if messages are displayed. 
%                  'off' (default) displays no output
%                  'on' gives some output  
%                  'iter' displays output at each iteration
%      speedup   - defines if speed-up is used.
%                  'off' (default) no speed-up is used
%                  'on' With SEXP or EXP covariance function in 2D case
%                  uses Kronecker product structure and approximates the
%                  full posterior with a low-rank approximation. Otherwise
%                  with SEXP, EXP, MATERN32 and MATERN52 covariance
%                  functions in 1D and 2D cases uses FFT/FFT2 matrix-vector
%                  multiplication speed-up in the Newton's algorithm.
%      cond_dens - defines if conditional density estimate is computed.
%                  'off' (default) no conditional density
%                  'on' computes for 2D the conditional median density
%                  estimate p(x2|x1) when the matrix [x1 x2] is given as
%                  input. 
%      basis     - defines if basis functions are used. 
%                       'gaussian' (default) uses linear and quadratic
%                         basis functions on latent space implying
%                         centering on Gaussian distribution
%                       'exp' uses linear basis function on latent
%                         space implying centering on exponential distribution
%                       'off' no basis functions
%      bounded   - in 1D case tells if the density is bounded from left
%                  or right (default is [0 0]). In unbounded case,
%                  decreasing tails are assumed.

% Copyright (c) 2011-2012 Jaakko Riihimäki and Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LGPDENS';
  ip=iparser(ip,'addRequired','x', @(x) isnumeric(x) ...
                 && (size(x,2)==1 || size(x,2)==2));
  ip=iparser(ip,'addOptional','xt',NaN, @(x) isnumeric(x) && size(x,2)==1 || size(x,2)==2);
  ip=iparser(ip,'addParamValue','gridn',[], @(x) isnumeric(x));
  ip=iparser(ip,'addParamValue','range',[], @(x) isempty(x)||isreal(x)&&(length(x)==2||length(x)==4));
  ip=iparser(ip,'addParamValue','gpcf',@gpcf_sexp,@(x) ischar(x) || isa(x,'function_handle'));
  ip=iparser(ip,'addParamValue','latent_method','Laplace', @(x) ismember(x,{'EP' 'Laplace' 'MCMC'}));
  ip=iparser(ip,'addParamValue','int_method','mode', @(x) ismember(x,{'mode' 'CCD', 'grid'}));
  ip=iparser(ip,'addParamValue','normalize',false, @islogical);
  ip=iparser(ip,'addParamValue','display', 'off', @(x) islogical(x) || ...
                   ismember(x,{'on' 'off' 'iter'}));
  ip=iparser(ip,'addParamValue','speedup',[], @(x) ismember(x,{'on' 'off'}));
  ip=iparser(ip,'addParamValue','cond_dens',[], @(x) ismember(x,{'on' 'off'}));
  ip=iparser(ip,'addParamValue','basis','gaussian', @(x) ismember(x,{'gaussian' 'exp' 'off'}));
  ip=iparser(ip,'addParamValue','bounded',[0 0], @(x) isnumeric(x) && min(size(x))==1 && max(size(x))==2);
  ip=iparser(ip,'parse',x,varargin{:});
  x=ip.Results.x;
  xt=ip.Results.xt;
  gridn=ip.Results.gridn;
  xrange=ip.Results.range;
  gpcf=ip.Results.gpcf;
  latent_method=ip.Results.latent_method;
  int_method=ip.Results.int_method;
  normalize=ip.Results.normalize;
  display=ip.Results.display;
  speedup=ip.Results.speedup;
  cond_dens=ip.Results.cond_dens;
  basis=ip.Results.basis;
  bounded=ip.Results.bounded;

  x(any(~isfinite(x),2),:)=[];
  [n,m]=size(x);
  if n<2
    error('Number of finite values in x is too small');
  end
  
  switch m
    case 1 % 1D
      if ~isempty(cond_dens) && strcmpi(cond_dens,'on')
        error('LGPDENS: the input x must be 2D if cond_dens option is ''on''.')
      end
      % Parameters for a grid
      if isempty(gridn)
        % number of points
        gridn=400;
      end
      xmin=min(x);xmax=max(x);
      if ~isempty(xrange)
        % extend given range to include min(x) and max(x)
        xmin=min(xmin,xrange(1));
        xmax=max(xmax,xrange(2));
      elseif ~isnan(xt)
        % use xt to define range and 
        % extend it to include min(x) and max(x)
        xmin=min(xmin,min(xt));
        xmax=max(xmax,max(xt));
      else
        xmin=min(xmin,mean(x)-3*std(x));
        xmax=max(xmax,mean(x)+3*std(x));
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
      
      % normalise, so that same prior is ok for different scales
      xxn=(xx-mean(xx))./std(xx);
      
      [Ef,Covf]=gpsmooth(xxn,yy,xxn,gpcf,latent_method,int_method,display,speedup,gridn,cond_dens,basis);
      
      if strcmpi(latent_method,'MCMC')
        PJR=zeros(size(Ef,1),size(Covf,3));
        for i1=1:size(Covf,3)
          qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Covf(:,:,i1),'upper'),Ef(:,i1)');
          if ~any(bounded)
            qii=find(qr(:,1)<qr(:,2)&qr(:,end-1)>qr(:,end));
          elseif bounded(1)&~bounded(2)
            qii=find(qr(:,end-1)>qr(:,end));
          elseif bounded(2)&~bounded(1)
            qii=find(qr(:,1)<qr(:,2));
          else
            qii=1:1000;
          end
          qr=qr(qii,:);
          qjr=exp(qr)';
          pjr=bsxfun(@rdivide,qjr,sum(qjr));
          pjr=pjr./xd;
          PJR(:,i1)=mean(pjr,2);
        end
        pjr=PJR;
      else
        qr=bsxfun(@plus,randn(2000,size(Ef,1))*chol(Covf,'upper'),Ef');
        if ~any(bounded)
          qii=find(qr(:,1)<qr(:,2)&qr(:,end-1)>qr(:,end));
        elseif bounded(1)&~bounded(2)
          qii=find(qr(:,end-1)>qr(:,end));
        elseif bounded(2)&~bounded(1)
          qii=find(qr(:,1)<qr(:,2));
        else
          qii=1:2000;
        end
        if numel(qii)>200
          qr=qr(qii,:);
        else
          warning('Rejection sampling to force decreasing tails for (semi)unbounded failed')
        end
        qjr=exp(qr)';
        pjr=bsxfun(@rdivide,qjr,sum(qjr(1:gridn,:)));
        pjr=pjr./xd;
      end
      pp=mean(pjr')';
      ppq=prctile(pjr',[5 95])';
      
      if nargout<1
        % no output, do the plot thing
        newplot
        hp=patch([xx; xx(end:-1:1)],[ppq(:,1); ppq(end:-1:1,2)],[.8 .8 .8]);
        set(hp,'edgecolor',[.8 .8 .8])
        xlim([xmin xmax])
        line(xx,pp,'linewidth',2);
      else
        p=pp;
        pq=ppq;
      end
      
    case 2 % 2D
      
      if ~isempty(cond_dens) && strcmpi(cond_dens,'on') && ~isempty(speedup) && strcmp(speedup, 'on')
        warning('No speed-up option available with the cond_dens option. Using full covariance instead.')
        speedup='off';
      end
      % Find unique points
      [xu,I,J]=unique(x,'rows');
      % and count number of repeated x's
      counts=crosstab(J); 
      nu=length(xu);
      
      % Parameters for a grid
      if isempty(gridn)
        % number of points in each direction
        gridn=20;
      end
      if numel(gridn)==1
        gridn(2)=gridn(1);
      end
      x1min=min(x(:,1));x1max=max(x(:,1));
      x2min=min(x(:,2));x2max=max(x(:,2));
      if ~isempty(xrange)
        % extend given range to include min(x) and max(x)
        x1min=min(x1min,xrange(1));
        x1max=max(x1max,xrange(2));
        x2min=min(x2min,xrange(3));
        x2max=max(x2max,xrange(4));
      elseif ~isnan(xt)
        % use xt to define range and 
        % extend it to include min(x) and max(x)
        x1min=min(x1min,min(xt(:,1)));
        x1max=max(x1max,max(xt(:,1)));
        x2min=min(x2min,min(xt(:,2)));
        x2max=max(x2max,max(xt(:,2)));
      else
        x1min=min(x1min,mean(x(:,1))-3*std(x(:,1)));
        x1max=max(x1max,mean(x(:,1))+3*std(x(:,1)));
        x2min=min(x2min,mean(x(:,2))-3*std(x(:,2)));
        x2max=max(x2max,mean(x(:,2))+3*std(x(:,2)));
      end
      
      % Discretize the data
      if isnan(xt)
        % Form regular grid to discretize the data
        zz1=linspace(x1min,x1max,gridn(1))';
        zz2=linspace(x2min,x2max,gridn(2))';
        [z1,z2]=meshgrid(zz1,zz2);
        z=[z1(:),z2(:)];
        nz=length(z);
        xx=z;
        if ~isempty(cond_dens) && strcmpi(cond_dens,'on')
          % use ntx2 times more grid points for predictions
          if gridn(2)>10
            ntx2=3;
          else
            ntx2=10;
          end
          zzt1=linspace(x1min,x1max,gridn(1))';
          zzt2=linspace(x2min,x2max,gridn(2)*ntx2)';
          [zt1,zt2]=meshgrid(zzt1,zzt2);
          zt=[zt1(:),zt2(:)];
          %nzt=length(zt);
          xt=zt;
        end
      else
        xx=xt;
        gridn=[length(unique(xx(:,1))) length(unique(xx(:,2)))];
        z1=reshape(xx(:,1),gridn(2),gridn(1));
        z2=reshape(xx(:,2),gridn(2),gridn(1));
        nz=numel(z1);
      end
      yy=zeros(nz,1);
      zi=interp2(z1,z2,reshape(1:nz,gridn(2),gridn(1)),xu(:,1),xu(:,2),'nearest');
      for i1=1:nu
        if ~isnan(zi(i1))
          yy(zi(i1),1)=yy(zi(i1),1)+counts(i1);
        end
      end
      %ye=ones(nz,1)./nz.*n;
      
      unx1=unique(xx(:,1));
      unx2=unique(xx(:,2));
      xd=(unx1(2)-unx1(1))*(unx2(2)-unx2(1));
      
      % normalise, so that same prior is ok for different scales
      xxn=bsxfun(@rdivide,bsxfun(@minus,xx,mean(xx,1)),std(xx,1));
      if ~isempty(cond_dens) && strcmpi(cond_dens,'on')
        xxtn=bsxfun(@rdivide,bsxfun(@minus,xt,mean(xx,1)),std(xx,1));
      end
      
      % [Ef,Covf]=gpsmooth(xxn,yy,[xxn; xtn],gpcf,latent_method,int_method);
      if ~isempty(cond_dens) && strcmpi(cond_dens,'on')
        [Ef,Covf]=gpsmooth(xxn,yy,xxtn,gpcf,latent_method,int_method,display,speedup,gridn,cond_dens,basis);
      else
        [Ef,Covf]=gpsmooth(xxn,yy,xxn,gpcf,latent_method,int_method,display,speedup,gridn,cond_dens,basis);
      end
      
      if strcmpi(latent_method,'MCMC')
        
        if ~isempty(cond_dens) && strcmpi(cond_dens,'on')
          unx2=(unique(xt(:,2)));
          xd2=(unx2(2)-unx2(1));
          PJR=zeros(size(Ef,1),size(Covf,3));
          for i1=1:size(Covf,3)
            qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Covf(:,:,i1),'upper'),Ef(:,i1)');
            qjr=exp(qr)';
            %pjr=bsxfun(@rdivide,qjr,sum(qjr));
            pjr=qjr;
            pjr2=reshape(pjr,[gridn(2)*ntx2 gridn(1) size(pjr,2)]);
            for j1=1:size(pjr2,3)
              pjr2(:,:,j1)=bsxfun(@rdivide,pjr2(:,:,j1),sum(pjr2(:,:,j1)))./xd2;
            end
            pjr=reshape(pjr2,[gridn(2)*ntx2*gridn(1) size(pjr,2)]);
            PJR(:,i1)=mean(pjr,2);
          end
          pjr=PJR;
          %qp=median(pjr2,3);
          %qp=bsxfun(@rdivide,qp,sum(qp,1));
          
        else  
          
          PJR=zeros(size(Ef,1),size(Covf,3));
          for i1=1:size(Covf,3)
            qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Covf(:,:,i1),'upper'),Ef(:,i1)');
            qjr=exp(qr)';
            pjr=bsxfun(@rdivide,qjr,sum(qjr));
            pjr=pjr./xd;
            PJR(:,i1)=mean(pjr,2);
          end
          pjr=PJR;
          %pjr=mean(PJR,2);
        end
        
      else
        if strcmpi(speedup,'on') && length(Covf)==2
          qr1=bsxfun(@plus,bsxfun(@times,randn(1000,size(Ef,1)),sqrt(Covf{1})'),Ef');
          qr2=randn(1000,size(Covf{2},1))*Covf{2};
          qr=qr1+qr2;
        else
          qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Covf,'upper'),Ef');
        end
        qjr=exp(qr)';
        if ~isempty(cond_dens) && strcmpi(cond_dens,'on') 
          pjr=zeros(size(qjr));
          unx2=unique(xt(:,2));
          xd2=(unx2(2)-unx2(1));
          for k1=1:size(qjr,2)
            qjrtmp=reshape(qjr(:,k1),[gridn(2)*ntx2 gridn(1)]);
            qjrtmp=bsxfun(@rdivide,qjrtmp,sum(qjrtmp));
            qjrtmp=qjrtmp./xd2;
            pjr(:,k1)=qjrtmp(:);
          end
        else
          pjr=bsxfun(@rdivide,qjr,sum(qjr));
          pjr=pjr./xd;
        end
      end
      
      %if ~isempty(cond_dens) && strcmpi(cond_dens,'on')
      %  pp=median(pjr')';
      %else
      pp=mean(pjr')';
      %end
      ppq=prctile(pjr',[5 95])';
      
      if nargout<1
        % no output, do the plot thing
        if ~isempty(cond_dens) && strcmpi(cond_dens,'on')
          pjr2=reshape(pjr,[gridn(2)*ntx2 gridn(1) size(pjr,2)]);
          %qp=median(pjr2,3);
          qp=mean(pjr2,3);
          qp=bsxfun(@rdivide,qp,sum(qp,1));
          qpc=cumsum(qp,1);
          PL=[.05 .1 .2 .5 .8 .9 .95];
          for i1=1:gridn(1)
            pc=qpc(:,i1);
            for pli=1:numel(PL),
              qi(pli)=find(pc>PL(pli),1);
            end,
            ql(:,i1)=unx2(qi);
          end
          hold on
          h1=plot(zz1,ql(4,:)','-', 'color', [0 0 255]./255,'linewidth',2);
          h2=plot(zz1,ql([3 5],:)','--', 'color', [0 127 0]./255,'linewidth',1);
          h3=plot(zz1,ql([2 6],:)','-.', 'color', [255 0 0]./255,'linewidth',1);
          h4=plot(zz1,ql([1 7],:)',':', 'color', [0 0 0]./255,'linewidth',1);
          hold off
          legend([h1 h2(1) h3(1) h4(1)],'.5','.2/.8','.1/.9','.05/.95')
          %plot(zz1,ql','linewidth',1)
          %legend('.05','.1','.2','.5','.8','.9','.95')
          xlim([x1min x1max])
          ylim([x2min x2max])
        else
          G=zeros(size(z1));
          G(:)=prctile(pjr',50);
          %contour(z1,z2,G);
          pp=G(:);
          p1=pp./sum(pp);
          pu=sort(p1,'ascend');
          pc=cumsum(pu);
          PL=[.05 .1 .2 .5 .8 .9 .95];
          qi=[];
          for pli=1:numel(PL)
            qi(pli)=find(pc>PL(pli),1);
          end
          pl=pu(qi).*sum(pp);
          contour(z1,z2,G,pl);
          %hold on, plot(x(:,1),x(:,2),'kx')
          %colorbar
        end
      else
        p=pp;
        pq=ppq;
      end
    otherwise
      error('X has to be Nx1 or Nx2')
  end
end

function [Ef,Covf] = gpsmooth(xx,yy,xxt,gpcf,latent_method,int_method,display,speedup,gridn,cond_dens,basis)
% Make inference with log Gaussian process and EP or Laplace approximation

  % gp_mc and gp_ia still uses numeric display option
  if strcmp(display,'off')
    displ=0;
  else
    displ=1;
  end
  
  nin = size(xx,2);
  % init gp
  if strfind(func2str(gpcf),'ppcs')
    % ppcs still have nin parameter...
    gpcf1 = gpcf('nin',nin);
  else
    gpcf1 = gpcf();
  end
  
  % weakly informative priors
  % prior based on guess of maximum differences in log densities
  pm = prior_sqrtt('s2',10^2,'nu',1);
  % Weakly informative prior states that probability is smaller for
  % lengthscales which are much smaller than Silverman's rule of thumb
  % or min grid distance
  h=max(diff(xx(1:2,end)).^2,1/sum(yy).^(1/5)/2);
  if size(xx,2)==2
    h=sqrt(h);
  end
  pl = prior_logt('s2', 2, 'mu', log(h),'nu',1);
  pa = prior_t('s2', 20^2, 'nu', 1);
  %pm = prior_sqrtt('s2', 10^2, 'nu', 4);
  %pl = prior_sinvchi2('s2', 1, 'nu', 1);
  %pl = prior_logunif();
  %pa = prior_t('s2', 10^2, 'nu', 4);
  % different covariance functions have different parameters
  if isfield(gpcf1,'magnSigma2')
     gpcf1 = gpcf(gpcf1, 'magnSigma2', 1, 'magnSigma2_prior', pm);
  end
  if isfield(gpcf1,'lengthScale')
     gpcf1 = gpcf(gpcf1, 'lengthScale', h*2, 'lengthScale_prior', pl);
  end
  if isfield(gpcf1,'alpha')
    gpcf1 = gpcf(gpcf1, 'alpha', 20, 'alpha_prior', pa);
  end
  if isfield(gpcf1,'biasSigma2')
    gpcf1 = gpcf(gpcf1, 'biasSigma2', 10, 'weightSigma2', 10,'biasSigma2_prior',prior_sinvchi2('s2',1^2,'nu',1),'weightSigma2_prior',prior_sinvchi2('s2',1^2,'nu',1));
  end
  
  if ~isempty(cond_dens) && strcmp(cond_dens, 'on')
    lik=lik_lgpc;
    lik.gridn=gridn;
  else
    lik=lik_lgp;
  end
  
  % Create the GP structure
  if ~isempty(basis) && strcmp(basis, 'off')
    gp = gp_set('lik', lik, 'cf', {gpcf1}, 'jitterSigma2', 1e-4);
  elseif strcmp(basis, 'exp')
    gpmflin = gpmf_linear('prior_mean',0,'prior_cov',100);
    gp = gp_set('lik', lik, 'cf', {gpcf1}, 'jitterSigma2', 1e-4, 'meanf', {gpmflin});
  else
    gpmflin = gpmf_linear('prior_mean',0,'prior_cov',100);
    gpmfsq = gpmf_squared('prior_mean',0,'prior_cov',100,'interactions','on');
    gp = gp_set('lik', lik, 'cf', {gpcf1}, 'jitterSigma2', 1e-4, 'meanf', {gpmflin,gpmfsq});
  end
  % First optimise hyperparameters using Laplace approximation
  gp = gp_set(gp, 'latent_method', 'Laplace');
  opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display',display);
  
  if ~isempty(speedup) && strcmp(speedup, 'on')
    gp.latent_opt.gridn=gridn;
    gp.latent_opt.pcg_tol=1e-12;
    if size(xx,2)==2 && (strcmp(gp.cf{1}.type,'gpcf_sexp') || strcmp(gp.cf{1}.type,'gpcf_exp'))
      % exclude eigenvalues smaller than 1e-6 or take 50%
      % eigenvalues at most
      gp.latent_opt.eig_tol=1e-6;
      gp.latent_opt.eig_prct=0.5;
      gp.latent_opt.kron=1;
      opt.LargeScale='off';
      if norm(xx-xxt)~=0
        warning('In the low-rank approximation the grid locations xx are used instead of xxt in predictions.')
        xxt=xx;
      end
    elseif strcmp(gp.cf{1}.type,'gpcf_sexp') || strcmp(gp.cf{1}.type,'gpcf_exp') || strcmp(gp.cf{1}.type,'gpcf_matern32') || strcmp(gp.cf{1}.type,'gpcf_matern52')
      gp.latent_opt.fft=1;
    end
  end

  gp=gp_optim(gp,xx,yy,'opt',opt, 'optimf', @fminlbfgs);
  %gradcheck(gp_pak(gp), @gpla_nd_e, @gpla_nd_g, gp, xx, yy);
  %exp(gp_pak(gp))
  
  if strcmpi(latent_method,'MCMC')
    gp = gp_set(gp, 'latent_method', 'MCMC');
    
    %if ~isempty(cond_dens) && strcmpi(cond_dens,'on')
    if size(xx,2)==2
      % add more jitter for 2D cases with MCMC
      gp = gp_set(gp, 'jitterSigma2', 1e-2);
      %error('LGPDENS: MCMC is not implemented if cond_dens option is ''on''.')
    end
    
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
    [r,g,opt]=gp_mc(gp, xx, yy, 'hmc_opt', hmc_opt, 'latent_opt', latent_opt, 'nsamples', 1, 'repeat', 15, 'display', displ);
    %[r,g,opt]=gp_mc(gp, xx, yy, 'latent_opt', latent_opt, 'nsamples', 1, 'repeat', 15);
    
    % re-set some of the sampling options
    hmc_opt.steps=4;
    hmc_opt.stepadj=0.05;
    %latent_opt.repeat = 5;
    hmc2('state', sum(100*clock));
    
    % The second stage sampling
    % Notice that previous record r is given as an argument
    [rgp,g,opt]=gp_mc(gp, xx, yy, 'hmc_opt', hmc_opt, 'nsamples', 500,'latent_opt', latent_opt, 'record', r,  'display', displ);
    % Remove burn-in
    rgp=thin(rgp,102,4);
    
    [Ef, Covf] = gpmc_jpreds(rgp, xx, yy, xxt);
     
  else
    if strcmpi(int_method,'mode')
      % Just make prediction for the test points
      [Ef,Covf] = gp_pred(gp, xx, yy, xxt);
    else
      % integrate over the hyperparameters
      %[~, ~, ~, Ef, Covf] = gp_ia(opt, gp, xx, yy, xt, param);
      gpia=gp_ia(gp, xx, yy, 'int_method', int_method, 'display', displ);
      [Ef, Covf]=gpia_jpred(gpia, xx, yy, xxt);
    end
  end
end
