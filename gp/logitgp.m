function [l,lq,xx] = logitgp(x,varargin)
%function [l,lq,xt] = logitgp(x,varargin)
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
%       'latent_method' is optional 'EP' (default) or 'Laplace'
%       'int_method' is optional 'mode' (default), 'CCD' or 'grid'
% 
%     P is the estimated density
%     PQ is the 5% and 95% percentiles of the density estimate
%     XT contains the used test points
  
% Copyright (c) 2011 Jaakko Riihimäki and Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LOGITGP';
  ip.addRequired('x', @(x) isnumeric(x) && size(x,2)==1 || size(x,2)==2);
  ip.addOptional('xt',NaN, @(x) isnumeric(x) && size(x,2)==1 || size(x,2)==2);
  ip.addParamValue('gridn',[], @(x) isnumeric(x));
  ip.addParamValue('range',[], @(x) isreal(x)&&(length(x)==2||length(x)==4));
  ip.addParamValue('gpcf',@gpcf_sexp,@(x) ischar(x) || isa(x,'function_handle'));
  ip.addParamValue('latent_method','Laplace', @(x) ismember(x,{'EP' 'Laplace'}))
  ip.addParamValue('int_method','mode', @(x) ismember(x,{'mode' 'CCD', 'grid','MCMC'}))
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
      %xd=xt(2)-xt(1);
      yy=hist(x,xx)';
      
      % weight for normalization
      % w=1/gridn;
      % ye=w.*ones(gridn,1);
      
      % normalise, so that same prior is ok for different scales
      xxn=(xx-mean(xx))./std(xx);
      if nargout>0 && ~sum(isnan(xt))
        %xt=xx;
        xtn=(xt-mean(xx))./std(xx);
        [Ef,Varf]=gpsmooth(xxn,yy,[xxn; xtn],gpcf,latent_method,int_method);
        qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Varf,'upper'),Ef');
        %qr=bsxfun(@plus,randn(5000,gridn)*chol(Varf,'upper'),Ef');
        qjr=exp(qr)';
        %qjr=w.*exp(qr)';
        %qjr=w.*exp(bsxfun(@plus,qr,meanf()'))';
        % and normalize each function
        pjr=bsxfun(@rdivide,qjr,sum(qjr(1:gridn,:)));
        pjr=pjr./xd;
        
        %plot(xx,prctile(pjr(1:gridn,:)',50),xx,prctile(pjr(1:gridn,:)',2.5),'b--',xx,prctile(pjr(1:gridn,:)',97.5),'b--')
        %hold on, plot(xt,prctile(pjr(gridn+1:end,:)',50),'r',xt,prctile(pjr(gridn+1:end,:)',2.5),'r--',xt,prctile(pjr(gridn+1:end,:)',97.5),'r--')
        %hold on,plot(x,zeros(size(x)),'kx')
        
        %l=mean(pjr(gridn+1:end,:)')';
        l=prctile(pjr(gridn+1:end,:)',[50])';
        lq=prctile(pjr(gridn+1:end,:)',[2.5 97.5])';
        
%       lm=exp(Ef+Varf/2)./A.*n;
%       lq5=exp(Ef-sqrt(Varf)*1.96)./A*n;
%       lq95=exp(Ef+sqrt(Varf)*1.96)./A*n;
%       lq=[lq5 lq95];
        
      else
        % smooth...
        %[Ef,Varf]=gpsmooth(xxn,yy,ye,xtn,gpcf,latent_method,int_method);
        %[Ef,Varf]=gpsmooth(xxn,yy,xtn,gpcf,latent_method,int_method);
        [Ef,Varf]=gpsmooth(xxn,yy,xxn,gpcf,latent_method,int_method);
        
        if strcmpi(int_method,'mode')
          %[Ef,Varf]=gpsmooth(xx,yy,ye,xt,gpcf,latent_method,int_method);
          % instead sample functions from the joint distribution
          qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Varf,'upper'),Ef');
          %qr=bsxfun(@plus,randn(5000,gridn)*chol(Varf,'upper'),Ef');
          qjr=exp(qr)';
          %qjr=w.*exp(qr)';
          %qjr=w.*exp(bsxfun(@plus,qr,meanf()'))';
          % and normalize each function
          pjr=bsxfun(@rdivide,qjr,sum(qjr));
          pjr=pjr./xd;
          
          l=mean(pjr')';
          lq=prctile(pjr',[2.5 97.5])';
          
          if nargout==0
            % and compute percentiles from these
            plot(xx,prctile(pjr',50),xx,prctile(pjr',2.5),'b--',xx,prctile(pjr',97.5),'b--')
            %plot(xt,prctile(pjr',50),xt,prctile(pjr',2.5),'b--',xt,prctile(pjr',97.5),'b--')
            %hold on,plot(x,zeros(size(x)),'kx')
          end
          
        elseif strcmpi(int_method,'MCMC')
          
          PJR=zeros(size(Ef,1),size(Varf,3));
          for i1=1:size(Varf,3)
            % instead sample functions from the joint distribution
            qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Varf(:,:,i1),'upper'),Ef(:,i1)');
            %qr=bsxfun(@plus,randn(5000,gridn)*chol(Varf,'upper'),Ef');
            qjr=exp(qr)';
            %qjr=w.*exp(qr)';
            %qjr=w.*exp(bsxfun(@plus,qr,meanf()'))';
            % and normalize each function
            pjr=bsxfun(@rdivide,qjr,sum(qjr));
            pjr=pjr./xd;
            PJR(:,i1)=mean(pjr,2);
          end
          pjr=PJR;
          
          % and compute percentiles from these
          plot(xx,prctile(pjr',50),xx,prctile(pjr',2.5),'b--',xx,prctile(pjr',97.5),'b--')
          %plot(xt,prctile(pjr',50),xt,prctile(pjr',2.5),'b--',xt,prctile(pjr',97.5),'b--')
          %hold on,plot(x,zeros(size(x)),'kx')
          
          l=mean(pjr')';
          lq5=prctile(pjr',2.5)';
          lq95=prctile(pjr',97.5)';
          lq=[lq5 lq95];
        end
      end
      
      
      % percentiles
      %sqrt(sum(Varf(:))/gridn);
      %fs=sqrt(diag(Varf));
      %fp5=Ef-1.96*fs;
      %fp95=Ef+1.96*fs;
      %p5=w.*exp(fp5+meanf())/sum(qj);
      %p95=w.*exp(fp95+meanf())/sum(qj);
      
      %qj=exp(Ef);
      %qj=w.*exp(Ef);
      %pj=qj./sum(qj);
      %p5=w.*exp(fp5)/sum(qj);
      %p95=w.*exp(fp95)/sum(qj);
      % marginal variances do not respect the fact that probability mass
      % has to sum to one, and thus overall level here is unknown
      
      %plot(xx,Ef,xx,fp5,xx,fp95)
      %plot(xx,pj,xx,p5,'b--',xx,p95,'b--')
      
      % instead sample functions from the joint distribution
      %qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Varf,'upper'),Ef');
      %qr=bsxfun(@plus,randn(5000,gridn)*chol(Varf,'upper'),Ef');
      %qjr=exp(qr)';
      %qjr=w.*exp(qr)';
      %qjr=w.*exp(bsxfun(@plus,qr,meanf()'))';
      % and normalize each function
      %pjr=bsxfun(@rdivide,qjr,sum(qjr));
      %pjr=pjr./xd;
      
      % and compute percentiles from these
      %plot(xx,prctile(pjr',50),xx,prctile(pjr',2.5),'b--',xx,prctile(pjr',97.5),'b--')
      %plot(xt,prctile(pjr',50),xt,prctile(pjr',2.5),'b--',xt,prctile(pjr',97.5),'b--')
      %hold on,plot(x,zeros(size(x)),'kx')
      
%       % compute mean and quantiles
%       A=range(xx);
%       lm=exp(Ef+Varf/2)./A.*n;
%       lq5=exp(Ef-sqrt(Varf)*1.96)./A*n;
%       lq95=exp(Ef+sqrt(Varf)*1.96)./A*n;
%       lq=[lq5 lq95];
% 
%       if nargout<1
%         % no output, do the plot thing
%         newplot
%         hp=patch([xt; xt(end:-1:1)],[lq(:,1); lq(end:-1:1,2)],[.9 .9 .9]);
%         set(hp,'edgecolor',[.9 .9 .9])
%         xlim([xmin xmax])
%         line(xt,lm,'linewidth',2);
%       else
%         l=lm;
%       end
%       
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
      % Form regular grid to discretize the data
      zz1=linspace(x1min,x1max,gridn(1))';
      zz2=linspace(x2min,x2max,gridn(2))';
      [z1,z2]=meshgrid(zz1,zz2);
      z=[z1(:),z2(:)];
      nz=length(z);
      % form data for GP (xx,yy,ye)
      xx=z;
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
      
      % Default test points
      if ~isnan(xt)
       % [xt1,xt2]=meshgrid(linspace(x1min,x1max,max(50,gridn)),...
       %                    linspace(x2min,x2max,max(50,gridn)));
       % xt=[xt1(:) xt2(:)];
        xtn=bsxfun(@rdivide,bsxfun(@minus,xt,mean(xx,1)),std(xx,1));
        
        [Ef,Varf]=gpsmooth(xxn,yy,[xxn; xtn],gpcf,latent_method,int_method);
        
        if strcmpi(int_method,'mode')
          % instead sample functions from the joint distribution
          qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Varf,'upper'),Ef');
          %qr=bsxfun(@plus,randn(5000,gridn)*chol(Varf,'upper'),Ef');
          qjr=exp(qr)';
          %qjr=w.*exp(qr)';
          %qjr=w.*exp(bsxfun(@plus,qr,meanf()'))';
          % and normalize each function
          pjr=bsxfun(@rdivide,qjr,sum(qjr(1:size(xxn,1),:)));
          pjr=pjr./xd;
          
          %G=zeros(size(z1));
          %G(:)=prctile(pjr',50);
          
          l=mean(pjr(size(xxn,1)+1:end,:)')';
          lq=prctile(pjr(size(xxn,1)+1:end,:)',[2.5 97.5])';
          
          p=G(:);
          p1=p./sum(p);
          pu=unique(p1);pu=pu(end:-1:1);
          pc=cumsum(pu);
          PL=[.05 .1 .2 .5 .8 .9 .95];
          qi=[];
          for pli=1:numel(PL)
            qi(pli)=find(pc>PL(pli),1);
          end
          pl=pu(qi).*sum(p);
          contour(z1,z2,G,pl);
          %hold on, plot(x(:,1),x(:,2),'kx')
          %colorbar
        elseif strcmpi(int_method,'MCMC')
          
          % instead sample functions from the joint distribution
          qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Varf,'upper'),Ef');
          %qr=bsxfun(@plus,randn(5000,gridn)*chol(Varf,'upper'),Ef');
          qjr=exp(qr)';
          %qjr=w.*exp(qr)';
          %qjr=w.*exp(bsxfun(@plus,qr,meanf()'))';
          % and normalize each function
          pjr=bsxfun(@rdivide,qjr,sum(qjr(1:size(xxn,1),:)));
          pjr=pjr./xd;
          
          %G=zeros(size(z1));
          %G(:)=prctile(pjr',50);
          
          l=mean(pjr(size(xxn,1)+1:end,:)')';
          lq=prctile(pjr(size(xxn,1)+1:end,:)',[2.5 97.5])';
          
          %contour(z1,z2,G);
          %hold on, plot(x(:,1),x(:,2),'kx')
          %colorbar  
            
        end
        
      else
        % smooth...
        [Ef,Varf]=gpsmooth(xxn,yy,xxn,gpcf,latent_method,int_method);
        %[Ef,Varf]=gpsmooth(xxn,yy,ye,xtn,gpcf,latent_method,int_method);
        % compute mean
        
        if strcmpi(int_method,'mode')
          
          % instead sample functions from the joint distribution
          qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Varf,'upper'),Ef');
          %qr=bsxfun(@plus,randn(5000,gridn)*chol(Varf,'upper'),Ef');
          qjr=exp(qr)';
          %qjr=w.*exp(qr)';
          %qjr=w.*exp(bsxfun(@plus,qr,meanf()'))';
          % and normalize each function
          pjr=bsxfun(@rdivide,qjr,sum(qjr));
          pjr=pjr./xd;
          
          G=zeros(size(z1));
          G(:)=prctile(pjr',50);
          
          l=mean(pjr')';
          lq5=prctile(pjr',2.5)';
          lq95=prctile(pjr',97.5)';
          lq=[lq5 lq95];
          
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
          
        elseif strcmpi(int_method,'MCMC')
          
          PJR=zeros(size(Ef,1),size(Varf,3));
          for i1=1:size(Varf,3)
            % instead sample functions from the joint distribution
            qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Varf(:,:,i1),'upper'),Ef(:,i1)');
            %qr=bsxfun(@plus,randn(5000,gridn)*chol(Varf,'upper'),Ef');
            qjr=exp(qr)';
            %qjr=w.*exp(qr)';
            %qjr=w.*exp(bsxfun(@plus,qr,meanf()'))';
            % and normalize each function
            pjr=bsxfun(@rdivide,qjr,sum(qjr));
            pjr=pjr./xd;
            PJR(:,i1)=mean(pjr,2);
          end
          pjr=mean(PJR,2);
          
          G=zeros(size(z1));
          %G(:)=prctile(pjr',50);
          G(:)=mean(PJR,2);
          
          
          l=mean(pjr')';
          lq5=prctile(pjr',2.5)';
          lq95=prctile(pjr',97.5)';
          lq=[lq5 lq95];
          
          contour(z1,z2,G);
          %hold on, plot(x(:,1),x(:,2),'kx')
          colorbar
        end
      end
      
      
      
%       
%       
%       % smooth...
%       [Ef,Varf]=gpsmooth(xxn,yy,xtn,gpcf,latent_method,int_method);
%       %[Ef,Varf]=gpsmooth(xxn,yy,ye,xtn,gpcf,latent_method,int_method);
%       % compute mean
%       
%       % instead sample functions from the joint distribution
%       qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Varf,'upper'),Ef');
%       %qr=bsxfun(@plus,randn(5000,gridn)*chol(Varf,'upper'),Ef');
%       qjr=exp(qr)';
%       %qjr=w.*exp(qr)';
%       %qjr=w.*exp(bsxfun(@plus,qr,meanf()'))';
%       % and normalize each function
%       pjr=bsxfun(@rdivide,qjr,sum(qjr));
%       pjr=pjr./xd;
%       
%       G=zeros(size(xt1));
%       G(:)=prctile(pjr',50);
%       
%       lq5=prctile(pjr',2.5);
%       lq95=prctile(pjr',97.5);
%       lq=[lq5 lq95];
%       
%       contour(xt1,xt2,G);
%       hold on, plot(x(:,1),x(:,2),'kx')
%       colorbar
      
%      pcolor(xt1,xt2,G);
%      shading flat
%      colormap('jet')
%      %cx=caxis;
%      %cx(1)=0;
%      %caxis(cx);
%      colorbar
      
      
      
%       A = range(xx(:,1)).*range(xx(:,2));
%       lm=exp(Ef+Varf/2)./A.*n;
%       lq5=exp(Ef-sqrt(Varf)*1.96)./A.*n;
%       lq95=exp(Ef+sqrt(Varf)*1.96)./A.*n;
%       lq=[lq5 lq95];
% 
%       if nargout<1
%         % no output, do the plot thing
%         G=zeros(size(xt1));
%         G(:)=lm;
%         pcolor(xt1,xt2,G);
%         shading flat
%         colormap('jet')
%         cx=caxis;
%         cx(1)=0;
%         caxis(cx);
%         colorbar
%       else
%         l=lm;
%       end
      
    otherwise
      error('X has to be Nx1 or Nx2')
  end

end

function [Ef,Varf] = gpsmooth(xx,yy,xxt,gpcf,latent_method,int_method)
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
  
  % Create the GP structure
  %gp = gp_de_set('lik', lik_logitgp, 'cf', {gpcf1}, 'jitterSigma2', 1e-4);
  gpmfco = gpmf_constant('prior_mean',0,'prior_cov',100);
  gpmflin = gpmf_linear('prior_mean',0,'prior_cov',100);
  gpmfsq = gpmf_squared('prior_mean',0,'prior_cov',100);
  %gp = gp_de_set('lik', lik_logitgp, 'cf', {gpcf1}, 'jitterSigma2', 1e-4);
  gp = gp_set('lik', lik_logitgp, 'cf', {gpcf1}, 'jitterSigma2', 1e-4, 'meanf', {gpmfco,gpmflin,gpmfsq});
  %gp = gp_de_set('lik', lik_logitgp, 'cf', {gpcf1}, 'jitterSigma2', 1e-4, 'meanf', {gpmfco,gpmflin});
  
  % Make prediction for the test points
  if strcmpi(int_method,'mode')
    % Set the approximate inference method
    gp = gp_set(gp, 'latent_method', latent_method);
    
    %gpla_nd_e(gp_pak(gp),gp, xx, yy)
    %gpla_nd_g(gp_pak(gp),gp, xx, yy);
    %gradcheck(gp_pak(gp),@gpla_nd_e,@gpla_nd_g,gp,xx,yy)
    %gradcheck(randn(size(gp_pak(gp))),@gpla_nd_e,@gpla_nd_g,gp,xx,yy)
    
    %opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter','Derivativecheck','on');
    opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','off','Derivativecheck','off');
    gp=gp_optim(gp,xx,yy,'opt',opt);
    
    %opt = scg2_opt;
    %opt.tolfun = 1e-3;
    %opt.tolx = 1e-3;
    %opt.display = 1;
    %opt.maxiter = 100;
    %opt.display=2;
    %fopt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','off');
    %mydeal = @(varargin)varargin{1:nargout};
    %wopt=fminscg(@(ww) gpla_de_eg(ww,gp,xx,yy), gp_pak(gp), fopt);
    %wopt=scg2(@gpla_de_e, gp_pak(gp), opt, @gpla_de_g, gp, xx, yy);
    %gp=gp_unpak(gp,wopt);
    %  % Optimize hyperparameters
    %  opt=optimset('TolX', 1e-3, 'Display', 'off');
    %  if exist('fminunc')
    %    gp = gp_optim(gp, xx, yy, 'z', ye, 'optimf', @fminunc, 'opt', opt);
    %  else
    %    gp = gp_optim(gp, xx, yy, 'z', ye, 'optimf', @fminscg, 'opt', opt);
    %  end
    
    
    % point estimate for the hyperparameters
    %[Ef,Varf] = gp_pred(gp, xx, yy, xt, 'z', ye);
    
    [Ef,Varf] = gp_pred(gp, xx, yy, xxt);
    %[Ef,Varf] = gpla_de_pred(gp, xx, yy, xxt);
    %[Ef,Varf] = gpla_de_pred(gp, xx, yy, xx, 'z', ye);
  elseif strcmpi(int_method,'MCMC')
      
    gp = gp_de_set(gp, 'latent_method', latent_method);
    
    %gpla_de_e(gp_pak(gp),gp, xx, yy);
    %gpla_de_g(gp_pak(gp),gp, xx, yy);
    %gradcheck(gp_pak(gp),@gpla_de_e,@gpla_de_g,gp,xx,yy)
    opt = scg2_opt;
    opt.tolfun = 1e-3;
    opt.tolx = 1e-3;
    opt.display = 1;
    opt.maxiter = 100;
    opt.display=2;
    
    wopt=scg2(@gpla_de_e, gp_pak(gp), opt, @gpla_de_g, gp, xx, yy);
    %clear gp
    %gpcf1 = gpcf(gpcf1, 'magnSigma2_prior', prior_fixed(),'lengthScale_prior', prior_fixed());
    %gpcf1 = gpcf_sexp(gpcf1, 'magnSigma2_prior', prior_fixed(),'lengthScale_prior', prior_fixed());
    %gpcf1 = gpcf_sexp();
    %gpcf1 = gpcf(gpcf1, 'magnSigma2_prior', prior_fixed(),'lengthScale_prior', prior_fixed());
    gp = gp_de_set('lik', lik_logitgp, 'cf', {gpcf1}, 'jitterSigma2', 1e-4, 'meanf', {gpmfco,gpmflin,gpmfsq});
    %gp = gp_de_set('lik', lik_logitgp, 'cf', {gpcf1}, 'jitterSigma2', 1e-4);
    
    %
    %gp = gp_de_set(gp, 'latent_method', {'MCMC', zeros(size(yy))', @scaled_mh});
    gp = gp_de_set(gp, 'latent_method', 'MCMC', 'jitterSigma2', 1e-4);
    gp=gp_unpak(gp,wopt);
    
    %gp = gp_de_set('lik', lik_logitgp, 'cf', {gpcf1}, 'jitterSigma2', 1e-4);
    %gp = gp_de_set('lik', lik_logitgp, 'cf', {gpcf1}, 'jitterSigma2', 1e-4, 'meanf', {gpmfco,gpmflin,gpmfsq});
    
    
    %gp.cf{1}.p.lengthScale=[];
    %gp.cf{1}.p.magnSigma2=[];
    
    % Set the parameters for MCMC
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
    %[r,g,opt]=gp_mc(gp, xx, yy, 'hmc_opt', hmc_opt, 'latent_opt', latent_opt, 'nsamples', 1, 'repeat', 15);
    [r,g,opt]=gp_mc(gp, xx, yy, 'latent_opt', latent_opt, 'nsamples', 1, 'repeat', 15);
    
    % re-set some of the sampling options
    hmc_opt.steps=4;
    hmc_opt.stepadj=0.05;
    %latent_opt.repeat = 5;
    hmc2('state', sum(100*clock));
    
    % The second stage sampling
    % Notice that previous record r is given as an argument
    [rgp,g,opt]=gp_mc(gp, xx, yy, 'nsamples', 500,'latent_opt', latent_opt, 'record', r);
    %[rgp,g,opt]=gp_mc(gp, xx, yy, 'nsamples', 500, 'hmc_opt', hmc_opt, 'latent_opt', latent_opt, 'record', r);
    % Remove burn-in
    %rgp=thin(rgp,102);
    rgp=thin(rgp,102,4);
    
    %[Ef, Varf] = gpmc_jpred(rgp, xx, yy, xxt);
    [Ef, Varf] = gpmc_jpreds(rgp, xx, yy, xxt);
     
  else
    % integrate over the hyperparameters
    %[~, ~, ~, Ef, Varf] = gp_ia(opt, gp, xx, yy, xt, param);
    [notused, notused, notused, Ef, Varf]=...
        gp_ia(gp, xx, yy, xt, 'z', ye, 'int_method', int_method);
  end
  
end
