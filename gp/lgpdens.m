function [p,pq,xx,pjr,gp,ess,eig,q,r] = lgpdens(x,varargin)
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
%
%  Reference
%
%    Jaakko Riihimäki and Aki Vehtari (2014). Laplace approximation
%    for logistic Gaussian process density estimation and
%    regression. Bayesian analysis, 9:425-448
%
% Copyright (c) 2011-2013,2015 Aki Vehtari
% Copyright (c) 2011-2013 Jaakko Riihimäki
% Copyright (c) 2013 Enrique Lelo de Larrea Andrade

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LGPDENS';
  ip.addRequired('x', @(x) isnumeric(x) ...
                 && (size(x,2)==1 || size(x,2)==2));
  ip.addOptional('xt',NaN, @(x) isnumeric(x) && size(x,2)==1 || size(x,2)==2);
  ip.addParamValue('gridn',[], @(x) isnumeric(x));
  ip.addParamValue('range',[], @(x) isempty(x)||isreal(x)&&(length(x)==2||length(x)==4));
  ip.addParamValue('gpcf',@gpcf_sexp,@(x) ischar(x) || isa(x,'function_handle'));
  ip.addParamValue('latent_method','Laplace', @(x) ismember(x,{'EP' 'Laplace' 'MCMC'}))
  ip.addParamValue('int_method','mode', @(x) ismember(x,{'mode' 'CCD', 'grid'}))
  ip.addParamValue('normalize',false, @islogical);
  ip.addParamValue('display', 'off', @(x) islogical(x) || ...
                   ismember(x,{'on' 'off' 'iter'}))
  ip.addParamValue('speedup',[], @(x) ismember(x,{'on' 'off'}));
  ip.addParamValue('rej_sampling','on', @(x) ismember(x,{'on' 'off'}));
  ip.addParamValue('imp_sampling','on', @(x) ismember(x,{'on' 'off' 'is' 'tis' 'psis'}));
  ip.addParamValue('cond_dens',[], @(x) ismember(x,{'on' 'off'}));
  ip.addParamValue('basis','gaussian', @(x) ismember(x,{'gaussian' 'exp' 'off'}));
  ip.addParamValue('bounded',[0 0], @(x) isnumeric(x) && min(size(x))==1 && max(size(x))==2);
  
  % additional undocumented parameters for importance sampling
  ip.addParamValue('n_is',8000, @(x) isnumeric(x) && x>=0);
  ip.addParamValue('n_scale',50, @(x) isnumeric(x) && x>=0);
  ip.addParamValue('contSplit','on', @(x) ismember(x,{'on' 'off'}));

  ip.parse(x,varargin{:});
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
  rej_sampling=ip.Results.rej_sampling;
  imp_sampling=ip.Results.imp_sampling;
  cond_dens=ip.Results.cond_dens;
  basis=ip.Results.basis;
  bounded=ip.Results.bounded;
  
  
  % additional undocumented parameters for importance sampling
  n_is = floor(ip.Results.n_is);
  n_scale = floor(ip.Results.n_scale);
  contSplit = ip.Results.contSplit;

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
      
      gp=gpsmooth(xxn,yy,gpcf,latent_method,int_method,display,speedup,gridn,cond_dens,basis,imp_sampling);
      
      if strcmpi(latent_method,'MCMC')
        [Ef, Covf] = gpmc_jpreds(gp, xxn, yy, xxn);
        imp_sampling='off';
        PJR=zeros([size(Ef,1) 0]);
        for i1=1:size(Covf,3)
          qr=bsxfun(@plus,randn(10,size(Ef,1))*chol(Covf(:,:,i1),'upper'),Ef(:,i1)');
          if strcmpi(rej_sampling,'on')
            if ~any(bounded)
              qii=find(qr(:,1)<qr(:,2)&qr(:,end-1)>qr(:,end));
            elseif bounded(1)&&~bounded(2)
              qii=find(qr(:,end-1)>qr(:,end));
            elseif bounded(2)&&~bounded(1)
              qii=find(qr(:,1)<qr(:,2));
            else
              qii=1:10;
            end
            if isempty(qii)
              continue
            end
            qr=qr(qii,:);
          end
          qjr=exp(qr)';
          pjr=bsxfun(@rdivide,qjr,sum(qjr));
          pjr=pjr./xd;
          PJR(:,end+1:end+size(pjr,2))=pjr;
        end
        pjr=PJR;
        if size(pjr,2)<20
          warning('Rejection sampling to force decreasing tails for (semi)unbounded unreliable')
        end
        
      elseif strcmpi(int_method,'CCD') || strcmpi(int_method,'grid')
        
        % with CCD or GRID integration
        gpia=gp_ia(gp, xxn, yy, 'int_method', int_method, 'display', 'off');
        [Ef, Covf]=gpia_jpreds(gpia, xxn, yy, xxn);
        nGP = numel(gpia);
        
        P_TH=zeros(1,nGP);
        for i1=1:nGP
          P_TH(i1) = gpia{i1}.ia_weight;
        end
        
        nw=resampstr(P_TH,n_is,1);
        
        qr=zeros(n_is,gridn);
        for i1=1:nGP
          qr(nw==i1,:)=bsxfun(@plus,randn(sum(nw==i1),length(Ef(i1,:)))*chol(1.0*Covf(:,:,i1),'upper'),Ef(i1,:));
        end
        if strcmpi(rej_sampling,'on')
          if ~any(bounded)
            qii=find(qr(:,1)<qr(:,2)&qr(:,end-1)>qr(:,end));
          elseif bounded(1)&&~bounded(2)
            qii=find(qr(:,end-1)>qr(:,end));
          elseif bounded(2)&&~bounded(1)
            qii=find(qr(:,1)<qr(:,2));
          else
            qii=1:n_is;
          end
          if numel(qii)>200
            qr=qr(qii,:);
            nw=nw(qii);
          else
            warning('Rejection sampling to force decreasing tails for (semi)unbounded unreliable')
          end
        end
        
        if ~strcmpi(imp_sampling,'off')
          % importance sampling using a normal as a proposal density
          % TODO: split normal
          % Geweke, J. (1989).  Bayesian inference in econometric models using
          % Monte Carlo integration. Econometrica 57:1317-1339.
          lps=zeros(size(qr,1),nGP);
          lpq=zeros(size(qr,1),nGP);
          for i1=1:nGP
            gpis=gp_set(gpia{i1},'latent_method','MCMC');
            ll=sum(bsxfun(@times,qr,yy'),2)-sum(yy)*log(sum(exp(qr),2));
            
            [~, C] = gp_trcov(gpis, xxn);
            [H,b,B]=mean_prep(gpis,xxn,[]);
            LN = chol(C + H'*B*H,'lower');
            MNM = LN\(H'*b-qr(1,:)');
            MNM = MNM'*MNM;
            Econst = gp_e(gp_pak(gpis), gpis, xxn, qr(1,:)') - 0.5*MNM;
            
            MNM = LN\qr';
            MNM = sum(MNM.^2)';
            e = 0.5*MNM + Econst;
            
            lps(:,i1)=-e+ll;
            lpq(:,i1) = mnorm_lpdf(qr,Ef(i1,:),1.0*Covf(:,:,i1));
          end
          lps2=lps-max(lps(:));
          lqs=sumlogs(bsxfun(@plus,lps2,log(P_TH)),2);

          lpq2=lpq-max(lpq(:));
          lqq=sumlogs(bsxfun(@plus,lpq2,log(P_TH)),2);
          
          lws=lqs-lqq;
          lws(isnan(lws)|isinf(lws))=-Inf;
          % compute Pareto smoothed log weights given raw log importance weights
          if ismember(imp_sampling,{'on' 'psis'})
              % compute Pareto smoothed log weights given raw log importance weights
              [lws,pk]=psislw(lws);
          elseif ismember(imp_sampling,{'tis'})
              lws=bsxfun(@minus,lws,sumlogs(lws));
              lwt=min(lws,(-1/2)*log(size(lws,1)));
              lwt=bsxfun(@minus,lwt,sumlogs(lwt));
              lws=lwt;
              pk=0;
          else
              % normalise raw log weights (basic IS weights)
              lws=bsxfun(@minus,lws,sumlogs(lws));
              pk=0;
          end
          if ismember(display,{'on','iter'})
              fprintf(' lgpdens: Pareto k=%.2f\n',pk);
          end
          if pk>0.7&pk<1
              % with the current proposal, it is quite common to get k\approx 0.6
              % which is not so bad, and thus don't warn until k>0.7
              warning('PSIS Pareto k estimate between 0.7 and 1 (%.1f)',pk)
          elseif pk>1
              warning('PSIS Pareto k estimate greater than 1 (%.1f)',pk)
          end
          % importance sampling weights
          ws=exp(lws);
          % Skare, O., Bolviken, E., and Holden, L. (2003). Improved
          % sampling-importance resampling and reduced bias importance
          % sampling. {\em Scandivanian Journal of Statistics} {\bf
          % 30}, 719--737.
          ws=ws./(sum(ws)-ws);
          ws=ws./sum(ws);
          
        end
        qjr=exp(qr)';
        pjr=bsxfun(@rdivide,qjr,sum(qjr(1:gridn,:)));
        pjr=pjr./xd;
        
      else
        
        % with a MAP estimate
        [Ef,Covf] = gp_pred(gp, xxn, yy, xxn);
        if strcmpi(imp_sampling,'off')
          qr=bsxfun(@plus,randn(n_is,size(Ef,1))*chol(1.0*Covf,'upper'),Ef');
        end
        
        if strcmpi(rej_sampling,'on') && strcmpi(imp_sampling,'off')
          if ~any(bounded)
            qii=find(qr(:,1)<qr(:,2)&qr(:,end-1)>qr(:,end));
          elseif bounded(1)&&~bounded(2)
            qii=find(qr(:,end-1)>qr(:,end));
          elseif bounded(2)&&~bounded(1)
            qii=find(qr(:,1)<qr(:,2));
          else
            qii=1:n_is;
          end
          if numel(qii)>200
            qr=qr(qii,:);
          else
            warning('Rejection sampling to force decreasing tails for (semi)unbounded unreliable')
          end
        end
        
        if ~strcmpi(imp_sampling,'off')
          % importance sampling using a split normal as a proposal density
          % Geweke, J. (1989).  Bayesian inference in econometric models using
          % Monte Carlo integration. Econometrica 57:1317-1339.

          nParam = size(Ef,1);
          [V, D] = svd(Covf);
          eig = diag(real(D));
          T = real(V) * sqrt(real(D));
          gpis=gp_set(gp,'latent_method','MCMC');
          [~, C] = gp_trcov(gpis, xxn);
          [H,b,B]=mean_prep(gpis,xxn,[]);
          LN = chol(C + H'*B*H,'lower');
          
          if isempty(n_scale)
            n_scale = nParam;
          end
          n_scale = min(n_scale, nParam);

          ll0 = (Ef' * yy) - sum(yy)*log(sum(exp(Ef)));
          e0 = gp_e(gp_pak(gpis), gpis, xxn, Ef);
          P0 = -e0 + ll0;
          
          e = randn(n_is,nParam);
          
          delta = -6:0.5:6;
          
          q = ones(1,nParam);
          r = ones(1,nParam);
            
          if n_scale > 0
            % Scaling in some eigendirections
            D = reshape(T(:,1:n_scale),numel(T(:,1:n_scale)),1);
            D = D * delta;
            D = reshape(D,nParam,n_scale*length(delta));
            D = bsxfun(@plus,D,Ef);
            D = D';
            ll = sum(bsxfun(@times,D,yy'),2)-sum(yy)*log(sum(exp(D),2));
            MNM = LN\(H'*b-D(1,:)');
            MNM = MNM'*MNM;
            Econst = gp_e(gp_pak(gpis), gpis, xxn, D(1,:)') - 0.5*MNM;
            MNM = LN\D';
            MNM = sum(MNM.^2)';
            en = 0.5*MNM + Econst;
            phat = -en' + ll';    
            
            f = phat;
            f(f >= P0) = NaN;
            f = reshape(f, n_scale, length(delta));
            f = bsxfun(@times, abs(delta),(2*(P0-f)).^(-0.5));
            f = f';
            qTemp = max(f(delta>0,:));
            rTemp = max(f(delta<0,:));
            qTemp(isnan(qTemp)) = 1;
            rTemp(isnan(rTemp)) = 1;
            q(1:n_scale) = qTemp;
            r(1:n_scale) = rTemp;
          end
          
          % Safety limits
          q=min(max(q,1/10),10);
          r=min(max(r,1/10),10);

          % Sample from split normal
          if strcmpi(contSplit,'on')
            % continuous split normal (Geweke (1989) used the discontinuous split normal)
            u = rand(n_is,nParam);
            c = r ./ (r+q);
            eta = (bsxfun(@times,q,double(bsxfun(@ge,u,c)))+bsxfun(@times,-r,double(bsxfun(@lt,u,c)))).*abs(e);
          else
            % discontinuous split normal (Geweke (1989) used this)
            eta = (bsxfun(@times,q,double(e>=0))+bsxfun(@times,r,double(e<0))).*e;
          end
          th = bsxfun(@plus,Ef',eta*T');
          
          % Rejection sampling
          % reject samples with non-decreasing tails, before continuing with IS
          % to avoid unnecessary computations
          if strcmpi(rej_sampling,'on')
            if ~any(bounded)
              qii=find(th(:,1)<th(:,2)&th(:,end-1)>th(:,end));
            elseif bounded(1)&&~bounded(2)
              qii=find(th(:,end-1)>th(:,end));
            elseif bounded(2)&&~bounded(1)
              qii=find(th(:,1)<th(:,2));
            else
              qii=1:n_is;
            end
            if numel(qii)>200
              th=th(qii,:);
              e = e(qii,:);
            else
              warning('Rejection sampling to force decreasing tails for (semi)unbounded unreliable')
            end
          end
          
          if strcmpi(contSplit,'on')
            % continuous split normal (Geweke (1989) used the discontinuous split normal)
            lp_th_appr = - 0.5*sum(e.^2,2)';
          else
            % discontinuous split normal (Geweke (1989) used this)
            C = sum(bsxfun(@times,log(q),double(e>=0))+bsxfun(@times,log(r),double(e<0)),2);
            lp_th_appr = (-C - 0.5*sum(e.^2,2))';
          end
          
          % Evaluate density
          ll = sum(bsxfun(@times,th,yy'),2)-sum(yy)*log(sum(exp(th),2));
          MNM = LN\(H'*b-th(1,:)');
          MNM = MNM'*MNM;
          Econst = gp_e(gp_pak(gpis), gpis, xxn, th(1,:)') - 0.5*MNM;
          MNM = LN\th';
          MNM = sum(MNM.^2)';
          en = 0.5*MNM + Econst;
          lp_th = -en' + ll';
          
          % log importance weights for the samples
          lws = lp_th(:) - lp_th_appr(:);
          if ismember(imp_sampling,{'on' 'psis'})
              % compute Pareto smoothed log weights given raw log importance weights
              [lws,pk]=psislw(lws);
          elseif ismember(imp_sampling,{'tis'})
              lws=bsxfun(@minus,lws,sumlogs(lws));
              lwt=min(lws,(-1/2)*log(size(lws,1)));
              lwt=bsxfun(@minus,lwt,sumlogs(lwt));
              lws=lwt;
              pk=0;
          else
              % normalise raw log weights (basic IS weights)
              lws=bsxfun(@minus,lws,sumlogs(lws));
              pk=0;
          end
          if ismember(display,{'on','iter'})
              fprintf(' lgpdens: Pareto k=%.2f\n',pk);
          end
          if pk>0.7&pk<1
              % with the current proposal, it is quite common to get k\approx 0.6
              % which is not so bad, and thus don't warn until k>0.7
              warning('PSIS Pareto k estimate between 0.7 and 1 (%.1f)',pk)
          elseif pk>1
              warning('PSIS Pareto k estimate greater than 1 (%.1f)',pk)
          end
          % importance sampling weights
          ws=exp(lws);
          % Skare, O., Bolviken, E., and Holden, L. (2003). Improved
          % sampling-importance resampling and reduced bias importance
          % sampling. {\em Scandivanian Journal of Statistics} {\bf
          % 30}, 719--737.
          ws=ws./(sum(ws)-ws);
          ws=ws./sum(ws);
          
          qr = th;
          
        end
        
        qjr=exp(qr)';
        pjr=bsxfun(@rdivide,qjr,sum(qjr(1:gridn,:)));
        pjr=pjr./xd;
      end
      
      if ~strcmpi(imp_sampling,'off')
        pp=wmean(pjr',ws)';
        ppq=wprctile(pjr', [5 95], ws)';
      else
        pp=mean(pjr')';
        ppq=prctile(pjr',[5 95])';
      end  
      
      if nargout<1
        % no output, do the plot thing
        newplot
        hp=patch([xx; xx(end:-1:1)],[ppq(:,1); ppq(end:-1:1,2)],[.8 .8 .8]);
        set(hp,'edgecolor',[.8 .8 .8])
        xlim([xmin xmax])
        line(xx,pp,'linewidth',2,'color',[.4 .4 .4]);
      else
        p=pp;
        pq=ppq;
      end
      
    case 2 % 2D
      rej_sampling='off'; % not implemented for 2D
      if ~isempty(cond_dens) && strcmpi(cond_dens,'on') && ~isempty(speedup) && strcmp(speedup, 'on')
        warning('No speed-up option available with the cond_dens option. Using full covariance instead.')
        speedup='off';
      end
      
      % if ~isempty(rej_sampling) && strcmpi(rej_sampling,'on')
      %   warning('No rejection sampling option available with 2D data.')
      %   rej_sampling='off';
      % end
      
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
          
          if strcmpi(latent_method,'Laplace') && ~strcmpi(imp_sampling,'off')
            % Enrique added this to make IS easier
            % TODO: allow different xx and xt
            xx = xt;
            nz = length(zt);
            gridn(2) = gridn(2)*ntx2;
            z1 = zt1;
            z2 = zt2;
          end
        end
      else
        xx=xt;
        zz1=unique(xx(:,1));
        zz2=unique(xx(:,2));
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
      
      gp=gpsmooth(xxn,yy,gpcf,latent_method,int_method,display,speedup,gridn,cond_dens,basis);
      
      if strcmpi(latent_method,'MCMC')
        imp_sampling='off';
        if ~isempty(cond_dens) && strcmpi(cond_dens,'on')
          [Ef, Covf] = gpmc_jpreds(gp, xxn, yy, xxtn);
        else
          [Ef, Covf] = gpmc_jpreds(gp, xxn, yy, xxn);
        end
        
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
        
      elseif strcmpi(int_method,'CCD') || strcmpi(int_method,'grid')
        error('CCD and grid integration for hyperparameters is not yet implemented for 2D')
      else
        % MAP for covariance function parameters
        if ~isempty(cond_dens) && strcmpi(cond_dens,'on')
          [Ef,Covf] = gp_pred(gp, xxn, yy, xxtn);
        else
          [Ef,Covf] = gp_pred(gp, xxn, yy, xxn);
        end
        
        if strcmpi(speedup,'on') && length(Covf)==2
          qr1=bsxfun(@plus,bsxfun(@times,randn(1000,size(Ef,1)),sqrt(Covf{1})'),Ef');
          qr2=randn(1000,size(Covf{2},1))*Covf{2};
          qr=qr1+qr2;
        else
          qr=bsxfun(@plus,randn(1000,size(Ef,1))*chol(Covf,'upper'),Ef');
        end
        if ~strcmpi(imp_sampling,'off')
          % importance sampling using a split normal as a proposal density
          % Geweke, J. (1989).  Bayesian inference in econometric models using
          % Monte Carlo integration. Econometrica 57:1317-1339.
          
          if strcmpi(speedup,'on') && length(Covf)==2
            % create full covariance to compute eigenvalues and vectors
            Covft=Covf{2}'*Covf{2};
            Covft(1:(size(Ef,1)+1):end)=Covft(1:(size(Ef,1)+1):end)+Covf{1}';
            Covf=Covft;
            clear Covft
          end
          nParam = size(Ef,1);
          [V, D] = svd(Covf);
          T = real(V) * sqrt(real(D));
          eig = diag(real(D));
          gpis=gp_set(gp,'latent_method','MCMC');
          [~, C] = gp_trcov(gpis, xxn);
          [H,b,B]=mean_prep(gpis,xxn,[]);
          LN = chol(C + H'*B*H,'lower');

          if isempty(n_scale)
            n_scale = nParam;
          end
          n_scale = min(n_scale, nParam);
          
          if ~isempty(cond_dens) && strcmpi(cond_dens,'on')
            ll0 = (Ef' * yy) - sum(sum(reshape(yy,gridn(2),gridn(1))).*log(sum(reshape(exp(Ef),gridn(2),gridn(1)))));
          else
            ll0 = (Ef' * yy) - sum(yy)*log(sum(exp(Ef)));
          end
          e0 = gp_e(gp_pak(gpis), gpis, xxn, Ef);
          P0 = -e0 + ll0;
          
          e = randn(n_is,nParam);
          
          delta = -6:0.5:6;
          
          q = ones(1,nParam);
          r = ones(1,nParam);
          
          if n_scale > 0
            
            % Scaling in some eigendirections
            D = reshape(T(:,1:n_scale),numel(T(:,1:n_scale)),1);
            D = D * delta;
            D = reshape(D,nParam,n_scale*length(delta));
            D = bsxfun(@plus,D,Ef);
            D = D';
            if ~isempty(cond_dens) && strcmpi(cond_dens,'on')
              n=sum(reshape(yy,gridn(2),gridn(1)));
              ll = sum(bsxfun(@times,D,yy'),2)-sum(bsxfun(@times,n',reshape(log(sum(reshape(exp(D)',gridn(2),gridn(1)*n_scale*length(delta)))),gridn(1),n_scale*length(delta))))';
            else
              ll = sum(bsxfun(@times,D,yy'),2)-sum(yy)*log(sum(exp(D),2));
            end
            MNM = LN\(H'*b-D(1,:)');
            MNM = MNM'*MNM;
            Econst = gp_e(gp_pak(gpis), gpis, xxn, D(1,:)') - 0.5*MNM;
            MNM = LN\D';
            MNM = sum(MNM.^2)';
            en = 0.5*MNM + Econst;
            phat = -en' + ll';
            
            f = phat;
            f(f >= P0) = NaN;
            f = reshape(f, n_scale, length(delta));
            f = bsxfun(@times, abs(delta),(2*(P0-f)).^(-0.5));
            f = f';
            qTemp = max(f(delta>0,:));
            rTemp = max(f(delta<0,:));
            qTemp(isnan(qTemp)) = 1;
            rTemp(isnan(rTemp)) = 1;
            q(1:n_scale) = qTemp;
            r(1:n_scale) = rTemp;
            
          end                           
          
          % Safety limits
          q=min(max(q,1/10),10);
          r=min(max(r,1/10),10);
          
          % Sample from split normal
          if strcmpi(contSplit,'on')
            % continuous split normal (Geweke (1989) used the discontinuous split normal)
            u = rand(n_is,nParam);
            c = r ./ (r+q);
            eta = (bsxfun(@times,q,double(bsxfun(@ge,u,c)))+bsxfun(@times,-r,double(bsxfun(@lt,u,c)))).*abs(e);
          else
            % discontinuous split normal (Geweke (1989) used this)
            eta=(bsxfun(@times,q,double(e>=0))+bsxfun(@times,r,double(e<0))).*e;
          end
          th=bsxfun(@plus,Ef',eta*T');
          
          if strcmpi(contSplit,'on')
            % continuous split normal (Geweke (1989) used the discontinuous split normal)
            lp_th_appr = - 0.5*sum(e.^2,2)';
          else
            % discontinuous split normal (Geweke (1989) used this)
            C = sum(bsxfun(@times,log(q),double(e>=0))+bsxfun(@times,log(r),double(e<0)),2);
            lp_th_appr = (-C - 0.5*sum(e.^2,2))';
          end
          
          % Evaluate density 
          if ~isempty(cond_dens) && strcmpi(cond_dens,'on')
            n=sum(reshape(yy,gridn(2),gridn(1)));
            ll = sum(bsxfun(@times,th,yy'),2)-sum(bsxfun(@times,n',reshape(log(sum(reshape(exp(th)',gridn(2),gridn(1)*n_is))),gridn(1),n_is)))';
          else
            ll = sum(bsxfun(@times,th,yy'),2)-sum(yy)*log(sum(exp(th),2));
          end
          MNM = LN\(H'*b-th(1,:)');
          MNM = MNM'*MNM;
          Econst = gp_e(gp_pak(gpis), gpis, xxn, th(1,:)') - 0.5*MNM;
          MNM = LN\th';
          MNM = sum(MNM.^2)';
          en = 0.5*MNM + Econst;
          lp_th = -en' + ll';
          
          % Importance weights for the samples
          lws = lp_th(:) - lp_th_appr(:);
          % subtract maximum before exp to avoid overflow
          ws = exp(lws - max(lws));
          % Skare, O., Bolviken, E., and Holden, L. (2003). Improved
          % sampling-importance resampling and reduced bias importance
          % sampling. {\em Scandivanian Journal of Statistics} {\bf
          % 30}, 719--737.
          ws=ws./(sum(ws)-ws);
          ws=ws./sum(ws);
          
          % Kong, A., Liu, J. S., and Wong, W. H. (1996). Sequential imputations
          % and Bayesian missing data problems. {\em Journal of the
          % American Statistical Association} {\bf 89}, 278--288.
          ess = 1/sum(ws.^2);
          
          if ess < 200;
            warning('The effective sample size of importance sampling is small (less than 200). Soft tresholding large weights')
            % has similar effect as resampling without replacement, but allows use of all samples
            ws=logitinv(ws*size(th,1)*.02)*2-1;
            ws=ws./sum(ws);
          end
          qr = th;
          
        end
        qjr=exp(qr)';
        if ~isempty(cond_dens) && strcmpi(cond_dens,'on') 
          pjr=zeros(size(qjr));
          unx2=unique(xt(:,2));
          xd2=(unx2(2)-unx2(1));
          for k1=1:size(qjr,2)
            if strcmpi(latent_method,'Laplace') && ~strcmpi(imp_sampling,'off')
              qjrtmp=reshape(qjr(:,k1),[gridn(2) gridn(1)]);
              % Enrique disabled ntx2
              % TODO: enable ntx2
            else
              qjrtmp=reshape(qjr(:,k1),[gridn(2)*ntx2 gridn(1)]);
            end
            qjrtmp=bsxfun(@rdivide,qjrtmp,sum(qjrtmp));
            qjrtmp=qjrtmp./xd2;
            pjr(:,k1)=qjrtmp(:);
          end
        else
          pjr=bsxfun(@rdivide,qjr,sum(qjr));
          pjr=pjr./xd;
        end
      end
      
      if ~strcmpi(imp_sampling,'off')
        pp=wmean(pjr',ws)';
        ppq=wprctile(pjr', [5 95], ws)';
      else
        pp=mean(pjr')';
        ppq=prctile(pjr',[5 95])';
      end  
      
      if nargout<1
        % no output, do the plot thing
        if ~isempty(cond_dens) && strcmpi(cond_dens,'on')
          if strcmpi(latent_method,'Laplace') && ~strcmpi(imp_sampling,'off')
            pjr2=reshape(pjr,[gridn(2) gridn(1) size(pjr,2)]);
            % Enrique disabled ntx2
            % TODO: enable ntx2
          else
            pjr2=reshape(pjr,[gridn(2)*ntx2 gridn(1) size(pjr,2)]);
          end
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
          newplot
          hold on
          h1=plot(zz1,ql(4,:)','-', 'color', [.6 .6 .6],'linewidth',2);
          h2=plot(zz1,ql([3 5],:)','--', 'color', [.4 .4 .4],'linewidth',1);
          h3=plot(zz1,ql([2 6],:)','-.', 'color', [.2 .2 .2],'linewidth',1);
          h4=plot(zz1,ql([1 7],:)',':', 'color', [0 0 0],'linewidth',1);
          % h1=plot(zz1,ql(4,:)','-', 'color', [0 0 255]./255,'linewidth',2);
          % h2=plot(zz1,ql([3 5],:)','--', 'color', [0 127 0]./255,'linewidth',1);
          % h3=plot(zz1,ql([2 6],:)','-.', 'color', [255 0 0]./255,'linewidth',1);
          % h4=plot(zz1,ql([1 7],:)',':', 'color', [0 0 0]./255,'linewidth',1);
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

function gp = gpsmooth(xx,yy,gpcf,latent_method,int_method,display,speedup,gridn,cond_dens,basis,imp_sampling)
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
  if size(xx,2)==2 %~isempty(cond_dens) && strcmp(cond_dens, 'on')
                   % prior based on guess of maximum differences in log densities
    pm = prior_sqrtt('s2',1000,'nu',4);
    % prior based on knowing that data has been scaled
    pl = prior_t('s2', 10, 'nu', 4);
  else
    % prior based on guess of maximum differences in log densities
    pm = prior_sqrtt('s2',10,'nu',4);
    % prior based on knowing that data has been scaled
    pl = prior_t('s2', 1, 'nu', 4);
  end
  % 
  pa = prior_t('s2', 20^2, 'nu', 1);
  % use modified Silverman's rule of thumb or min grid distance
  % as initial guess for the length scale
  h=max(diff(xx(1:2,end)).^2,1/sum(yy).^(1/5)/2);
  % different covariance functions have different parameters
  if isfield(gpcf1,'magnSigma2')
    gpcf1 = gpcf(gpcf1, 'magnSigma2', 1, 'magnSigma2_prior', pm);
  end
  if isfield(gpcf1,'lengthScale')
    gpcf1 = gpcf(gpcf1, 'lengthScale', h*repmat(2,[1 size(xx,2)]), 'lengthScale_prior', pl);
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
    gp = gp_set('lik', lik, 'cf', {gpcf1}, 'jitterSigma2', 1e-6);
  elseif strcmp(basis, 'exp')
    gpmflin = gpmf_linear('prior_mean',0,'prior_cov',100);
    gp = gp_set('lik', lik, 'cf', {gpcf1}, 'jitterSigma2', 1e-6, 'meanf', {gpmflin});
  else
    gpmflin = gpmf_linear('prior_mean',0,'prior_cov',100);
    gpmfsq = gpmf_squared('prior_mean',0,'prior_cov',100,'interactions','on');
    gp = gp_set('lik', lik, 'cf', {gpcf1}, 'jitterSigma2', 1e-6, 'meanf', {gpmflin,gpmfsq});
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
    elseif strcmp(gp.cf{1}.type,'gpcf_sexp') || strcmp(gp.cf{1}.type,'gpcf_exp') || strcmp(gp.cf{1}.type,'gpcf_matern32') || strcmp(gp.cf{1}.type,'gpcf_matern52')
      gp.latent_opt.fft=1;
    end
  end

  gp=gp_optim(gp,xx,yy,'opt',opt, 'optimf', @fminlbfgs);
  %gradcheck(gp_pak(gp), @gpla_nd_e, @gpla_nd_g, gp, xx, yy);
  %[w,s]=gp_pak(gp);
  %exp(gp_pak(gp));
  
  if strcmpi(latent_method,'MCMC')
    % gpia could be used to get integartion limits for slice sampling
    % gpia=gp_ia(gp, xx, yy, 'int_method', 'CCD', 'display', displ);
    % for i1=1:numel(gpia)
    %   ws(i1,:)=gp_pak(gpia{i1});
    % end
    % w0=ws(1,:);
    % minw=min(ws);
    % maxw=max(ws);
    % minw=w0-(w0-minw)*3;
    % maxw=w0+(maxw-w0)*3;
    % sample latents from the approximate posterior
    fs=gp_rnd(gp,xx,yy,xx);
    % elliptical slice sampling for latents
    %gp = gp_set(gp, 'latent_method', 'MCMC', 'latent_opt', struct('method', @esls));
    % scaled Metropolis-Hastings for latents
    gp = gp_set(gp, 'latent_method', 'MCMC', 'latent_opt', struct('method', @scaled_mh));
    % intialise the latent values
    gp.latentValues=fs;
    % options for HMC-NUTS
    hmc_opt.nuts = 1; % True or false
    hmc_opt.Madapt = 20; % Number of step-size adaptation steps (Burn-in)
    latent_opt.display=0;
    latent_opt.repeat = 100;
    latent_opt.sample_latent_scale = 0.1;
    % options for slice sampling
    % slsopt=sls_opt;
    % slsopt.mmlimits=[minw;maxw];
    % slsopt.method='multi';
    % slsopt.latent_opt.repeat=100;
    if size(xx,2)==2
      % add more jitter for 2D cases with MCMC
      gp = gp_set(gp, 'jitterSigma2', 1e-2);
    end
    
    [rgp,g,opt]=gp_mc(gp, xx, yy, 'hmc_opt', hmc_opt, 'latent_opt', latent_opt, 'nsamples', 510, 'repeat', 10, 'display', displ);
    % estimate the effective number of samples
    %neff=510./[geyer_imse(log(rgp.cf{1}.magnSigma2)) geyer_imse(log(rgp.cf{1}.lengthScale))];
    % Remove burn-in
    gp=thin(rgp,11,5);
    
  end
end
