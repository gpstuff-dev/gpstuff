function [Ef, Varf, xtnn, xt1, xt2] = gp_cpred(gp,x,y,xt,ind,varargin)
%GP_CPRED Conditional predictions using specific covariates
%
%  Description
%    GP_CPRED(GP,X,Y,XT,IND,OPTIONS) does predictions using only
%    covariates specified in vector IND. Other covariates are fixed to
%    either mean, median or values chosen by user. Returns predictions for
%    latent values, variance and corresponding inputs. If IND=0, only time
%    is used as a covariate for Cox-PH model.
%
%   OPTIONS is optional parameter-value pair
%      method - which value to fix the not used covariates, 'median'
%               (default), 'mean' or 'mode'
%      var    - vector specifying optional values for not used covariates,
%               elements corresponding to mean/median values should 
%               be set to NaN. 
%      plot   - option for plotting, 'off' (default) or 'on'
%      normdata - a structure with fields xmean, xstd, ymean, and ystd
%               to allow plotting in the original data scale (see
%               functions normdata and denormdata)
%      target - option for choosing what is computed 'mu' (default),
%               'f' or 'cdf'
%      tr     - Euclidean distance treshold for not using grid points when
%               doing predictions with 2 covariates, default 0.25
%      predcf - an index vector telling which covariance functions are 
%               used for prediction. Default is all (1:gpcfn). 
%               See additional information below.
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case. 
%      zt     - optional observed quantity in triplet (xt_i,yt_i,zt_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, the expected 
%               value for the ith case. 


ip=inputParser;
ip.FunctionName = 'GP_CPRED';
ip.addRequired('gp',@(x) isstruct(x) || iscell(x));
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('xt',  @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('yt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addRequired('ind', @(x) ~isempty(x) && isvector(x))
ip.addParamValue('var',  [], @(x) isreal(x))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('predcf', [], @(x) isempty(x) || ...
                 isvector(x) && isreal(x) && all(isfinite(x)&x>0))
ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                 (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
ip.addParamValue('method', 'median', @(x)  ismember(x, {'median', 'mean' 'mode'}))
ip.addParamValue('plot', 'off', @(x)  ismember(x, {'on', 'off'}))
ip.addParamValue('tr', 0.25, @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('target', 'mu', @(x) ismember(x,{'f','mu','cdf'}))
ip.addParamValue('prct', [5 50 95], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('normdata', struct(), @(x) isempty(x) || isstruct(x))
ip.addParamValue('xlabels', [], @(x) isempty(x) || iscell(x));
ip.parse(gp, x, y, xt, ind, varargin{:});
zt=ip.Results.zt;
options=struct();
options.predcf=ip.Results.predcf;
options.prct=ip.Results.prct;
options.tstind=ip.Results.tstind;
method = ip.Results.method;
vars = ip.Results.var;
plot_results = ip.Results.plot;
tr = ip.Results.tr;
target = ip.Results.target;
if strcmp(target,'f')
    options = rmfield(options,'prct');
end
yt=ip.Results.yt;
if ~isempty(yt)
  options.yt=yt;
end
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
% normdata
nd=ip.Results.normdata;
ipnd=inputParser;
ipnd.FunctionName = 'normdata';
ipnd.addParamValue('xmean',zeros(1,size(x,2)),@(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))));
ipnd.addParamValue('xstd',ones(1,size(x,2)),@(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))));
ipnd.addParamValue('xlog',zeros(1,size(x,2)),@(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))));
ipnd.addParamValue('ymean',0,@(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))));
ipnd.addParamValue('ystd',1,@(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))));
ipnd.addParamValue('ylog',0,@(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))));
ipnd.parse(nd);
nd=ipnd.Results;

[tmp, nin] = size(x);

if iscell(gp)
  liktype=gp{1}.lik.type;
else
  liktype=gp.lik.type;
end

if isequal(liktype, 'Coxph') && isequal(target,'mu')
    target='f';
    warning('GP_CPRED: Target ''mu'' not applicable for a Cox-PH model. Switching to target ''f''')
end

if ~isempty(vars) && (~isvector(vars) || length(vars) ~= nin)
  error('Vector defining fixed variable values must be same length as number of covariates')
end

xto=xt; [n,tmp]=size(xto);

if length(ind)==1
  
  if ind~=0
    [xtnn, iu] = unique(xt(:,ind));
  else
    xtnn = xt(1,:);
    iu = 1;
  end
  if ~isempty(z)
    options.zt = options.zt(iu);
  end
  if ~isempty(yt)
    options.yt = options.yt(iu);
  end
  xt = repmat(feval(method,xt), size(xtnn,1), 1);
  if ~isempty(vars)
    xt(:,~isnan(vars)) = repmat(vars(~isnan(vars)), length(xtnn), 1);
  end
  
  if ind>0
    xt(:,ind) = xtnn;
  end
  if ~strcmp(liktype, 'Coxph')
    switch target
      case 'f'
        [Ef, Varf] = gp_pred(gp, x, y, xt, options);
      case 'mu'
        prctmu = denormdata(gp_predprctmu(gp, x, y, xt, options),nd.ymean,nd.ystd);
        Ef = prctmu; Varf = [];
      case 'cdf'
        cdf = gp_predcdf(gp, x, y, xt, options);
        Ef = cdf; Varf = [];
    end
  else
    [Ef1,Ef2,Covf] = pred_coxph(gp,x,y,xt,'z',options.z,'zt',options.zt);
    nt=size(Ef1,1);
    if ind>0
      % conditional posterior given Ef1=E[Ef1]
      Ef = Ef2; 
      Varf = diag(Covf(nt+1:end,nt+1:end)-Covf(nt+1:end,1:nt)*(Covf(1:nt,1:nt)\Covf(1:nt,nt+1:end)));
    else
      % conditional posterior given Ef2=E[Ef2]
      Ef = Ef1; 
      Varf = diag(Covf(1:nt, 1:nt)-Covf(1:nt,nt+1:end)*(Covf(nt+1:end,nt+1:end)\Covf(nt+1:end,1:nt)));
      xtnn = gp.lik.xtime;
    end
  end
  if isequal(plot_results, 'on')
    if ind>0
      if ind>=1&numel(nd.xmean)>=ind
        xtnn=denormdata(xtnn,nd.xmean(ind),nd.xstd(ind));
      end
      deltadist=gp_finddeltadist(gp);
      if ~ismember(ind,deltadist)
        switch target
          case 'f'
            plot(xtnn, Ef, 'ob', xtnn, Ef, '-k', xtnn, Ef-1.64*sqrt(Varf), '--b', xtnn, Ef+1.64*sqrt(Varf), '--b')
          case 'mu'
            plot(xtnn, prctmu(:,2), 'ob', xtnn, prctmu(:,2), '-k', xtnn, prctmu(:,1), '--b', xtnn, prctmu(:,3), '--b')
          case 'cdf'
            plot(xtnn, Ef, 'o-b')
        end
      else
        switch target
          case 'f'
            plot(xtnn, Ef, 'ob', [xtnn xtnn]',[Ef-1.64*sqrt(Varf) Ef+1.64*sqrt(Varf)]', '-b')
            xlim([1.5*xtnn(1)-0.5*xtnn(2) 1.5*xtnn(end)-.5*xtnn(end-1)])
            set(gca,'xtick',xtnn)
          case 'mu'
            plot(xtnn, prctmu(:,2), 'ob', [xtnn xtnn]',[prctmu(:,1) prctmu(:,3)]', '-b')
            xlim([1.5*xtnn(1)-0.5*xtnn(2) 1.5*xtnn(end)-.5*xtnn(end-1)])
            set(gca,'xtick',xtnn)
          case 'cdf'
            plot(xtnn, Ef, 'ob')
        end
      end
    else
      % use stairs for piecewise constant baseline hazard
      xtnn = gp.lik.stime;
      [xx,yy]=stairs(xtnn, [Ef;Ef(end)]);
      [xx,yyl]=stairs(xtnn, [Ef-1.64*sqrt(Varf);Ef(end)-1.64*sqrt(Varf(end))]);
      [xx,yyu]=stairs(xtnn, [Ef+1.64*sqrt(Varf);Ef(end)+1.64*sqrt(Varf(end))]);
      plot(xx, yy, '-k', xx, yyl, '--b', xx, yyu, '--b')
    end
  end
  
elseif length(ind)==2
  
  uu1=unique(xt(:,ind(1)));
  uu2=unique(xt(:,ind(2)));
  nu1=numel(uu1);
  nu2=numel(uu2);
  if nu1==2 || nu2==2
    % First or second covariate binary
    
    if nu1>2 && nu2==2
      % switch indeces, so that binary covariate is first
      tmp=ind(1);ind(1)=ind(2);ind(2)=tmp;
      tmp=uu1;uu1=uu2;uu2=tmp;
    end
    
    xt1=xt(xt(:,ind(1))==uu1(1),:);
    xt2=xt(xt(:,ind(1))==uu1(2),:);
    [xtnn1, iu1] = unique(xt1(:,ind(2)));
    [xtnn2, iu2] = unique(xt2(:,ind(2)));
    
    options1=options;
    options2=options;
    if ~isempty(z)
      options1.zt = options.zt(iu1);
      options2.zt = options.zt(iu2);
    end

    xt1 = repmat(feval(method,xt1), length(xtnn1), 1);
    xt2 = repmat(feval(method,xt2), length(xtnn2), 1);
    if ~isempty(vars)
      xt1(:,~isnan(vars)) = repmat(vars(~isnan(vars)), length(xtnn1), 1);
      xt2(:,~isnan(vars)) = repmat(vars(~isnan(vars)), length(xtnn2), 1);
    end
    xt1(:,ind(1)) = uu1(1); xt1(:,ind(2)) = xtnn1;
    xt2(:,ind(1)) = uu1(2); xt2(:,ind(2)) = xtnn2;
    
    if ~strcmp(liktype, 'Coxph')
      switch target
        case 'f'
          [Ef1, Varf1] = gp_pred(gp, x, y, xt1);
          [Ef2, Varf2] = gp_pred(gp, x, y, xt2);
        case 'mu'
          prctmu1 = denormdata(gp_predprctmu(gp, x, y, xt1, options1),nd.ymean,nd.ystd);
          prctmu2 = denormdata(gp_predprctmu(gp, x, y, xt2, options2),nd.ymean,nd.ystd);
      end
    else
      [Ef11,Ef12,Covf] = pred_coxph(gp,x,y,xt1, 'z', z);
      Ef1 = Ef12; Varf1 = diag(Covf(size(Ef11,1)+1:end,size(Ef11,1)+1:end));
      [Ef21,Ef22,Covf] = pred_coxph(gp,x,y,xt2, 'z', z);
      Ef2 = Ef22; Varf2 = diag(Covf(size(Ef21,1)+1:end,size(Ef21,1)+1:end));
    end
    
    if isequal(plot_results, 'on')
      xtnn1=denormdata(xtnn1,nd.xmean(ind(2)),nd.xstd(ind(2)));
      xtnn2=denormdata(xtnn2,nd.xmean(ind(2)),nd.xstd(ind(2)));
      if nu1>2 && nu2==2
        lstyle10='or';lstyle11='-r';lstyle12='--r';
        lstyle20='ob';lstyle21='-b';lstyle22='--b';
      else
        lstyle10='ob';lstyle11='-b';lstyle12='--b';
        lstyle20='or';lstyle21='-r';lstyle22='--r';
      end
      deltadist=gp_finddeltadist(gp);
      if ~ismember(ind(2),deltadist)
        switch target
          case 'f'
            plot(xtnn1, Ef1, lstyle10, xtnn1, Ef1, lstyle11, xtnn1, Ef1-1.64*sqrt(Varf1), lstyle12, xtnn1, Ef1+1.64*sqrt(Varf1), lstyle12); hold on;
            plot(xtnn2, Ef2, lstyle20, xtnn2, Ef2, lstyle21, xtnn2, Ef2-1.64*sqrt(Varf2), lstyle22, xtnn2, Ef2+1.64*sqrt(Varf2), lstyle22); hold off;
          case 'mu'
            plot(xtnn1, prctmu1(:,2), lstyle10, xtnn1, prctmu1(:,2), lstyle11, xtnn1, prctmu1(:,1), lstyle12, xtnn1, prctmu1(:,3), lstyle12); hold on;
            plot(xtnn2, prctmu2(:,2), lstyle20, xtnn2, prctmu2(:,2), lstyle21, xtnn2, prctmu2(:,1), lstyle22, xtnn2, prctmu2(:,3), lstyle22); hold off;
        end
      else
        delta=(diff(xtnn1(1:2))/10);
        switch target
          case 'f'
            plot(xtnn1-delta, Ef1, lstyle10, [xtnn1 xtnn1]'-delta, [Ef1-1.64*sqrt(Varf1) Ef1+1.64*sqrt(Varf1)]', lstyle11); hold on;
            plot(xtnn2+delta, Ef2, lstyle20, [xtnn2 xtnn2]'+delta, [Ef2-1.64*sqrt(Varf2) Ef2+1.64*sqrt(Varf2)]', lstyle21); hold off;
            xlim([1.5*xtnn1(1)-0.5*xtnn1(2) 1.5*xtnn1(end)-.5*xtnn1(end-1)])
          case 'mu'
            plot(xtnn1-delta, prctmu1(:,2), lstyle10, [xtnn1 xtnn1]'-delta, [prctmu1(:,1) prctmu1(:,3)]', lstyle11); hold on;
            plot(xtnn2+delta, prctmu2(:,2), lstyle20, [xtnn2 xtnn2]'+delta, [prctmu2(:,1) prctmu2(:,3)]', lstyle21); hold off;
            xlim([1.5*xtnn1(1)-0.5*xtnn1(2) 1.5*xtnn1(end)-.5*xtnn1(end-1)])
        end
      end
    end
    switch target
      case 'f'
        Ef = {Ef1  Ef2}; Varf = {Varf1 Varf2}; xtnn={xtnn1 xtnn2};
      case 'mu'
        Ef = {prctmu1 prctmu2}; Varf = {[] []}; xtnn={xtnn1 xtnn2};
    end
    
  else
    % both the first and the second covariate are non-binary
    deltadist=gp_finddeltadist(gp);
    if ~ismember(ind(1),deltadist)
      xtnn1 = linspace(min(xt(:,ind(1))), max(xt(:,ind(1))), 20);
    else
      xtnn1 = unique(xt(:,ind(1)));
    end
    if ~ismember(ind(2),deltadist)
      xtnn2 = linspace(min(xt(:,ind(2))), max(xt(:,ind(2))), 20);
    else
      xtnn2 = unique(xt(:,ind(2)));
    end
    [XT1, XT2] = meshgrid(xtnn1, xtnn2); XT1=XT1(:); XT2=XT2(:);
    xtnn1=denormdata(xtnn1,nd.xmean(ind(1)),nd.xstd(ind(1)));
    xtnn2=denormdata(xtnn2,nd.xmean(ind(2)),nd.xstd(ind(2)));
    if ~isempty(z)
      options.zt = repmat(options.zt(1), size(XT1));
    end
    xt = repmat(feval(method,xt), length(XT1), 1);
    if ~isempty(vars)
      xt(:,~isnan(vars)) = repmat(vars(~isnan(vars)), length(XT1), 1);
    end
    xt(:,ind) = [XT1 XT2];
    if ~strcmp(liktype, 'Coxph')
      switch target
        case 'f'
          [Ef, Varf] = gp_pred(gp, x, y, xt);
        case 'mu'
          prctmu = gp_predprctmu(gp, x, y, xt, options, 'prct', 50);
          Ef = prctmu; Varf = [];
        case 'cdf'
          cdf = gp_predcdf(gp, x, y, xt, options);
          Ef = cdf; Varf = [];
      end
    else
      [Ef1,Ef2,Covf] = pred_coxph(gp,x,y,xt, options);
      Ef = Ef2; Varf = diag(Covf(size(Ef1,1)+1:end,size(Ef1,1)+1:end));
    end
    
    indd = zeros(size(Ef));
    
    for i2=1:n
      for i3=1:numel(XT1)
        if sqrt(sum((xto(i2,ind)-xt(i3,ind)).^2)) < tr
          indd(i3) = 1;
        end
      end
    end
    
    XT1(indd==0) = NaN; XT2(indd==0) = NaN; Ef(indd==0) = NaN; Varf(indd==0) = NaN;
    
    if isequal(plot_results, 'on')
      xtnn1=denormdata(xtnn1,nd.xmean(ind(1)),nd.xstd(ind(1)));
      xtnn2=denormdata(xtnn2,nd.xmean(ind(2)),nd.xstd(ind(2)));
      surf(reshape(XT1,numel(xtnn2),numel(xtnn1)), reshape(XT2,numel(xtnn2),numel(xtnn1)), reshape(Ef,numel(xtnn2),numel(xtnn1)))
      view(2)
      axis tight
      shading flat
      colormap(mapcolor(Ef,repmat(nanmedian(Ef(:)),[1 2])))
      colorbar('EastOutside')
    end
    
    %xtnn = [XT1(indd==1), XT2(indd==1)]; Ef = Ef(indd==1); Varf = Varf(indd==1);
    xtnn = [XT1, XT2];
  end
  
else
  error('Only 1 or 2 covariates can be defined for predicting')
end


end

