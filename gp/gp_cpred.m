function [Ef, Varf, xtnn] = gp_cpred(gp,x,y,xt,ind,varargin)
%GP_CPRED Conditional predictions using specific covariates
%
%  Description
%    GP_CPRED(GP,X,Y,XT,IND,OPTIONS) does predictions using only
%    covariates specified in vector IND. Other covariates are fixed to
%    either mean, median or values chosen by user. Returns predictions for
%    latent values, variance and corresponding inputs. If IND=0, only time
%    is used as a covariate for coxph model.
%
%   OPTIONS is optional parameter-value pair
%      predcf - an index vector telling which covariance functions are 
%               used for prediction. Default is all (1:gpcfn). 
%               See additional information below.
%      method - which value to fix the not used covariates, 'mean'
%               (default) or 'median'
%      var    - vector specifying optional values for not used covariates,
%               elements corresponding to mean/median values should 
%               be set to NaN. 
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case. 
%      plot   - Option for plotting, 'off' (default) or 'on'
%      tr     - Euclidean distance treshold for not using grid points when
%               doing predictions with 2 covariates, default 0.25


ip=inputParser;
ip.FunctionName = 'GP_CPRED';
ip=iparser(ip,'addRequired','gp',@isstruct);
ip=iparser(ip,'addRequired','x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))));
ip=iparser(ip,'addRequired','y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))));
ip=iparser(ip,'addRequired','xt',  @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))));
ip=iparser(ip,'addRequired','ind', @(x) ~isempty(x) && isvector(x));
ip=iparser(ip,'addParamValue','var',  [], @(x) isreal(x));
ip=iparser(ip,'addParamValue','z', [], @(x) isreal(x) && all(isfinite(x(:))));
ip=iparser(ip,'addParamValue','predcf', [], @(x) isempty(x) || ...
                 isvector(x) && isreal(x) && all(isfinite(x)&x>0));
ip=iparser(ip,'addParamValue','tstind', [], @(x) isempty(x) || iscell(x) ||...
                 (isvector(x) && isreal(x) && all(isfinite(x)&x>0)));
ip=iparser(ip,'addParamValue','method', 'mean', @(x)  ismember(x, {'median', 'mean'}));
ip=iparser(ip,'addParamValue','plot', 'off', @(x)  ismember(x, {'on', 'off'}));
ip=iparser(ip,'addParamValue','tr', 0.25, @(x) isreal(x) && all(isfinite(x(:))));
ip=iparser(ip,'addParamValue','target', 'f', @(x) ismember(x,{'f','mu'}));
ip=iparser(ip,'parse',gp, x, y, xt, ind, varargin{:});
options=struct();
options.predcf=ip.Results.predcf;
options.tstind=ip.Results.tstind;
method = ip.Results.method;
vars = ip.Results.var;
plot_results = ip.Results.plot;
tr = ip.Results.tr;
target = ip.Results.target;
z=ip.Results.z;
if ~isempty(z)
  options.zt=z;
  options.z=z;
end

[tmp, nin] = size(x);

if iscell(gp)
  liktype=gp{1}.lik.type;
else
  liktype=gp.lik.type;
end

if isequal(liktype, 'Coxph') && isequal(target,'mu')
  error('GP_CPRED: Target ''mu'' not applicable for a Cox-PH model')
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
  meanxt=mean(xt);
  if isequal(method, 'mean')
    xt = repmat(meanxt, size(xtnn,1), 1);
  else
    xt = repmat(median(xt), size(xtnn,1), 1);
  end
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
        prctmu = gp_predprctmu(gp, x, y, xt, options);
        Ef = prctmu; Varf = [];
    end
  else
    [Ef1,Ef2,Covf] = pred_coxph(gp,x,y,xt, options);
    if ind>0
      Ef = Ef2; Varf = diag(Covf(size(Ef1,1)+1:end,size(Ef1,1)+1:end));
    else
      Ef = Ef1; Varf = diag(Covf(1:size(Ef1,1), 1:size(Ef1,1)));
      xtnn = gp.lik.xtime;
    end
  end
  if isequal(plot_results, 'on')
    if ind>0
      switch target
        case 'f'
          plot(xtnn, Ef, 'ob', xtnn, Ef, '-k', xtnn, Ef-1.64*sqrt(Varf), '--b', xtnn, Ef+1.64*sqrt(Varf), '--b')
        case 'mu'
          plot(xtnn, prctmu(:,2), 'ob', xtnn, prctmu(:,2), '-k', xtnn, prctmu(:,1), '--b', xtnn, prctmu(:,3), '--b')
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

    if isequal(method, 'mean')
      xt1 = repmat(mean(xt1), length(xtnn1), 1);
      xt2 = repmat(mean(xt2), length(xtnn2), 1);
    else
      xt1 = repmat(median(xt1), length(xtnn1), 1);
      xt2 = repmat(median(xt2), length(xtnn2), 1);
    end
    if ~isempty(vars)
      xt1(:,~isnan(vars)) = repmat(vars(~isnan(vars)), length(xtnn1), 1);
      xt2(:,~isnan(vars)) = repmat(vars(~isnan(vars)), length(xtnn2), 1);
    end
    xt1(:,ind(1)) = uu1(1); xt1(:,ind(2)) = xtnn1;
    xt2(:,ind(1)) = uu1(2); xt2(:,ind(2)) = xtnn2;
    
    if ~strcmp(liktype, 'Coxph')
      switch target
        case 'f'
          [Ef1, Varf1] = gp_pred(gp, x, y, xt1, options1);
          [Ef2, Varf2] = gp_pred(gp, x, y, xt2, options2);
        case 'mu'
          prctmu1 = gp_predprctmu(gp, x, y, xt1, options1);
          prctmu2 = gp_predprctmu(gp, x, y, xt2, options2);
      end
    else
      [Ef11,Ef12,Covf] = pred_coxph(gp,x,y,xt1, options1);
      Ef1 = Ef12; Varf1 = diag(Covf(size(Ef11,1)+1:end,size(Ef11,1)+1:end));
      [Ef21,Ef22,Covf] = pred_coxph(gp,x,y,xt2, options2);
      Ef2 = Ef22; Varf2 = diag(Covf(size(Ef21,1)+1:end,size(Ef21,1)+1:end));
    end
    
    if isequal(plot_results, 'on')
      if nu1>2 && nu2==2
        lstyle10='or';lstyle11='-r';lstyle12='--r';
        lstyle20='ob';lstyle21='-b';lstyle22='--b';
      else
        lstyle10='ob';lstyle11='-b';lstyle12='--b';
        lstyle20='or';lstyle21='-r';lstyle22='--r';
      end
      switch target
        case 'f'
          plot(xtnn1, Ef1, lstyle10, xtnn1, Ef1, lstyle11, xtnn1, Ef1-1.64*sqrt(Varf1), lstyle12, xtnn1, Ef1+1.64*sqrt(Varf1), lstyle12); hold on;
          plot(xtnn2, Ef2, lstyle20, xtnn2, Ef2, lstyle21, xtnn2, Ef2-1.64*sqrt(Varf2), lstyle22, xtnn2, Ef2+1.64*sqrt(Varf2), lstyle22);
        case 'mu'
          plot(xtnn1, prctmu1(:,2), lstyle20, xtnn1, prctmu1(:,2), lstyle11, xtnn1, prctmu1(:,1), lstyle12, xtnn1, prctmu1(:,3), lstyle12); hold on;
          plot(xtnn2, prctmu2(:,2), lstyle20, xtnn2, prctmu2(:,2), lstyle21, xtnn2, prctmu2(:,1), lstyle22, xtnn2, prctmu2(:,3), lstyle22);
      end
    end
    switch target
      case 'f'
        Ef = {Ef1  Ef2}; Varf = {Varf1 Varf2}; xtnn={xtnn1 xtnn2};
      case 'mu'
        Ef = {prctmu1 prctmu2}; Varf = {[] []}; xtnn={xtnn1 xtnn2};
    end
    
  else
    % first or second covariate is not binary
    xtnn1 = linspace(min(xt(:,ind(1))), max(xt(:,ind(1))), 20);
    xtnn2 = linspace(min(xt(:,ind(2))), max(xt(:,ind(2))), 20);
    [XT1, XT2] = meshgrid(xtnn1, xtnn2); XT1=XT1(:); XT2=XT2(:);
    if ~isempty(z)
      options.zt = repmat(options.zt(1), 400, 1 );
    end
    if isequal(method, 'mean')
      xt = repmat(mean(xt), length(XT1), 1);
    else
      xt = repmat(median(xt), length(XT1), 1);
    end
    if ~isempty(vars)
      xt(:,~isnan(vars)) = repmat(vars(~isnan(vars)), length(XT1), 1);
    end
    xt(:,ind) = [XT1 XT2];
    if ~strcmp(liktype, 'Coxph')
      [Ef, Varf] = gp_pred(gp, x, y, xt, options);
    else
      [Ef1,Ef2,Covf] = pred_coxph(gp,x,y,xt, options);
      Ef = Ef2; Varf = diag(Covf(size(Ef1,1)+1:end,size(Ef1,1)+1:end));
    end
    
    indd = zeros(size(Ef));
    
    for i2=1:n
      for i3=1:400
        if sqrt(sum((xto(i2,ind)-xt(i3,ind)).^2)) < tr
          indd(i3) = 1;
        end
      end
    end
    
    XT1(indd==0) = NaN; XT2(indd==0) = NaN; Ef(indd==0) = NaN; Varf(indd==0) = NaN;
    
    if isequal(plot_results, 'on')
      contourf(reshape(XT1,20,20), reshape(XT2,20,20), reshape(Ef,20,20))
    end
    
    xtnn = [XT1(indd==1), XT2(indd==1)]; Ef = Ef(indd==1); Varf = Varf(indd==1);
  end
  
else
  error('Only 1 or 2 covariates can be defined for predicting')
end


end

