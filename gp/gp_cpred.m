function [Ef, Varf, xtnn] = gp_cpred(gp,x,y,xt, ind,varargin)
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
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('xt',  @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('ind', @(x) ~isempty(x) && isvector(x))
ip.addParamValue('var',  [], @(x) isreal(x))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('predcf', [], @(x) isempty(x) || ...
                 isvector(x) && isreal(x) && all(isfinite(x)&x>0))
ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                 (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
ip.addParamValue('method', 'mean', @(x)  ismember(x, {'median', 'mean'}))
ip.addParamValue('plot', 'off', @(x)  ismember(x, {'on', 'off'}))
ip.addParamValue('tr', 0.25, @(x) isreal(x) && all(isfinite(x(:))))
ip.parse(gp, x, y, xt, ind, varargin{:});
predcf=ip.Results.predcf;
tstind=ip.Results.tstind;
method = ip.Results.method;
var = ip.Results.var;
plot_results = ip.Results.plot;
tr = ip.Results.tr;
options=struct();
z = ip.Results.z;
if ~isempty(ip.Results.z)
  options.zt=ip.Results.z;
  options.z=ip.Results.z;
end

[tmp, nin] = size(x);

if ~isempty(var) && (~isvector(var) || length(var) ~= nin)
  error('Vector defining fixed variable values must be same length as number of covariates')
end

xto=xt; [n,tmp]=size(xto);

if length(ind)==1
  
  if ind~=0
    [xtnn, iu] = unique(xt(:,ind));
  else
    xtnn = xt(1,:);
    iu = 1;
    ind=1:4;
  end
  if ~isempty(z)
%     options.z = options.z(iu);
    options.zt = options.zt(iu);
  end
  meanxt=mean(xt);
  if isequal(method, 'mean')
    xt = repmat(meanxt, size(xtnn,1), 1);
  else
    xt = repmat(median(xt), size(xtnn,1), 1);
  end
  if ~isempty(var)
    xt(:,~isnan(var)) = repmat(var(~isnan(var)), length(xtnn), 1);
  end
  
  xt(:,ind) = xtnn;
  if ~strcmp(gp.lik.type, 'Coxph')
    [Ef, Varf] = gp_pred(gp, x, y, xt, 'predcf', predcf, 'tstind', tstind, options);
  else
    [Ef1,Ef2,Covf] = pred_coxph(gp,x,y,xt, 'predcf', predcf, 'tstind', tstind, options);
    if isscalar(ind)
      Ef = Ef2; Varf = diag(Covf(size(Ef1,1)+1:end,size(Ef1,1)+1:end));
    else
      Ef = Ef1; Varf = diag(Covf(1:size(Ef1,1), 1:size(Ef1,1)));
      xtnn = gp.lik.xtime;
    end
  end
  if isequal(plot_results, 'on')
    plot(xtnn, Ef, 'xb', xtnn, Ef, '-k', xtnn, Ef-sqrt(Varf), '--b', xtnn, Ef+sqrt(Varf), '--b')
  end
  
elseif length(ind)==2
  
  if sum(xt(:,ind(1))==-1) + sum(xt(:,ind(1))==1) == n
    % First (or first and second) covariate binary
    
    [xtnn1, iu1] = unique(xt(xt(:,ind(1))==-1,ind(2)));
    [xtnn2, iu2] = unique(xt(xt(:,ind(1))==1,ind(2)));
    
    options1=options; options2 = options;
    if ~isempty(z)
%       options1.z = options.z(iu1);
      options1.zt = options.zt(iu1);
    end

    meanxt=mean(xt);
    if isequal(method, 'mean')
      xt = repmat(meanxt, length(xtnn1), 1);
    else
      xt = repmat(median(xt), length(xtnn1), 1);
    end
    if ~isempty(var)
      xt(:,~isnan(var)) = repmat(var(~isnan(var)), length(xtnn1), 1);
    end
    xt(:,ind(1)) = -1*ones(length(xtnn1),1); xt(:,ind(2)) = xtnn1;
    if ~strcmp(gp.lik.type, 'Coxph')
      [Ef1, Varf1] = gp_pred(gp, x, y, xt, 'predcf', predcf, 'tstind', tstind, options1);
    else
      [Ef11,Ef12,Covf] = pred_coxph(gp,x,y,xt, 'predcf', predcf, 'tstind', tstind, options1);
      Ef1 = Ef12; Varf1 = diag(Covf(size(Ef11,1)+1:end,size(Ef11,1)+1:end));
    end
    
    if ~isempty(z)
%       options2.z = options.z(iu2);
      options2.zt = options.zt(iu2);
    end
    if isequal(method, 'mean')
      xt = repmat(meanxt, length(xtnn2), 1);
    else
      xt = repmat(median(xt), length(xtnn2), 1);
    end
    if ~isempty(var)
      xt(:,~isnan(var)) = repmat(var(~isnan(var)), length(xtnn2), 1);
    end
    xt(:,ind(1)) = ones(length(xtnn2),1); xt(:,ind(2)) = xtnn2;
    if ~strcmp(gp.lik.type, 'Coxph')
      [Ef2, Varf2] = gp_pred(gp, x, y, xt, 'predcf', predcf, 'tstind', tstind, options2);
    else
      [Ef21,Ef22,Covf] = pred_coxph(gp,x,y,xt, 'predcf', predcf, 'tstind', tstind, options2);
      Ef2 = Ef22; Varf2 = diag(Covf(size(Ef21,1)+1:end,size(Ef21,1)+1:end));
    end
    if isequal(plot_results, 'on')
      plot(xtnn1, Ef1, 'xb', xtnn1, Ef1, '-k', xtnn1, Ef1-sqrt(Varf1), '--k', xtnn1, Ef1+sqrt(Varf1), '--k'); hold on;
      plot(xtnn2, Ef2, 'xb', xtnn2, Ef2, '-r', xtnn2, Ef2-sqrt(Varf2), '--r', xtnn2, Ef2+sqrt(Varf2), '--r');
    end
    Ef = [Ef1; Ef2]; Varf = [Varf1; Varf2]; xtnn=[xtnn1;xtnn2];
    
  elseif sum(xt(:,ind(2))==-1) + sum(xt(:,ind(2))==1) == n 
    % Second covariate binary
    
    [xtnn1, iu1] = unique(xt(xt(:,ind(2))==-1,ind(1)));
    [xtnn2, iu2] = unique(xt(xt(:,ind(2))==1,ind(1)));
    
    if ~isempty(z)
      options1.z = options.z;
      options1.zt = options.zt(iu1);
    end
    
    meanxt=mean(xt);
    if isequal(method, 'mean')
      xt = repmat(meanxt, length(xtnn1), 1);
    else
      xt = repmat(median(xt), length(xtnn1), 1);
    end
    if ~isempty(var)
      xt(:,~isnan(var)) = repmat(var(~isnan(var)), length(xtnn1), 1);
    end
    xt(:,ind(2)) = -1*ones(length(xtnn1),1); xt(:,ind(1)) = xtnn1;
    if ~strcmp(gp.lik.type, 'Coxph')
      [Ef1, Varf1] = gp_pred(gp, x, y, xt, 'predcf', predcf, 'tstind', tstind, options1);
    else
      [Ef11,Ef12,Covf] = pred_coxph(gp,x,y,xt, 'predcf', predcf, 'tstind', tstind, options1);
      Ef1 = Ef12; Varf1 = diag(Covf(size(Ef11,1)+1:end,size(Ef11,1)+1:end));
    end
    
    if ~isempty(z)
      options2.z = options.z;
      options2.zt = options.zt(iu2);
    end
    if isequal(method, 'mean')
      xt = repmat(meanxt, length(xtnn2), 1);
    else
      xt = repmat(median(xt), length(xtnn2), 1);
    end
    if ~isempty(var)
      xt(:,~isnan(var)) = repmat(var(~isnan(var)), length(xtnn2), 1);
    end
    xt(:,ind(2)) = ones(length(xtnn2),1); xt(:,ind(1)) = xtnn2;
    if ~strcmp(gp.lik.type, 'Coxph')
      [Ef2, Varf2] = gp_pred(gp, x, y, xt, 'predcf', predcf, 'tstind', tstind, options2);
    else
      [Ef21,Ef22,Covf] = pred_coxph(gp,x,y,xt, 'predcf', predcf, 'tstind', tstind, options2);
      Ef2 = Ef22; Varf2 = diag(Covf(size(Ef21,1)+1:end,size(Ef21,1)+1:end));
    end
    if isequal(plot_results, 'on')
      plot(xtnn1, Ef1, '-k', xtnn1, Ef1-sqrt(Varf1), '--k', xtnn1, Ef1+sqrt(Varf1), '--k'); hold on;
      plot(xtnn2, Ef2, '-r', xtnn2, Ef2-sqrt(Varf2), '--r', xtnn2, Ef2+sqrt(Varf2), '--r');
    end
    Ef = [Ef1; Ef2]; Varf = [Varf1; Varf2]; xtnn=[xtnn1;xtnn2];
    
  else
    meanxt=mean(xt);
    xtnn1 = linspace(min(xt(:,ind(1))), max(xt(:,ind(1))), 20);
    xtnn2 = linspace(min(xt(:,ind(2))), max(xt(:,ind(2))), 20);
    [XT1, XT2] = meshgrid(xtnn1, xtnn2); XT1=XT1(:); XT2=XT2(:);
    if ~isempty(z)
%       options.z = repmat(options.z(1), 400, 1);
      options.zt = repmat(options.zt(1), 400, 1 );
    end
    if isequal(method, 'mean')
      xt = repmat(meanxt, length(XT1), 1);
    else
      xt = repmat(median(xt), length(XT1), 1);
    end
    if ~isempty(var)
      xt(:,~isnan(var)) = repmat(var(~isnan(var)), length(XT1), 1);
    end
    xt(:,ind) = [XT1 XT2];
    if ~strcmp(gp.lik.type, 'Coxph')
      [Ef, Varf] = gp_pred(gp, x, y, xt, 'predcf', predcf, 'tstind', tstind, options);
    else
      [Ef1,Ef2,Covf] = pred_coxph(gp,x,y,xt, 'predcf', predcf, 'tstind', tstind, options);
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

