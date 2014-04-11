function u_g = gp_refpred(gp1, gp2, x, y, varargin)
% GP_REFPRED Reference predictive approximation to the expected utility of 
%            single predictions.
% 
%   Description
%     u = GP_REFPRED(GP1, GP2, X, Y, OPTIONS) evaluates reference
%     predictive approximation between models GP1 and GP2. Here GP1 is the
%     reference model and GP2 is the candidate model.
%
%   OPTIONS is optional parameter-value pair
%      z        - optional observed quantity in triplet (x_i,y_i,z_i)
%                 Some likelihoods may use this. For example, in case of 
%                 Poisson likelihood we have z_i=E_i, that is, expected value 
%                 for ith case. 
%      method   - method for inference, 'posterior' (default) uses posterior
%                 predictive density, 'loo' uses leave-one-out predictive 
%                 density (approximative), 'kfcv' uses loo cross-validation 
%                 posterior predictive density, 'joint' uses joint
%                 posterior predictive density for latent values
%                 (non-Gaussian likelihood) or observations (Gaussian
%                 likelihood)
%      x2,y2,z2 - Optional values for candidate model gp2. 
%                 If only subset of these is specified, remaining variables
%                 are set from x,y,z.
%     
%   See also
%     GP_LOOPRED, GP_KFCV   
%
%   References
%     Vehtari & Ojanen (2011). Bayesian preditive methods for model
%     assesment and selection. In preparation.
%
% Copyright (c) 2011-2012 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GP_REFPRED';
  ip.addRequired('gp1',@(x) isstruct(x) || iscell(x));
  ip.addRequired('gp2',@(x) isstruct(x) || iscell(x));
  ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('x2', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('y2', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z2', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('method', 'posterior', @(x) ismember(x,{'posterior' 'kfcv' 'loo' 'joint'}))
  ip.addParamValue('form', 'mean', @(x) ismember(x,{'mean','all'}))
  ip.parse(gp1, gp2, x, y, varargin{:});
  % pass these forward
  options=struct();
  x2 = ip.Results.x2;
  y2 = ip.Results.y2;
  z2 = ip.Results.z2;
  z = ip.Results.z;
  method = ip.Results.method;
  form = ip.Results.form;
  if ~isempty(ip.Results.z)
    options.zt=ip.Results.z;
    options.z=ip.Results.z;
  end
  if ~isempty(ip.Results.z2)
    options2.zt=ip.Results.z2;
    options2.z=ip.Results.z2;
  else
    options2 = options;
    z2 = z;
  end
  [tn, nin] = size(x);
  u_g = zeros(size(y));
  opt = optimset('TolX', 1e-4, 'TolFun', 1e-4);
  if isempty(x2)
    x2 = x;
  end
  if isempty(y2)
    y2 = y;
  end

  if isstruct(gp1)
    % Single gp or MCMC
    
    switch gp1.type
      case {'FULL' 'VAR' 'DTC' 'SOR'}
        tstind = [];
      case {'FIC' 'CS+FIC'}
        tstind = 1:tn;
      case 'PIC'
        tstind = gp1.tr_index;
    end
    if ~isfield(gp1,'etr')
      model1 = 1;
      if isfield(gp1.lik.fh, 'trcov')
        switch method
          case 'joint'
            
          otherwise
            fh1 = @(f,Ey,Vary) norm_pdf(f,Ey,sqrt(Vary));
        end
      else
        fh1 = @(gp,Ef,Varf,f,z) exp(predvec(gp,Ef,(Varf),f,z));
      end
      
      switch method
        case 'posterior'
          if ~isequal(gp1.lik.type, 'Coxph')
            [Ef1, Varf1, tmp, Ey1, Vary1] = gp_pred(gp1,x,y,x,'yt',y, 'tstind', tstind, options);
          else
            [Ef1, Varf1] = gp_pred(gp1,x,y,x,'yt',y, 'tstind', tstind, options);
          end
        case 'loo'
          if ~isfield(gp1.lik.fh, 'trcov') && ~isfield(gp1.lik, 'type_nd')
            gp1 = gp_set(gp1, 'latent_method', 'EP');
          end
          [Ef1, Varf1, tmp, Ey1, Vary1] = gp_loopred(gp1,x,y, 'z', z);
        case 'kfcv'
          [tmp, preds] = gp_kfcv(gp1, x, y, 'tstindex', tstind, 'opt', opt, 'display', 'iter', 'k', tn, options);
          [Ef1, Varf1, Ey1, Vary1] = deal(preds.Eft,preds.Varft,preds.Eyt,preds.Varyt);
        case 'joint'
          [Ef1, Covf1] = gp_jpred(gp1,x,y,x,'yt',y, 'tstind', tstind, options);
      end
        
    else
      model1 = 2;
      if isfield(gp1.lik.fh, 'trcov')
        fh1 = @(f,Ey,Vary) mean(multi_npdf(f,Ey,(Vary)),1);
      else
        fh1 = @(gp,Ef,Varf,f,z) mean(exp(predvec(gp,Ef,(Varf),f,z)),1);
      end
      nsamples = length(gp1.edata);
      if strcmp(gp1.type, 'PIC')
        tr_index = gp1.tr_index;
        gp1 = rmfield(gp1, 'tr_index');
      else
        tr_index = [];
      end
      
      for j = 1:nsamples
        Gp = take_nth(gp1,j);
        if  strcmp(gp1.type, 'FIC') | strcmp(gp1.type, 'PIC')  || strcmp(gp1.type, 'CS+FIC') || strcmp(gp1.type, 'VAR') || strcmp(gp1.type, 'DTC') || strcmp(gp1.type, 'SOR')
          Gp.X_u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
        end
        Gp.tr_index = tr_index;
        gp_array1{j} = Gp;
        switch method
          case 'posterior'
            [Ef1(:,j), Varf1(:,j), tmp, Ey1(:,j), Vary1(:,j)] = gpmc_pred(Gp, x, y, x, 'yt', y, 'tstind', tstind, options);
          case 'loo'
            [Ef1(:,j), Varf1(:,j), tmp, Ey1(:,j), Vary1(:,j)] = gp_loopred(Gp, x, y, 'z', z);
          case 'kfcv'
            [tmp, pred] = gp_kfcv(Gp, x, y, 'tstindex', tstind, 'k', tn, 'opt', opt, 'display', 'iter', options);
            [Ef1(:,j), Varf1(:,j), Ey1(:,j), Vary1(:,j)] = deal(preds.Eft, preds.Varft, preds.Eyt, preds.Varyt);
        end
      end
      if isequal(method, 'joint')
        [Ef1, Covf1] = gp_jpred(gp1, x, y, x, 'yt', y, 'tstind', tstind, options);
      end
      gp1 = gp_array1;
    end
  else
    % GP IA
    switch gp1{1}.type
      case {'FULL' 'VAR' 'DTC' 'SOR'}
        tstind = [];
      case {'FIC' 'CS+FIC'}
        tstind = 1:tn;
      case 'PIC'
        tstind = gp1{1}.tr_index;
    end
    model1 = 3;
    nsamples = length(gp1);
    for j = 1:nsamples
      Gp = gp1{j};
      weight1(j) = Gp.ia_weight;
      w(j,:) = gp_pak(Gp);
      switch method
        case 'posterior'
          [Ef1(:,j), Varf1(:,j), tmp, Ey1(:,j), Vary1(:,j)] = gp_pred(Gp, x, y, x, 'yt', y, 'tstind', tstind, options);
        case 'loo'
          [Ef1(:,j), Varf1(:,j), tmp, Ey1(:,j), Vary1(:,j)] = gp_pred(Gp, x, y, 'z', z);
        case 'kfcv'
          [tmp, preds] = gp_pred(Gp, x, y, 'tstindex', tstind, 'k', tn, 'opt', opt, 'display', 'iter', options);
          [Ef1(:,j), Varf1(:,j), tmp, Ey1(:,j), Vary1(:,j)] = deal(preds.Eft, preds.Varft, preds.Eyt, preds.Varyt);
      end
    end
    if isequal(method, 'joint')
        [Ef1, Covf1] = gp_jpred(gp1, x, y, x, 'yt', y, 'tstind', tstind, options);
    end
    if isfield(gp1{1}.lik.fh, 'trcov')
      fh1 = @(f,Ey,Vary) sum(bsxfun(@times, multi_npdf(f,Ey,(Vary)),weight1'),1);
    else
      fh1 = @(gp,Ef,Varf,f,z) (sum(bsxfun(@times, exp(predvec(gp,Ef,(Varf),f,z)),weight1'),1));
    end
    
  end
  
  if isstruct(gp2)
    % Single gp or MCMC
    switch gp2.type
      case {'FULL' 'VAR' 'DTC' 'SOR'}
        tstind = [];
      case {'FIC' 'CS+FIC'}
        tstind = 1:tn;
      case 'PIC'
        tstind = gp2.tr_index;
    end
    if ~isfield(gp2,'etr')
      model2 = 1;
      if isfield(gp2.lik.fh, 'trcov')
        fh2 = @(f,Ey,Vary) norm_lpdf(f,Ey,sqrt(Vary));
      else
        fh2 = @(gp,Ef,Varf,f,z) predvec(gp,Ef,(Varf),f,z);
      end
      switch method
        case 'posterior'
          if ~isequal(gp2.lik.type, 'Coxph')
            [Ef2, Varf2, tmp, Ey2, Vary2] = gp_pred(gp2,x2,y2,x2,'yt',y2, 'tstind', tstind, options2);
          else
            [Ef2, Varf2] = gp_pred(gp2,x2,y2,x2,'yt',y2, 'tstind', tstind, options2);
          end
        case 'loo'
          if ~isfield(gp2.lik.fh, 'trcov') && ~isfield(gp2.lik, 'type_nd')
            gp1 = gp_set(gp2, 'latent_method', 'EP');
          end
          [Ef2, Varf2, tmp, Ey2, Vary2] = gp_loopred(gp2,x2,y2, 'z', z2);
        case 'kfcv'
          [tmp, preds] = gp_kfcv(gp2, x2, y2, 'tstindex', tstind, 'k', tn, 'opt', opt, 'display', 'iter', options2);
          [Ef2, Varf2, Ey2, Vary2] = deal(preds.Eft,preds.Varft,preds.Eyt,preds.Varyt);
        case 'joint'
          [Ef2, Covf2] = gp_jpred(gp2,x2,y2,x2,'yt',y2, 'tstind', tstind, options2);
      end
    else
      model2 = 2;
      if isfield(gp2.lik.fh, 'trcov')
        fh2 = @(f,Ey,Vary) log(mean(multi_npdf(f,Ey,(Vary)),1));
      else
        fh2 = @(gp,Ef,Varf,f,z) log(mean(exp(predvec(gp,Ef,(Varf),f,z)),1));
      end
      nsamples = length(gp2.edata);
      if strcmp(gp2.type, 'PIC')
        tr_index = gp2.tr_index;
        gp2 = rmfield(gp2, 'tr_index');
      else
        tr_index = [];
      end
      
      for j = 1:nsamples
        Gp = take_nth(gp2,j);
        if  strcmp(gp2.type, 'FIC') | strcmp(gp2.type, 'PIC')  || strcmp(gp2.type, 'CS+FIC') || strcmp(gp2.type, 'VAR') || strcmp(gp2.type, 'DTC') || strcmp(gp2.type, 'SOR')
          Gp.X_u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
        end
        Gp.tr_index = tr_index;
        gp_array2{j} = Gp;
        switch method
          case 'posterior'
            [Ef2(:,j), Varf2(:,j), tmp, Ey2(:,j), Vary2(:,j)] = gpmc_pred(Gp, x2, y2, x2, 'yt', y2, 'tstind', tstind, options2);
          case 'loo'
            [Ef2(:,j), Varf2(:,j), tmp, Ey2(:,j), Vary2(:,j)] = gp_loopred(Gp, x2, y2, 'z', z2);
          case 'kfcv'
            [Ef2(:,j), Varf2(:,j), tmp, Ey2(:,j), Vary2(:,j)] = gp_kfcv(Gp, x2, y2, 'tstindex', tstind, 'k', tn, 'opt', opt, 'display', 'iter', options2);
        end
        
      end
      if isequal(method, 'joint')
        [Ef2, Covf2] = gp_jpred(gp2, x2, y2, x2, 'yt', y2, 'tstind', tstind, options2);
      end
      gp2 = gp_array2;
    end
  else
    % GP IA
    switch gp2{1}.type
      case {'FULL' 'VAR' 'DTC' 'SOR'}
        tstind = [];
      case {'FIC' 'CS+FIC'}
        tstind = 1:tn;
      case 'PIC'
        tstind = gp2{1}.tr_index;
    end
    model2 = 3;
    nsamples = length(gp2);
    for j = 1:nsamples
      Gp = gp2{j};
      weight2(j) = Gp.ia_weight;
      w(j,:) = gp_pak(Gp);
      switch method
        case 'posterior'
          [Ef2(:,j), Varf2(:,j), tmp, Ey2(:,j), Vary2(:,j)] = gp_pred(Gp, x2, y2, x2, 'yt', y2, 'tstind', tstind, options2);
        case 'loo'
          [Ef2(:,j), Varf2(:,j), tmp, Ey2(:,j), Vary2(:,j)] = gp_loopred(Gp, x2, y2, 'z', z2);
        case 'kfcv'
          [Ef2(:,j), Varf2(:,j), tmp, Ey2(:,j), Vary2(:,j)] = gp_pred(Gp, x2, y2, x2, 'yt', y2, 'tstindex', tstind, 'k', tn, 'opt', opt, 'display', 'off', options2);
      end
    end
    if isequal(method, 'joint')
      [Ef2, Covf2] = gp_jpred(gp2, x2, y2, x2, 'yt', y2, 'tstind', tstind, options2);
    end
    if isfield(gp2{1}.lik.fh, 'trcov')
      fh2 = @(f,Ey,Vary) log(sum(bsxfun(@times, multi_npdf(f,Ey,(Vary)),weight2'),1));
    else
      fh2 = @(gp,Ef,Varf,f,z) log(sum(bsxfun(@times, exp(predvec(gp,Ef,(Varf),f,z)),weight2'),1));
    end
    
  end
  
  if ((isstruct(gp1) && isfield(gp1.lik.fh, 'trcov')) || (iscell(gp1) && isfield(gp1{1}.lik.fh,'trcov')))
    % Gaussian likelihood
    
    switch method
      case 'joint'
        u_g = -0.5.*((Ey1 - Ey2)'*(Covy2\(Ey1-Ey2)) + sum(sum(inv(Covy2).*Covy1)) ...
          + tn*log(2*pi) + 2*sum(log(diag(chol(Covy2)))));
      otherwise
        for i=1:tn
          m1 = Ey1(i,:); m2=Ey1(i,:).^2 + Vary1(i,:);
          u_g(i) = mean(-1./(2.*Vary2(i,:))*m2 + Ey2(i,:)./Vary2(i,:)*m1 - Ey2(i,:).^2./(2.*Vary2(i,:)) - 0.5*log(2*pi*Vary2(i,:)));
        end
    end
 
  else
    % Non-Gaussian likelihood
    
    switch method
      case 'joint'
        % Joint refpred of latent values
        u_g = -0.5.*((Ef1 - Ef2)'*(Covf2\(Ef1-Ef2)) + sum(sum(inv(Covf2).*Covf1)) ...
          + tn*log(2*pi) + 2*sum(log(diag(chol(Covf2)))));
        
      otherwise
        if ismember(gp1.lik.type, {'Binomial', 'Poisson', 'Probit', 'Logit', 'Negbin', 'Negbinztr'})
          % Discrete likelihoods
          for i=1:tn
            if ~isempty(z)
              z1 = z(i);
              z12 = z2(i);
            else
              z1 = [];
              z12 = [];
            end
            if model1~=3
              [tmp, tmp, int] = int_limits(gp1, Ef1(i,:), z1);
            else
              [minf maxf] = int_limits(gp1,Ef1(i,:),z1);
              minf = sum(minf.*weight1);
              maxf = sum(maxf.*weight1);
              int = minf:maxf;
            end
            u_g(i) = sum(fh1(gp1,Ef1(i,:),Varf1(i,:),int,z1).*fh2(gp2,Ef2(i,:),Varf2(i,:),int,z12));
          end
        else
          % Continuous likelihoods
          for i=1:tn
            if ~isempty(z)
              z1 = z(i);
              z12 = z2(i);
            else
              z1 = [];
              z12 = [];
            end
            if model1~=3
              if ismember(gp1.lik.type, {'Student-t', 'Weibull', 'Coxph'})
                [minf, maxf] = int_limits(gp1, Ef1(i), z1);
              else
                minf = mean(Ey1(i) - 12.*sqrt(Vary1(i)),2);
                minf(minf<0)=0;
                maxf = mean(Ey1(i) + 12.*sqrt(Vary1(i)),2);
              end
            else
              minf = sum(bsxfun(@times, weight1, Ey1(i,:)-12.*sqrt(Vary1(i,:))),2);
              maxf = sum(bsxfun(@times, weight1, Ey1(i,:)+12.*sqrt(Vary1(i,:))),2);
            end
            if ~isequal(gp1.lik.type, 'Coxph')
              u_g(i) = quadgk(@(f) fh1(gp1,Ef1(i,:),Varf1(i,:),f,z1).*fh2(gp2,Ef2(i,:),Varf2(i,:),f,z12), minf, maxf, 'absTol', 1e-3);
            else
              ntime1=size(gp1.lik.xtime,1);
              ntime2=size(gp2.lik.xtime,1);
              u_g(i) = quadgk(@(f) fh1(gp1,Ef1([1:ntime1 i],:),Varf1([1:ntime1 i+ntime1],[1:ntime1 i+ntime1]),f,z1).*fh2(gp2,Ef2([1:ntime2 i],:),Varf2([1:ntime2 i+ntime2],[1:ntime2 i+ntime2]),f,z12), minf, maxf, 'absTol', 1e-3);
            end
          end
        end
    end
  end
  if isequal(form, 'mean')
    u_g = mean(u_g);
  end
end

function predvec = predvec(gp, Ef, Varf, f, z)
  % Compute vector of lpyts from lik.fh.predy when numel(Ef)=numel(Varf)=1
  % and numel(f) != 1.
  if isstruct(gp)
    if ~isfield(gp, 'etr')
      % single gp
      predvec=zeros(size(f));
      for i1=1:numel(f)
        predvec(i1)=gp.lik.fh.predy(gp.lik,Ef,Varf,f(i1),z);
      end
    end
  else
    % ia & mc
    predvec=zeros(length(gp), length(f));
    for i=1:numel(f)
      for j=1:numel(gp)
        predvec(j,i) = gp{j}.lik.fh.predy(gp{j}.lik, Ef(j), Varf(j), f(i), z);
      end  
    end
  end
end

function mpdf = multi_npdf(f, mean, sigma2)
% for every element in f, compute means calculated with 
% norm_pdf(f(i), mean, sqrt(sigma2)). If mean and sigma2
% are vectors, returns length(mean) x length(f) matrix. 
  
  mpdf = zeros(length(mean), length(f));
  for i=1:length(f)
    mpdf(:,i) = norm_pdf(f(i), mean, sqrt(sigma2));
  end
end

function [minf, maxf, interval] = int_limits(gp, Ef, z)
% Return integration limits for quadgk and interval for discrete integration.
  if isstruct(gp)
    gplik = gp.lik;
  else
    gplik = gp{1}.lik;
  end
  switch gplik.type
    
    case 'Binomial'
      p = exp(Ef)./(1+exp(Ef));
      minf = binoinv(0.0001, z, p);
      maxf = binoinv(0.9999, z, p);
      interval = minf:maxf;
    case 'Poisson'
      lambda = z.*exp(Ef);
      minf = poissinv(0.0001, lambda);
      maxf = poissinv(0.9999, lambda);
      interval=minf:maxf;
    case {'Probit' 'Logit'}
      minf = -1*ones(size(Ef));
      maxf = ones(size(Ef));
      interval = [-1 1];
    case {'Negbin' 'Negbinztr'}
      r = gplik.disper;
      p = z.*exp(Ef);
      minf = nbininv(0.0001, r, p);
      maxf = nbininv(0.9999, r, p);
      interval = minf:maxf;
    case 'Student-t'
      [n, n2] = size(Ef);
      nu = gp.lik.nu;
      minf = repmat(tinv(0.01, nu), n, n2);
      maxf = repmat(tinv(0.99, nu), n, n2);
      interval = [];
    case 'Weibull'
      % Probably not very sensible...
      minf = 1e-5;
      maxf = 1e5;
      interval = maxf;
    case 'Coxph'
      minf = 0;
      maxf = 1;
      interval = maxf;
  end

end
