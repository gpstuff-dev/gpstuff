function u_g = gp_refpred(gp1, gp2, x, y, varargin)
% GP_REFPRED Reference predictive approximation to the expected utility of 
%            single predictions.
% 
%   Description
%     
%
%   OPTIONS is optional parameter-value pair
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case. 
%      method - method for inference, 'posterior' (default) uses posterior
%               predictive density, 'loo' uses leave-one-out predictive 
%               density (approximative), 'kfcv' uses loo cross-validation 
%               posterior predictive density, 'joint' uses joint
%               posterior predictive density for latent values
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
  ip.addParamValue('method', 'posterior', @(x) ismember(x,{'posterior' 'kfcv' 'loo'}))
  ip.parse(gp1, gp2, x, y, varargin{:});
  % pass these forward
  options=struct();
  z = ip.Results.z;
  if ~isempty(ip.Results.z)
    options.zt=ip.Results.z;
    options.z=ip.Results.z;
  end
  [tn, nin] = size(x);
  u_g = zeros(size(y));
  opt = optimset('TolX', 1e-4, 'TolFun', 1e-4);

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
          [Ef1, Varf1, tmp, Ey1, Vary1] = gp_pred(gp1,x,y,x,'yt',y, 'tstind', tstind, options);
        case 'loo'
          if ~isfield(gp1.lik.fh, 'trcov')
            gp1 = gp_set(gp1, 'latent_method', 'EP');
          end
          [Ef1, Varf1, tmp, Ey1, Vary1] = gp_loopred(gp1,x,y, options);
        case 'kfcv'
          [~, preds] = gp_kfcv(gp1, x, y, 'tstindex', tstind, 'opt', opt, 'display', 'off', 'k', tn, options);
          [Ef1, Varf1, Ey1, Vary1] = deal(preds.Eft,preds.Varft,preds.Eyt,preds.Varyt);
        case 'joint'
          [Ef1, Covf1, tmp, Ey1, Covy1] = gp_jpred(gp1,x,y,x,'yt',y, 'tstind', tstind, options);
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
      
      Ef = zeros(tn, nsamples);
      Varf = zeros(tn, nsamples);
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
            [Ef1(:,j), Varf1(:,j), tmp, Ey1(:,j), Vary1(:,j)] = gp_loopred(Gp, x, y, options);
          case 'kfcv'
            [tmp, pred] = gp_kfcv(Gp, x, y, 'tstindex', tstind, 'k', tn, 'opt', opt, 'display', 'off', options);
            [Ef1(:,j), Varf1(:,j), Ey1(:,j), Vary1(:,j)] = deal(preds.Eft, preds.Varft, preds.Eyt, preds.Varyt);
        end
      end
      if isequal(method, 'joint')
        [Ef1, Covf1, tmp, Ey1, Covy1] = gp_jpred(gp1, x, y, x, 'yt', y, 'tstind', tstind, options);
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
          [Ef1(:,j), Varf1(:,j), tmp, Ey1(:,j), Vary1(:,j)] = gp_pred(Gp, x, y, options);
        case 'kfcv'
          [tmp, preds] = gp_pred(Gp, x, y, 'tstindex', tstind, 'k', tn, 'opt', opt, 'display', 'off', options);
          [Ef1(:,j), Varf1(:,j), tmp, Ey1(:,j), Vary1(:,j)] = deal(preds.Eft, preds.Varft, preds.Eyt, preds.Varyt);
      end
    end
    if isequal(method, 'joint')
        [Ef1, Covf1, tmp, Ey1, Covy1] = gp_jpred(gp1, x, y, x, 'yt', y, 'tstind', tstind, options);
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
          [Ef2, Varf2, tmp, Ey2, Vary2] = gp_pred(gp2,x,y,x,'yt',y, 'tstind', tstind, options);
        case 'loo'
          if ~isfield(gp2.lik.fh, 'trcov')
            gp1 = gp_set(gp2, 'latent_method', 'EP');
          end
          [Ef2, Varf2, tmp, Ey2, Vary2] = gp_loopred(gp2,x,y, options);
        case 'kfcv'
          [tmp, preds] = gp_kfcv(gp2, x, y, 'tstindex', tstind, 'opt', opt, 'k', tn, 'opt', opt, 'display', 'off', options);
          [Ef2, Varf2, Ey2, Vary2] = deal(preds.Eft,preds.Varft,preds.Eyt,preds.Varyt);
        case 'joint'
          [Ef2, Covf2, tmp, Ey2, Covy2] = gp_jpred(gp2,x,y,x,'yt',y, 'tstind', tstind, options);
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
      
      Ef = zeros(tn, nsamples);
      Varf = zeros(tn, nsamples);
      for j = 1:nsamples
        Gp = take_nth(gp2,j);
        if  strcmp(gp2.type, 'FIC') | strcmp(gp2.type, 'PIC')  || strcmp(gp2.type, 'CS+FIC') || strcmp(gp2.type, 'VAR') || strcmp(gp2.type, 'DTC') || strcmp(gp2.type, 'SOR')
          Gp.X_u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
        end
        Gp.tr_index = tr_index;
        gp_array2{j} = Gp;
        switch method
          case 'posterior'
            [Ef2(:,j), Varf2(:,j), tmp, Ey2(:,j), Vary2(:,j)] = gpmc_pred(Gp, x, y, x, 'yt', y, 'tstind', tstind, options);
          case 'loo'
            [Ef2(:,j), Varf2(:,j), tmp, Ey2(:,j), Vary2(:,j)] = gp_loopred(Gp, x, y, options);
          case 'kfcv'
            [Ef2(:,j), Varf2(:,j), tmp, Ey2(:,j), Vary2(:,j)] = gp_kfcv(Gp, x, y, 'tstindex', tstind, 'k', tn, 'opt', opt, 'display', 'off', options);
        end
        
      end
      if isequal(method, 'joint')
        [Ef2, Covf2, tmp, Ey2, Covy2] = gp_jpred(gp2, x, y, x, 'yt', y, 'tstind', tstind, options);
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
          [Ef2(:,j), Varf2(:,j), tmp, Ey2(:,j), Vary2(:,j)] = gp_pred(Gp, x, y, x, 'yt', y, 'tstind', tstind, options);
        case 'loo'
          [Ef2(:,j), Varf2(:,j), tmp, Ey2(:,j), Vary2(:,j)] = gp_loopred(Gp, x, y, options);
        case 'kfcv'
          [Ef2(:,j), Varf2(:,j), tmp, Ey2(:,j), Vary2(:,j)] = gp_pred(Gp, x, y, x, 'yt', y, 'tstindex', tstind, 'k', tn, 'opt', opt, 'display', 'off', options);
      end
    end
    if isequal(method, 'joint')
      [Ef2, Covf2, tmp, Ey2, Covy2] = gp_jpred(gp2, x, y, x, 'yt', y, 'tstind', tstind, options);
    end
    if isfield(gp2{1}.lik.fh, 'trcov')
      fh2 = @(f,Ey,Vary) log(sum(bsxfun(@times, multi_npdf(f,Ey,(Vary)),weight2'),1));
    else
      fh2 = @(gp,Ef,Varf,f,z) log(sum(bsxfun(@times, exp(predvec(gp,Ef,(Varf),f,z)),weight2'),1));
    end
    
  end
  
  if ((isstruct(gp1) && isfield(gp1.lik.fh, 'trcov')) || (iscell(gp1) && isfield(gp1{1}.lik.fh,'trcov')))
    % Gaussian likelihood
    
    % Integration limits from gp1 predictions
    if model1~=3
      minf = mean(Ey1 - 6.*sqrt(Vary1),2);
      maxf = mean(Ey1 + 6.*sqrt(Vary1),2);
    else
      minf = sum(bsxfun(@times, weight1, Ey1-6.*sqrt(Vary1)),2);
      maxf = sum(bsxfun(@times, weight1, Ey1+6.*sqrt(Vary1)),2);
    end
    
    switch method
      case 'joint'
        u_g = -0.5.*((Ey1 - Ey2)'*(Covy2\(Ey1-Ey2)) + sum(sum(inv(Covy2).*Covy1))) ...
          -(tn/2*log(2*pi) + sum(diag(Covy2)));
      otherwise
        for i=1:tn
          u_g(i) = quadgk(@(f) fh1(f,Ey1(i,:),Vary1(i,:)).*fh2(f,Ey2(i,:),Vary2(i,:)), minf(i), maxf(i));
        end
    end
 
  else
    % Non-Gaussian likelihood
    
    switch method
      case 'joint'
        % Joint refpred of latent values
        u_g = -0.5.*((Ef1 - Ef2)'*(Covf2\(Ef1-Ef2)) + sum(sum(inv(Covf2).*Covf1))) ...
          -(tn/2*log(2*pi) + sum(diag(Covf2)));
        
      otherwise
        if ismember(gp1.lik.type, {'Binomial', 'Poisson', 'Probit', 'Logit', 'Negbin', 'Negbinztr'})
          % Discrete likelihoods
          for i=1:tn
            if ~isempty(z)
              z1 = z(i);
            else
              z1 = [];
            end
            if model1~=3
              [tmp, tmp, int] = int_limits(gp1, Ef1(i,:), z1);
            else
              [minf maxf] = int_limits(gp1,Ef1(i,:),z1);
              minf = sum(minf.*weight1);
              maxf = sum(maxf.*weight1);
              int = minf:maxf;
            end
            u_g(i) = sum(fh1(gp1,Ef1(i,:),Varf1(i,:),int,z1).*fh2(gp2,Ef2(i,:),Varf2(i,:),int,z1));
          end
        else
          % Continuous likelihoods
          for i=1:tn
            if ~isempty(z)
              z1 = z(i);
            else
              z1 = [];
            end
            if model1~=3
              if ismember(gp1.lik.type, {'Student-t', 'Weibull'})
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
            u_g(i) = quadgk(@(f) fh1(gp1,Ef1(i,:),Varf1(i,:),f,z1).*fh2(gp2,Ef2(i,:),Varf2(i,:),f,z1), minf, maxf, 'absTol', 1e-3);
          end
        end
    end
  end
  u_g = mean(u_g);
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
      lambda = z.*exp(Ef(:,i1));
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
  end

end
