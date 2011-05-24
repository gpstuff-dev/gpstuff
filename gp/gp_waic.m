function waic = gp_waic(gp, x, y, varargin)
% GP_WAIC The widely applicable information criterion of GP model
% 
%   Description
%     WAIC = GP_WAIC(GP, X, Y) evaluates WAIC defined by Watanabe(2010). WAIC
%     is evaluated with focus on either latent variables(single gp) or
%     parameters(MCM, IA). X contains training inputs and Y contains training
%     outputs.
% 
%   WAIC is evaluated as follows
%        
%          WAIC(n) = BtL(n) + Vn/n
%        
%     where BtL(n) is Bayesian training loss,  Vn is functional variance
%     and n is the number of training inputs.
%
%          BtL = -1/n*sum(log(p(yt | xt, x, y)))
%
%          Vn = sum(E[log(p(y|th))^2] - E[log(p(y|th))]^2)
%     
%     1) GP is a record structure from gp_mc or an array of GPs from gp_ia.
%        Focus is now parameters and latent variables.
%
%     2) GP is Gaussian process structure. In this case the focus is in the
%        latent variables and the parameters are considered fixed. 
%     
%               
%
%   See also
%     GP_DIC, DEMO_MODELASSESMENT1, DEMO_MODELASSESMENT2
%
%   References
%     
%     Watanabe(2010). Equations of states in singular statistical
%     estimation. Neural Networks 23 (2010), 20-34
%
%     Watanabe(2010). Asymptotic Equivalance of Bayes Cross Validation and
%     Widely applicable Information Criterion in Singular Learning Theory.
%     Journal of Machine Learning Research 11 (2010), 3571-3594.
%     
%

% Copyright (c) 2011 Ville Tolvanen



  ip=inputParser;
  ip.FunctionName = 'GP_WAIC';
  ip.addRequired('gp',@(x) isstruct(x) || iscell(x));
  ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addOptional('focus', 'param', @(x) ismember(x,{'param','latent','all'}))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.parse(gp, x, y, varargin{:});
  focus=ip.Results.focus;
  % pass these forward
  options=struct();
  z = ip.Results.z;
  if ~isempty(ip.Results.z)
    options.zt=ip.Results.z;
    options.z=ip.Results.z;
  end
  
  [tn, nin] = size(x);
  
  % ====================================================
  if isstruct(gp)     % Single GP or MCMC solution
    switch gp.type
      case {'FULL' 'VAR' 'DTC' 'SOR'}
        tstind = [];
      case {'FIC' 'CS+FIC'}
        tstind = 1:tn;
      case 'PIC'
        tstind = gp.tr_index;
    end

    if isfield(gp, 'etr')
      % MCMC solution
      if nargin < 4 || isempty(focus)
        focus = 'param';
      end
    else
      % A single GP
      focus = 'latent';
    end     
    
    % Define the error and prediction functions
    if ~isfield(gp.lik.fh,'trcov') && isfield(gp, 'latent_method')
      switch gp.latent_method
        case 'Laplace'
          fh_pred = @gpla_pred;
        case 'EP'
          fh_pred = @gpep_pred;
        case 'MCMC'
          fh_pred = @gpmc_pred;
      end
    else
      fh_pred = @gp_pred;
    end
    
    switch focus
      
      case {'param' 'all'}
        % MCMC solution
        
        [~, ~, lpyt] = gp_pred(gp,x,y, x, 'yt', y, 'tstind', tstind, options);
        BtL = -1/tn * sum(lpyt);
        
        nsamples = length(gp.edata);
        if strcmp(gp.type, 'PIC')
          tr_index = gp.tr_index;
          gp = rmfield(gp, 'tr_index');
        else
          tr_index = [];
        end
        
        Ef = zeros(tn, nsamples);
        Varf = zeros(tn, nsamples);
        sigma2 = zeros(tn, nsamples);
        for i = 1:nsamples
          Gp = take_nth(gp,i);
          if  strcmp(gp.type, 'FIC') | strcmp(gp.type, 'PIC')  || strcmp(gp.type, 'CS+FIC') || strcmp(gp.type, 'VAR') || strcmp(gp.type, 'DTC') || strcmp(gp.type, 'SOR')
            Gp.X_u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
          end
          Gp.tr_index = tr_index;

          gp_array{i} = Gp;
%           w(i,:) = gp_pak(Gp);
          [Ef(:,i), Varf(:,i)] = feval(fh_pred, Gp, x, y, x, 'yt', y, 'tstind', tstind, options);
          if isfield(gp.lik.fh,'trcov')
            sigma2(:,i) = repmat(Gp.lik.sigma2,1,tn);
          end
        end
        
        if isfield(gp.lik.fh,'trcov')
          % gaussian likelihood
          for i=1:tn
            fmin = mean(Ef(i,:) - 9*sqrt(Varf(i,:)));
            fmax = mean(Ef(i,:) + 9*sqrt(Varf(i,:)));
            Elog(i) = quadgk(@(f) mean(multi_npdf(f,Ef(i,:),(Varf(i,:))) ...
              .*bsxfun(@minus,-bsxfun(@rdivide,(repmat((y(i)-f),nsamples,1)).^2,(2.*sigma2(i,:))'), 0.5*log(2*sigma2(i,:))').^2), fmin, fmax);
            Elog2(i) = quadgk(@(f) mean(multi_npdf(f,Ef(i,:),(Varf(i,:))) ...
              .*bsxfun(@minus,-bsxfun(@rdivide,(repmat((y(i)-f),nsamples,1)).^2,(2.*sigma2(i,:))'), 0.5*log(2*sigma2(i,:))')), fmin, fmax);
          end
          Elog2 = Elog2.^2;
          Vn = sum(Elog-Elog2);
          waic = BtL + Vn/tn;
        else
          % non-gaussian likelihood
          for i=1:tn
            if ~isempty(z)
              z1 = z(i);
            else
              z1 = [];
            end
            fmin = mean(Ef(i,:) - 9*sqrt(Varf(i,:)));
            fmax = mean(Ef(i,:) + 9*sqrt(Varf(i,:)));
            Elog(i) = quadgk(@(f) mean(multi_npdf(f,Ef(i,:),(Varf(i,:))) ...
              .*llvec(gp_array, y(i), f, z1).^2), fmin, fmax);
            Elog2(i) = quadgk(@(f) mean(multi_npdf(f,Ef(i,:),(Varf(i,:))) ...
              .*llvec(gp_array, y(i), f, z1)), fmin, fmax);
          end
          Elog2 = Elog2.^2;
          Vn = sum(Elog-Elog2);
          waic = BtL + Vn/tn;
        end
        
          
      case 'latent'     
        % A single GP solution -> focus on latent variables
        [Ef, Varf, lpyt] = feval(fh_pred, gp, x, y, x, 'yt', y, 'tstind', tstind, options);
        BtL = - 1/tn*sum(lpyt);              % Bayes training loss.
%           n = 5000;                    % gp_rnd sample size
%           [sampf] = gp_rnd(gp, x, y, x, 'tstind', tstind, 'nsamp', n, options);
        if isfield(gp.lik.fh,'trcov')
          % Gaussian likelihood
          sigma2 = gp.lik.sigma2;
          
%           Elog = mean((repmat(-0.5*log(2*pi*sigma2), 1, n) - (repmat(y,1,n) - sampf).^2./(2.*repmat(sigma2,1,n))).^2,2);
%           Elog2 = mean((repmat(-0.5*log(2*pi*sigma2),1,n) - (repmat(y,1,n) - sampf).^2./(2.*repmat(sigma2,1,n))),2).^2;
          
          for i=1:tn
            fmin = Ef(i)-9*sqrt(Varf(i));
            fmax = Ef(i)+9*sqrt(Varf(i));
            Elog(i) = quadgk(@(f) norm_pdf(f,Ef(i),sqrt(Varf(i))).*(-0.5*log(2*pi*sigma2)- (y(i) - f).^2/(2.*sigma2)).^2, fmin, fmax);
            Elog2(i) = quadgk(@(f) norm_pdf(f,Ef(i),sqrt(Varf(i))).*(-0.5*log(2*pi*sigma2)- (y(i) - f).^2/(2.*sigma2)), fmin, fmax);
          end
          Elog2 = Elog2.^2;
          
          Vn = sum(Elog - Elog2);
          waic = BtL + 1/tn * Vn;
          
%           k = Ef;
%           a = 1./(2*Varf);
%           C = 0.0997356./ (sigma2.^2.*sqrt(Varf));
%           I4 = 3./(4.*a.^2).*sqrt(pi./a);
%           I2 = (6.*k.^2 + 2.*sigma2.*log(2.*pi.*sigma2)) * 1./(2.*a).*sqrt(pi./a);
%           I0 = (k.^2+sigma2.*log(2.*sigma2.*pi)).^2.*sqrt(pi./a);
%           Elog = C.*(I4+I2+I0);
%           C = -1./(2.*sigma2).*1./sqrt(2*pi.*Varf);
%           I2 = 1./(2*a).*sqrt(pi./a);
%           I0 = (k.^2+sigma2.*log(2*pi*sigma2)).*sqrt(a./pi);
%           Elog2 = (C.*(I2+I0)).^2;
%           Vn = sum(Elog - Elog2);
%           waic = BtL + 1/tn * Vn;

        else
          % Non-Gaussian likelihood
          for i=1:tn
            if ~isempty(z)
              z1 = z(i);
            else
              z1 = [];
            end
            fmin = Ef(i)-9*sqrt(Varf(i));
            fmax = Ef(i)+9*sqrt(Varf(i));
            Elog(i) = quadgk(@(f) norm_pdf(f, Ef(i), sqrt(Varf(i))).*llvec(gp, y(i), f, z1).^2 ,...
              fmin, fmax);
            Elog2(i) = quadgk(@(f) norm_pdf(f, Ef(i), sqrt(Varf(i))).*llvec(gp, y(i), f, z1) ,...
              fmin, fmax);
          end
          Elog2 = Elog2.^2;
          Vn = sum(Elog-Elog2);
          waic = BtL + 1/tn * Vn;
              
        end
          
   
    end
  
  elseif iscell(gp)
      
    % gp_ia solution
    
    switch gp{1}.type
      case {'FULL' 'VAR' 'DTC' 'SOR'}
        tstind = [];
      case {'FIC' 'CS+FIC'}
        tstind = 1:tn;
      case 'PIC'
        tstind = gp{1}.tr_index;
    end
    
    % Define the error and prediction functions
    if isstruct(gp{1}.lik) && isfield(gp{1}, 'latent_method')
      switch gp{1}.latent_method
        case 'Laplace'
          fh_pred = @gpla_pred;
          fh_e = @gpla_e;
        case 'EP'
          fh_pred = @gpep_pred;
          fh_e = @gpep_e;
      end
    else
      fh_e = @gp_e;
      fh_pred = @gp_pred;
    end
    
    [~, ~, lpyt] = gp_pred(gp,x,y, x, 'yt', y, 'tstind', tstind, options);
    BtL = -1/tn * sum(lpyt);
    
    nsamples = length(gp);
    for i = 1:nsamples
      Gp = gp{i};
      weight(i) = Gp.ia_weight;
      w(i,:) = gp_pak(Gp);
      [Ef(:,i), Varf(:,i)] = feval(fh_pred, Gp, x, y, x, 'yt', y, 'tstind', tstind, options);
      if isfield(Gp.lik.fh,'trcov')
        sigma2(:,i) = repmat(Gp.lik.sigma2,1,tn);
      end
    end
    if isfield(gp{1}.lik.fh,'trcov')
      % gaussian likelihood
      for i=1:tn
        fmin = sum(weight.*Ef(i,:) - 9*weight.*sqrt(Varf(i,:)));
        fmax = sum(weight.*Ef(i,:) + 9*weight.*sqrt(Varf(i,:)));
        Elog(i) = quadgk(@(f) sum(bsxfun(@times, multi_npdf(f,Ef(i,:),(Varf(i,:))),weight') ...
          .*bsxfun(@minus,-bsxfun(@rdivide,(repmat((y(i)-f),nsamples,1)).^2,(2.*sigma2(i,:))'), 0.5*log(2*sigma2(i,:))').^2), fmin, fmax);
        Elog2(i) = quadgk(@(f) sum(bsxfun(@times, multi_npdf(f,Ef(i,:),(Varf(i,:))),weight') ...
          .*bsxfun(@minus,-bsxfun(@rdivide,(repmat((y(i)-f),nsamples,1)).^2,(2.*sigma2(i,:))'), 0.5*log(2*sigma2(i,:))')), fmin, fmax);
      end
      Elog2 = Elog2.^2;
      Vn = sum(Elog-Elog2);
      waic = BtL + Vn/tn;
    else
      % non-gaussian likelihood
      for i=1:tn
        if ~isempty(z)
          z1 = z(i);
        else
          z1 = [];
        end
        fmin = sum(weight.*Ef(i,:) - 9*weight.*sqrt(Varf(i,:)));
        fmax = sum(weight.*Ef(i,:) + 9*weight.*sqrt(Varf(i,:)));
        Elog(i) = quadgk(@(f) sum(bsxfun(@times, multi_npdf(f,Ef(i,:),(Varf(i,:))),weight') ...
          .*llvec(gp, y(i), f, z1).^2), fmin, fmax);
        Elog2(i) = quadgk(@(f) sum(bsxfun(@times, multi_npdf(f,Ef(i,:),(Varf(i,:))),weight') ...
          .*llvec(gp, y(i), f, z1)), fmin, fmax);
      end
      Elog2 = Elog2.^2;
      Vn = sum(Elog-Elog2);
      waic = BtL + Vn/tn;
        

    end
        

  end

end

% function lls=llvec(fhlikll,lik,y,fs,z)
%   % compute vector of lls for vector argument fs
%   % used by quadgk
%   lls=zeros(size(fs));
%   for i1=1:numel(fs)
%     lls(i1)=fhlikll(lik,y,fs(i1),z);
%   end
% end

function lls=llvec(gp, y, fs, z)
  % compute vector of lls for vector argument fs
  % used by quadgk. in case of ia or mc, return matrix with rows corresponding to
  % one gp and columns corresponding to all of the gp's.
  
  if isstruct(gp)
    if ~isfield(gp, 'etr')
      % single gp
      lls=zeros(size(fs));
      for i1=1:numel(fs)
        lls(i1)=gp.lik.fh.ll(gp.lik,y,fs(i1),z);
      end
%     else
%       % mc
%       lls=zeros(length(gp), length(fs));
%       for i=1:numel(fs)
%         for j=1:numel(gp.edata)
%           Gp = take_nth(gp, j);
%           lls(j,i) = Gp.lik.fh.ll(Gp.lik, y, fs(i), z);
%         end
%       end
    end
  else
    % ia & mc
    lls=zeros(length(gp), length(fs));
    for i=1:numel(fs)
      for j=1:numel(gp)
        lls(j,i) = gp{j}.lik.fh.ll(gp{j}.lik, y, fs(i), z);
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
