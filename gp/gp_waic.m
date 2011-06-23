function waic = gp_waic(gp, x, y, varargin)
%GP_WAIC The widely applicable information criterion (WAIC) for GP model
% 
%  Description
%    WAIC = GP_WAIC(GP, X, Y) evaluates WAIC defined by Watanabe(2010) 
%    given a Gaussian process model GP, training inputs X and
%    training outputs Y.
% 
%    WAIC is evaluated as follows when using the variance form
%        
%      WAIC(n) = BtL(n) + Vn/n
%        
%    where BtL(n) is Bayesian training loss,  Vn is functional variance
%    and n is the number of training inputs.
%
%      BtL = -1/n*sum(log(p(yt | xt, x, y)))
%      Vn = sum(E[log(p(y|th))^2] - E[log(p(y|th))]^2)
%
%    When using the Gibbs training loss, WAIC is evaluated as follows
%
%          WAIC(n) = 2*GtL(n) + BtL(n)
%
%    where BtL(n) is as above and GtL is Gibbs training loss
%
%          GtL(n) = -E[mean(log(p(y|th)))].
%     
%    GP can be a Gaussian process structure, a record structure
%    from GP_MC or an array of GPs from GP_IA.
%
%   OPTIONS is optional parameter-value pair
%      method - Method to evaluate waic, '1' = Variance form, '2' = Gibbs
%               training loss (default = 1)
%      form   - Return form, 'full' or 'single' (default = 'full')
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case. 
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
  ip.addParamValue('focus', 'param', @(x) ismember(x,{'param','latent','all'}))
  ip.addParamValue('method', '1', @(x) ismember(x,{'1','2'}))
  ip.addParamValue('form', 'full', @(x) ismember(x,{'full','single'}))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.parse(gp, x, y, varargin{:});
  focus=ip.Results.focus;
  method=ip.Results.method;
  form=ip.Results.form;
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
        BtL = -lpyt;
        GtL = zeros(tn,1);
        Elog = zeros(tn,1);
        Elog2 = zeros(tn,1);
        
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
        for j = 1:nsamples
          Gp = take_nth(gp,j);
          if  strcmp(gp.type, 'FIC') | strcmp(gp.type, 'PIC')  || strcmp(gp.type, 'CS+FIC') || strcmp(gp.type, 'VAR') || strcmp(gp.type, 'DTC') || strcmp(gp.type, 'SOR')
            Gp.X_u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
          end
          Gp.tr_index = tr_index;

          gp_array{j} = Gp;
          %           w(j,:) = gp_pak(Gp);
          [Ef(:,j), Varf(:,j)] = fh_pred(Gp, x, y, x, 'yt', y, 'tstind', tstind, options);
          if isfield(gp.lik.fh,'trcov')
            sigma2(:,j) = repmat(Gp.lik.sigma2,1,tn);
          end
        end
        
        if strcmp(method,'1')
          % Evaluate WAIC using the variance form
          
          if isfield(gp.lik.fh,'trcov')
            % Gaussian likelihood
            for i=1:tn
              fmin = mean(Ef(i,:) - 9*sqrt(Varf(i,:)));
              fmax = mean(Ef(i,:) + 9*sqrt(Varf(i,:)));
              Elog(i) = quadgk(@(f) mean(multi_npdf(f,Ef(i,:),(Varf(i,:))) ...
                                         .*bsxfun(@minus,-bsxfun(@rdivide,(repmat((y(i)-f),nsamples,1)).^2,(2.*sigma2(i,:))'), 0.5*log(2*sigma2(i,:))').^2), fmin, fmax);
              Elog2(i) = quadgk(@(f) mean(multi_npdf(f,Ef(i,:),(Varf(i,:))) ...
                                          .*bsxfun(@minus,-bsxfun(@rdivide,(repmat((y(i)-f),nsamples,1)).^2,(2.*sigma2(i,:))'), 0.5*log(2*sigma2(i,:))')), fmin, fmax);
            end
            Elog2 = Elog2.^2;
            Vn = (Elog-Elog2);
            if strcmp(form, 'full')
              Vn = sum(Vn)/tn;
              BtL = sum(BtL)/tn;
            end
            waic = BtL + Vn;
          else
            % non-Gaussian likelihood
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
            Vn = (Elog-Elog2);
            if strcmp(form, 'full')
              Vn = sum(Vn)/tn;
              BtL = sum(BtL)/tn;
            end
            waic = BtL + Vn;
          end
          
        else
          % Evaluate WAIC using the expected value form via Gibbs training
          % loss
          
          if isfield(gp.lik.fh,'trcov')
            % Gaussian likelihood
            for i=1:tn
              fmin = mean(Ef(i,:) - 9*sqrt(Varf(i,:)));
              fmax = mean(Ef(i,:) + 9*sqrt(Varf(i,:)));
              GtL(i) = quadgk(@(f) mean(multi_npdf(f,Ef(i,:),(Varf(i,:))) ...
                                        .*bsxfun(@minus,-bsxfun(@rdivide,(repmat((y(i)-f),nsamples,1)).^2,(2.*sigma2(i,:))'), 0.5*log(2*sigma2(i,:))')), fmin, fmax);
            end
            if strcmp(form, 'full')
              GtL = 1/tn*sum(GtL);
              BtL = 1/tn*sum(BtL);
            end
            waic = -2*GtL-BtL;
          else
            % non-Gaussian likelihood
            for i=1:tn
              if ~isempty(z)
                z1 = z(i);
              else
                z1 = [];
              end
              fmin = mean(Ef(i,:) - 9*sqrt(Varf(i,:)));
              fmax = mean(Ef(i,:) + 9*sqrt(Varf(i,:)));
              GtL(i) = quadgk(@(f) mean(multi_npdf(f,Ef(i,:),(Varf(i,:))) ...
                                        .*llvec(gp_array, y(i), f, z1)), fmin, fmax);
            end
            if strcmp(form, 'full')
              GtL = 1/tn*sum(GtL);
              BtL = 1/tn*sum(BtL);
            end
            waic = -2*GtL-BtL;
          end
        end
        
        
      case 'latent'     
        % A single GP solution -> focus on latent variables
        [Ef, Varf, lpyt] = fh_pred(gp, x, y, x, 'yt', y, 'tstind', tstind, options);
        BtL = -lpyt;              % Bayes training loss.

        GtL = zeros(tn,1);
        Elog = zeros(tn,1);
        Elog2 = zeros(tn,1);

        if strcmp(method,'1')          
          % Estimate WAIC with variance form
          
          if isfield(gp.lik.fh,'trcov')
            % Gaussian likelihood
            sigma2 = gp.lik.sigma2;
            
            for i=1:tn
              fmin = Ef(i)-9*sqrt(Varf(i));
              fmax = Ef(i)+9*sqrt(Varf(i));
              
              % Use moments to calculate Elog, Elog2. Faster than guadgk
              % above.
              
              [m0, m1, m2, m3, m4] = moments(@(f) norm_pdf(f,Ef(i),sqrt(Varf(i))), fmin, fmax);          
              Elog2(i) = (-0.5*log(2*pi*sigma2) - y(i).^2./(2.*sigma2))*m0 - 1./(2.*sigma2) * m2 + y(i)./sigma2 * m1;
              Elog(i) = (1/4 * m4 - y(i) * m3 + (3*y(i).^2./2+0.5*log(2*pi*sigma2).*sigma2) * m2 ...
                         - (y(i).^3 + y(i).*log(2*pi*sigma2).*sigma2) * m1 + (y(i).^4/4 + 0.5*y(i).^2*log(2*pi*sigma2).*sigma2 ...
                                                                + 0.25*log(2*pi*sigma2).^2.*sigma2.^2) * m0) ./ sigma2.^2;
              
            end
            Elog2 = Elog2.^2;
            Vn = Elog-Elog2;
            if strcmp(form,'full')
              BtL = sum(BtL)/tn;
              Vn = sum(Vn)/tn;
            end
            waic = BtL + Vn;
            
            % Analytic solution
            
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
            Vn = Elog-Elog2;
            if strcmp(form, 'full')
              Vn = 1/tn * sum(Vn);
              BtL = 1/tn * sum(BtL);
            end
            waic = BtL + 1/tn * Vn;
            
          end
          
        else
          % WAIC using the expected value form via Gibbs training loss GtL
          
          if isfield(gp.lik.fh,'trcov')
            % Gaussian likelihood
            sigma2 = gp.lik.sigma2;
            for i=1:tn
              fmin = Ef(i)-9*sqrt(Varf(i));
              fmax = Ef(i)+9*sqrt(Varf(i));
              
              %               GtL(i) = quadgk(@(f) norm_pdf(f,Ef(i),sqrt(Varf(i))).*(-0.5*log(2*pi*sigma2)- (y(i) - f).^2/(2.*sigma2)), fmin, fmax);

              % Use moments to calculate GtL, faster than guadk above.
              [m0, m1, m2] = moments(@(f) norm_pdf(f,Ef(i),sqrt(Varf(i))), fmin, fmax);
              GtL(i) = (-0.5*log(2*pi*sigma2) - y(i).^2./(2.*sigma2))*m0 - 1./(2.*sigma2) * m2 + y(i)./sigma2 * m1;
            end
            if strcmp(form,'full')
              GtL = 1/tn * sum(GtL);
              BtL = 1/tn * sum(BtL);
            end
            waic = -2*GtL - BtL;
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
              GtL(i) = quadgk(@(f) norm_pdf(f, Ef(i), sqrt(Varf(i))).*llvec(gp, y(i), f, z1) ,...
                              fmin, fmax);
            end
            if strcmp(form,'full')
              GtL = -1/tn * sum(GtL);
              BtL = 1/tn * sum(BtL);
            end
            waic = 2*GtL-BtL;
          end
          
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
    BtL = -lpyt;
    GtL = zeros(tn,1);
    Elog = zeros(tn,1);
    Elog2 = zeros(tn,1);
    
    nsamples = length(gp);
    for j = 1:nsamples
      Gp = gp{j};
      weight(j) = Gp.ia_weight;
      w(j,:) = gp_pak(Gp);
      [Ef(:,j), Varf(:,j)] = fh_pred(Gp, x, y, x, 'yt', y, 'tstind', tstind, options);
      if isfield(Gp.lik.fh,'trcov')
        sigma2(:,j) = repmat(Gp.lik.sigma2,1,tn);
      end
    end
    if strcmp(method,'1')
      % Evaluate WAIC using the variance form
      
      if isfield(gp{1}.lik.fh,'trcov')
        % Gaussian likelihood
        for i=1:tn
          fmin = sum(weight.*Ef(i,:) - 9*weight.*sqrt(Varf(i,:)));
          fmax = sum(weight.*Ef(i,:) + 9*weight.*sqrt(Varf(i,:)));
          Elog(i) = quadgk(@(f) sum(bsxfun(@times, multi_npdf(f,Ef(i,:),(Varf(i,:))),weight') ...
                                    .*bsxfun(@minus,-bsxfun(@rdivide,(repmat((y(i)-f),nsamples,1)).^2,(2.*sigma2(i,:))'), 0.5*log(2*sigma2(i,:))').^2), fmin, fmax);
          Elog2(i) = quadgk(@(f) sum(bsxfun(@times, multi_npdf(f,Ef(i,:),(Varf(i,:))),weight') ...
                                     .*bsxfun(@minus,-bsxfun(@rdivide,(repmat((y(i)-f),nsamples,1)).^2,(2.*sigma2(i,:))'), 0.5*log(2*sigma2(i,:))')), fmin, fmax);
        end
        Elog2 = Elog2.^2;
        Vn = (Elog-Elog2);
        if strcmp(form, 'full')
          Vn = sum(Vn)/tn;
          BtL = sum(BtL)/tn;
        end
        waic = BtL + Vn;
      else
        % non-Gaussian likelihood
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
        Vn = (Elog-Elog2);
        if strcmp(form, 'full')
          Vn = sum(Vn)/tn;
          BtL = sum(BtL)/tn;
        end
        waic = BtL + Vn;
        
      end
      
    else
      % Evaluate WAIC using the expected value form via Gibbs training loss
      
      if isfield(gp{1}.lik.fh,'trcov')
        % Gaussian likelihood
        for i=1:tn
          fmin = sum(weight.*Ef(i,:) - 9*weight.*sqrt(Varf(i,:)));
          fmax = sum(weight.*Ef(i,:) + 9*weight.*sqrt(Varf(i,:)));
          GtL(i) = quadgk(@(f) sum(bsxfun(@times, multi_npdf(f,Ef(i,:),(Varf(i,:))),weight') ...
                                   .*bsxfun(@minus,-bsxfun(@rdivide,(repmat((y(i)-f),nsamples,1)).^2,(2.*sigma2(i,:))'), 0.5*log(2*sigma2(i,:))')), fmin, fmax);
        end
        if strcmp(form, 'full')
          GtL = 1/tn*sum(GtL);
          BtL = 1/tn*sum(BtL);
        end
        waic = -2*GtL-BtL;
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
          GtL(i) = quadgk(@(f) sum(bsxfun(@times, multi_npdf(f,Ef(i,:),(Varf(i,:))),weight') ...
                                   .*llvec(gp, y(i), f, z1)), fmin, fmax);
        end
        if strcmp(form, 'full')
          GtL = 1/tn*sum(GtL);
          BtL = 1/tn*sum(BtL);
        end
        waic = -2*GtL-BtL;
      end
    end
    

  end

end

function lls=llvec(gp, y, fs, z)
% Compute a vector of lls for vector argument fs used by quadgk. In
% case of IA or MC, return a matrix with rows corresponding to one
% GP and columns corresponding to all of the GP's.
  
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

function [m_0, m_1, m_2, m_3, m_4] = moments(fun, a, b, rtol, atol, minsubs)
% QUAD_MOMENTS Calculate the 0th, 1st and 2nd moment of a given
%              (unnormalized) probability distribution
%
%   [m_0, m_1, m_2] = quad_moments(fun, a, b, varargin) 
%   Inputs:
%      fun  = Function handle to the unnormalized probability distribution
%      a,b  = integration limits [a,b]
%      rtol = relative tolerance for the integration (optional, default 1e-6)
%      atol = absolute tolerance for the integration (optional, default 1e-10)
%               
%   Returns the first three moments:
%      m0  = int_a^b fun(x) dx
%      m1  = int_a^b x*fun(x) dx / m0
%      m2  = int_a^b x^2*fun(x) dx / m0
%
%   The function uses an adaptive Gauss-Kronrod quadrature. The same set of 
%   integration points and intervals are used for each moment. This speeds up 
%   the evaluations by factor 3, since the function evaluations are done only 
%   once.
% 
%   The quadrature method is described by:
%   L.F. Shampine, "Vectorized Adaptive Quadrature in Matlab",
%   Journal of Computational and Applied Mathematics, 211, 2008, 
%   pp. 131-140.

%   Copyright (c) 2010 Jarno Vanhatalo, Jouni Hartikainen
  
% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

  maxsubs = 650;
  
  if nargin < 4
    rtol = 1.e-6;
  end
  if nargin < 5
    atol = 1.e-10;
  end
  if nargin < 6
    minsubs = 10;
  end
  
  rtol = max(rtol,100*eps);
  atol = max(atol,0);
  minsubs = max(minsubs,2); % At least two subintervals are needed
  
  % points and weights
  points15 = [0.2077849550078985; 0.4058451513773972; 0.5860872354676911; ...
              0.7415311855993944; 0.8648644233597691; 0.9491079123427585; ...
              0.9914553711208126];
  points = [-points15(end:-1:1); 0; points15];
  
  w15 = [0.2044329400752989, 0.1903505780647854, 0.1690047266392679, ...
         0.1406532597155259, 0.1047900103222502, 0.06309209262997855, ...
         0.02293532201052922];
  w = [w15(end:-1:1), 0.2094821410847278, w15];
  
  w7 = [0,0.3818300505051189,0,0.2797053914892767,0,0.1294849661688697,0];
  ew = w - [w7(end:-1:1), 0.4179591836734694, w7];
  
  samples = numel(w);
  
  % split the interval.
  if b-a <= 0
    c = a; a = b; b=c;
    warning('The start of the integration interval was less than the end of it.')
  end
  apu = a + (1:(minsubs-1))./minsubs*(b-a);
  apu = [a,apu,b];
  subs = [apu(1:end-1);apu(2:end)];
  
  % Initialize partial sums.
  Ifx_ok = 0;
  Ifx1_ok = 0;
  Ifx2_ok = 0;
  Ifx3_ok = 0;
  Ifx4_ok = 0;
  % The main loop
  while true
    % subintervals and their midpoints
    midpoints = sum(subs)/2;   
    halfh = diff(subs)/2;  
    x = bsxfun(@plus,points*halfh,midpoints);
    x = reshape(x,1,[]);
    
    fx = fun(x);
    fx1 = fx.*x;
    fx2 = fx.*x.^2;
    fx3 = fx.*x.^3;
    fx4 = fx.*x.^4;

    fx = reshape(fx,samples,[]);
    fx1 = reshape(fx1,samples,[]);
    fx2 = reshape(fx2,samples,[]);
    fx3 = reshape(fx3,samples,[]);
    fx4 = reshape(fx4,samples,[]);
    
    % Subintegrals.
    Ifxsubs = (w*fx) .* halfh;
    errsubs = (ew*fx) .* halfh;
    Ifxsubs1 = (w*fx1) .* halfh;
    Ifxsubs2 = (w*fx2) .* halfh;
    Ifxsubs3 = (w*fx3) .* halfh;
    Ifxsubs4 = (w*fx4) .* halfh;

    % Ifx and tol.
    Ifx = sum(Ifxsubs) + Ifx_ok;
    Ifx1 = sum(Ifxsubs1) + Ifx1_ok;
    Ifx2 = sum(Ifxsubs2) + Ifx2_ok;
    Ifx3 = sum(Ifxsubs3) + Ifx3_ok;
    Ifx4 = sum(Ifxsubs4) + Ifx4_ok;
    tol = max(atol,rtol*abs(Ifx));
    
    % determine the indices ndx of Ifxsubs for which the
    % errors are acceptable and remove those from subs
    ndx = find(abs(errsubs) <= (2/(b-a)*halfh*tol));
    subs(:,ndx) = [];
    if isempty(subs)
      break
    end
    
    % Update the integral.
    Ifx_ok = Ifx_ok + sum(Ifxsubs(ndx));
    Ifx1_ok = Ifx1_ok + sum(Ifxsubs1(ndx));
    Ifx2_ok = Ifx2_ok + sum(Ifxsubs2(ndx));
    Ifx3_ok = Ifx3_ok + sum(Ifxsubs3(ndx));
    Ifx4_ok = Ifx4_ok + sum(Ifxsubs4(ndx));

    
    % Quit if too many subintervals.
    nsubs = 2*size(subs,2);
    if nsubs > maxsubs
      warning('quad_moments: Reached the limit on the maximum number of intervals in use.');
      break
    end
    midpoints(ndx) = []; 
    subs = reshape([subs(1,:); midpoints; midpoints; subs(2,:)],2,[]); % Divide the remaining subintervals in half
  end
  
  % Scale moments
  m_0 = Ifx;
  m_1 = Ifx1./Ifx;
  m_2 = Ifx2./Ifx;
  m_3 = Ifx3./Ifx;
  m_4 = Ifx4./Ifx;
end
