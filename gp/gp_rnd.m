function [sampft, sampyt] = gp_rnd(gp, x, y, varargin)
%GP_RND  Random draws from the posterior Gaussian process
%
%  Description
%    [SAMPFT, SAMPYT] = GP_RND(GP, X, Y, XT, OPTIONS) takes a
%    Gaussian process structure, record structure (from gp_mc) or
%    array (from gp_ia) GP together with a matrix XT of input
%    vectors, matrix X of training inputs and vector Y of training
%    targets, and returns a random sample SAMPFT and SAMPYT from
%    the posterior distribution p(ft|x,y,xt) and the predictive
%    distribution p(yt|x,y,xt) at locations XT.
%
%    OPTIONS is optional parameter-value pair
%      nsamp     - determines the number of samples (default = 1).
%      predcf    - index vector telling which covariance functions are 
%                  used for prediction. Default is all (1:gpcfn)
%      tstind    - a vector defining, which rows of X belong to which 
%                  training block in *IC type sparse models. Default is [].
%                  See also GP_PRED.
%      z         - optional observed quantity in triplet (x_i,y_i,z_i)
%                  Some likelihoods may use this. For example, in case of 
%                  Poisson likelihood we have z_i=E_i, that is, expected
%                  value
%                  for ith case. 
%      fcorr     - Method used for latent marginal posterior corrections. 
%                  Default is 'off'. Possible methods are 'fact' for EP
%                  and either 'fact', 'cm2' or 'lr' for Laplace. If method is
%                  'on', 'fact' is used for EP and 'cm2' for Laplace.
%                  See GP_PREDCM and Cseke & Heskes (2011) for more information.
%      splitnormal - determines if the samples are drawn from
%                  split-Normal approximation (Geweke, 1989) in the
%                  case of non-Gaussian likelihood.
%                  Possible values are 'on' (default) and 'off'. 
%      n_scale   - the maximum number of the most significant principal
%                  component directories to scale in the split-Normal
%                  approximation (default 50).
%
%    If likelihood is non-Gaussian and gp.latent_method is Laplace the
%    samples are drawn from split-Normal approximation (see option
%    splitnormal), and if gp.latent_method is EP the samples are drawn
%    from the normal approximation.
%
%  Reference
%    Cseke & Heskes (2011). Approximate Marginals in Latent Gaussian
%    Models. Journal of Machine Learning Research 12 (2011), 417-454
%
%    Geweke, J. (1989).  Bayesian inference in econometric models using
%    Monte Carlo integration. Econometrica 57:1317-1339.
%
%  See also
%    GP_PRED, GP_PAK, GP_UNPAK

%  Internal comments
%    - sampling with FIC, PIC and CS+FIC forms full nxn matrix and
%       works only when sampling for the training inputs
%    - The code is not optimized
%
% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2008      Jouni Hartikainen
% Copyright (c) 2011      Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GP_RND';
ip.addRequired('gp',@(x) isstruct(x) || iscell(x));
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('xt', [], @(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('predcf', [], @(x) isempty(x) || ...
                 isvector(x) && isreal(x) && all(isfinite(x)&x>0))
ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                 (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
ip.addParamValue('nsamp', 1, @(x) isreal(x) && isscalar(x))
ip.addParamValue('fcorr', 'off', @(x) ismember(x, {'fact','cm2','off','on','lr'}))
ip.addParamValue('splitnormal', 'on', @(x) (islogical(x) && isscalar(x))|| ...
                 ismember(x,{'on' 'off' 'full'}))
ip.addParamValue('n_scale',50, @(x) isnumeric(x) && x>=0);
if numel(varargin)==0 || isnumeric(varargin{1})
  % inputParser should handle this, but it doesn't
  ip.parse(gp, x, y, varargin{:});
else
  ip.parse(gp, x, y, [], varargin{:});
end
xt=ip.Results.xt;
z=ip.Results.z;
zt=ip.Results.zt;
tn = size(x,1);
predcf=ip.Results.predcf;
tstind=ip.Results.tstind;
nsamp=ip.Results.nsamp;
fcorr=ip.Results.fcorr;
splitnormal = ip.Results.splitnormal;
n_scale = min(tn, floor(ip.Results.n_scale));
if isempty(xt)
  xt=x;
  if isempty(tstind)
    if iscell(gp)
      gptype=gp{1}.type;
    else
      gptype=gp.type;
    end
    switch gptype
      case {'FULL' 'VAR' 'DTC' 'SOR'}
        tstind = [];
      case {'FIC' 'CS+FIC'}
        tstind = 1:size(x,1);
      case 'PIC'
        if iscell(gp)
          tstind = gp{1}.tr_index;
        else
          tstind = gp.tr_index;
        end
    end
  end
end

if isfield(gp, 'latent_method')
  if iscell(gp)
      gplatmet=gp{1}.latent_method;
  else
      gplatmet=gp.latent_method;
  end
  if ~strcmp(gplatmet, 'Laplace') && strcmp(splitnormal,'on')
      % splitnormal is applicable only with Laplace
      splitnormal='off';
  end
end


sampyt=[];
if isstruct(gp) && numel(gp.jitterSigma2)==1
  % Single GP
  if isfield(gp, 'monotonic') && gp.monotonic
      [gp,x,y,z,xt,zt] = gp.fh.setUpDataForMonotonic(gp,x,y,z,xt,zt);
  end
  
  if (isfield(gp.lik.fh,'trcov') && ~isfield(gp,'lik_mono')) ...
      || isfield(gp, 'latentValues')
    % ====================================================
    %    Gaussian likelihood or MCMC with latent values
    % ====================================================
    
    if nargout > 1
      [Eft, Covft, ~, Eyt, Covyt] ...
        = gp_jpred(gp,x,y,xt,'z',z,'predcf',predcf,'tstind',tstind);
    else
      [Eft, Covft] = gp_jpred(gp,x,y,xt,'z',z, ...
                              'predcf',predcf,'tstind',tstind);
    end
    rr = randn(size(Eft,1),nsamp);
    sampft = bsxfun(@plus, Eft, chol(Covft,'lower')*rr);
    if nargout > 1
      sampyt = bsxfun(@plus, Eyt, chol(Covyt,'lower')*rr);
    end
    
  else
    % =============================
    %    non-Gaussian likelihood
    % =============================
    
    if nargout > 1
      warning('Sampling of yt is not implemented for non-Gaussian likelihoods');
    end
    if strcmp(splitnormal,'on') && isfield(gp,'meanf')
        warning('Split-Normal is not implemented for mean functions');
        splitnormal='off';
    end
    if strcmp(splitnormal,'on') && isfield(gp.lik, 'nondiagW')
        warning('Split-Normal is not implemented for likelihoods with nondiagonal Hessian')
        splitnormal='off';
    end
    if strcmp(splitnormal,'on') && ~isequal(fcorr, 'off')
        warning('Split-Normal is not used when latent marginal posterior corrections are used')
        splitnormal='off';
    end
    
    if strcmp(splitnormal,'on')
      % ------------------
      %    Splitnormal on
      % ------------------
      
      switch gp.type
        
        % ---------------------------
        case 'FULL'
          
          [e,~,~, param] = gpla_e(gp_pak(gp), gp, x, y, 'z', z);
          if isnan(e)
            error('Laplace-algorithm returned NaN');
          end
          [Ef, L, W] = deal(param.f, param.L, param.La2);

          % Evaluate the covariance
          K = gp_trcov(gp,x);
          K_ss = gp_trcov(gp,xt,predcf);
          K_nf = gp_cov(gp,xt,x,predcf);
          if W >= 0
            % Likelihood is log concave
            if issparse(K) && issparse(L)
              if issparse(W)
                sqrtWK = sqrt(W)*K;
              else
                sqrtWK = sparse(1:tn,1:tn,sqrt(W),tn,tn)*K;
              end
              Covf = K - sqrtWK'*ldlsolve(L,sqrtWK);
            else
              V = linsolve(L,bsxfun(@times,sqrt(W),K),struct('LT',true));
              Covf = K - V'*V;
            end
          else
            % Likelihood is not log concave
            V = bsxfun(@times,L,W');
            Covf = K - K*(diag(W) - V'*V)*K;
          end
          
        % ---------------------------
        case 'FIC'
          
          if ~isempty(tstind) && length(tstind) ~= tn
            error('tstind (if provided) has to be of same length as x.')
          end

          % Ensure the direction of tstind
          tstind = tstind(:);

          [e,~,~, param] = gpla_e(gp_pak(gp), gp, x, y, 'z', z);
          if isnan(e)
            error('Laplace-algorithm returned NaN');
          end
          Ef = param.f;

          u = gp.X_u;
          K_uu = gp_trcov(gp,u,predcf);
          K_uu = (K_uu+K_uu')./2;          % ensure the symmetry
          Luu = chol(K_uu, 'lower');

          B = Luu\gp_cov(gp,u,x,predcf);
          B2 = Luu\gp_cov(gp,u,xt,predcf);
          K = B'*B;
          K_ss = B2'*B2;
          K_nf = B2'*B;
          clear K_uu Luu B B2

          % If the prediction is made for training set, evaluate Lav
          % also for prediction points.
          if ~isempty(tstind)
            kss = gp_trvar(gp,xt,predcf);
            % Replace diagonals
            K(1:length(K)+1:numel(K)) = kss(tstind);
            K_ss(1:length(K_ss)+1:numel(K_ss)) = kss;
            % Replace common samples in K_nf
            K_nf(tstind+(0:size(xt,1):(tn-1)*size(xt,1))') = kss(tstind);
            clear kss
          else
            % Add lambda
            % Replace diagonals
            K(1:length(K)+1:numel(K)) = gp_trvar(gp,x,predcf);
            K_ss(1:length(K_ss)+1:numel(K_ss)) = gp_trvar(gp,xt,predcf);
          end

          Wsqrt = -gp.lik.fh.llg2(gp.lik, y, Ef, 'latent', z);
          if any(Wsqrt < 0)
            error('FIC not implemented for non-log-concave likelihoods')
          end
          Wsqrt = sqrt(Wsqrt);
          L = chol(eye(size(K)) + bsxfun(@times,bsxfun(@times,Wsqrt,K),Wsqrt'), 'lower');

          % Evaluate the covariance for the posterior of f
          V = linsolve(L,bsxfun(@times,Wsqrt,K),struct('LT',true));
          Covf = K - V'*V;
          
        % ---------------------------
        case {'PIC' 'PIC_BLOCK'}
              
          u = gp.X_u;
          K_fu = gp_cov(gp,x,u,predcf);         % f x u
          K_uu = gp_trcov(gp,u,predcf);         % u x u, noiseles covariance K_uu
          K_uu = (K_uu+K_uu')./2;               % ensure the symmetry of K_uu

          ind = gp.tr_index;
          Luu = chol(K_uu)';
          B=Luu\(K_fu');
          B2 = Luu\(gp_cov(gp,u,xt,predcf));
          clear Luu

          [e,~,~, p] = gpla_e(gp_pak(gp), gp, x, y, 'z', z);
          if isnan(e)
            error('Laplace-algorithm returned NaN');
          end
          [Ef, La2] = deal(p.f, p.La2);

          % Evaluate the variance
          sqrtW = -gp.lik.fh.llg2(gp.lik, y, Ef, 'latent', z);
          if any(sqrtW < 0)
            error('PIC not implemented for non-log-concave likelihoods')
          end
          sqrtW = sqrt(sqrtW);
          % Components for (I + W^(1/2)*(Qff + La2)*W^(1/2))^(-1) = Lahat^(-1) - L2*L2'
          Lahat = cell(length(ind),1);
          for i=1:length(ind)
            Lahat{i} = eye(length(ind{i})) + bsxfun(@times,bsxfun(@times,sqrtW(ind{i}),La2{i}),sqrtW(ind{i})');
          end
          sKfu = bsxfun(@times, sqrtW, K_fu);
          iLasKfu = zeros(size(K_fu));
          for i=1:length(ind)
            iLasKfu(ind{i},:) = Lahat{i}\sKfu(ind{i},:);
          end
          A2 = K_uu + sKfu'*iLasKfu; A2=(A2+A2')./2;
          L2 = iLasKfu/chol(A2);

          % NOTE!
          % This is done with full matrices at the moment.
          % Needs to be rewritten.
          K_ss = B2'*B2;
          K_nf = B2'*B;
          K = B'*B;
          C = -L2*L2';
          for i=1:length(ind)
            % Originally 
            % >> La = gp_trcov(gp, xt(tstind{i},:), predcf) - B2(:,tstind{i})'*B2(:,tstind{i});
            % >> K_ss(ind{i},ind{i}) =  K_ss(ind{i},ind{i}) + La;
            % changed into (works if xt is not the same as x)
            % >> La = gp_trcov(gp, xt(tstind{i},:), predcf) - B2(:,tstind{i})'*B2(:,tstind{i});
            % >> K_ss(tstind{i},tstind{i}) =  K_ss(tstind{i},tstind{i}) + La;
            % which is implemented in the line bellow
            K_ss(tstind{i},tstind{i}) =  gp_trcov(gp, xt(tstind{i},:), predcf);
            K_nf(tstind{i},ind{i}) = gp_cov(gp, xt(tstind{i},:), x(ind{i},:),predcf);
            K(ind{i},ind{i}) = K(ind{i},ind{i}) + La2{i};
            C(ind{i},ind{i}) =  C(ind{i},ind{i}) + inv(Lahat{i});
          end

          Covf = K - K * bsxfun(@times,bsxfun(@times,sqrtW,C),sqrtW') * K;
          
        % ---------------------------
        case 'CS+FIC'
          
          if ~isempty(tstind) && length(tstind) ~= tn
            error('tstind (if provided) has to be of same length as x.')
          end

          % Ensure the direction of tstind
          tstind = tstind(:);

          % Indexes to all non-compact support and compact support covariances.
          cscf = cellfun(@(x) isfield(x,'cs'), gp.cf);
          predcf1 = find(~cscf);
          predcf2 = find(cscf);
          if ~isempty(predcf)
            predcf1 = intersect(predcf1,predcf);
            predcf2 = intersect(predcf2,predcf);
          end

          % Laplace approximation
          [e,~,~, param] = gpla_e(gp_pak(gp), gp, x, y, 'z', z);
          if isnan(e)
            error('Laplace-algorithm returned NaN');
          end
          Ef = param.f;

          u = gp.X_u;

          K_uu = gp_trcov(gp,u,predcf1);
          K_uu = (K_uu+K_uu')./2;
          Luu = chol(K_uu)';

          B=Luu\(gp_cov(gp,u,x,predcf1));
          B2=Luu\(gp_cov(gp,u,xt,predcf1));
          K = B'*B;
          K_ss = B2'*B2;
          K_nf = B2'*B;
          clear K_uu Luu B B2

          if ~isempty(tstind)
            % Add Lambda
            kss = gp_trvar(gp,xt,predcf1);
            % Replace diagonals
            K(1:length(K)+1:numel(K)) = kss(tstind);
            K_ss(1:length(K_ss)+1:numel(K_ss)) = kss;
            % Replace common samples in K_nf
            K_nf(tstind+(0:size(xt,1):(tn-1)*size(xt,1))') = kss(tstind);

            % Add CS covariance
            kss = gp_trcov(gp, xt, predcf2);
            K = K + kss(tstind,tstind);
            K_ss = K_ss + kss;
            clear kss

          else
            % Add lambda
            % Replace diagonals
            K(1:length(K)+1:numel(K)) = gp_trvar(gp,x,predcf1);
            K_ss(1:length(K_ss)+1:numel(K_ss)) = gp_trvar(gp,xt,predcf1);

            % Add CS covariance
            K = K + gp_trcov(gp, x, predcf2);
            K_ss = K_ss + gp_trcov(gp, xt, predcf2);

          end

          % Add CS covariance for K_nf
          K_nf = K_nf + gp_cov(gp, xt, x, predcf2);

          sqrtW = -gp.lik.fh.llg2(gp.lik, y, Ef, 'latent', z);
          if any(sqrtW < 0)
            error('CS+FIC not implemented for non-log-concave likelihoods')
          end
          sqrtW = sqrt(sqrtW);
          L = chol(eye(size(K)) + bsxfun(@times,bsxfun(@times,sqrtW,K),sqrtW'), 'lower');
          V = linsolve(L,bsxfun(@times,sqrtW,K),struct('LT',true));
          Covf = K - V'*V;
          
        otherwise
          error('Split-Normal not implemented for %s', gp.type)
          
      end
      
      % Scaling n_scale principal component directions
      % Split-normal approximation: Geweke (1989)

      % N.B. svd of sparse matrix is not generally sparse (unless Covf is
      % block diagonal or can be made such by reordering the dimensions).
      % Scaling using the Cholesky factorisation could possibly be done
      % sparsely but the directions would be different (not split-Normal).
      % Thus full matrices has to be used.
      [V, D, ~] = svd(full(Covf));
      T = real(V) * sqrt(real(D)); % Ensuring the real

      L = chol(K,'lower');

      % Optimise the skewness
      delta = [-6:0.5:-0.5, 0.5:0.5:6];
      D = bsxfun(@plus, kron(delta,T(:,1:n_scale)), Ef);
      ll = zeros(1,n_scale*length(delta)); % Here looping is faster than cellfun
      for i = 1:n_scale*length(delta)
        ll(i) = 2*gp.lik.fh.ll(gp.lik,y,D(:,i),z);
      end
      ll = ll - sum(D.*(L'\(L\D)));
      ll0 = 2*gp.lik.fh.ll(gp.lik, y, Ef, z) - Ef'*(L'\(L\Ef));
      ll(ll >= ll0) = NaN;
      f = bsxfun(@rdivide, abs(delta), reshape(sqrt(ll0-ll),n_scale,length(delta)));
      clear D V ll ll0
      q = max(f(:,delta>0),[],2);
      r = max(f(:,delta<0),[],2);
      q(isnan(q)) = 1;
      r(isnan(r)) = 1;
      % Safety limits [1/10, 10]
      q = min(max(q,0.1),10);
      r = min(max(r,0.1),10);
      if n_scale < tn
        q = [q;ones(tn-n_scale,1)];
        r = [r;ones(tn-n_scale,1)];
      end
      
      % Draw samples from split-Normal approximated posterior of f and
      % sample from the conditional distribution of ft for each sample of f
      u = rand(tn,nsamp);
      c = r./(r+q);
      Covft = K_ss - K_nf*(L'\(L\K_nf'));
      Covft_ind = find(diag(Covft)>gp.jitterSigma2);
      if length(Covft_ind) == size(xt,1)
        % The following evaluates samples of ft directly by
        % Eft + chol(Covft)*randn,
        % where Eft = K_nf*K\f and Covft = K_ss - K_nf*K\K_nf'
        sampft = K_nf*(L'\(L\ ...
          bsxfun(@plus,Ef,T*((bsxfun(@times,q,double(bsxfun(@ge,u,c))) ...
          +bsxfun(@times,-r,double(bsxfun(@lt,u,c)))).*abs(randn(tn,nsamp)))) ...
          )) + chol(Covft,'lower')*randn(size(xt,1),nsamp);
      else
        % Zero variance dimensions in p(ft|X,xt,f)
        % Evaluate first Eft for each f
        sampft = K_nf*(L'\(L\ ...
            bsxfun(@plus,Ef,T*((bsxfun(@times,q,double(bsxfun(@ge,u,c))) ...
            +bsxfun(@times,-r,double(bsxfun(@lt,u,c)))).*abs(randn(tn,nsamp)))) ));
        % Add variability only from non-zero variance dimensions
        sampft(Covft_ind,:) = sampft(Covft_ind,:) ...
          + chol(Covft(Covft_ind,Covft_ind),'lower')*randn(length(Covft_ind),nsamp);
      end
      
    else
      % -------------------
      %    split-Normal off
      % -------------------
      [Eft, Covft] = gp_jpred(gp,x,y,xt,'z',z, ...
        'predcf',predcf, 'tstind',tstind, 'fcorr','off');
      sampft = bsxfun(@plus, Eft, chol(Covft,'lower')*randn(size(xt,1),nsamp));
      
      if ~isequal(fcorr, 'off')
        % Do marginal corrections for samples
        [pc_predm, fvecm] = gp_predcm(gp, x, y, xt, 'z', z, 'ind', 1:size(xt,1), 'fcorr', fcorr);
        for i=1:size(xt,1)
          % Remove NaNs and zeros
          pc_pred=pc_predm(:,i);
          dii=isnan(pc_pred)|pc_pred==0;
          pc_pred(dii)=[];
          fvec=fvecm(:,i);
          fvec(dii)=[];
          % compute cdf
          cumsumpc = cumsum(pc_pred)/sum(pc_pred);
          % Remove non-unique values from grid vector & distribution
          [cumsumpc, inds] = unique(cumsumpc);
          fvec = fvec(inds);
          % use inverse cdf to make marginal transformation
          fsnc = normcdf(sampft(i,:), Eft(i,1), sqrt(diag(Covft(i,i))));
          sampft(i,:) = interp1(cumsumpc,fvec,fsnc);
        end
      end
    end
      
    
    
  end

elseif isstruct(gp) && numel(gp.jitterSigma2)>1
  % MCMC
  nmc=size(gp.jitterSigma2,1);
  % resample nsamp cases from nmc samples
  % deterministic resampling has low variance and small bias for
  % equal weights
  gi=resampdet(ones(nmc,1),nsamp,1);
  sampft = zeros(size(xt,1),nsamp);
  if nargout>=2
    sampyt = zeros(size(xt,1),nsamp);
  end
  for i1=1:nmc
    sampind = find(gi==i1);
    nsampi = length(sampind);
    if nsampi>0
      Gp = take_nth(gp,i1);
      if isfield(Gp,'latent_method') && isequal(Gp.latent_method,'MCMC')
        Gp = rmfield(Gp,'latent_method');
      end
      if isfield(gp, 'latentValues') && ~isempty(gp.latentValues)
        % Non-Gaussian likelihood. The latent variables should be used in
        % place of observations
        for ni=1:nsampi
          if ni==1
            % use stored latent values
            f = gp.latentValues(i1,:);
          else
            % need to resample latent values
            opt=scaled_mh();
            f=scaled_mh(f, opt, Gp, x, y, z);
          end
          if nargout<2
            sampft(:,sampind(ni)) = gp_rnd(Gp, x, f', xt, 'nsamp', 1, ...
                             'z', z, 'zt', zt, 'predcf', predcf, ...
                             'tstind', tstind);
          else
            [tsampft, tsampyt] = gp_rnd(Gp, x, f', xt, 'nsamp', ...
                                        1, 'z', z, 'zt', zt, ...
                                        'predcf', predcf, 'tstind', tstind);
            
            sampft(:,sampind(ni)) = tsampft;
            sampyt(:,sampind(ni)) = tsampyt;
          end

          
        end
      else         
        % Gaussian likelihood
        if nargout<2
          sampft(:,sampind) = gp_rnd(Gp, x, y, xt, 'nsamp', nsampi, ...
                           'z', z, 'zt', zt, 'predcf', predcf, ...
                           'tstind', tstind);
        else
          [tsampft, tsampyt] = gp_rnd(Gp, x, y, xt, 'nsamp', ...
                                      nsampi, 'z', z, 'zt', zt, ...
                                      'predcf', predcf, 'tstind', tstind);
          sampft(:,sampind) = tsampft;
          sampyt(:,sampind) = tsampyt;
        end
      end
    end
  end
elseif iscell(gp)
  % gp_ia
  ngp=length(gp);
  if isfield(gp{1},'ia_weight')
    gw = zeros(ngp,1);
    for i1=1:ngp
      gw(i1)=gp{i1}.ia_weight;
    end
  else
    gw=ones(ngp,1);
  end
  % resample nsamp cases from nmc samples
  % stratified resampling has has higher variance than deterministic
  % resampling, but has a smaller bias, and thus it should be more
  % suitable for unequal weights
  gi=resampstr(gw,nsamp,1);
  sampft = zeros(size(xt,1),nsamp);
  if nargout>=2
    sampyt = zeros(size(xt,1),nsamp);
  end
  for i1 = 1:ngp
    nsampi=sum(gi==i1);
    if nsampi>0
      if nargout<2
        sampft(:,gi==i1) = gp_rnd(gp{i1}, x, y, xt, 'nsamp', nsampi, ...
                         'z', z, 'zt', zt, 'predcf', predcf, ...
                         'tstind', tstind, 'fcorr', fcorr);
      else
        [tsampft, tsampyt] = gp_rnd(gp{i1}, x, y, xt, 'nsamp', nsampi, ...
                                    'z', z, 'zt', zt, 'predcf', predcf, ...
                                    'tstind', tstind, 'fcorr', fcorr);
        sampft(:,gi==i1) = tsampft;
        sampyt(:,gi==i1) = tsampyt;
      end
    end
  end
end
