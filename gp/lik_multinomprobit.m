function lik = lik_multinomprobit(varargin)
%LIK_MULTINOMPROBIT  Create a multinomial probit likelihood structure 
%
%  Description
%    LIK = LIK_MULTINOMPROBIT creates the multinomial probit
%    likelihood for multi-class classification problem. The observed
%    class label with C classes is given as 1xC vector where C-1
%    entries are 0 and the observed class label is 1.
%
%    Note that when using LIK_MULTINOMPROBIT, gp_predy returns Ey=[]
%    and Vary=[], as we think that they are not worth computing,
%    because they do not describe the predictive distribution well.
%
%  See also
%    GP_SET, LIK_*
%
% Copyright (c) 2011-2013 Jaakko Riihimäki, Pasi Jylänki, Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_MULTINOMPROBIT';
  ip.addOptional('lik', [], @isstruct);
  ip.parse(varargin{:});
  lik=ip.Results.lik;

  if isempty(lik)
    init=true;
    lik.type = 'Multinomprobit';
    lik.nondiagW = true;
  else
    if ~isfield(lik,'type') && ~isequal(lik.type,'Multinomprobit')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end
  
  if init
    % Set the function handles to the nested functions
    lik.fh.pak = @lik_multinomprobit_pak;
    lik.fh.unpak = @lik_multinomprobit_unpak;
    lik.fh.ll = @lik_multinomprobit_ll;
    lik.fh.llg = @lik_multinomprobit_llg;    
    lik.fh.llg2 = @lik_multinomprobit_llg2;
    lik.fh.llg3 = @lik_multinomprobit_llg3;
    lik.fh.tiltedMoments = @lik_multinomprobit_tiltedMoments;
    lik.fh.predy = @lik_multinomprobit_predy;
    lik.fh.recappend = @lik_multinomprobit_recappend;
  end
  

  function [w,s,h] = lik_multinomprobit_pak(lik)
  %LIK_MULTINOMPROBIT_PAK  Combine likelihood parameters into one vector.
  %
  %  Description 
  %    W = LIK_MULTINOMPROBIT_PAK(LIK) takes a likelihood structure LIK and
  %    returns an empty verctor W. If multinomprobit likelihood had
  %    parameters this would combine them into a single row vector
  %    W (see e.g. lik_negbin).
  %     
  %
  %  See also
  %    LIK_MULTINOMPROBIT_UNPAK, GP_PAK
    
    w = []; s = {}; h=[];
  end


  function [lik, w] = lik_multinomprobit_unpak(lik, w)
  %LIK_MULTINOMPROBIT_UNPAK  Extract likelihood parameters from the vector.
  %
  %  Description
  %    W = LIK_MULTINOMPROBIT_UNPAK(W, LIK) Doesn't do anything.
  % 
  %    If MULTINOMPROBIT likelihood had parameters this would extracts them
  %    parameters from the vector W to the LIK structure.
  %     
  %
  %  See also
  %    LIK_MULTINOMPROBIT_PAK, GP_UNPAK

    lik=lik;
    w=w;
  end


  function ll = lik_multinomprobit_ll(lik, y, f2, z)
  %LIK_MULTINOMPROBIT_LL  Log likelihood
  %
  %  Description
  %    LL = LIK_MULTINOMPROBIT_LL(LIK, Y, F) takes a likelihood structure
  %    LIK, class labels Y (NxC matrix), and latent values F (NxC
  %    matrix). Returns the log likelihood, log p(y|f,z).
  %
  %  See also
  %    LIK_MULTINOMPROBIT_LLG, LIK_MULTINOMPROBIT_LLG3, LIK_MULTINOMPROBIT_LLG2, GPLA_E
    
    if ~isempty(find(y~=1 & y~=0))
      error('lik_multinomprobit: The class labels have to be {0,1}')
    end
    
    [n,nout]=size(y);
    minf=-6;
    maxf=6;

    ind=1:nout;
    ll=0;
    
    for i1=1:n
      indr=ind(y(i1,:)==0);
      indi=ind(y(i1,:)==1);
      if nout==3
        integrand = @(u) normpdf(u).*normcdf(u+f2(i1,indi)-f2(i1,indr(1))).*normcdf(u+f2(i1,indi)-f2(i1,indr(2)));
        ll =ll + log(quadgk(integrand,minf,maxf));
      else
        error('lik_multinomprobit: Log likelihood not yet implemented')
      end
    end
    
  end

  function llg = lik_multinomprobit_llg(lik, y, f2, param, z)
  %LIK_MULTINOMPROBIT_LLG  Log likelihood
  %
  %  Description
  %    LL = LIK_MULTINOMPROBIT_LL(LIK, Y, F, PARAM) takes a likelihood structure
  %    LIK, class labels Y (NxC matrix), and latent values F (NxC
  %    matrix). Returns the gradient of the log likelihood
  %    with respect to PARAM. At the moment PARAM can be 'param' or
  %    'latent'. This subfunction is needed when using Laplace
  %    approximation or MCMC for inference with non-Gaussian
  %    likelihoods.
  %
  %  See also
  %    LIK_MULTINOMPROBIT_LLG, LIK_MULTINOMPROBIT_LLG3, LIK_MULTINOMPROBIT_LLG2, GPLA_E
  
    error('llg not implememented for multinomial probit likelihood');
  end

  function llg2 = lik_multinomprobit_llg2(lik, y, f2, param, z)
  %LIK_MULTINOMPROBIT_LLG2  Log likelihood
  %
  %  Description
  %    LL = LIK_MULTINOMPROBIT_LL(LIK, Y, F) takes a likelihood structure
  %    LIK, class labels Y (NxC matrix), and latent values F (NxC
  %    matrix). Returns the Hessian of the log
  %    likelihood with respect to PARAM. At the moment PARAM can be
  %    only 'latent'. G2 is a vector with diagonal elements of the
  %    Hessian matrix (off diagonals are zero). This subfunction
  %    is needed when using Laplace approximation or EP for inference
  %    with non-Gaussian likelihoods.
  %
  %  See also
  %    LIK_MULTINOMPROBIT_LLG, LIK_MULTINOMPROBIT_LLG3, LIK_MULTINOMPROBIT_LLG2, GPLA_E
  
    error('llg2 not implememented for multinomial probit likelihood');
  end

  function llg3 = lik_multinomprobit_llg3(lik, y, f2, param, z)
  %LIK_MULTINOMPROBIT_LLG3  Log likelihood
  %
  %  Description
  %    LL = LIK_MULTINOMPROBIT_LL(LIK, Y, F) takes a likelihood structure
  %    LIK, class labels Y (NxC matrix), and latent values F (NxC
  %    matrix). Returns the third gradients of the
  %    log likelihood with respect to PARAM. At the moment PARAM
  %    can be only 'latent'. G3 is a vector with third gradients.
  %    This subfunction is needed when using Laplace appoximation
  %    for inference with non-Gaussian likelihoods.
  %
  %  See also
  %    LIK_MULTINOMPROBIT_LLG, LIK_MULTINOMPROBIT_LLG3, LIK_MULTINOMPROBIT_LLG2, GPLA_E
  
    error('llg3 not implememented for multinomial probit likelihood');
  end

  function [alphatilde, betatilde, m_1, sigm2hati1, lm_0] = lik_multinomprobit_tiltedMoments(lik, y, i1, sigm2_i, mu_i, z, alphatildeold, betatildeold)

    % Approximating tilted moments for fully-coupled nested EP with a
    % multinomial probit likelihood as described in Appendix A of the
    % paper:
    %
    %   Jaakko Riihimäki, Pasi Jylänki and Aki Vehtari (2013). Nested
    %   Expectation Propagation for Gaussian Process Classification with a
    %   Multinomial Probit Likelihood. Journal of Machine Learning Research
    %   14:75-109, 2013.
    %
    
    % tolerance for convergence
    tol=1e-6;

    % number of output classes
    nout=size(y,2);

    % observed class label
    ci=find(y(i1,:)==1);
    % other class labels excluding the observed class label
    cni=find(y(i1,:)==0);
    
    % w=[u f_1 f_2 ... f_C]^T
    Btilde=zeros(nout-1,nout+1);
    Btilde(:,1)=1;
    Btilde(:,ci+1)=1;
    for k1=1:(nout-1)
      Btilde(k1,cni(k1)+1)=-1;
    end
    % Note that in the notation of the paper (Riihimäki et al., 2013) 
    % matrix Btilde is transposed and the vector w is defined 
    % w=[f_1 f_2 ... f_C u]^T, where the auxiliary variable u is the
    % last argument in the vector w (instead of being the first argument as
    % in this implementation).
    
    % site parameters
    if nargin > 6
      % incremental update (only one inner-loop iteration per site)
      max_ep_iter=1;
      alphatilde=alphatildeold;
      betatilde=betatildeold;
      mw0=[0; mu_i];
      Vw0=[1 zeros(1,nout); zeros(nout,1) sigm2_i];

      % prior cholesky
      Lprior=chol(Vw0,'lower');
      iLmprior=Lprior\eye(nout+1);

      Lpost=chol(Btilde'*diag(alphatildeold)*Btilde+(iLmprior'*iLmprior),'lower');
      iLpost=Lpost\eye(nout+1);
      Vw=(iLpost'*iLpost);
      mw=Vw*(Btilde'*betatildeold+Lprior'\(Lprior\mw0));
    else
      % standard update (inner-loops are run until convergence)
      max_ep_iter=20;
      alphatilde=zeros(nout-1,1);
      betatilde=zeros(nout-1,1);
      mw=[0; mu_i];
      Vw=[1 zeros(1,nout); zeros(nout,1) sigm2_i];
      % priors
      mw0=mw;
      Vw0=Vw;
    end

    Zw=zeros(nout-1,1);

    % vectors for cavity parameters
    mcvec=zeros(nout-1,1);
    vcvec=zeros(nout-1,1);

    % vectors for tilted moment parameters
    mhatvec=zeros(nout-1,1);
    vhatvec=zeros(nout-1,1);

    betatilde0=Inf; alphatilde0=Inf;
    ep_iter=1;

    while ep_iter <= max_ep_iter && (sum(abs(betatilde0-betatilde)>tol) || sum(abs(alphatilde0-alphatilde)>tol))

      betatilde0=betatilde; alphatilde0=alphatilde;

      for k1=1:(nout-1)

        btilde = Btilde(k1,:)';

        % 1. Cavity evaluations:
        vi=btilde'*Vw*btilde;
        mi=btilde'*mw;

        vc=1./(1./vi-alphatilde(k1));
        mc=vc*(mi./vi-betatilde(k1));

        % store cavity parameters
        vcvec(k1)=vc;
        mcvec(k1)=mc;

        
        % 2. Tilted moments:
        sqvc=sqrt(1+vc);
        zi=mc./sqvc;
        %normcdfzw=normcdf(zi);
        normcdfzw=0.5*erfc(-zi./sqrt(2));
        %normpdfzw=normpdf(zi);
        normpdfzw=exp(-0.5*zi.^2)./sqrt(2*pi);

        rhoi=normpdfzw./normcdfzw./sqvc;
        bi=rhoi.^2+zi.*rhoi./sqvc;

        Zw(k1) = normcdfzw;
        mhat=rhoi*vc+mc;
        vhat=vc-vc.^2*bi;

        % store tilted moment parameters
        mhatvec(k1)=mhat;
        vhatvec(k1)=vhat;


        % 3. Site updates (without damping):
        dalphatilde=1./vhat-1./vi;
        dbetatilde=mhat./vhat-mi./vi;

        alphatilde(k1)=alphatilde(k1)+dalphatilde;
        betatilde(k1)=betatilde(k1)+dbetatilde;


        % 4. Sequential rank-1 covariance update:
        ui=Vw*btilde;
        Vw = Vw - ui.*dalphatilde./(1+vi.*dalphatilde)*ui';
        mw = mw + ui./(1+dalphatilde*vi)*(dbetatilde-dalphatilde*mi);
        
      end
      ep_iter=ep_iter+1;
    end
    
    muhati=mw(2:end);
    sigm2hati=Vw(2:end,2:end);

    % posterior marginals
    mivec=Btilde*mw;
    vivec=sum(Btilde'.*(Vw*Btilde'))';

    % posterior cholesky
    Lpost=chol(Vw,'lower');
    iLm=Lpost\mw;

    % prior cholesky
    Lprior=chol(Vw0,'lower');
    iLmprior=Lprior\mw0;
    
    % posterior mean
    m_1=muhati;
    % posterior covariance
    sigm2hati1=sigm2hati;
    
    if nargout == 5
      % normalization of the tilted distribution
      lm_0 = 0.5*(iLm'*iLm) + sum(log(diag(Lpost))) - 0.5*(iLmprior'*iLmprior) - sum(log(diag(Lprior))) + ...
        sum(log(Zw)) + ...
        sum( 0.5*mcvec.^2./vcvec + 0.5*log(vcvec) - 0.5*mivec.^2./vivec - 0.5*log(vivec) );
    end
    
  end
  
  function [lpy, Ey, Vary] = lik_multinomprobit_predy(lik, Ef, Varf, yt, zt)
  %LIK_MULTINOMPROBIT_PREDY  Returns the predictive mean, variance and density of
  %y
  %
  %  Description         
  %    LPY = LIK_MULTINOMPROBIT_PREDY(LIK, EF, VARF YT, ZT)
  %    Returns the predictive density of YT, that is
  %        p(yt | y, zt) = \int p(yt | f, zt) p(f|y) df.
  %    This requires also the succes counts YT, numbers of trials ZT.
  %
  %    [LPY, EY, VARY] = LIK_MULTINOMPROBIT_PREDY(LIK, EF, VARF) takes a
  %    likelihood structure LIK, posterior mean EF and posterior
  %    Variance VARF of the latent variable and returns the
  %    posterior predictive mean EY and variance VARY of the
  %    observations related to the latent variables
  %        
  %  See also 
  %    GPEP_PRED, GPLA_PRED, GPMC_PRED
  
  
  %if ~isempty(find(yt~=1 & yt~=0))
  %  error('lik_multinomprobit: The class labels have to be {0,1}')
  %end
  
  ntest=size(Varf,3);
  nout=size(Varf,1);
  
  M0=zeros(ntest,nout);
  for i1=1:ntest
    sigm2_i=Varf(:,:,i1);
    mu_i=Ef(i1:ntest:end);
    % approximate the predictive class probabilities
    for k1=1:nout
      ytmp=zeros(1,nout); ytmp(k1)=1;
      [tmp,tmp,tmp,tmp,M0(i1,k1)] = lik.fh.tiltedMoments(lik, ytmp, 1, sigm2_i, mu_i);
    end
  end
  M0=exp(M0);
  lpy=log(bsxfun(@rdivide,M0,sum(M0,2)));  
  % Don't compute Ey and Vary as they don't describe the predictive distribution well
  Ey=[];
  Vary=[];
  
  end

  function reclik = lik_multinomprobit_recappend(reclik, ri, lik)
  %RECAPPEND  Append the parameters to the record
  %
  %  Description 
  %    RECLIK = LIK_MULTINOMPROBIT_RECAPPEND(RECLIK, RI, LIK) takes a
  %    likelihood record structure RECLIK, record index RI and
  %    likelihood structure LIK with the current MCMC samples of
  %    the parameters. Returns RECLIK which contains all the old
  %    samples and the current samples from LIK.
  % 
  %  See also
  %    GP_MC

    if nargin == 2
      reclik.type = 'Multinomprobit';
      reclik.nondiagW = true;

      % Set the function handles
      reclik.fh.pak = @lik_multinomprobit_pak;
      reclik.fh.unpak = @lik_multinomprobit_unpak;
      reclik.fh.ll = @lik_multinomprobit_ll;
      reclik.fh.llg = @lik_multinomprobit_llg;
      reclik.fh.llg2 = @lik_multinomprobit_llg2;
      reclik.fh.llg3 = @lik_multinomprobit_llg3;
      reclik.fh.tiltedMoments = @lik_multinomprobit_tiltedMoments;
      reclik.fh.predy = @lik_multinomprobit_predy;
      reclik.fh.recappend = @lik_multinomprobit_recappend;
      return
    end
    
  end
end
