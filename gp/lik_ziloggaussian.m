function lik = lik_ziloggaussian(varargin)
%LIK_ZILOGGAUSSIAN    Create a zero-inflated log-Gaussian likelihood structure 
%
%  Description
%    LIK = LIK_ZILOGGAUSSIAN('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates a zero-inflated log-Gaussian likelihood structure in
%    which the named parameters have the specified values. Any unspecified
%    parameters are set to default values.  
%  
%    LIK = LIK_ZILOGGAUSSIAN(LIK,'PARAM1',VALUE1,'PARAM2,VALUE2,...)
%    modify a likelihood structure with the named parameters
%    altered with the specified values.
%
%    Parameters for a zero-inflated log-Gaussian likelihood [default]
%      sigma2       - sigma2sion parameter r [10]
%      sigma2_prior - prior for sigma2 [prior_logunif]
%  
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%    The likelihood is defined as follows:
%     
%                                      p ,    when y=0
%          (1-p)*log-Gaussian(y|f,sigma2),    when y>0,
%
%      where the probability p is given by a binary classifier with Logit
%      likelihood and log-Gaussian is the log-Gaussian distribution
%      parametrized for the i'th observation as
%                  
%    The latent value vector f=[f1^T f2^T]^T has length 2*N, where N is the
%    number of observations. The latents f1 are associated with the
%    classification process and the latents f2 with log-Gaussian count
%    process.
%
%
%  See also
%    GP_SET, LIK_*, PRIOR_*
%
% Copyright (c) 2007-2010, 2016 Jarno Vanhatalo
% Copyright (c) 2007-2010 Jouni Hartikainen
% Copyright (c) 2010 Aki Vehtari
% Copyright (c) 2011 Jaakko Riihimäki

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_ZILOGGAUSSIAN';
  ip.addOptional('lik', [], @isstruct);
  ip.addParamValue('sigma2',1, @(x) isscalar(x) && x>0);
  ip.addParamValue('sigma2_prior',prior_logunif(), @(x) isstruct(x) || isempty(x));
  ip.parse(varargin{:});
  lik=ip.Results.lik;
  
  if isempty(lik)
    init=true;
    lik.type = 'Ziloggaussian';
    lik.nondiagW=true;
  else
    if ~isfield(lik,'type') || ~isequal(lik.type,'Ziloggaussian')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end

  % Initialize parameters
  if init || ~ismember('sigma2',ip.UsingDefaults)
    lik.sigma2 = ip.Results.sigma2;
  end
  % Initialize prior structure
  if init
    lik.p=[];
  end
  if init || ~ismember('sigma2_prior',ip.UsingDefaults)
    lik.p.sigma2=ip.Results.sigma2_prior;
  end
  
  if init
    % Set the function handles to the nested functions
    lik.fh.pak = @lik_ziloggaussian_pak;
    lik.fh.unpak = @lik_ziloggaussian_unpak;
    lik.fh.lp = @lik_ziloggaussian_lp;
    lik.fh.lpg = @lik_ziloggaussian_lpg;
    lik.fh.ll = @lik_ziloggaussian_ll;
    lik.fh.llg = @lik_ziloggaussian_llg;    
    lik.fh.llg2 = @lik_ziloggaussian_llg2;
    lik.fh.llg3 = @lik_ziloggaussian_llg3;
    lik.fh.predy = @lik_ziloggaussian_predy;
    lik.fh.invlink = @lik_ziloggaussian_invlink;
    lik.fh.recappend = @lik_ziloggaussian_recappend;
  end

  function [w,s,h] = lik_ziloggaussian_pak(lik)
  %LIK_ZILOGGAUSSIAN_PAK  Combine likelihood parameters into one vector.
  %
  %  Description 
  %    W = LIK_ZILOGGAUSSIAN_PAK(LIK) takes a likelihood structure LIK and
  %    combines the parameters into a single row vector W. This is a 
  %    mandatory subfunction used for example in energy and gradient 
  %    computations.
  %     
  %       w = log(lik.sigma2)
  %
  %   See also
  %   LIK_ZILOGGAUSSIAN_UNPAK, GP_PAK
    
    w=[];s={};h=[];
    if ~isempty(lik.p.sigma2)
      w = log(lik.sigma2);
      s = [s; 'log(ziloggaussian.sigma2)'];
      h = 0;
      [wh, sh, hh] = feval(lik.p.sigma2.fh.pak, lik.p.sigma2);
      w = [w wh];
      s = [s; sh];
      h = [h hh];
    end
  end


  function [lik, w] = lik_ziloggaussian_unpak(lik, w)
  %LIK_ZILOGGAUSSIAN_UNPAK  Extract likelihood parameters from the vector.
  %
  %  Description
  %    [LIK, W] = LIK_ZILOGGAUSSIAN_UNPAK(W, LIK) takes a likelihood
  %    structure LIK and extracts the parameters from the vector W
  %    to the LIK structure. This is a mandatory subfunction used for 
  %    example in energy and gradient computations.
  %     
  %   Assignment is inverse of  
  %       w = log(lik.sigma2)
  %
  %   See also
  %   LIK_ZILOGGAUSSIAN_PAK, GP_UNPAK

    if ~isempty(lik.p.sigma2)
      lik.sigma2 = exp(w(1));
      w = w(2:end);
      [p, w] = feval(lik.p.sigma2.fh.unpak, lik.p.sigma2, w);
      lik.p.sigma2 = p;
    end
  end


  function lp = lik_ziloggaussian_lp(lik, varargin)
  %LIK_ZILOGGAUSSIAN_LP  log(prior) of the likelihood parameters
  %
  %  Description
  %    LP = LIK_ZILOGGAUSSIAN_LP(LIK) takes a likelihood structure LIK and
  %    returns log(p(th)), where th collects the parameters. This 
  %    subfunction is needed when there are likelihood parameters.
  %
  %  See also
  %    LIK_ZILOGGAUSSIAN_LLG, LIK_ZILOGGAUSSIAN_LLG3, LIK_ZILOGGAUSSIAN_LLG2, GPLA_E
    

  % If prior for sigma2sion parameter, add its contribution
    lp=0;
    if ~isempty(lik.p.sigma2)
      lp = feval(lik.p.sigma2.fh.lp, lik.sigma2, lik.p.sigma2) +log(lik.sigma2);
    end
    
  end

  
  function lpg = lik_ziloggaussian_lpg(lik)
  %LIK_ZILOGGAUSSIAN_LPG  d log(prior)/dth of the likelihood 
  %                parameters th
  %
  %  Description
  %    E = LIK_ZILOGGAUSSIAN_LPG(LIK) takes a likelihood structure LIK and
  %    returns d log(p(th))/dth, where th collects the parameters. This
  %    subfunction is needed when there are likelihood parameters.
  %
  %  See also
  %    LIK_ZILOGGAUSSIAN_LLG, LIK_ZILOGGAUSSIAN_LLG3, LIK_ZILOGGAUSSIAN_LLG2, GPLA_G
    
    lpg=[];
    if ~isempty(lik.p.sigma2)            
      % Evaluate the gprior with respect to sigma2
      ggs = feval(lik.p.sigma2.fh.lpg, lik.sigma2, lik.p.sigma2);
      lpg = ggs(1).*lik.sigma2 + 1;
      if length(ggs) > 1
        lpg = [lpg ggs(2:end)];
      end
    end
  end  
  
  function ll = lik_ziloggaussian_ll(lik, y, ff, ~)
  %LIK_ZILOGGAUSSIAN_LL  Log likelihood
  %
  %  Description
  %    LL = LIK_ZILOGGAUSSIAN_LL(LIK, Y, F, Z) takes a likelihood
  %    structure LIK, outcomes Y and latent values F. Returns the log
  %    likelihood, log p(y|f). This subfunction is needed when using
  %    Laplace approximation or MCMC for inference with non-Gaussian
  %    likelihoods. This subfunction is also used in information criteria
  %    (DIC, WAIC) computations.
  %
  %  See also
  %    LIK_ZILOGGAUSSIAN_LLG, LIK_ZILOGGAUSSIAN_LLG3, LIK_ZILOGGAUSSIAN_LLG2, GPLA_E
        
    f=ff(:);
    n=size(y,1);
    f1=f(1:n);
    f2=f((n+1):2*n);
    y0ind=y==0;
    yind=y>0;
    
    s2 = lik.sigma2;
    expf1=exp(f1);
    
    % for y = 0
    lly0 = sum(  -log(1+expf1(y0ind)) + log( expf1(y0ind) ) );
    % for y > 0
    lly = sum( -log(1+expf1(yind)) +  -1/2*log(2*pi*s2) - log(y(yind)) - 1./(2*s2).*(log(y(yind))-f2(yind)).^2 );
    
    ll=lly0+lly;
  
  end

  function llg = lik_ziloggaussian_llg(lik, y, ff, param, ~)
  %LIK_ZILOGGAUSSIAN_LLG  Gradient of the log likelihood
  %
  %  Description 
  %    LLG = LIK_ZILOGGAUSSIAN_LLG(LIK, Y, F, PARAM) takes a likelihood
  %    structure LIK, outcomes Y and latent values F. Returns the gradient
  %    of the log likelihood with respect to PARAM. At the moment PARAM can
  %    be 'param' or 'latent'. This subfunction is needed when using
  %    Laplace approximation or MCMC for inference with non-Gaussian
  %    likelihoods. 
  %
  %  See also
  %    LIK_ZILOGGAUSSIAN_LL, LIK_ZILOGGAUSSIAN_LLG2, LIK_ZILOGGAUSSIAN_LLG3, GPLA_E

  
    f=ff(:);
    n=size(y,1);
    f1=f(1:n);
    f2=f((n+1):2*n);
    y0ind=y==0;
    yind=y>0;
    
    s2 = lik.sigma2;
    expf1=exp(f1);
    r = log(y(yind))-f2(yind);
    
    switch param
      case 'param'
          llg = sum(-1./(2.*s2) + r.^2./(2.*s2^2));
          % correction for the log transformation
          llg = llg.*s2;
      case 'latent'
        llg1=zeros(n,1);
        llg2=zeros(n,1);
        
        llg1(y0ind)= 1 - expf1(y0ind)./(1+expf1(y0ind));
        llg2(y0ind)= 0;
        
        llg1(yind) = -expf1(yind)./(1+expf1(yind));
        llg2(yind) = 1./s2.*r;
        
        llg=[llg1; llg2];
                
    end
    
  end

  function llg2 = lik_ziloggaussian_llg2(lik, y, ff, param, ~)
  %LIK_ZILOGGAUSSIAN_LLG2  Second gradients of the log likelihood
  %
  %  Description        
  %    LLG2 = LIK_ZILOGGAUSSIAN_LLG2(LIK, Y, F, PARAM) takes a likelihood
  %    structure LIK, outcomes Y and latent values F. Returns the Hessian
  %    of the log likelihood with respect to PARAM. At the moment PARAM can
  %    be only 'latent'. Second gradients form a matrix of size 2N x 2N as
  %    [diag(LLG2_11) diag(LLG2_12); diag(LLG2_12) diag(LLG2_22)],
  %    but the function returns only vectors of diagonal elements as
  %    LLG2 = [LLG2_11 LLG2_12; LLG2_12 LLG2_22] (2Nx2 matrix) since off
  %    diagonals of the blocks are zero. This subfunction is needed when 
  %    using Laplace approximation or EP for inference with non-Gaussian 
  %    likelihoods.
  %
  %  See also
  %    LIK_ZILOGGAUSSIAN_LL, LIK_ZILOGGAUSSIAN_LLG, LIK_ZILOGGAUSSIAN_LLG3, GPLA_E

  f=ff(:);
  
  n=size(y,1);
  f1=f(1:n);
  f2=f((n+1):2*n);
  y0ind=y==0;
  yind=y>0;
  
  s2 = lik.sigma2;
  expf1=exp(f1);
  
  switch param
      case 'param'
          
      case 'latent'
          
          llg2_11=zeros(n,1);
          llg2_12=zeros(n,1);
          llg2_22=zeros(n,1);
          
          llg2_11(y0ind) = -expf1(y0ind)./(1+expf1(y0ind)).^2;
          llg2_11(yind) = -expf1(yind)./(1+expf1(yind)).^2;
          llg2_22(yind) = repmat(-1./s2,sum(yind),1);
          
          llg2 = [llg2_11 llg2_12; llg2_12 llg2_22];
                    
      case 'latent+param'
          
          
          llg2_1=zeros(n,1);
          llg2_2=zeros(n,1);
          
          r = log(y(yind))-f2(yind);
          llg2_2(yind) = -1./s2^2.*r;
         
          llg2=[llg2_1; llg2_2];
          
          % correction due to the log transformation
          llg2 = llg2.*lik.sigma2;
  end
  end

  function llg3 = lik_ziloggaussian_llg3(lik, y, ff, param, ~)
  %LIK_ZILOGGAUSSIAN_LLG3  Third gradients of the log likelihood
  %
  %  Description
  %    LLG3 = LIK_ZILOGGAUSSIAN_LLG3(LIK, Y, F, PARAM) takes a likelihood
  %    structure LIK, outcomes Y and latent values F and returns the third
  %    gradients of the log likelihood with respect to PARAM. At the moment
  %    PARAM can be only 'latent'. LLG3 is a 2-by-2-by-2-by-N array of with
  %    third gradients, where LLG3(:,:,1,i) is the third derivative wrt f1
  %    for the i'th observation and LLG3(:,:,2,i) is the third derivative
  %    wrt f2 for the i'th observation. This subfunction is needed when
  %    using Laplace approximation for inference with non-Gaussian
  %    likelihoods. 
  %
  %  See also
  %    LIK_ZILOGGAUSSIAN_LL, LIK_ZILOGGAUSSIAN_LLG, LIK_ZILOGGAUSSIAN_LLG2, GPLA_E, GPLA_G
  
  f=ff(:);
  
  n=size(y,1);
  f1=f(1:n);
  f2=f((n+1):2*n);
  y0ind=y==0;
  yind=y>0;
  
  s2 = lik.sigma2;
  expf1=exp(f1);
  
  switch param
      case 'param'
          
      case 'latent'
          nl=2;
          llg3=zeros(nl,nl,nl,n);
          
          expf1y0ind=expf1(y0ind);
          expf1y0ind2=expf1y0ind.^2;
          
          % y=0:
          % thrid derivative derivative wrt f1 (11)
          llg3(1,1,1,y0ind) = (2.*expf1y0ind2)./(expf1y0ind + 1).^3 - expf1y0ind./(expf1y0ind + 1).^2 ;
          
          % y>0:
          llg3(1,1,1,yind) = -expf1(yind).*(1-expf1(yind))./(1+expf1(yind)).^3;
          
      case 'latent2+param'
          
          llg3_11=zeros(n,1);
          llg3_12=zeros(n,1);
          llg3_22=zeros(n,1);
          
          llg3_22(yind) = repmat(1./s2^2,sum(yind),1);
          
          llg3 = [diag(llg3_11) diag(llg3_12); diag(llg3_12) diag(llg3_22)];
          
          % correction due to the log transformation
          llg3 = llg3.*s2;
  end
  
  end

  function [lpyt,Ey, Vary] = lik_ziloggaussian_predy(lik, Ef, Covf, yt, ~)
  %LIK_ZILOGGAUSSIAN_PREDY  Returns the predictive mean, variance and density of y
  %
  %  Description         
  %    [EY, VARY] = LIK_ZILOGGAUSSIAN_PREDY(LIK, EF, VARF) takes a
  %    likelihood structure LIK, posterior mean EF and posterior
  %    covariance COVF of the latent variable and returns the
  %    posterior predictive mean EY and variance VARY of the
  %    observations related to the latent variables. This 
  %    subfunction is needed when computing posterior predictive 
  %    distributions for future observations.
  %        
  %    [Ey, Vary, PY] = LIK_ZILOGGAUSSIAN_PREDY(LIK, EF, VARF YT, ZT)
  %    Returns also the predictive density of YT, that is 
  %        p(yt | zt) = \int p(yt | f, zt) p(f|y) df.
  %    This requires also the test outcomes YT. This subfunction is needed
  %    when computing posterior predictive distributions for future
  %    observations. 
  %
  %  See also
  %    GPLA_PRED, GPEP_PRED, GPMC_PRED

    ntest=size(Ef,1)/2;
    
    s2 = lik.sigma2;
    Py = [];    
    S=10000;
    for i1=1:ntest
      Sigm_tmp=Covf(i1:ntest:(2*ntest),i1:ntest:(2*ntest));
      Sigm_tmp=(Sigm_tmp+Sigm_tmp')./2;
      f_star=mvnrnd(Ef(i1:ntest:(2*ntest)), Sigm_tmp, S);
      
      expf1=exp(f_star(:,1));
      
      if yt(i1)==0
        Py(i1)=mean(exp(-log(1+expf1) + log( expf1 )));
      else
        Py(i1)=mean(exp( -log(1+expf1) +  -1/2*log(2*pi*s2) - log(yt(i1)) - 1./(2*s2).*(log(yt(i1))-f_star(:,2)).^2 ));
      end
      Ey(i1) = mean(exp(f_star(:,2)+s2/2 - log(1+expf1)));
      Vary(i1) = mean( exp(2*f_star(:,2)+s2 - log(1+expf1)).*(exp(s2)-1) );
    end
    %Ey = [];
    %Vary = [];
    lpyt=log(Py);
  end

  function p = lik_ziloggaussian_invlink(lik, f, ~)
  %LIK_ZILOGGAUSSIAN_INVLINK  Returns values of inverse link function
  %             
  %  Description 
  %    P = LIK_ZILOGGAUSSIAN_INVLINK(LIK, F) takes a likelihood structure LIK and
  %    latent values F and returns the values of inverse link function P.
  %    This subfunction is needed when using function gp_predprctmu.
  %
  %     See also
  %     LIK_ZILOGGAUSSIAN_LL, LIK_ZILOGGAUSSIAN_PREDY
  
    p = 1./(1+exp(-f));
  end
  
  function reclik = lik_ziloggaussian_recappend(reclik, ri, lik)
  %RECAPPEND  Append the parameters to the record
  %
  %  Description 
  %    RECLIK = GPCF_ZILOGGAUSSIAN_RECAPPEND(RECLIK, RI, LIK) takes a
  %    likelihood record structure RECLIK, record index RI and
  %    likelihood structure LIK with the current MCMC samples of
  %    the parameters. Returns RECLIK which contains all the old
  %    samples and the current samples from LIK. This subfunction
  %    is needed when using MCMC sampling (gp_mc).
  % 
  %  See also
  %    GP_MC

  % Initialize record
    if nargin == 2
      reclik.type = 'Ziloggaussian';
      reclik.nondiagW=true;

      % Initialize parameter
      reclik.sigma2 = [];

      % Set the function handles
      reclik.fh.pak = @lik_ziloggaussian_pak;
      reclik.fh.unpak = @lik_ziloggaussian_unpak;
      reclik.fh.lp = @lik_ziloggaussian_lp;
      reclik.fh.lpg = @lik_ziloggaussian_lpg;
      reclik.fh.ll = @lik_ziloggaussian_ll;
      reclik.fh.llg = @lik_ziloggaussian_llg;    
      reclik.fh.llg2 = @lik_ziloggaussian_llg2;
      reclik.fh.llg3 = @lik_ziloggaussian_llg3;
      reclik.fh.predy = @lik_ziloggaussian_predy;
      reclik.fh.invlink = @lik_ziloggaussian_invlink;
      reclik.fh.recappend = @lik_ziloggaussian_recappend;
      reclik.p=[];
      reclik.p.sigma2=[];
      if ~isempty(ri.p.sigma2)
        reclik.p.sigma2 = ri.p.sigma2;
      end
      return
    end
    
    reclik.sigma2(ri,:)=lik.sigma2;
    if ~isempty(lik.p.sigma2)
        reclik.p.sigma2 = feval(lik.p.sigma2.fh.recappend, reclik.p.sigma2, ri, lik.p.sigma2);
    end
  end
end


