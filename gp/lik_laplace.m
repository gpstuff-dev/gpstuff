function lik = lik_laplace(varargin)
%LIK_Laplace  Create a Laplace likelihood structure
%
%  Description
%    LIK = LIK_LAPLACE('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates a laplace likelihood structure in which the named
%    parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%    LIK = LIK_Laplace(LIK,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a likelihood function structure with the named
%    parameters altered with the specified values.
%
%    Parameters for Laplace likelihood function [default]
%      scale       - scale parameter of the laplace distribution [0.2]
%      scale_prior - prior for scale [prior_logunif]
%
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc. 
%
%    The likelihood is defined as follows:
%                      __ n
%      p(y|f, scale) = || i=1 1/(2*scale)*exp(-abs(y-f)/scale)
%    
%    Note that because the form of the likelihood, second order derivatives
%    with respect to latent values are 0. Because this, EP should be used
%    instead of Laplace approximation.    
%
%  See also
%    GP_SET, PRIOR_*, LIK_*

% Copyright (c) 2014 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_LAPLACE';
  ip.addOptional('lik', [], @isstruct);
  ip.addParamValue('scale',0.2, @(x) isscalar(x) && x>0);
  ip.addParamValue('scale_prior',prior_logunif(), @(x) isstruct(x) || isempty(x));
  ip.parse(varargin{:});
  lik=ip.Results.lik;

  if isempty(lik)
    init=true;
    lik.type = 'Laplace';
  else
    if ~isfield(lik,'type') || ~isequal(lik.type,'Laplace')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end
  
  % Initialize parameters
  if init || ~ismember('scale',ip.UsingDefaults)
    lik.scale = ip.Results.scale;
  end
  % Initialize prior structure
  if init
    lik.p=[];
  end
  if init || ~ismember('scale_prior',ip.UsingDefaults)
    lik.p.scale=ip.Results.scale_prior;
  end
  if init
    % Set the function handles to the subfunctions
    lik.fh.pak = @lik_laplace_pak;
    lik.fh.unpak = @lik_laplace_unpak;
    lik.fh.lp = @lik_laplace_lp;
    lik.fh.lpg = @lik_laplace_lpg;
    lik.fh.ll = @lik_laplace_ll;
    lik.fh.llg = @lik_laplace_llg;    
    lik.fh.llg2 = @lik_laplace_llg2;
    lik.fh.llg3 = @lik_laplace_llg3;
    lik.fh.tiltedMoments = @lik_laplace_tiltedMoments;
    lik.fh.siteDeriv = @lik_laplace_siteDeriv;
    lik.fh.predy = @lik_laplace_predy;
    lik.fh.invlink = @lik_laplace_invlink;
    lik.fh.recappend = @lik_laplace_recappend;
  end

end

function [w s h] = lik_laplace_pak(lik)
%LIK_LAPLACE_PAK  Combine likelihood parameters into one vector.
%
%  Description
%    W = LIK_LAPLACE_PAK(LIK) takes a likelihood structure LIK
%    and combines the parameters into a single row vector W.
%    This is a mandatory subfunction used for example in 
%    energy and gradient computations.
%
%       w = [ log(lik.scale)
%             (hyperparameters of lik.magnSigma2)]'
%     
%  See also
%    LIK_LAPLACE_UNPAK

  w = []; s = {}; h=[];
  if ~isempty(lik.p.scale)
    w = [w log(lik.scale)];
    s = [s; 'log(laplace.scale)'];
    h = [h 0];
    % Hyperparameters of scale
    [wh, sh, hh] = lik.p.scale.fh.pak(lik.p.scale);    
    w = [w wh];
    s = [s; sh];
    h = [h hh];
  end    

end

function [lik, w] = lik_laplace_unpak(lik, w)
%LIK_LAPLACE_UNPAK  Extract likelihood parameters from the vector.
%
%  Description
%    W = LIK_LAPLACE_UNPAK(W, LIK) takes a likelihood structure
%    LIK and extracts the parameters from the vector W to the LIK
%    structure. This is a mandatory subfunction used for example 
%    in energy and gradient computations.
%
%    Assignment is inverse of  
%       w = [ log(lik.scale)
%             (hyperparameters of lik.magnSigma2)]'
%
%  See also
%    LIK_LAPLACE_PAK
  
  if ~isempty(lik.p.scale)
    lik.scale = exp(w(1));
    w = w(2:end);
    
    % Hyperparameters of scale
    [p, w] = lik.p.scale.fh.unpak(lik.p.scale, w);
    lik.p.scale = p;
  end
end

function lp = lik_laplace_lp(lik)
%LIK_LAPLACE_LP  Evaluate the log prior of likelihood parameters
%
%  Description
%    LP = LIK_LAPLACE_LP(LIK) takes a likelihood structure LIK and
%    returns log(p(th)), where th collects the parameters. This
%    subfunction is needed when there are likelihood parameters.
%
%  See also
%    LIK_LAPLACE_PAK, LIK_LAPLACE_UNPAK, LIK_LAPLACE_G, GP_E

  lp = 0;

  if ~isempty(lik.p.scale)
    likp=lik.p;
    lp = likp.scale.fh.lp(lik.scale, likp.scale) + log(lik.scale);
  end
end

function lpg = lik_laplace_lpg(lik)
%LIK_LAPLACE_LPG  Evaluate gradient of the log prior with respect
%                  to the parameters.
%
%  Description
%    LPG = LIK_LAPLACE_LPG(LIK) takes a Laplace likelihood
%    function structure LIK and returns LPG = d log (p(th))/dth,
%    where th is the vector of parameters. This subfunction is 
%    needed when there are likelihood parameters.
%
%  See also
%    LIK_LAPLACE_PAK, LIK_LAPLACE_UNPAK, LIK_LAPLACE_E, GP_G

  lpg = [];

  if ~isempty(lik.p.scale)
    likp=lik.p;
    
    lpgs = likp.scale.fh.lpg(lik.scale, likp.scale);
    lpg = lpgs(1).*lik.scale + 1;
    if length(lpgs) > 1
      lpg = [lpg lpgs(2:end)];
    end            
  end
end

function ll = lik_laplace_ll(lik, y, f, z)
%LIK_LAPLACE_LL  Log likelihood
%
%  Description
%    LL = LIK_LAPLACE_LL(LIK, Y, F, Z) takes a likelihood
%    structure LIK, observations Y and latent values F. 
%    Returns the log likelihood, log p(y|f,z). This subfunction 
%    is needed when using Laplace approximation or MCMC for 
%    inference with non-Gaussian likelihoods. This subfunction 
%    is also used in information criteria (DIC, WAIC) computations.
%
%  See also
%    LIK_LAPLACE_LLG, LIK_LAPLACE_LLG3, LIK_LAPLACE_LLG2, GPLA_E
  
  scale=lik.scale;
  ll = sum(-log(2*scale) - abs(y-f)./scale);

end

function llg = lik_laplace_llg(lik, y, f, param, z)
%LIK_LAPLACE_LLG  Gradient of the log likelihood
%
%  Description 
%    LLG = LIK_LAPLACE_LLG(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, observations Y and latent values F. Returns 
%    the gradient of the log likelihood with respect to PARAM. 
%    At the moment PARAM can be 'param' or 'latent'. This subfunction 
%    is needed when using Laplace approximation or MCMC for inference 
%    with non-Gaussian likelihoods.
%
%  See also
%    LIK_LAPLACE_LL, LIK_LAPLACE_LLG2, LIK_LAPLACE_LLG3, GPLA_E

  error('Laplace likelihood is not differentiable. Use EP or gradient free MCMC.')
end

function llg2 = lik_laplace_llg2(lik, y, f, param, z)
%LIK_LAPLACE_LLG2  Second gradients of the log likelihood
%
%  Description        
%    LLG2 = LIK_LAPLACE_LLG2(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, observations Y and latent values F. Returns 
%    the Hessian of the log likelihood with respect to PARAM. 
%    At the moment PARAM can be 'param' or 'latent'. LLG2 is 
%    a vector with diagonal elements of the Hessian matrix 
%    (off diagonals are zero). This subfunction is needed 
%    when using Laplace approximation or EP for inference 
%    with non-Gaussian likelihoods.
%
%  See also
%    LIK_LAPLACE_LL, LIK_LAPLACE_LLG, LIK_LAPLACE_LLG3, GPLA_E

  
  error('Laplace likelihood is not differentiable. Use EP or gradient free MCMC.')
end    

function llg3 = lik_laplace_llg3(lik, y, f, param, z)
%LIK_LAPLACE_LLG3  Third gradients of the log likelihood
%
%  Description
%    LLG3 = LIK_LAPLACE_LLG3(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, observations Y and latent values F and 
%    returns the third gradients of the log likelihood with 
%    respect to PARAM. At the moment PARAM can be 'param' or 
%    'latent'. LLG3 is a vector with third gradients. This 
%    subfunction is needed when using Laplace approximation for 
%    inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_LAPLACE_LL, LIK_LAPLACE_LLG, LIK_LAPLACE_LLG2, GPLA_E, GPLA_G

  error('Laplace likelihood is not differentiable. Use EP or gradient free MCMC.')
end

function [logM_0, m_1, sigm2hati1] = lik_laplace_tiltedMoments(lik, y, i1, sigma2_i, myy_i, z)
%LIK_LAPLACE_TILTEDMOMENTS  Returns the marginal moments for EP algorithm
%
%  Description
%    [M_0, M_1, M2] = LIK_LAPLACE_TILTEDMOMENTS(LIK, Y, I, S2,
%    MYY, Z) takes a likelihood structure LIK, observations
%    Y, index I and cavity variance S2 and mean MYY. Returns 
%    the zeroth moment M_0, mean M_1 and variance M_2 of the 
%    posterior marginal (see Rasmussen and Williams (2006): 
%    Gaussian processes for Machine Learning, page 55). This 
%    subfunction is needed when using EP for inference with 
%    non-Gaussian likelihoods.
%
%  See also
%    GPEP_E
  
  yy = y(i1);
  scale = lik.scale;
  logM_0=zeros(size(yy));
  m_1=zeros(size(yy));
  sigm2hati1=zeros(size(yy));
  
  for i=1:length(i1)
    if isscalar(sigma2_i)
      sigma2ii = sigma2_i;
    else
      sigma2ii = sigma2_i(i);
    end
    
    % get a function handle of an unnormalized tilted distribution
    % (likelihood * cavity = Quantile-GP * Gaussian)
    % and useful integration limits
    [tf,minf,maxf]=init_laplace_norm(yy(i),myy_i(i),sigma2ii,scale);
    
    % Integrate with quadrature
    RTOL = 1.e-6;
    ATOL = 1.e-10;
    [m_0, m_1(i), m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
    if isnan(m_0)
      logM_0=NaN;
      return
    end
    sigm2hati1(i) = m_2 - m_1(i).^2;
    
    % If the second central moment is less than cavity variance
    % integrate more precisely. Theoretically for log-concave
    % likelihood should be sigm2hati1 < sigm2_i.
    if sigm2hati1(i) >= sigma2ii
      ATOL = ATOL.^2;
      RTOL = RTOL.^2;
      [m_0, m_1(i), m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
      sigm2hati1(i) = m_2 - m_1(i).^2;
      if sigm2hati1(i) >= sigma2ii
        warning('lik_laplace_tilted_moments: sigm2hati1 >= sigm2_i');
      end
    end
    logM_0(i) = log(m_0);
  end
end

function [g_i] = lik_laplace_siteDeriv(lik, y, i1, sigm2_i, myy_i, z)
%LIK_LAPLACE_SITEDERIV  Evaluate the expectation of the gradient
%                      of the log likelihood term with respect
%                      to the likelihood parameters for EP 
%
%  Description [M_0, M_1, M2] =
%    LIK_LAPLACE_SITEDERIV(LIK, Y, I, S2, MYY, Z) takes a
%    likelihood structure LIK, observations Y, index I 
%    and cavity variance S2 and mean MYY. Returns E_f 
%    [d log p(y_i|f_i) /d a], where a is the likelihood 
%    parameter and the expectation is over the marginal posterior.
%    This term is needed when evaluating the gradients of 
%    the marginal likelihood estimate Z_EP with respect to 
%    the likelihood parameters (see Seeger (2008):
%    Expectation propagation for exponential families).This 
%    subfunction is needed when using EP for inference with 
%    non-Gaussian likelihoods and there are likelihood parameters.
%
%  See also
%    GPEP_G


  yy = y(i1);
  scale=lik.scale;
  
  % get a function handle of an unnormalized tilted distribution 
  % (likelihood * cavity = Quantile-GP * Gaussian)
  % and useful integration limits
  [tf,minf,maxf]=init_laplace_norm(yy,myy_i,sigm2_i,scale);
  % additionally get function handle for the derivative
  td = @deriv;
  
  % Integrate with quadgk
  [m_0, fhncnt] = quadgk(tf, minf, maxf);
  [g_i, fhncnt] = quadgk(@(f) td(f).*tf(f)./m_0, minf, maxf);
  g_i = g_i.*scale;

  function g = deriv(f)

    g = -1/scale + abs(yy-f)./scale^2;
    
  end
end

function [lpy, Ey, Vary] = lik_laplace_predy(lik, Ef, Varf, yt, zt)
%LIK_LAPLACE_PREDY  Returns the predictive mean, variance and density of y
%
%  Description  
%    LPY = LIK_LAPLACE_PREDY(LIK, EF, VARF YT, ZT)
%    Returns logarithm of the predictive density PY of YT, that is 
%        p(yt | zt) = \int p(yt | f, zt) p(f|y) df.
%    This subfunction is needed when computing posterior predictive 
%    distributions for future observations.
%
%    [LPY, EY, VARY] = LIK_LAPLACE_PREDY(LIK, EF, VARF) takes a
%    likelihood structure LIK, posterior mean EF and posterior
%    Variance VARF of the latent variable and returns the
%    posterior predictive mean EY and variance VARY of the
%    observations related to the latent variables. This 
%    subfunction is needed when computing posterior predictive 
%    distributions for future observations.
%        

%
%  See also
%    GPLA_PRED, GPEP_PRED, GPMC_PRED


  scale=lik.scale;
  
  Ey=Ef;
  Vary=2*scale^2 + Varf;
  
  % Evaluate the posterior predictive densities of the given observations
  lpy = zeros(length(yt),1);
  if (size(Ef,2) > 1) && (size(Ef,2) > 1) && size(yt,2) == 1
    % Approximate integral with sum of grid points when using corrected
    % marginal posterior
    for i1=1:length(yt)
      py = arrayfun(@(f) exp(lik.fh.ll(lik, yt(i1), f, [])), Ef(i1,:));
      pf = Varf(i1,:)./sum(Varf(i1,:));
      lpy(i1) = log(sum(py.*pf));
    end
  else
    for i1=1:length(yt)
      % get a function handle of the likelihood times posterior
      % (likelihood * posterior = Quantile-GP * Gaussian)
      % and useful integration limits
      [pdf,minf,maxf]=init_laplace_norm(...
        yt(i1),Ef(i1),Varf(i1),scale);
      % integrate over the f to get posterior predictive distribution
      lpy(i1) = log(quadgk(pdf, minf, maxf));
    end
  end
end


function [df,minf,maxf] = init_laplace_norm(yy,myy_i,sigm2_i,scale)
%INIT_LAPLACE_NORM
%
%  Description
%    Return function handle to a function evaluating
%    Laplace * Gaussian which is used for evaluating
%    (likelihood * cavity) or (likelihood * posterior) Return
%    also useful limits for integration. This is private function
%    for lik_laplace. This subfunction is needed by subfunctions
%    tiltedMoments, siteDeriv and predy.
%  
%  See also
%    LIK_LAPLACE_TILTEDMOMENTS, LIK_LAPLACE_SITEDERIV,
%    LIK_LAPLACE_PREDY
  
  sigma=scale;
% avoid repetitive evaluation of constant part
  ldconst = log(1/(2*sigma)) ...
            - log(sigm2_i)/2 - log(2*pi)/2;
  % Create function handle for the function to be integrated
  df = @laplace_norm;
  % use log to avoid underflow, and derivates for faster search
  ld = @log_laplace_norm;
  ldg = @log_laplace_norm_g;
%   ldg2 = @log_laplace_norm_g2;

  % Set the limits for integration
  % Quantile-GP likelihood is log-concave so the laplace_norm
  % function is unimodal, which makes things easier
  if yy==0
    % with yy==0, the mode of the likelihood is not defined
    % use the mode of the Gaussian (cavity or posterior) as a first guess
    modef = myy_i;
  else
    % use precision weighted mean of the Gaussian approximation
    % of the Quantile-GP likelihood and Gaussian
    modef = (myy_i/sigm2_i + yy/scale)/(1/sigm2_i + 1/scale);
  end
  % find the mode of the integrand using Newton iterations
  % few iterations is enough, since the first guess in the right direction
  niter=8;       % number of Newton iterations 
  
  minf=modef-6*sigm2_i;
  while ldg(minf) < 0
    minf=minf-2*sigm2_i;
  end
  maxf=modef+6*sigm2_i;
  while ldg(maxf) > 0
    maxf=maxf+2*sigm2_i;
  end
  for ni=1:niter
%     h=ldg2(modef);
    modef=0.5*(minf+maxf);
    if ldg(modef) < 0
      maxf=modef;
    else
      minf=modef;
    end
  end
  % integrand limits based on Gaussian approximation at mode
  minf=modef-6*sqrt(sigm2_i);
  maxf=modef+6*sqrt(sigm2_i);
  modeld=ld(modef);
  iter=0;
  % check that density at end points is low enough
  lddiff=20; % min difference in log-density between mode and end-points
  minld=ld(minf);
  step=1;
  while minld>(modeld-lddiff)
    minf=minf-step*sqrt(sigm2_i);
    minld=ld(minf);
    iter=iter+1;
    step=step*2;
    if iter>100
      error(['lik_laplace -> init_laplace_norm: ' ...
             'integration interval minimun not found ' ...
             'even after looking hard!'])
    end
  end
  maxld=ld(maxf);
  iter=0;
  step=1;
  while maxld>(modeld-lddiff)
    maxf=maxf+step*sqrt(sigm2_i);
    maxld=ld(maxf);
    iter=iter+1;
    step=step*2;
    if iter>100
      error(['lik_laplace -> init_laplace_norm: ' ...
             'integration interval maximun not found ' ...
             'even after looking hard!'])
    end
  end
  
  function integrand = laplace_norm(f)
  % Laplace * Gaussian
    integrand = exp(ldconst ...
                    -abs(yy-f)./scale ...
                    -0.5*(f-myy_i).^2./sigm2_i);
  end
  
  function log_int = log_laplace_norm(f)
  % log(Laplace * Gaussian)
  % log_laplace_norm is used to avoid underflow when searching
  % integration interval
    log_int = ldconst...
              -abs(yy-f)./scale ...
              -0.5*(f-myy_i).^2./sigm2_i;
  end
  
  function g = log_laplace_norm_g(f)
  % d/df log(Laplace * Gaussian)
  % derivative of log_laplace_norm
    g = -1/scale ...
        + (myy_i - f)./sigm2_i;
  end
  
  
end

function mu = lik_laplace_invlink(lik, f, z)
%LIK_LAPLACE_INVLINK  Returns values of inverse link function
%             
%  Description 
%    MU = LIK_LAPLACE_INVLINK(LIK, F) takes a likelihood structure LIK and
%    latent values F and returns the values MU of inverse link function.
%    This subfunction is needed when using function gp_predprctmu.
%
%     See also
%     LIK_LAPLACE_LL, LIK_LAPLACE_PREDY
  
  mu = f;
end

function reclik = lik_laplace_recappend(reclik, ri, lik)
%RECAPPEND  Append the parameters to the record
%
%  Description 
%    RECLIK = LIK_LAPLACE_RECAPPEND(RECLIK, RI, LIK) takes a
%    likelihood record structure RECLIK, record index RI and
%    likelihood structure LIK with the current MCMC samples of
%    the parameters. Returns RECLIK which contains all the old
%    samples and the current samples from LIK.  This subfunction
%    is needed when using MCMC sampling (gp_mc).
% 
%  See also
%    GP_MC

  if nargin == 2
    % Initialize the record
    reclik.type = 'Laplace';

    % Initialize parameter
    reclik.scale = [];   

    % Set the function handles
    reclik.fh.pak = @lik_laplace_pak;
    reclik.fh.unpak = @lik_laplace_unpak;
    reclik.fh.lp = @lik_laplace_lp;
    reclik.fh.lpg = @lik_laplace_lpg;
    reclik.fh.ll = @lik_laplace_ll;
    reclik.fh.llg = @lik_laplace_llg;    
    reclik.fh.llg2 = @lik_laplace_llg2;
    reclik.fh.llg3 = @lik_laplace_llg3;
    reclik.fh.tiltedMoments = @lik_laplace_tiltedMoments;
    reclik.fh.predy = @lik_laplace_predy;
    reclik.fh.invlink = @lik_laplace_invlink;
    reclik.fh.recappend = @lik_laplace_recappend;
    reclik.p=[];
    reclik.p.scale=[];
    if ~isempty(ri.p.scale)
      reclik.p.scale = ri.p.scale;
    end
  else
        
    % Append to the record
    reclik.scale(ri,:)=lik.scale;    
    if ~isempty(lik.p.scale)
      reclik.p.scale = lik.p.scale.fh.recappend(reclik.p.scale, ri, lik.p.scale);
    end
  end
end
