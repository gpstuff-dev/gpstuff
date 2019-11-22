function lik = lik_gaussian(varargin)
%LIK_GAUSSIAN  Create a Gaussian likelihood structure
%
%  Description
%    LIK = LIK_GAUSSIAN('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates a Gaussian likelihood structure in which the named
%    parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%    LIK = LIK_GAUSSIAN(LIK,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a likelihood function structure with the named
%    parameters altered with the specified values.
%
%    Parameters for Gaussian likelihood function [default]
%      sigma2       - variance [0.1]
%      sigma2_prior - prior for sigma2 [prior_logunif]
%      n            - number of observations per input (See using average
%                     observations below)
%
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%    Using average observations
%    The lik_gaussian can be used to model data where each input vector is
%    attached to an average of varying number of observations. That is, we
%    have input vectors x_i, average observations y_i and sample sizes n_i.
%    Each observation is distributed  
%
%        y_i ~ N(f(x_i), sigma2/n_i)
%
%    The model is constructed as lik_gaussian('n', n), where n is the same
%    length as y and collects the sample sizes. 
%
%  See also
%    GP_SET, PRIOR_*, LIK_*

% Internal note: Because Gaussian noise can be combined
% analytically to the covariance matrix, lik_gaussian is internally
% little between lik_* and gpcf_* functions.
%
% Copyright (c) 2007-2017 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_GAUSSIAN';
  ip.addOptional('lik', [], @(x) isstruct(x) || isempty(x));
  ip.addParamValue('sigma2',0.1, @(x) isscalar(x) && x>=0);
  ip.addParamValue('sigma2_prior',prior_logunif(), @(x) isstruct(x) || isempty(x));
  ip.addParamValue('n',[], @(x) isreal(x) && all(x>0));
  ip.parse(varargin{:});
  lik=ip.Results.lik;

  if isempty(lik)
    init=true;
    lik.type = 'Gaussian';
  else
    if ~isfield(lik,'type') || ~isequal(lik.type,'Gaussian')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end
  
  % Initialize parameters
  if init || ~ismember('sigma2',ip.UsingDefaults)
    lik.sigma2 = ip.Results.sigma2;
  end
  if init || ~ismember('n',ip.UsingDefaults)
    lik.n = ip.Results.n;
  end
  % Initialize prior structure
  if init
    lik.p=[];
  end
  if init || ~ismember('sigma2_prior',ip.UsingDefaults)
    lik.p.sigma2=ip.Results.sigma2_prior;
  end
  if init
    % Set the function handles to the subfunctions
    lik.fh.pak = @lik_gaussian_pak;
    lik.fh.unpak = @lik_gaussian_unpak;
    lik.fh.ll = @lik_gaussian_ll;
    lik.fh.llg = @lik_gaussian_llg;    
    lik.fh.llg2 = @lik_gaussian_llg2;
    lik.fh.llg3 = @lik_gaussian_llg3;
    lik.fh.lp = @lik_gaussian_lp;
    lik.fh.lpg = @lik_gaussian_lpg;
    lik.fh.cfg = @lik_gaussian_cfg;
    lik.fh.tiltedMoments = @lik_gaussian_tiltedMoments;
    lik.fh.trcov  = @lik_gaussian_trcov;
    lik.fh.trvar  = @lik_gaussian_trvar;
    lik.fh.predy = @lik_gaussian_predy;    
    lik.fh.siteDeriv = @lik_gaussian_siteDeriv;
    lik.fh.recappend = @lik_gaussian_recappend;
  end

end

function [w, s, h] = lik_gaussian_pak(lik)
%LIK_GAUSSIAN_PAK  Combine likelihood parameters into one vector.
%
%  Description
%    W = LIK_GAUSSIAN_PAK(LIK) takes a likelihood structure LIK
%    and combines the parameters into a single row vector W.
%    This is a mandatory subfunction used for example in energy 
%    and gradient computations.
%
%       w = [ log(lik.sigma2)
%             (hyperparameters of lik.magnSigma2)]'
%     
%  See also
%    LIK_GAUSSIAN_UNPAK

  w = []; s = {}; h=[];
  if ~isempty(lik.p.sigma2)
    w = [w log(lik.sigma2)];
    s = [s; 'log(gaussian.sigma2)'];
    h = [h 0];
    % Hyperparameters of sigma2
    [wh, sh, hh] = lik.p.sigma2.fh.pak(lik.p.sigma2);
    w = [w wh];
    s = [s; sh];
    h = [h hh];
  end    

end

function [lik, w] = lik_gaussian_unpak(lik, w)
%LIK_GAUSSIAN_UNPAK  Extract likelihood parameters from the vector.
%
%  Description
%    W = LIK_GAUSSIAN_UNPAK(W, LIK) takes a likelihood structure
%    LIK and extracts the parameters from the vector W to the LIK
%    structure. This is a mandatory subfunction used for example 
%    in energy and gradient computations.
%
%    Assignment is inverse of  
%       w = [ log(lik.sigma2)
%             (hyperparameters of lik.magnSigma2)]'
%
%  See also
%    LIK_GAUSSIAN_PAK
  
  if ~isempty(lik.p.sigma2)
    lik.sigma2 = exp(w(1));
    w = w(2:end);
    
    % Hyperparameters of sigma2
    [p, w] = lik.p.sigma2.fh.unpak(lik.p.sigma2, w);
    lik.p.sigma2 = p;
  end
end




function logLik = lik_gaussian_ll(lik, y, f, ~)
%LIK_GAUSSIAN_LL    Log likelihood
%
%  Description
%    E = LIK_GAUSSIAN_LL(LIK, Y, F, Z) takes a likelihood data
%    structure LIK, incedence counts Y, expected counts Z, and
%    latent values F. Returns the log likelihood, log p(y|f,z).
%    This subfunction is needed when using Laplace approximation
%    or MCMC for inference with non-Gaussian likelihoods. This 
%    subfunction is also used in information criteria (DIC, WAIC)
%    computations.
%
%  See also
%    LIK_GAUSSIAN_LLG, LIK_GAUSSIAN_LLG3, LIK_GAUSSIAN_LLG2, GPLA_E

  s2 = lik.sigma2;
  r2 = (f-y).^2;  
  logLik =  sum(-0.5 * r2./s2 - 0.5*log(s2) - 0.5*log(2*pi));
  
end


function llg = lik_gaussian_llg(lik, y, f, param, ~)
%LIK_GAUSSIAN_LLG    Gradient of the log likelihood
%
%  Description 
%    G = LIK_GAUSSIAN_LLG(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, incedence counts Y, expected counts Z
%    and latent values F. Returns the gradient of the log
%    likelihood with respect to PARAM. At the moment PARAM can be
%    'param' or 'latent'. This subfunction is needed when using 
%    Laplace approximation or MCMC for inference with non-Gaussian 
%    likelihoods.
%
%  See also
%    LIK_GAUSSIAN_LL, LIK_GAUSSIAN_LLG2, LIK_GAUSSIAN_LLG3, GPLA_E
  
 
switch param
    case 'param'
        % there is also correction due to the log transformation
        s2 = lik.sigma2;
        r2 = (f-y).^2;
        llg =  0.5.* sum( r2./s2 -1 );

    case 'latent'
        s2 = lik.sigma2;
        r = f-y;
        llg =  - r./s2 ;
end
end


function llg2 = lik_gaussian_llg2(lik, y, f, param, ~)
%LIK_GAUSSIAN_LLG2  Second gradients of the log likelihood
%
%  Description        
%    G2 = LIK_GAUSSIAN_LLG2(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, incedence counts Y, expected counts Z,
%    and latent values F. Returns the Hessian of the log
%    likelihood with respect to PARAM. At the moment PARAM can be
%    only 'latent'. G2 is a vector with diagonal elements of the
%    Hessian matrix (off diagonals are zero). This subfunction
%    is needed when using Laplace approximation or EP for inference 
%    with non-Gaussian likelihoods.

%
%  See also
%    LIK_GAUSSIAN_LL, LIK_GAUSSIAN_LLG, LIK_GAUSSIAN_LLG3, GPLA_E

   
switch param
    case 'latent'
        s2 = lik.sigma2;
        llg2 =  - ones(size(y))./s2 ;
    case 'latent+param'
        % there is also correction due to the log transformation
        s2 = lik.sigma2;
        r = f-y;
        llg2 =  r./s2 ;
        
end
end    

function llg3 = lik_gaussian_llg3(lik, y, f, param, z)
%LIK_GAUSSIAN_LLG3  Third gradients of the log likelihood
%
%  Description
%    G3 = LIK_GAUSSIAN_LLG3(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, incedence counts Y, expected counts Z
%    and latent values F and returns the third gradients of the
%    log likelihood with respect to PARAM. At the moment PARAM
%    can be only 'latent'. G3 is a vector with third gradients.
%    This subfunction is needed when using Laplace approximation 
%    for inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_GAUSSIAN_LL, LIK_GAUSSIAN_LLG, LIK_GAUSSIAN_LLG2, GPLA_E, GPLA_G

switch param
    case 'latent'
        llg3 = zeros(size(y));
    case 'latent2+param'
        % there is also correction due to the log transformation
        s2 = lik.sigma2;
        llg3 = ones(size(y))./s2 ;
end
end


function lp = lik_gaussian_lp(lik)
%LIK_GAUSSIAN_LP  Evaluate the log prior of likelihood parameters
%
%  Description
%    LP = LIK_T_LP(LIK) takes a likelihood structure LIK and
%    returns log(p(th)), where th collects the parameters.
%    This subfunctions is needed when there are likelihood
%    parameters.
%
%  See also
%    LIK_GAUSSIAN_PAK, LIK_GAUSSIAN_UNPAK, LIK_GAUSSIAN_G, GP_E

  lp = 0;

  if ~isempty(lik.p.sigma2)
    likp=lik.p;
    lp = likp.sigma2.fh.lp(lik.sigma2, likp.sigma2) + log(lik.sigma2);
  end
end

function lpg = lik_gaussian_lpg(lik)
%LIK_GAUSSIAN_LPG  Evaluate gradient of the log prior with respect
%                  to the parameters.
%
%  Description
%    LPG = LIK_GAUSSIAN_LPG(LIK) takes a Gaussian likelihood
%    function structure LIK and returns LPG = d log (p(th))/dth,
%    where th is the vector of parameters. This subfunction is 
%    needed when there are likelihood parameters.
%
%  See also
%    LIK_GAUSSIAN_PAK, LIK_GAUSSIAN_UNPAK, LIK_GAUSSIAN_E, GP_G

  lpg = [];

  if ~isempty(lik.p.sigma2)
    likp=lik.p;
    
    lpgs = likp.sigma2.fh.lpg(lik.sigma2, likp.sigma2);
    lpg = lpgs(1).*lik.sigma2 + 1;
    if length(lpgs) > 1
      lpg = [lpg lpgs(2:end)];
    end            
  end
end



function DKff = lik_gaussian_cfg(lik, x, x2)
%LIK_GAUSSIAN_CFG  Evaluate gradient of covariance with respect to
%                 Gaussian noise
%
%  Description
%    Gaussian likelihood is a special case since it can be
%    analytically combined with covariance functions and thus we
%    compute gradient of covariance instead of gradient of likelihood.
%
%    DKff = LIK_GAUSSIAN_CFG(LIK, X) takes a Gaussian likelihood
%    function structure LIK, a matrix X of input vectors and
%    returns DKff, the gradients of Gaussian noise covariance
%    matrix Kff = k(X,X) with respect to th (cell array with
%    matrix elements). This subfunction is needed only in Gaussian 
%    likelihood.
%
%    DKff = LIK_GAUSSIAN_CFG(LIK, X, X2) takes a Gaussian
%    likelihood function structure LIK, a matrix X of input
%    vectors and returns DKff, the gradients of Gaussian noise
%    covariance matrix Kff = k(X,X) with respect to th (cell
%    array with matrix elements). This subfunction is needed 
%    only in Gaussian likelihood.
%
%  See also
%    LIK_GAUSSIAN_PAK, LIK_GAUSSIAN_UNPAK, LIK_GAUSSIAN_E, GP_G

  DKff = {};
  if ~isempty(lik.p.sigma2)
      if isempty(lik.n)
          DKff{1}=lik.sigma2;
      else
          n=size(x,1);
          DKff{1} = sparse(1:n, 1:n, lik.sigma2./lik.n, n, n);
      end
  end
end

function DKff  = lik_gaussian_ginput(lik, x, t, g_ind, gdata_ind, gprior_ind, varargin)
%LIK_GAUSSIAN_GINPUT  Evaluate gradient of likelihood function with 
%                     respect to x.
%
%  Description
%    DKff = LIK_GAUSSIAN_GINPUT(LIK, X) takes a likelihood
%    function structure LIK, a matrix X of input vectors and
%    returns DKff, the gradients of likelihood matrix Kff =
%    k(X,X) with respect to X (cell array with matrix elements).
%    This subfunction is needed only in Gaussian likelihood.
%
%    DKff = LIK_GAUSSIAN_GINPUT(LIK, X, X2) takes a likelihood
%    function structure LIK, a matrix X of input vectors and
%    returns DKff, the gradients of likelihood matrix Kff =
%    k(X,X2) with respect to X (cell array with matrix elements).
%    This subfunction is needed only in Gaussian likelihood.
%
%  See also
%    LIK_GAUSSIAN_PAK, LIK_GAUSSIAN_UNPAK, LIK_GAUSSIAN_E, GP_G

end

function C = lik_gaussian_trcov(lik, x)
%LIK_GAUSSIAN_TRCOV  Evaluate training covariance matrix
%                    corresponding to Gaussian noise
%
%  Description
%    C = LIK_GAUSSIAN_TRCOV(GP, TX) takes in covariance function
%    of a Gaussian process GP and matrix TX that contains
%    training input vectors. Returns covariance matrix C. Every
%    element ij of C contains covariance between inputs i and j
%    in TX. This subfunction is needed only in Gaussian likelihood.
%
%  See also
%    LIK_GAUSSIAN_COV, LIK_GAUSSIAN_TRVAR, GP_COV, GP_TRCOV

  [n, m] =size(x);
  n1=n+1;

  if isempty(lik.n)
      C = sparse(1:n,1:n,ones(n,1).*lik.sigma2,n,n);
  else  
      C = sparse(1:n, 1:n, lik.sigma2./lik.n, n, n);
  end

end

function C = lik_gaussian_trvar(lik, x)
%LIK_GAUSSIAN_TRVAR  Evaluate training variance vector
%                    corresponding to Gaussian noise
%
%  Description
%    C = LIK_GAUSSIAN_TRVAR(LIK, TX) takes in covariance function
%    of a Gaussian process LIK and matrix TX that contains
%    training inputs. Returns variance vector C. Every element i
%    of C contains variance of input i in TX. This subfunction is 
%    needed only in Gaussian likelihood.
%
%
%  See also
%    LIK_GAUSSIAN_COV, GP_COV, GP_TRCOV

  [n, m] =size(x);
  if isempty(lik.n)
      C=repmat(lik.sigma2,n,1);
  else
      C=lik.sigma2./lik.n(:);
  end

end


function [lpy, Ey, Vary] = lik_gaussian_predy(lik, Ef, Varf, yt, zt)
%LIK_Gaussian_PREDY    Returns the predictive mean, variance and density of y
%
%  Description  
%    LPY = LIK_POISSON_PREDY(LIK, EF, VARF YT, ZT)
%    Returns also the predictive density of YT, that is 
%        p(yt | y,zt) = \int p(yt | f, zt) p(f|y) df.
%    This requires also the incedence counts YT, expected counts ZT.
%    This subfunction is needed when computing posterior predictive 
%    distributions for future observations.
%
%    [LPY, EY, VARY] = LIK_POISSON_PREDY(LIK, EF, VARF, YT, ZT) 
%    takes a likelihood structure LIK, posterior mean EF and 
%    posterior variance VARF of the latent variable and returns the
%    posterior predictive mean EY and variance VARY of the
%    observations related to the latent variables. This subfunction
%    is needed when computing posterior predictive distributions for 
%    future observations.
%        
%
%  See also 
%    GPLA_PRED, GPEP_PRED, GPMC_PRED

  lpy = zeros(size(Ef));
  Ey = Ef;
  EVary = lik.sigma2;
  VarEy = Varf; 
  Vary = EVary + VarEy;
  lpy = norm_lpdf(yt,Ey,sqrt(Vary));

end

function [logM_0, m_1, sigm2hati1] = lik_gaussian_tiltedMoments(lik, y, i1, sigm2_i, myy_i, z)
%LIK_PROBIT_TILTEDMOMENTS  Returns the marginal moments for EP algorithm
%
%  Description
%    [M_0, M_1, M2] = LIK_PROBIT_TILTEDMOMENTS(LIK, Y, I, S2,
%    MYY) takes a likelihood structure LIK, class labels Y, index
%    I and cavity variance S2 and mean MYY. Returns the zeroth
%    moment M_0, mean M_1 and variance M_2 of the posterior
%    marginal (see Rasmussen and Williams (2006): Gaussian
%    processes for Machine Learning, page 55). This subfunction 
%    is needed when using EP for inference with non-Gaussian 
%    likelihoods.
%
%  See also
%    GPEP_E

%   m_1=myy_i;
%   sigm2hati1=sigm2_i;
%   logM_0=zeros(size(y));
  
s2 = lik.sigma2;
tau = 1./s2 + 1./sigm2_i;
w = (y(i1)./s2 + myy_i./sigm2_i)./tau;
%Zi = 1 ./( sqrt( 2.*pi.*(s2+sigm2_i) ) ) .* exp( -0.5*(y(i1)-myy_i).^2./(s2+sigm2_i) );

m_1 = w;
sigm2hati1 = 1./tau;
%logM_0 = log(Zi)
logM_0 = -0.5*log( 2.*pi) - 0.5*log( (s2+sigm2_i) )  + ( -0.5*(y(i1)-myy_i).^2./(s2+sigm2_i) );

end

function [g_i] = lik_gaussian_siteDeriv(lik, y, i1, sigm2_i, myy_i, z)
%LIK_NEGBIN_SITEDERIV  Evaluate the expectation of the gradient
%                      of the log likelihood term with respect
%                      to the likelihood parameters for EP 
%
%  Description [M_0, M_1, M2] =
%    LIK_NEGBIN_SITEDERIV(LIK, Y, I, S2, MYY, Z) takes a
%    likelihood structure LIK, incedence counts Y, expected
%    counts Z, index I and cavity variance S2 and mean MYY. 
%    Returns E_f [d log p(y_i|f_i) /d a], where a is the
%    likelihood parameter and the expectation is over the
%    marginal posterior. This term is needed when evaluating the
%    gradients of the marginal likelihood estimate Z_EP with
%    respect to the likelihood parameters (see Seeger (2008):
%    Expectation propagation for exponential families). This 
%    subfunction is needed when using EP for inference with 
%    non-Gaussian likelihoods and there are likelihood parameters.
%
%  See also
%    GPEP_G

s2 = lik.sigma2;
tau = 1/s2 + 1/sigm2_i;
w = (y(i1)/s2 + myy_i/sigm2_i)/tau;
%Zi = 1/( sqrt(2*pi*(s2+sigm2_i)) )*exp( -0.5*(y(i1)-myy_i)^2/(s2+sigm2_i) );

%g_i = 0.5*( (1/tau + w.^2 -2*w*y(i1) + y(i1).^2 ) /s2 - 1)/s2 * s2;
g_i = 0.5*( (1/tau + w.^2 -2*w*y(i1) + y(i1).^2 ) /s2 - 1);


end



function reclik = lik_gaussian_recappend(reclik, ri, lik)
%RECAPPEND  Record append
%
%  Description
%    RECLIK = LIK_GAUSSIAN_RECAPPEND(RECLIK, RI, LIK) takes a
%    likelihood function record structure RECLIK, record index RI
%    and likelihood function structure LIK with the current MCMC
%    samples of the parameters. Returns RECLIK which contains all
%    the old samples and the current samples from LIK. This 
%    subfunction is needed when using MCMC sampling (gp_mc).
%
%  See also
%    GP_MC and GP_MC -> RECAPPEND

  if nargin == 2
    % Initialize the record
    reclik.type = 'Gaussian';
    
    % Initialize the parameters
    reclik.sigma2 = []; 
    reclik.n = []; 
    
    % Set the function handles
    reclik.fh.pak = @lik_gaussian_pak;
    reclik.fh.unpak = @lik_gaussian_unpak;
    reclik.fh.lp = @lik_gaussian_lp;
    reclik.fh.lpg = @lik_gaussian_lpg;
    reclik.fh.ll= @lik_gaussian_ll;
    reclik.fh.cfg = @lik_gaussian_cfg;
    reclik.fh.trcov  = @lik_gaussian_trcov;
    reclik.fh.trvar  = @lik_gaussian_trvar;
    lik.fh.predy = @lik_gaussian_predy;
    reclik.fh.recappend = @lik_gaussian_recappend;  
    reclik.p=[];
    reclik.p.sigma2=[];
    if ~isempty(ri.p.sigma2)
      reclik.p.sigma2 = ri.p.sigma2;
    end
  else
    % Append to the record
    likp = lik.p;

    % record sigma2
    reclik.sigma2(ri,:)=lik.sigma2;
    if isfield(likp,'sigma2') && ~isempty(likp.sigma2)
      reclik.p.sigma2 = likp.sigma2.fh.recappend(reclik.p.sigma2, ri, likp.sigma2);
    end
    % record n if given
    if isfield(lik,'n') && ~isempty(lik.n)
      reclik.n(ri,:)=lik.n(:)';
    end
  end
end
