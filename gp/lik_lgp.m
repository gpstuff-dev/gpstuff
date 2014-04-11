function lik = lik_lgp(varargin)
%LIK_LGP  Create a logistic Gaussian process likelihood structure 
%
%  Description
%    LIK = LIK_LGP creates a logistic Gaussian process likelihood structure
%
%    The likelihood is defined as follows:
%               __ n
%      p(y|f) = || i=1 exp(f_i) / Sum_{j=1}^n exp(f_j),
%
%      where f contains latent values.
%
%  Reference
%
%    Jaakko Riihimäki and Aki Vehtari (2014). Laplace approximation
%    for logistic Gaussian process density estimation and
%    regression. Bayesian analysis, in press.
%
%  See also
%    LGPDENS, GP_SET, LIK_*
%
% Copyright (c) 2011 Jaakko Riihimäki and Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_LGP';
  ip.addOptional('lik', [], @isstruct);
  ip.parse(varargin{:});
  lik=ip.Results.lik;

  if isempty(lik)
    init=true;
    lik.type = 'LGP';
    lik.nondiagW = true;
  else
    if ~isfield(lik,'type') || ~isequal(lik.type,'LGP')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end

  if init
    % Set the function handles to the subfunctions
    lik.fh.pak = @lik_lgp_pak;
    lik.fh.unpak = @lik_lgp_unpak;
    lik.fh.ll = @lik_lgp_ll;
    lik.fh.llg = @lik_lgp_llg;    
    lik.fh.llg2 = @lik_lgp_llg2;
    lik.fh.llg3 = @lik_lgp_llg3;
    lik.fh.tiltedMoments = @lik_lgp_tiltedMoments;
    lik.fh.predy = @lik_lgp_predy;
    lik.fh.invlink = @lik_lgp_invlink;
    lik.fh.recappend = @lik_lgp_recappend;
  end

end

function [w,s,h] = lik_lgp_pak(lik)
%LIK_LGP_PAK  Combine likelihood parameters into one vector.
%
%  Description 
%    W = LIK_LGP_PAK(LIK) takes a likelihood structure LIK
%    and returns an empty verctor W. If LGP likelihood had
%    parameters this would combine them into a single row vector
%    W (see e.g. lik_negbin). This is a mandatory subfunction 
%    used for example in energy and gradient computations.
%     
%  See also
%    LIK_LGP_UNPAK, GP_PAK

  w = []; s = {}; h=[];
end


function [lik, w] = lik_lgp_unpak(lik, w)
%LIK_LGP_UNPAK  Extract likelihood parameters from the vector.
%
%  Description
%    W = LIK_LGP_UNPAK(W, LIK) Doesn't do anything.
%
%    If LGP likelihood had parameters this would extract them
%    parameters from the vector W to the LIK structure. This 
%    is a mandatory subfunction used for example in energy 
%    and gradient computations.
%     
%
%  See also
%    LIK_LGP_PAK, GP_UNPAK

  lik=lik;
  w=w;
  
end


function logLik = lik_lgp_ll(lik, y, f, z)
%LIK_LGP_LL    Log likelihood
%
%  Description
%    E = LIK_LGP_LL(LIK, Y, F, Z) takes a likelihood data
%    structure LIK, incedence counts Y, expected counts Z, and
%    latent values F. Returns the log likelihood, log p(y|f,z).
%    This subfunction is needed when using Laplace approximation 
%    or MCMC for inference with non-Gaussian likelihoods. This 
%    subfunction is also used in information criteria (DIC, WAIC) 
%    computations.
%
%  See also
%    LIK_LGP_LLG, LIK_LGP_LLG3, LIK_LGP_LLG2, GPLA_E

  n=sum(y);
  qj=exp(f);
  logLik = sum(f.*y)-n*log(sum(qj));
end


function deriv = lik_lgp_llg(lik, y, f, param, z)
%LIK_LGP_LLG    Gradient of the log likelihood
%
%  Description 
%    G = LIK_LGP_LLG(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, incedence counts Y, expected counts Z
%    and latent values F. Returns the gradient of the log
%    likelihood with respect to PARAM. At the moment PARAM can be
%    'param' or 'latent'. This subfunction is needed when using Laplace 
%    approximation or MCMC for inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_LGP_LL, LIK_LGP_LLG2, LIK_LGP_LLG3, GPLA_E
  
  switch param
    case 'latent'
      n=sum(y);
      qj=exp(f);
      pj=qj./sum(qj);
      deriv=y-n*pj;
  end
end


function g2 = lik_lgp_llg2(lik, y, f, param, z)
%function g2 = lik_lgp_llg2(lik, y, f, param, z)
%LIK_LGP_LLG2  Second gradients of the log likelihood
%
%  Description        
%    G2 = LIK_LGP_LLG2(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, incedence counts Y, expected counts Z,
%    and latent values F. Returns the Hessian of the log
%    likelihood with respect to PARAM. At the moment PARAM can be
%    only 'latent'. G2 is a vector with diagonal elements of the
%    Hessian matrix (off diagonals are zero). This subfunction 
%    is needed when using Laplace approximation or EP for 
%    inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_LGP_LL, LIK_LGP_LLG, LIK_LGP_LLG3, GPLA_E

  switch param
    case 'latent'
      qj=exp(f);
      
      % g2 is not the second gradient of the log likelihood but only a
      % vector to form the exact gradient term in gpla_nd_e, gpla_nd_g and
      % gpla_nd_pred functions
      g2=qj./sum(qj);
  end
end    

function g3 = lik_lgp_llg3(lik, y, f, param, z)
%LIK_LGP_LLG3  Third gradients of the log likelihood
%
%  Description
%    G3 = LIK_LGP_LLG3(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, incedence counts Y, expected counts Z
%    and latent values F and returns the third gradients of the
%    log likelihood with respect to PARAM. At the moment PARAM
%    can be only 'latent'. G3 is a vector with third gradients.
%    This subfunction is needed when using Laplace approximation 
%    for inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_LGP_LL, LIK_LGP_LLG, LIK_LGP_LLG2, GPLA_E, GPLA_G
  
  switch param
    case 'latent'
      qj=exp(f);
      
      % g3 is not the third gradient of the log likelihood but only a
      % vector to form the exact gradient term in gpla_nd_e, gpla_nd_g and
      % gpla_nd_pred functions
      g3=qj./sum(qj);
      
      %n=sum(y);
      %nf=size(f,1);
      %g3d=zeros(nf,nf);
      %for i1=1:nf
      %  g3dtmp=-g3*g3(i1);
      %  g3dtmp(i1)=g3dtmp(i1)+g3(i1);
      %  g3d(:,i1)=g3dtmp;
      %  %g3i1= n*(-diag(g3d(:,i1)) + bsxfun(@times,g3,g3d(:,i1)') + bsxfun(@times,g3d(:,i1),g3'));
      %end
  end
end

function [logM_0, m_1, sigm2hati1] = lik_lgp_tiltedMoments(lik, y, i1, sigm2_i, myy_i, z)
%LIK_LGP_TILTEDMOMENTS  Returns the marginal moments for EP algorithm
%
%  Description
%    [M_0, M_1, M2] = LIK_LGP_TILTEDMOMENTS(LIK, Y, I, S2,
%    MYY, Z) takes a likelihood structure LIK, incedence counts
%    Y, expected counts Z, index I and cavity variance S2 and
%    mean MYY. Returns the zeroth moment M_0, mean M_1 and
%    variance M_2 of the posterior marginal (see Rasmussen and
%    Williams (2006): Gaussian processes for Machine Learning,
%    page 55). This subfunction is needed when using EP for 
%    inference with non-Gaussian likelihoods.
%
%  See also
%    GPEP_E

  error('Not implemented')
  
end


function [lpy, Ey, Vary] = lik_lgp_predy(lik, Ef, Varf, yt, zt)
%LIK_LGP_PREDY    Returns the predictive mean, variance and density of y
%
%  Description  
%    LPY = LIK_LGP_PREDY(LIK, EF, VARF YT, ZT)
%    Returns also the predictive density of YT, that is 
%        p(yt | y,zt) = \int p(yt | f, zt) p(f|y) df.
%    This requires also the incedence counts YT, expected counts ZT.
%    This subfunction is needed when computing posterior predictive 
%    distributions for future observations.
%
%    [LPY, EY, VARY] = LIK_LGP_PREDY(LIK, EF, VARF) takes a
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

  error('Not implemented')
  
end

function [df,minf,maxf] = init_lgp_norm(yy,myy_i,sigm2_i,myy)
%INIT_LGP_NORM
%
%  Description
%    Return function handle to a function evaluating LGP *
%    Gaussian which is used for evaluating (likelihood * cavity)
%    or (likelihood * posterior) Return also useful limits for
%    integration. This is private function for lik_lgp. This 
%    subfunction is needed by sufunctions tiltedMoments, siteDeriv 
%    and predy.
%  
%  See also
%    LIK_LGP_TILTEDMOMENTS, LIK_LGP_PREDY
  
% Not applicable

end

function mu = lik_lgp_invlink(lik, f, z)
%LIK_LGP_INVLINK  Returns values of inverse link function
%             
%  Description 
%    P = LIK_LGP_INVLINK(LIK, F) takes a likelihood structure LIK and
%    latent values F and returns the values MU of inverse link function.
%    This subfunction is needed when using function gp_predprctmu.
%
%     See also
%     LIK_LGP_LL, LIK_LGP_PREDY
  
  mu = exp(f);
  mu = mu./sum(mu);
  
end

function reclik = lik_lgp_recappend(reclik, ri, lik)
%RECAPPEND  Append the parameters to the record
%
%  Description 
%    RECLIK = LIK_LGP_RECAPPEND(RECLIK, RI, LIK) takes a
%    likelihood record structure RECLIK, record index RI and
%    likelihood structure LIK with the current MCMC samples of
%    the parameters. Returns RECLIK which contains all the old
%    samples and the current samples from LIK. This subfunction 
%    is needed when using MCMC sampling (gp_mc).
% 
%  See also
%    GP_MC

  if nargin == 2
    reclik.type = 'LGP';

    % Set the function handles
    reclik.fh.pak = @lik_lgp_pak;
    reclik.fh.unpak = @lik_lgp_unpak;
    reclik.fh.ll = @lik_lgp_ll;
    reclik.fh.llg = @lik_lgp_llg;    
    reclik.fh.llg2 = @lik_lgp_llg2;
    reclik.fh.llg3 = @lik_lgp_llg3;
    reclik.fh.tiltedMoments = @lik_lgp_tiltedMoments;
    reclik.fh.predy = @lik_lgp_predy;
    reclik.fh.invlink = @lik_lgp_invlink;
    reclik.fh.recappend = @lik_lgp_recappend;
    return
  end
end
