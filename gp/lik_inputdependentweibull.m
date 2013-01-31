function lik = lik_inputdependentweibull(varargin)
%LIK_INPUTDEPENDENTWEIBULL    Create a right censored input dependent Weibull likelihood structure 
%
%  Description
%    LIK = LIK_INPUTDEPENDENTWEIBULL('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates a likelihood structure for right censored input dependent
%    Weibull survival model in which the named parameters have the
%    specified values. Any unspecified parameters are set to default 
%    values.
%  
%    LIK = LIK_INPUTDEPENDENTWEIBULL(LIK,'PARAM1',VALUE1,'PARAM2,VALUE2,...)
%    modify a likelihood structure with the named parameters
%    altered with the specified values.
%
%    Parameters for Weibull likelihood [default]
%      shape       - shape parameter r [1]
%      shape_prior - prior for shape [prior_logunif]
%  
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%    The likelihood is defined as follows:
%                  __ n
%      p(y|f1,f2, z) = || i=1 [ (r*exp(f2_i))^(1-z_i) exp( (1-z_i)*(-f1_i)
%                           +(1-z_i)*((r*exp(f2_i))-1)*log(y_i)
%                           -exp(-f1_i)*y_i^(r*exp(f2_i))) ]
%
%    where r is the shape parameter of Weibull distribution.
%    z is a vector of censoring indicators with z = 0 for uncensored event
%    and z = 1 for right censored event. Here the second latent variable f2
%    implies the input dependance to the shape parameter in the original
%    Weibull likelihood.
%
%    When using the Weibull likelihood you need to give the vector z
%    as an extra parameter to each function that requires also y. 
%    For example, you should call gpla_e as follows: gpla_e(w, gp,
%    x, y, 'z', z)
%
%  See also
%    GP_SET, LIK_*, PRIOR_*
%

% Copyright (c) 2011 Jaakko Riihimäki
% Copyright (c) 2011 Aki Vehtari
% Copyright (c) 2012 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_INPUTDEPENDENTWEIBULL';
  ip.addOptional('lik', [], @isstruct);
  ip.addParamValue('shape',1, @(x) isscalar(x) && x>0);
  ip.addParamValue('shape_prior',prior_logunif(), @(x) isstruct(x) || isempty(x));
  ip.parse(varargin{:});
  lik=ip.Results.lik;
  
  if isempty(lik)
    init=true;
    lik.nondiagW=true;
    lik.type = 'Inputdependent-Weibull';
  else
    if ~isfield(lik,'type') && ~isequal(lik.type,'Weibull')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end

  % Initialize parameters
  if init || ~ismember('shape',ip.UsingDefaults)
    lik.shape = ip.Results.shape;
  end
  % Initialize prior structure
  if init
    lik.p=[];
  end
  if init || ~ismember('shape_prior',ip.UsingDefaults)
    lik.p.shape=ip.Results.shape_prior;
  end
  
  if init
    % Set the function handles to the subfunctions
    lik.fh.pak = @lik_inputdependentweibull_pak;
    lik.fh.unpak = @lik_inputdependentweibull_unpak;
    lik.fh.lp = @lik_inputdependentweibull_lp;
    lik.fh.lpg = @lik_inputdependentweibull_lpg;
    lik.fh.ll = @lik_inputdependentweibull_ll;
    lik.fh.llg = @lik_inputdependentweibull_llg;    
    lik.fh.llg2 = @lik_inputdependentweibull_llg2;
    lik.fh.llg3 = @lik_inputdependentweibull_llg3;
    lik.fh.invlink = @lik_inputdependentweibull_invlink;
    lik.fh.predy = @lik_inputdependentweibull_predy;
    lik.fh.recappend = @lik_inputdependentweibull_recappend;
  end

end

function [w,s] = lik_inputdependentweibull_pak(lik)
%LIK_INPUTDEPENDENTWEIBULL_PAK  Combine likelihood parameters into one vector.
%
%  Description 
%    W = LIK_INPUTDEPENDENTWEIBULL_PAK(LIK) takes a likelihood structure LIK and
%    combines the parameters into a single row vector W. This is a 
%    mandatory subfunction used for example in energy and gradient 
%    computations.
%     
%       w = log(lik.shape)
%
%   See also
%   LIK_INPUTDEPENDENTWEIBULL_UNPAK, GP_PAK
  
  w=[];s={};
  if ~isempty(lik.p.shape)
    w = log(lik.shape);
    s = [s; 'log(weibull.shape)'];
    [wh sh] = lik.p.shape.fh.pak(lik.p.shape);
    w = [w wh];
    s = [s; sh];
  end
end


function [lik, w] = lik_inputdependentweibull_unpak(lik, w)
%LIK_INPUTDEPENDENTWEIBULL_UNPAK  Extract likelihood parameters from the vector.
%
%  Description
%    [LIK, W] = LIK_INPUTDEPENDENTWEIBULL_UNPAK(W, LIK) takes a likelihood
%    structure LIK and extracts the parameters from the vector W
%    to the LIK structure. This is a mandatory subfunction used 
%    for example in energy and gradient computations.
%     
%   Assignment is inverse of  
%       w = log(lik.shape)
%
%   See also
%   LIK_INPUTDEPENDENTWEIBULL_PAK, GP_UNPAK

  if ~isempty(lik.p.shape)
    lik.shape = exp(w(1));
    w = w(2:end);
    [p, w] = lik.p.shape.fh.unpak(lik.p.shape, w);
    lik.p.shape = p;
  end
end


function lp = lik_inputdependentweibull_lp(lik, varargin)
%LIK_INPUTDEPENDENTWEIBULL_LP  log(prior) of the likelihood parameters
%
%  Description
%    LP = LIK_INPUTDEPENDENTWEIBULL_LP(LIK) takes a likelihood structure LIK and
%    returns log(p(th)), where th collects the parameters. This 
%    subfunction is needed when there are likelihood parameters.
%
%  See also
%    LIK_INPUTDEPENDENTWEIBULL_LLG, LIK_INPUTDEPENDENTWEIBULL_LLG3, LIK_INPUTDEPENDENTWEIBULL_LLG2, GPLA_E
  

% If prior for shape parameter, add its contribution
  lp=0;
  if ~isempty(lik.p.shape)
    lp = lik.p.shape.fh.lp(lik.shape, lik.p.shape) +log(lik.shape);
  end
  
end


function lpg = lik_inputdependentweibull_lpg(lik)
%LIK_INPUTDEPENDENTWEIBULL_LPG  d log(prior)/dth of the likelihood 
%                parameters th
%
%  Description
%    E = LIK_INPUTDEPENDENTWEIBULL_LPG(LIK) takes a likelihood structure LIK and
%    returns d log(p(th))/dth, where th collects the parameters.
%    This subfunction is needed when there are likelihood parameters.
%
%  See also
%    LIK_INPUTDEPENDENTWEIBULL_LLG, LIK_INPUTDEPENDENTWEIBULL_LLG3, LIK_INPUTDEPENDENTWEIBULL_LLG2, GPLA_G
  
  lpg=[];
  if ~isempty(lik.p.shape)            
    % Evaluate the gprior with respect to shape
    ggs = lik.p.shape.fh.lpg(lik.shape, lik.p.shape);
    lpg = ggs(1).*lik.shape + 1;
    if length(ggs) > 1
      lpg = [lpg ggs(2:end)];
    end
  end
end  

function ll = lik_inputdependentweibull_ll(lik, y, ff, z)
%LIK_INPUTDEPENDENTWEIBULL_LL  Log likelihood
%
%  Description
%    LL = LIK_INPUTDEPENDENTWEIBULL_LL(LIK, Y, F, Z) takes a likelihood
%    structure LIK, survival times Y, censoring indicators Z, and
%    latent values F. Returns the log likelihood, log p(y|f,z).
%    This subfunction is needed when using Laplace approximation 
%    or MCMC for inference with non-Gaussian likelihoods. This 
%    subfunction is also used in information criteria (DIC, WAIC) 
%    computations.
%
%  See also
%    LIK_INPUTDEPENDENTWEIBULL_LLG, LIK_INPUTDEPENDENTWEIBULL_LLG3, LIK_INPUTDEPENDENTWEIBULL_LLG2, GPLA_E
  
  if isempty(z)
    error(['lik_inputdependentweibull -> lik_inputdependentweibull_ll: missing z!    '... 
           'Weibull likelihood needs the censoring    '...
           'indicators as an extra input z. See, for         '...
           'example, lik_inputdependentweibull and gpla_e.               ']);
  end

  f=ff(:);
  n=size(y,1);
  f1=f(1:n);
  f2=f((n+1):2*n);
  expf2=exp(f2);
  expf2(isinf(expf2))=realmax;
  a = lik.shape;
  ll = sum((1-z).*(log(a*expf2) + (a*expf2-1).*log(y)-f1) - exp(-f1).*y.^(a*expf2));

end

function llg = lik_inputdependentweibull_llg(lik, y, ff, param, z)
%LIK_INPUTDEPENDENTWEIBULL_LLG  Gradient of the log likelihood
%
%  Description 
%    LLG = LIK_INPUTDEPENDENTWEIBULL_LLG(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, survival times Y, censoring indicators Z and
%    latent values F. Returns the gradient of the log likelihood
%    with respect to PARAM. At the moment PARAM can be 'param' or
%    'latent'. This subfunction is needed when using Laplace 
%    approximation or MCMC for inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_INPUTDEPENDENTWEIBULL_LL, LIK_INPUTDEPENDENTWEIBULL_LLG2, LIK_INPUTDEPENDENTWEIBULL_LLG3, GPLA_E

  if isempty(z)
    error(['lik_inputdependentweibull -> lik_inputdependentweibull_llg: missing z!    '... 
           'Weibull likelihood needs the censoring    '...
           'indicators as an extra input z. See, for         '...
           'example, lik_inputdependentweibull and gpla_e.               ']);
  end

  f=ff(:);
  n=size(y,1);
  f1=f(1:n);
  f2=f((n+1):2*n);
  expf2=exp(f2);
  expf2(isinf(expf2))=realmax;
  a = lik.shape;
  switch param
    case 'param'      
      llg = sum((1-z).*(1./a + expf2.*log(y)) - exp(-f1).*y.^(a.*expf2).*log(y).*expf2);
      % correction for the log transformation
      llg = llg.*lik.shape;
    case 'latent'
      llg1 = -(1-z) + exp(-f1).*y.^(a.*expf2);
      llg2 = (1-z).*(1 + a.*expf2.*log(y)) - exp(-f1).*y.^(a.*expf2).*log(y).*a.*expf2;
      llg = [llg1; llg2];
  end
end

function llg2 = lik_inputdependentweibull_llg2(lik, y, ff, param, z)
%LIK_INPUTDEPENDENTWEIBULL_LLG2  Second gradients of the log likelihood
%
%  Description        
%    LLG2 = LIK_INPUTDEPENDENTWEIBULL_LLG2(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, survival times Y, censoring indicators Z, and
%    latent values F. Returns the hessian of the log likelihood
%    with respect to PARAM. At the moment PARAM can be only
%    'latent'. LLG2 is a vector with diagonal elements of the
%    Hessian matrix (off diagonals are zero). This subfunction 
%    is needed when using Laplace approximation or EP for 
%    inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_INPUTDEPENDENTWEIBULL_LL, LIK_INPUTDEPENDENTWEIBULL_LLG, LIK_INPUTDEPENDENTWEIBULL_LLG3, GPLA_E

  if isempty(z)
    error(['lik_inputdependentweibull -> lik_inputdependentweibull_llg2: missing z!   '... 
           'Weibull likelihood needs the censoring   '...
           'indicators as an extra input z. See, for         '...
           'example, lik_inputdependentweibull and gpla_e.               ']);
  end

  a = lik.shape;
  f=ff(:);
  n=size(y,1);
  f1=f(1:n);
  f2=f((n+1):2*n);
  expf2=exp(f2);
  expf2(isinf(expf2))=realmax;
  switch param
    case 'param'
      
    case 'latent'
      t1=exp(-f1).*y.^(a.*expf2);
      t2=log(y).*a.*expf2;
      t3=t1.*t2;
      
      llg2_11 = -t1;
      llg2_12 = t3;
      llg2_22 = (1-z).*t2 - (t2 + 1).*t3;
      
      llg2 = [llg2_11 llg2_12; llg2_12 llg2_22];

    case 'latent+param'      
      t1=expf2.*log(y);
      t2=exp(-f1).*y.^(a.*expf2);
      t3=t1.*t2;
      llg2_1 = t3;
      llg2_2 = (1-z).*t1 - (t1.*a + 1).*t3;
      llg2 = [llg2_1; llg2_2];
      % correction due to the log transformation
      llg2 = llg2.*lik.shape;
  end
end    

function llg3 = lik_inputdependentweibull_llg3(lik, y, ff, param, z)
%LIK_INPUTDEPENDENTWEIBULL_LLG3  Third gradients of the log likelihood
%
%  Description
%    LLG3 = LIK_INPUTDEPENDENTWEIBULL_LLG3(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, survival times Y, censoring indicators Z and
%    latent values F and returns the third gradients of the log
%    likelihood with respect to PARAM. At the moment PARAM can be
%    only 'latent'. LLG3 is a vector with third gradients. This 
%    subfunction is needed when using Laplace approximation for 
%    inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_INPUTDEPENDENTWEIBULL_LL, LIK_INPUTDEPENDENTWEIBULL_LLG, LIK_INPUTDEPENDENTWEIBULL_LLG2, GPLA_E, GPLA_G

  if isempty(z)
    error(['lik_inputdependentweibull -> lik_inputdependentweibull_llg3: missing z!   '... 
           'Weibull likelihood needs the censoring    '...
           'indicators as an extra input z. See, for         '...
           'example, lik_inputdependentweibull and gpla_e.               ']);
  end

  a = lik.shape;
  f=ff(:);
  n=size(y,1);
  f1=f(1:n);
  f2=f((n+1):2*n);
  expf2=exp(f2);
  expf2(isinf(expf2))=realmax;
  switch param
    case 'param'
      
    case 'latent'
      t1=a.*expf2.*log(y);
      t2=exp(-f1).*y.^(a.*expf2);
      t3=t2.*t1;
      t4=t3.*t1;
      
      nl=2;
      llg3=zeros(nl,nl,nl,n);

      llg3(1,1,1,:) = t2;
      
      llg3(2,2,1,:) = t4 + t3;
      llg3(2,1,2,:) = llg3(2,2,1,:);
      llg3(1,2,2,:) = llg3(2,2,1,:);
      
      llg3(2,1,1,:) = -t3;
      llg3(1,2,1,:) = llg3(2,1,1,:);
      llg3(1,1,2,:) = llg3(2,1,1,:);
      
      llg3(2,2,2,:) = (1-z).*t1 - t4.*t1 - 3.*t4 - t3;
      
    case 'latent2+param'
      t1 = log(y).*expf2;
      t2 = exp(-f1).*y.^(a*expf2);
      t3 = t2.*t1;
      t4 = t3.*t1;
      llg3_11 = -t3;
      llg3_12 = a.*t4 + t3;
      llg3_22 = (1-z).*t1 - a.^2.*t4.*t1 - 3.*a.*t4 - t3;
      
      llg3 = [diag(llg3_11) diag(llg3_12); diag(llg3_12) diag(llg3_22)];
      % correction due to the log transformation
      llg3 = llg3.*lik.shape;
  end
end

% function [logM_0, m_1, sigm2hati1] = lik_inputdependentweibull_tiltedMoments(lik, y, i1, sigm2_i, myy_i, z)
% %LIK_INPUTDEPENDENTWEIBULL_TILTEDMOMENTS  Returns the marginal moments for EP algorithm
% %
% %  Description
% %    [M_0, M_1, M2] = LIK_INPUTDEPENDENTWEIBULL_TILTEDMOMENTS(LIK, Y, I, S2,
% %    MYY, Z) takes a likelihood structure LIK, survival times
% %    Y, censoring indicators Z, index I and cavity variance S2 and
% %    mean MYY. Returns the zeroth moment M_0, mean M_1 and
% %    variance M_2 of the posterior marginal (see Rasmussen and
% %    Williams (2006): Gaussian processes for Machine Learning,
% %    page 55). This subfunction is needed when using EP for 
% %    inference with non-Gaussian likelihoods.
% %
% %  See also
% %    GPEP_E
%   
%   if isempty(z)
%     error(['lik_inputdependentweibull -> lik_inputdependentweibull_tiltedMoments: missing z!'... 
%            'Weibull likelihood needs the censoring            '...
%            'indicators as an extra input z. See, for                 '...
%            'example, lik_inputdependentweibull and gpep_e.                       ']);
%   end
%   
%   yy = y(i1);
%   yc = 1-z(i1);
%   r = lik.shape;
%   
%   % get a function handle of an unnormalized tilted distribution 
%   % (likelihood * cavity = Weibull * Gaussian)
%   % and useful integration limits
%   [tf,minf,maxf]=init_weibull_norm(yy,myy_i,sigm2_i,yc,r);
%   
%   % Integrate with quadrature
%   RTOL = 1.e-6;
%   ATOL = 1.e-10;
%   [m_0, m_1, m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
%   sigm2hati1 = m_2 - m_1.^2;
%   
%   % If the second central moment is less than cavity variance
%   % integrate more precisely. Theoretically for log-concave
%   % likelihood should be sigm2hati1 < sigm2_i.
%   if sigm2hati1 >= sigm2_i
%     ATOL = ATOL.^2;
%     RTOL = RTOL.^2;
%     [m_0, m_1, m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
%     sigm2hati1 = m_2 - m_1.^2;
%     if sigm2hati1 >= sigm2_i
%       error('lik_inputdependentweibull_tilted_moments: sigm2hati1 >= sigm2_i');
%     end
%   end
%   logM_0 = log(m_0);
%   
% end

% function [g_i] = lik_inputdependentweibull_siteDeriv(lik, y, i1, sigm2_i, myy_i, z)
% %LIK_INPUTDEPENDENTWEIBULL_SITEDERIV  Evaluate the expectation of the gradient
% %                      of the log likelihood term with respect
% %                      to the likelihood parameters for EP 
% %
% %  Description [M_0, M_1, M2] =
% %    LIK_INPUTDEPENDENTWEIBULL_SITEDERIV(LIK, Y, I, S2, MYY, Z) takes a
% %    likelihood structure LIK, survival times Y, expected
% %    counts Z, index I and cavity variance S2 and mean MYY. 
% %    Returns E_f [d log p(y_i|f_i) /d a], where a is the
% %    likelihood parameter and the expectation is over the
% %    marginal posterior. This term is needed when evaluating the
% %    gradients of the marginal likelihood estimate Z_EP with
% %    respect to the likelihood parameters (see Seeger (2008):
% %    Expectation propagation for exponential families). This 
% %    subfunction is needed when using EP for inference with 
% %    non-Gaussian likelihoods and there are likelihood parameters.
% %
% %  See also
% %    GPEP_G
% 
%   if isempty(z)
%     error(['lik_inputdependentweibull -> lik_inputdependentweibull_siteDeriv: missing z!'... 
%            'Weibull likelihood needs the censoring        '...
%            'indicators as an extra input z. See, for             '...
%            'example, lik_inputdependentweibull and gpla_e.                   ']);
%   end
% 
%   yy = y(i1);
%   yc = 1-z(i1);
%   r = lik.shape;
%   
%   % get a function handle of an unnormalized tilted distribution 
%   % (likelihood * cavity = Weibull * Gaussian)
%   % and useful integration limits
%   [tf,minf,maxf]=init_weibull_norm(yy,myy_i,sigm2_i,yc,r);
%   % additionally get function handle for the derivative
%   td = @deriv;
%   
%   % Integrate with quadgk
%   [m_0, fhncnt] = quadgk(tf, minf, maxf);
%   [g_i, fhncnt] = quadgk(@(f) td(f).*tf(f)./m_0, minf, maxf);
%   g_i = g_i.*r;
% 
%   function g = deriv(f)
%     g = yc.*(1./r + log(yy)) - exp(-f).*yy.^r.*log(yy);
%   end
% end

function [lpy, Ey, Vary] = lik_inputdependentweibull_predy(lik, Ef, Varf, yt, zt)
%LIK_INPUTDEPENDENTWEIBULL_PREDY  Returns the predictive mean, variance and density of y
%
%  Description   
%    LPY = LIK_INPUTDEPENDENTWEIBULL_PREDY(LIK, EF, VARF YT, ZT)
%    Returns logarithm of the predictive density PY of YT, that is 
%        p(yt | zt) = \int p(yt | f, zt) p(f|y) df.
%    This requires also the survival times YT, censoring indicators ZT.
%    This subfunction is needed when computing posterior predictive 
%    distributions for future observations.
%
%    [LPY, EY, VARY] = LIK_INPUTDEPENDENTWEIBULL_PREDY(LIK, EF, VARF) takes a
%    likelihood structure LIK, posterior mean EF and posterior
%    Variance VARF of the latent variable and returns the
%    posterior predictive mean EY and variance VARY of the
%    observations related to the latent variables. This subfunction
%    is needed when computing posterior predictive distributions for 
%    future observations.
%        
%
%  See also
%    GPLA_PRED, GPEP_PRED, GPMC_PRED

  if isempty(zt)
    error(['lik_inputdependentweibull -> lik_inputdependentweibull_predy: missing zt!'... 
           'Weibull likelihood needs the censoring    '...
           'indicators as an extra input zt. See, for         '...
           'example, lik_inputdependentweibull and gpla_e.               ']);
  end

  yc = 1-zt;
  r = lik.shape;
  
  Ef = Ef(:);
  ntest = 0.5*size(Ef,1);
  Ef1=Ef(1:ntest); Ef2=Ef(ntest+1:end);
  %     Varf1=squeeze(Varf(1,1,:)); Varf2=squeeze(Varf(2,2,:));
  if size(Varf,2) == size(Varf,1)
    Varf1=diag(Varf(1:ntest,1:ntest));Varf2=diag(Varf(ntest+1:end,ntest+1:end));
  else
    Varf1=Varf(:,1); Varf2=Varf(:,2);
  end
  
  Ey=[];
  Vary=[];

  % Evaluate the posterior predictive densities of the given observations
  lpy = zeros(length(yt),1);
    
  for i2=1:ntest
    m1=Ef1(i2); m2=Ef2(i2);
    s1=sqrt(Varf1(i2)); s2=sqrt(Varf2(i2));
    % Function handle for Weibull * Gaussian_f1 * Gaussian_f2
    pd=@(f1,f2) exp(yc(i2).*((log(r) + f2) + (r.*exp(f2)-1).*log(yt(i2))-f1) - exp(-f1).*yt(i2).^(r*exp(f2))) ...
      .*norm_pdf(f1,Ef1(i2),sqrt(Varf1(i2))).*norm_pdf(f2,Ef2(i2),sqrt(Varf2(i2)));
    % Integrate over latent variables
    lpy(i2) = log(dblquad(pd, m1-6.*s1, m1+6.*s1, m2-6.*s2, m2+6.*s2));
  end
  
%   for i1=1:length(yt)   
%     % get a function handle of the likelihood times posterior
%     % (likelihood * posterior = Weibull * Gaussian)
%     % and useful integration limits
%     [pdf,minf,maxf]=init_weibull_norm(...
%       yt(i1),Ef(i1),Varf(i1),yc(i1),r);
%     % integrate over the f to get posterior predictive distribution
%     lpy(i1) = log(quadgk(pdf, minf, maxf));
%   end
end

% function [df,minf,maxf] = init_weibull_norm(yy,myy_i,sigm2_i,yc,r)
% %INIT_WEIBULL_NORM
% %
% %  Description
% %    Return function handle to a function evaluating
% %    Weibull * Gaussian which is used for evaluating
% %    (likelihood * cavity) or (likelihood * posterior) Return
% %    also useful limits for integration. This is private function
% %    for lik_inputdependentweibull. This subfunction is needed by subfunctions
% %    tiltedMoments, siteDeriv and predy.
% %  
% %  See also
% %    LIK_INPUTDEPENDENTWEIBULL_TILTEDMOMENTS, LIK_INPUTDEPENDENTWEIBULL_SITEDERIV,
% %    LIK_INPUTDEPENDENTWEIBULL_PREDY
%   
% % avoid repetitive evaluation of constant part
%   ldconst = yc*log(r)+yc*(r-1)*log(yy)...
%             - log(sigm2_i)/2 - log(2*pi)/2;
%   
%    
%   
%   % Create function handle for the function to be integrated
%   df = @weibull_norm;
%   % use log to avoid underflow, and derivates for faster search
%   ld = @log_weibull_norm;
%   ldg = @log_weibull_norm_g;
%   ldg2 = @log_weibull_norm_g2;
% 
%   % Set the limits for integration
%   if yc==0
%     % with yy==0, the mode of the likelihood is not defined
%     % use the mode of the Gaussian (cavity or posterior) as a first guess
%     modef = myy_i;
%   else
%     % use precision weighted mean of the Gaussian approximation
%     % of the Weibull likelihood and Gaussian
%     mu=-log(yc./(yy.^r));
%     %s2=1./(yc+1./sigm2_i);
%     s2=1./yc;
%     modef = (myy_i/sigm2_i + mu/s2)/(1/sigm2_i + 1/s2);
%   end
%   % find the mode of the integrand using Newton iterations
%   % few iterations is enough, since first guess is in the right direction
%   niter=4;       % number of Newton iterations
%   mindelta=1e-6; % tolerance in stopping Newton iterations
%   for ni=1:niter
%     g=ldg(modef);
%     h=ldg2(modef);
%     delta=-g/h;
%     modef=modef+delta;
%     if abs(delta)<mindelta
%       break
%     end
%   end
%   % integrand limits based on Gaussian approximation at mode
%   modes=sqrt(-1/h);
%   minf=modef-8*modes;
%   maxf=modef+8*modes;
%   modeld=ld(modef);
%   iter=0;
%   % check that density at end points is low enough
%   lddiff=20; % min difference in log-density between mode and end-points
%   minld=ld(minf);
%   step=1;
%   while minld>(modeld-lddiff)
%     minf=minf-step*modes;
%     minld=ld(minf);
%     iter=iter+1;
%     step=step*2;
%     if iter>100
%       error(['lik_inputdependentweibull -> init_weibull_norm: ' ...
%              'integration interval minimun not found ' ...
%              'even after looking hard!'])
%     end
%   end
%   maxld=ld(maxf);
%   step=1;
%   while maxld>(modeld-lddiff)
%     maxf=maxf+step*modes;
%     maxld=ld(maxf);
%     iter=iter+1;
%     step=step*2;
%     if iter>100
%       error(['lik_inputdependentweibull -> init_weibull_norm: ' ...
%              'integration interval maximun not found ' ...
%              'even after looking hard!'])
%     end
%   end
%   
%   function integrand = weibull_norm(f)
%   % Weibull * Gaussian
%     integrand = exp(ldconst ...
%                     -yc.*f -exp(-f).*yy.^r ...
%                     -0.5*(f-myy_i).^2./sigm2_i);
%   end
% 
%   function log_int = log_weibull_norm(f)
%   % log(Weibull * Gaussian)
%   % log_weibull_norm is used to avoid underflow when searching
%   % integration interval
%     log_int = ldconst ...
%               -yc.*f -exp(-f).*yy.^r ...
%               -0.5*(f-myy_i).^2./sigm2_i;
%   end
% 
%   function g = log_weibull_norm_g(f)
%   % d/df log(Weibull * Gaussian)
%   % derivative of log_weibull_norm
%     g = -yc + exp(-f).*yy.^r ...
%         + (myy_i - f)./sigm2_i;
%   end
% 
%   function g2 = log_weibull_norm_g2(f)
%   % d^2/df^2 log(Weibull * Gaussian)
%   % second derivate of log_weibull_norm
%     g2 = - exp(-f).*yy.^r ...
%          -1/sigm2_i;
%   end
% 
% end

% function cdf = lik_inputdependentweibull_predcdf(lik, Ef, Varf, yt)
% %LIK_INPUTDEPENDENTWEIBULL_PREDCDF  Returns the predictive cdf evaluated at yt 
% %
% %  Description   
% %    CDF = LIK_INPUTDEPENDENTWEIBULL_PREDCDF(LIK, EF, VARF, YT)
% %    Returns the predictive cdf evaluated at YT given likelihood
% %    structure LIK, posterior mean EF and posterior Variance VARF
% %    of the latent variable. This subfunction is needed when using
% %    functions gp_predcdf or gp_kfcv_cdf.
% %
% %  See also
% %    GP_PREDCDF
% 
%   r = lik.shape;
%   
%   % Evaluate the posterior predictive cdf at given yt
%   cdf = zeros(length(yt),1);
%   for i1=1:length(yt)
%     % Get a function handle of the likelihood times posterior
%     % (likelihood * posterior = Weibull * Gaussian)
%     % and useful integration limits.
%     % yc=0 when evaluating predictive cdf
%     [sf,minf,maxf]=init_weibull_norm(...
%       yt(i1),Ef(i1),Varf(i1),0,r);
%     % integrate over the f to get posterior predictive distribution
%     cdf(i1) = 1-quadgk(sf, minf, maxf);
%   end
% end

function p = lik_inputdependentweibull_invlink(lik, f)
%LIK_INPUTDEPENDENTWEIBULL Returns values of inverse link function
%             
%  Description 
%    P = LIK_INPUTDEPENDENTWEIBULL_INVLINK(LIK, F) takes a likelihood structure LIK and
%    latent values F and returns the values of inverse link function P.
%    This subfunction is needed when using function gp_predprctmu.
%
%     See also
%     LIK_INPUTDEPENDENTWEIBULL_LL, LIK_INPUTDEPENDENTWEIBULL_PREDY

p = exp(f);
end

function reclik = lik_inputdependentweibull_recappend(reclik, ri, lik)
%RECAPPEND  Append the parameters to the record
%
%  Description 
%    RECLIK = GPCF_WEIBULL_RECAPPEND(RECLIK, RI, LIK) takes a
%    likelihood record structure RECLIK, record index RI and
%    likelihood structure LIK with the current MCMC samples of
%    the parameters. Returns RECLIK which contains all the old
%    samples and the current samples from LIK. This subfunction
%    is needed when using MCMC sampling (gp_mc).
% 
%  See also
%    GP_MC

  if nargin == 2
    % Initialize the record
    reclik.type = 'Inputdependent-Weibull';
    reclik.nondiagW=true;

    % Initialize parameter
%     reclik.shape = [];

    % Set the function handles
    reclik.fh.pak = @lik_inputdependentweibull_pak;
    reclik.fh.unpak = @lik_inputdependentweibull_unpak;
    reclik.fh.lp = @lik_t_lp;
    reclik.fh.lpg = @lik_t_lpg;
    reclik.fh.ll = @lik_inputdependentweibull_ll;
    reclik.fh.llg = @lik_inputdependentweibull_llg;    
    reclik.fh.llg2 = @lik_inputdependentweibull_llg2;
    reclik.fh.llg3 = @lik_inputdependentweibull_llg3;
    reclik.fh.invlink = @lik_inputdependentweibull_invlink;
    reclik.fh.predy = @lik_inputdependentweibull_predy;
    reclik.fh.recappend = @lik_inputdependentweibull_recappend;
    reclik.p=[];
    reclik.p.shape=[];
    if ~isempty(ri.p.shape)
      reclik.p.shape = ri.p.shape;
    end
  else
    % Append to the record
    reclik.shape(ri,:)=lik.shape;
    if ~isempty(lik.p.shape)
      reclik.p.shape = lik.p.shape.fh.recappend(reclik.p.shape, ri, lik.p.shape);
    end
  end
end

