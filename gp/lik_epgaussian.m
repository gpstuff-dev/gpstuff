function lik = lik_epgaussian(varargin)
%LIK_EPGAUSSIAN  Create a EP-Gaussian likelihood structure 
%
%  Description
%    LIK = LIK_EPGAUSSIAN creates EP-Gaussian likelihood structure used
%    in models with input dependent noise/magnitude.
%
%  See also
%    GP_SET, LIK_*
%

% Copyright (c) 2013 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_EPGAUSSIAN';  
  ip.addOptional('lik', [], @isstruct);
  ip.addParamValue('sigma2',0.1, @(x) isscalar(x) && x>0);
  ip.addParamValue('sigma2_prior',prior_fixed(), @(x) isstruct(x) || isempty(x));
  ip.addParamValue('int_likparam', true, @(x) islogical(x));
  ip.addParamValue('inputparam',false, @(x) islogical(x));
  ip.addParamValue('int_magnitude',false, @(x) islogical(x));
  ip.addParamValue('inputmagnitude',false, @(x) islogical(x));
  ip.parse(varargin{:});
  lik=ip.Results.lik;

  if isempty(lik)
    init=true;
    lik.type = 'EP-Gaussian';
  else
    if ~isfield(lik,'type') || ~isequal(lik.type,'EP-Gaussian')
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
  if init || ~ismember('int_likparam', ip.UsingDefaults)
    lik.int_likparam=ip.Results.int_likparam;
  end
  if init || ~ismember('inputparam', ip.UsingDefaults)
    lik.inputparam=ip.Results.inputparam;
  end
  if init || ~ismember('int_magnitude', ip.UsingDefaults)
    lik.int_magnitude=ip.Results.int_magnitude;
  end
  if init || ~ismember('inputmagnitude', ip.UsingDefaults)
    lik.inputmagnitude=ip.Results.inputmagnitude;
  end

  if init
    % Set the function handles to the subfunctions
    lik.fh.pak = @lik_epgaussian_pak;
    lik.fh.unpak = @lik_epgaussian_unpak;
    lik.fh.lp = @lik_epgaussian_lp;
    lik.fh.lpg = @lik_epgaussian_lpg;
    lik.fh.ll = @lik_epgaussian_ll;
    lik.fh.llg = @lik_epgaussian_llg;    
    lik.fh.llg2 = @lik_epgaussian_llg2;
    lik.fh.llg3 = @lik_epgaussian_llg3;
    lik.fh.siteDeriv = @lik_epgaussian_siteDeriv;
    lik.fh.tiltedMoments = @lik_epgaussian_tiltedMoments;
    lik.fh.predy = @lik_epgaussian_predy;
    lik.fh.predprcty = @lik_epgaussian_predprcty;
    lik.fh.invlink = @lik_epgaussian_invlink;
    lik.fh.recappend = @lik_epgaussian_recappend;
  end

end

function [w, s, h] = lik_epgaussian_pak(lik)
%LIK_EPGAUSSIAN_PAK  Combine likelihood parameters into one vector.
%
%  Description 
%    W = LIK_EPGAUSSIAN_PAK(LIK) takes a likelihood structure LIK
%    and returns an empty verctor W. If EP-Gaussian likelihood had
%    parameters this would combine them into a single row vector
%    W (see e.g. likelih_negbin). This is a mandatory subfunction 
%    used for example in energy and gradient computations.
%
%  See also
%    LIK_EPGAUSSIAN_UNPAK, GP_PAK

  w = []; s = {}; h=[];
  if ~isempty(lik.p.sigma2)
    w = log(lik.sigma2);
    s = [s; 'log(epgaussian.sigma2)'];
    h = [h 0];
    [wh, sh, hh] = lik.p.sigma2.fh.pak(lik.p.sigma2);
    w = [w wh];
    s = [s; sh];
    h = [h hh];
  end
end


function [lik, w] = lik_epgaussian_unpak(lik, w)
%LIK_EPGAUSSIAN_UNPAK  Extract likelihood parameters from the vector.
%
%  Description
%    W = LIK_EPGAUSSIAN_UNPAK(W, LIK) Doesn't do anything.
% 
%    If EP-Gaussian likelihood had parameters this would extracts
%    them parameters from the vector W to the LIK structure. 
%    This is a mandatory subfunction used for example in energy 
%    and gradient computations.
%
%  See also
%    LIK_EPGAUSSIAN_PAK, GP_UNPAK
  
  if ~isempty(lik.p.sigma2)
    lik.sigma2 = exp(w(1));
    w = w(2:end);
    [p, w] = lik.p.sigma2.fh.unpak(lik.p.sigma2, w);
    lik.p.sigma2 = p;
  end
  
end

function lp = lik_epgaussian_lp(lik, varargin)
%LIK_NEGBIN_LP  log(prior) of the likelihood parameters
%
%  Description
%    LP = LIK_NEGBIN_LP(LIK) takes a likelihood structure LIK and
%    returns log(p(th)), where th collects the parameters. This
%    subfunction is needed if there are likelihood parameters.
%
%  See also
%    LIK_NEGBIN_LLG, LIK_NEGBIN_LLG3, LIK_NEGBIN_LLG2, GPLA_E
  

% If prior for dispersion parameter, add its contribution
  lp=0;
  if ~isempty(lik.p.sigma2)
    lp = lik.p.sigma2.fh.lp(lik.sigma2, lik.p.sigma2) +log(lik.sigma2);
  end
  
end


function lpg = lik_epgaussian_lpg(lik)
%LIK_NEGBIN_LPG  d log(prior)/dth of the likelihood 
%                parameters th
%
%  Description
%    E = LIK_NEGBIN_LPG(LIK) takes a likelihood structure LIK and
%    returns d log(p(th))/dth, where th collects the parameters.
%    This subfunction is needed if there are likelihood parameters.
%
%  See also
%    LIK_NEGBIN_LLG, LIK_NEGBIN_LLG3, LIK_NEGBIN_LLG2, GPLA_G
  
  lpg=[];
  if ~isempty(lik.p.sigma2)           
    lpg=0;
    % Evaluate the gprior with respect to disper
    ggs = lik.p.sigma2.fh.lpg(lik.sigma2, lik.p.sigma2);
    lpg = ggs(1).*lik.sigma2 + 1;
    if length(ggs) > 1
      lpg = [lpg ggs(2:end)];
    end
  end
end  



function ll = lik_epgaussian_ll(lik, y, f, z)
%LIK_EPGAUSSIAN_LL  Log likelihood
%
%  Description
%    LL = LIK_EPGAUSSIAN_LL(LIK, Y, F, Z) takes a likelihood
%    structure LIK, succes counts Y, numbers of trials Z, and
%    latent values F. Returns the log likelihood, log p(y|f,z).
%    This subfunction is needed when using Laplace approximation
%    or MCMC for inference with non-Gaussian likelihoods. This 
%    subfunction is also used in information criteria (DIC, WAIC)
%    computations.
%
%  See also
%    LIK_EPGAUSSIAN_LLG, LIK_EPGAUSSIAN_LLG3, LIK_EPGAUSSIAN_LLG2, GPLA_E
  
%   if isempty(z)
%     error(['lik_epgaussian -> lik_epgaussian_ll: missing z!'... 
%            'EP-Gaussian likelihood needs the expected number of   '...
%            'occurrences as an extra input z. See, for         '...
%            'example, lik_epgaussian and gpla_e.             ']);
%   end

%   sigma2=lik.sigma2;
%   ll=sum(norm_lpdf(y,f,sqrt(sigma2)));  

%   
%   expf = exp(f);
%   p = expf ./ (1+expf);
%   N = z;
%   ll =  sum(gammaln(N+1)-gammaln(y+1)-gammaln(N-y+1)+y.*log(p)+(N-y).*log(1-p));
end


function llg = lik_epgaussian_llg(lik, y, f, param, z)
%LIK_EPGAUSSIAN_LLG    Gradient of the log likelihood
%
%  Description 
%    LLG = LIK_EPGAUSSIAN_LLG(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, succes counts Y, numbers of trials Z and
%    latent values F. Returns the gradient of the log likelihood
%    with respect to PARAM. At the moment PARAM can be 'param' or
%    'latent'. This subfunction is needed when using Laplace 
%    approximation or MCMC for inference with non-Gaussian 
%    likelihoods.
%
%  See also
%    LIK_EPGAUSSIAN_LL, LIK_EPGAUSSIAN_LLG2, LIK_EPGAUSSIAN_LLG3, GPLA_E

%   if isempty(z)
%     error(['lik_epgaussian -> lik_epgaussian_llg: missing z!'... 
%            'EP-Gaussian likelihood needs the expected number of   '...
%            'occurrences as an extra input z. See, for         '...
%            'example, lik_epgaussian and gpla_e.             ']);
%   end
%   
%   switch param
%     case 'latent'
%       expf = exp(f);
%       N = z;
%       
%       llg = y./(1+expf) - (N-y).*expf./(1+expf);
%   end
end


function llg2 = lik_epgaussian_llg2(lik, y, f, param, z)
%LIK_EPGAUSSIAN_LLG2  Second gradients of the log likelihood
%
%  Description        
%    LLG2 = LIK_EPGAUSSIAN_LLG2(LIK, Y, F, PARAM) takes a
%    likelihood structure LIK, succes counts Y, numbers of trials
%    Z, and latent values F. Returns the Hessian of the log
%    likelihood with respect to PARAM. At the moment PARAM can be
%    only 'latent'. G2 is a vector with diagonal elements of the
%    Hessian matrix (off diagonals are zero). This subfunction
%    is needed when using Laplace approximation or EP for inference 
%    with non-Gaussian likelihoods.
%
%  See also
%    LIK_EPGAUSSIAN_LL, LIK_EPGAUSSIAN_LLG, LIK_EPGAUSSIAN_LLG3, GPLA_E

%   if isempty(z)
%     error(['lik_epgaussian -> lik_epgaussian_llg2: missing z!'... 
%            'EP-Gaussian likelihood needs the expected number of    '...
%            'occurrences as an extra input z. See, for          '...
%            'example, lik_epgaussian and gpla_e.              ']);
%   end
%   
%   switch param
%     case 'latent'
%       expf = exp(f);
%       N = z;
% 
%       llg2 = -N.*expf./(1+expf).^2;
%   end
end


function llg3 = lik_epgaussian_llg3(lik, y, f, param, z)
%LIK_EPGAUSSIAN_LLG3  Third gradients of the log likelihood
%
%  Description
%    LLG3 = LIK_EPGAUSSIAN_LLG3(LIK, Y, F, PARAM) takes a
%    likelihood structure LIK, succes counts Y, numbers of trials
%    Z and latent values F and returns the third gradients of the
%    log likelihood with respect to PARAM. At the moment PARAM
%    can be only 'latent'. G3 is a vector with third gradients.
%    This subfunction is needed when using Laplace appoximation 
%    for inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_EPGAUSSIAN_LL, LIK_EPGAUSSIAN_LLG, LIK_EPGAUSSIAN_LLG2, GPLA_E, GPLA_G
  
%   if isempty(z)
%     error(['lik_epgaussian -> lik_epgaussian_llg3: missing z!'... 
%            'EP-Gaussian likelihood needs the expected number of    '...
%            'occurrences as an extra input z. See, for          '...
%            'example, lik_epgaussian and gpla_e.              ']);
%   end
%   
%   switch param
%     case 'latent'
%       expf = exp(f);
%       N = z;
%       llg3 = N.*(expf.*(expf-1))./(1+expf).^3;
%   end
end

function [g_i] = lik_epgaussian_siteDeriv(lik, y, i1, sigm2_i, myy_i, z)
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

%   if isempty(z)
%     error(['lik_negbin -> lik_negbin_siteDeriv: missing z!'... 
%            'Negbin likelihood needs the expected number of        '...
%            'occurrences as an extra input z. See, for             '...
%            'example, lik_negbin and gpla_e.                   ']);
%   end

  yy = y(i1);
  s2 = lik.sigma2;
  
%   % get a function handle of an unnormalized tilted distribution 
%   % (likelihood * cavity = Negative-binomial * Gaussian)
%   % and useful integration limits
%   [tf,minf,maxf]=init_negbin_norm(yy,myy_i,sigm2_i,avgE,r);
%   % additionally get function handle for the derivative
%   td = @deriv;
%   
%   % Integrate with quadgk
%   [m_0, fhncnt] = quadgk(tf, minf, maxf);
%   [g_i, fhncnt] = quadgk(@(f) td(f).*tf(f)./m_0, minf, maxf);
%   g_i = g_i.*r;

  lik.sigma2=s2+1e-5;
  d1=lik.fh.tiltedMoments(lik,y,i1,sigm2_i,myy_i,z);
  lik.sigma2=s2-1e-5;
  d2=lik.fh.tiltedMoments(lik,y,i1,sigm2_i,myy_i,z);
  g_i=(d1-d2)./(2*1e-5);
  g_i=g_i.*s2;
  
  mu_i=myy_i;
  for i=1:length(i1)
    
    phat_phi=@(phi) 1./sqrt(2*pi.*(s2+sigm2_i(i,1)./exp(-phi))) ...
      .*exp(-1./(2.*(s2+sigm2_i(i,1)./exp(-phi))).*(yy(i)-exp(phi./2).*mu_i(i,1)).^2) ...
      .*norm_pdf(phi, mu_i(i,2), sqrt(sigm2_i(i,2)));
    
    minf=mu_i(i,:)-6.*sqrt(sigm2_i(i,:));
    maxf=mu_i(i,:)+6.*sqrt(sigm2_i(i,:));
    
    m_0=quad_moments(phat_phi, minf(2), maxf(2), 1e-12, 1e-12);
    
%     g_1=dblquad(@(f,phi) deriv(f,phi).*norm_pdf(yy(i), exp(0.5.*phi).*f, sqrt(s2)).*norm_pdf(f,mu_i(i,1),sqrt(sigm2_i(i,1))).* ...
%       norm_pdf(phi,mu_i(i,2),sqrt(sigm2_i(i,2))),minf(1),maxf(1),minf(2),maxf(2),1e-12);
    
    np=51;
    z=minf(2):(maxf(2)-minf(2))/(np-1):maxf(2);
    dz=z(2)-z(1);
    tmp=dz/3;
    c=ones(np,1);
    c([1 np])=tmp;
    c(2:2:np-1)=4*tmp;
    c(3:2:np-2)=2*tmp;
    f1=(exp(z)./s2+1./sigm2_i(i,1)).^-1.*(exp(z/2).*yy(i)./s2 ...
      + mu_i(i,1)./sigm2_i(i,1)).*phat_phi(z);
    f1=-f1.*1./s2.^2.*exp(0.5.*z).*yy(i);
    f2=(exp(z)./s2+1./sigm2_i(i,1)).^-1.*(1 + ...
      (exp(z)./s2+1./sigm2_i(i,1)).^-1.*(exp(z/2).*yy(i)./s2 ...
      + mu_i(i,1)./sigm2_i(i,1)).^2).*phat_phi(z);
    f2=f2.*1./(2*s2^2).*exp(z);
    f0=(-1./(2.*s2)+1./(2*s2^2).*yy(i).^2).*phat_phi(z);
    
    g_i=(f0+f1+f2)*c;
    
    g_i=(g_i./m_0).*s2;
    
  end
    

  function g = deriv(f,phi)
    g = -1./(2.*s2)+1./(2*s2.^2)*(exp(0.5.*phi).*f - yy(i)).^2;
  end
end

function [logM_0, m_1, sigm2hati1] = lik_epgaussian_tiltedMoments(lik, y, i1, sigm2_i, mu_i, z)
%LIK_EPGAUSSIAN_TILTEDMOMENTS  Returns the marginal moments for EP algorithm
%
%  Description
%    [M_0, M_1, M2] = LIK_EPGAUSSIAN_TILTEDMOMENTS(LIK, Y, I, S2,
%    MYY, Z) takes a likelihood structure LIK, succes counts Y,
%    numbers of trials Z, index I and cavity variance S2 and mean
%    MYY. Returns the zeroth moment M_0, mean M_1 and variance
%    M_2 of the posterior marginal (see Rasmussen and Williams
%    (2006): Gaussian processes for Machine Learning, page 55).
%    This subfunction is needed when using EP for inference with
%    non-Gaussian likelihoods.
%
%  See also
%    GPEP_E
  
%  if isempty(z)
%    error(['lik_epgaussian -> lik_epgaussian_tiltedMoments: missing z!'... 
%           'EP-Gaussian likelihood needs the expected number of               '...
%           'occurrences as an extra input z. See, for                     '...
%           'example, lik_epgaussian and gpla_e.                         ']);
%  end
  
  yy = y(i1);
  %N = z(i1);
  logM_0=zeros(size(yy));
  
  
  if ~isfield(lik, 'int_magnitude') || ~lik.int_magnitude
    m_1=zeros(size(yy,1),2);
    sigm2hati1=zeros(size(yy,1),2);
    for i=1:length(i1)
      % Create function handle for the function to be integrated
      % (likelihood * cavity) and useful integration limits
      
      % tf = phat_i(theta) = \int phat_i(f_i, theta) df_i
      tf = @(theta) 1./(2*pi).*(sigm2_i(i,1)+exp(theta)).^(-1/2) ...
        .*exp(-1./(2*(sigm2_i(i,1)+exp(theta))).*(yy(i)-mu_i(i,1)).^2) ...
        .*sigm2_i(i,2).^(-1/2).*exp(-1./(2.*sigm2_i(i,2)).*(theta-mu_i(i,2)).^2);
      
      minf=mu_i(i,:)-7.*sqrt(sigm2_i(i,:));
      maxf=mu_i(i,:)+7.*sqrt(sigm2_i(i,:));
      if any(~isreal(minf))
        logM_0(i)=0;
        m_1(i,:)=mu_i(i,:);
        sigm2hati1(i,:)=sigm2_i(i,:);
        continue;
      end
      
      % Integrate with quadrature
      RTOL = 1.e-10;
      ATOL = 1.e-10;
      [m_0, m_1(i,2), m_2] = quad_moments(tf,minf(2), maxf(2), RTOL, ATOL);
      sigm2hati1(i,2) = m_2 - m_1(i,2).^2;
      
      % If the second central moment is less than cavity variance
      % integrate more precisely. Theoretically for log-concave
      % likelihood should be sigm2hati1 < sigm2_i.
      if sigm2hati1(i,2) >= sigm2_i(i,2)
        ATOL = ATOL*1e-4;
        RTOL = RTOL*1e-4;
        [m_0tmp, m_1(i,2), m_2] = quad_moments(@(x) tf(x)/m_0, minf(2), maxf(2), RTOL, ATOL);
        m_0=m_0*m_0tmp;
        sigm2hati1(i,2) = m_2 - m_1(i,2).^2;
%         if sigm2hati1(i,2) >= sigm2_i(i,2)
%           sigm2hati1(i,2)=sigm2_i(i,2);
%         end
      end
      logM_0(i) = log(m_0);
      
      %     fh2=@(theta) 1./(1./sigm2_i(i,1)+1./exp(theta)).* ...
      %       (yy(i)./exp(theta) + mu_i(i,1)./sigm2_i(i,1)).*tf(theta);
      %         m_1(i,1)=quadgk(fh2,minf(2),maxf(2), 'RelTol',RTOL, 'AbsTol', ATOL);
      
      %     fh3=@(theta) ((1./(1/sigm2_i(i,1)+1./exp(theta)).* ...
      %       (yy(i)./exp(theta) + mu_i(i,1)./sigm2_i(i,1))).^2 + ...
      %       (1./(1/sigm2_i(i,1)+1./exp(theta)))).*tf(theta);
      %         m2=quadgk(fh3,minf(2),maxf(2), 'RelTol',RTOL, 'AbsTol', ATOL);
      
      np=101;
      z=minf(2):(maxf(2)-minf(2))/(np-1):maxf(2);
      dz=z(2)-z(1);
      tmp=dz/3;
      c=ones(np,1);
      c([1 np])=tmp;
      c(2:2:np-1)=4*tmp;
      c(3:2:np-2)=2*tmp;
      
      % fz = phat_i(theta) evalueted in grid z
      fz = 1./(2*pi) ...
        .*(sigm2_i(i,1)+exp(z)).^(-1/2).*exp(-1./(2*(sigm2_i(i,1)+exp(z))) ...
        .*(yy(i)-mu_i(i,1)).^2).*sigm2_i(i,2).^(-1/2).*exp(-1./(2.*sigm2_i(i,2)) ...
        .*(z-mu_i(i,2)).^2);
      % g1g2 = E(f_i | theta)
      % (g1g2).^2+g1 = E(f_i^2 | theta)
      g1 = 1./(1./sigm2_i(i,1)+1./exp(z));
      g2 = (yy(i)./exp(z) + mu_i(i,1)./sigm2_i(i,1));
      g1g2 = g1.*g2;
      
      % fz1 = E(f_i) = \int E(f_i | theta)phat_i(theta) dtheta
      fz1 =g1g2.*fz;
      % fz2 = E(f_i^2) = \int E(f_i.^2 | theta)phat_i(theta) dtheta
      fz2 =((g1g2).^2 + g1).*fz;
      
      m_1(i,1)=fz1*c;
      m2=fz2*c;
      
      m_1(i,1)=m_1(i,1)./m_0;
      sigm2hati1(i,1)= m2./m_0 - m_1(i,1).^2;
    end
  elseif ~isfield(lik, 'int_likparam') || ~lik.int_likparam
    
    s2=lik.sigma2;
    m_1=zeros(size(yy,1),2);
    sigm2hati1=zeros(size(yy,1),2);
    for i=1:length(i1)
      
      phat_phi=@(phi) 1./sqrt(2*pi.*(s2+sigm2_i(i,1)./exp(-phi))) ...
        .*exp(-1./(2.*(s2+sigm2_i(i,1)./exp(-phi))).*(yy(i)-exp(phi./2).*mu_i(i,1)).^2) ...
        .*norm_pdf(phi, mu_i(i,2), sqrt(sigm2_i(i,2)));
      
      minf=mu_i(i,:)-6.*sqrt(sigm2_i(i,:));
      maxf=mu_i(i,:)+6.*sqrt(sigm2_i(i,:));
      
      [m_0, m_1p, m_2p]=quad_moments(phat_phi, minf(2), maxf(2), 1e-12, 1e-12);
      
%       m_1f=dblquad(@(f,phi) f.*norm_pdf(yy(i), exp(0.5.*phi).*f, sqrt(s2)).*norm_pdf(f,mu_i(i,1),sqrt(sigm2_i(i,1))).* ...
%         norm_pdf(phi,mu_i(i,2),sqrt(sigm2_i(i,2))),minf(1),maxf(1),minf(2),maxf(2),1e-12);
%       m_2f=dblquad(@(f,phi) f.^2.*norm_pdf(yy(i), exp(0.5.*phi).*f, sqrt(s2)).*norm_pdf(f,mu_i(i,1),sqrt(sigm2_i(i,1))).* ...
%         norm_pdf(phi,mu_i(i,2),sqrt(sigm2_i(i,2))),minf(1),maxf(1),minf(2),maxf(2),1e-12);
%       m_1f=quadgk(@(phi) (exp(phi)./s2+1./sigm2_i(i,1)).^-1.*(exp(phi/2).*yy(i)./s2 ...
%         + mu_i(i,1)./sigm2_i(i,1)).*phat_phi(phi), minf(2),maxf(2),'RelTol',1e-12,'AbsTol',1e-12);
%       
%       m_2f=quadgk(@(phi) (exp(phi)./s2+1./sigm2_i(i,1)).^-1.*(1 + ...
%         (exp(phi)./s2+1./sigm2_i(i,1)).^-1.*(exp(phi/2).*yy(i)./s2 ...
%         + mu_i(i,1)./sigm2_i(i,1)).^2).*phat_phi(phi), minf(2),maxf(2),'RelTol',1e-12,'AbsTol',1e-12);
      
      np=51;
      z=minf(2):(maxf(2)-minf(2))/(np-1):maxf(2);
      dz=z(2)-z(1);
      tmp=dz/3;
      c=ones(np,1);
      c([1 np])=tmp;
      c(2:2:np-1)=4*tmp;
      c(3:2:np-2)=2*tmp;
      f1=(exp(z)./s2+1./sigm2_i(i,1)).^-1.*(exp(z/2).*yy(i)./s2 ...
        + mu_i(i,1)./sigm2_i(i,1)).*phat_phi(z);
      f2=(exp(z)./s2+1./sigm2_i(i,1)).^-1.*(1 + ...
        (exp(z)./s2+1./sigm2_i(i,1)).^-1.*(exp(z/2).*yy(i)./s2 ...
        + mu_i(i,1)./sigm2_i(i,1)).^2).*phat_phi(z);
      
      m_1f=f1*c;
      m_2f=f2*c;
      logM_0(i,:)=log(m_0);
      
      m_1(i,:)=[m_1f/m_0 m_1p];
      sigm2hati1(i,:)=[m_2f/m_0 m_2p] - m_1(i,:).^2;
      
%       if any(sigm2hati1(i,2) >= sigm2_i(i,2))
%         sigm2hati1(i,sigm2hati1(i,2) >= sigm2_i(i,2))=sigm2_i(i,sigm2hati1(i,2) >= sigm2_i(i,2));
%       end
      
      
    end
    
    
  else
    if ~isfield(lik, 'joint_mean_magnitude') || ~lik.joint_mean_magnitude
      m_1=zeros(size(yy,1),3);
      sigm2hati1=zeros(size(yy,1),3);
      for i=1:length(i1)
        phat_thetaphi=@(theta, phi) 1./sqrt(2*pi*(exp(theta)+exp(phi).*sigm2_i(i,1))) ...
          .*exp(-1./(2*(exp(theta)+exp(phi).*sigm2_i(i,1))).*(yy(i)-exp(0.5.*phi).*mu_i(i,1)).^2) ...
          .*1./sqrt(2*pi*sigm2_i(i,2)).*exp(-1./(2*sigm2_i(i,2)).*(theta-mu_i(i,2)).^2) ...
          .*1./sqrt(2*pi*sigm2_i(i,3)).*exp(-1./(2*sigm2_i(i,3)).*(phi-mu_i(i,3)).^2);
        
        minf=mu_i(i,:)-[4 6 6].*sqrt(sigm2_i(i,:));
        maxf=mu_i(i,:)+[4 6 6].*sqrt(sigm2_i(i,:));
        if any(~isreal(minf))
          logM_0(i)=0;
          m_1(i,:)=mu_i(i,:);
          sigm2hati1(i,:)=sigm2_i(i,:);
          continue;
        end
        minf(2)=max(minf(2),-20); maxf(2)=min(maxf(2),20);
        minf(3)=max(minf(3),-10); maxf(3)=min(maxf(3),10);
        
        %       m_0=dblquad(phat_thetaphi, minf(2), maxf(2), minf(3), maxf(3), 1e-8);
        %       m_1(i,2)=dblquad(@(x,y) x.*phat_thetaphi(x,y), minf(2), maxf(2), minf(3), maxf(3), 1e-8);
        %       m_2t=dblquad(@(x,y) x.^2.*phat_thetaphi(x,y), minf(2), maxf(2), minf(3), maxf(3), 1e-8);
        %       m_1(i,3)=dblquad(@(x,y) y.*phat_thetaphi(x,y), minf(2), maxf(2), minf(3), maxf(3), 1e-8);
        %       m_2p=dblquad(@(x,y) y.^2.*phat_thetaphi(x,y), minf(2), maxf(2), minf(3), maxf(3), 1e-8);
        
        np=51;
        z1=minf(2):(maxf(2)-minf(2))/(np-1):maxf(2);
        dz1=z1(2)-z1(1);
        z2=minf(3):(maxf(3)-minf(3))/(np-1):maxf(3);
        dz2=z2(2)-z2(1);
        tmp=dz1*dz2/9;
        c=ones(np,1);
        c(2:2:np-1)=4;
        c(3:2:np-2)=2;
        cc=bsxfun(@times, c, c').*tmp;
        [tg,pg]=meshgrid(z1,z2);
        fz=1./sqrt(2*pi*sigm2_i(i,2)).*exp(-1./(2*sigm2_i(i,2)).*(tg-mu_i(i,2)).^2) ...
          .*1./sqrt(2*pi*sigm2_i(i,3)).*exp(-1./(2*sigm2_i(i,3)).*(pg-mu_i(i,3)).^2);
        nz= 1./sqrt(2*pi*(exp(tg)+exp(pg).*sigm2_i(i,1))) ...
          .*exp(-1./(2*(exp(tg)+exp(pg).*sigm2_i(i,1))).*(yy(i)-exp(0.5.*pg).*mu_i(i,1)).^2);
        %fz=phat_thetaphi(tg,pg);
        m_0=sum(sum(cc.*nz.*fz));
        m_1(i,2)=sum(sum(tg.*cc.*nz.*fz));
        m_1(i,3)=sum(sum(pg.*cc.*nz.*fz));
        m_2t=sum(sum(tg.^2.*cc.*nz.*fz));
        m_2p=sum(sum(pg.^2.*cc.*nz.*fz));
        
        logM_0(i,:)=log(m_0);
        %       sigm2hati1(i,2)=m_2t-m_1(i,2).^2;
        %       sigm2hati1(i,3)=m_2p-m_1(i,3).^2;
        
        %       m_1f = dblquad(@(x,y) ((1./sigm2_i(i,1)+exp(y-x)).^-1.*(yy(i)./exp(x-y) + mu_i(i,1)./sigm2_i(i,1))) ...
        %         .*phat_thetaphi(x,y), minf(2), maxf(2), minf(3), maxf(3), 1e-8);
        %       m_2f = dblquad(@(x,y) (1./sigm2_i(i,1)+exp(y-x)).^-1.*(1 ...
        %         + (1./sigm2_i(i,1)+exp(y-x)).^-1.*(yy(i)./exp(x-y) + mu_i(i,1)./sigm2_i(i,1)).^2) ...
        %         .*phat_thetaphi(x,y), minf(2), maxf(2), minf(3), maxf(3), 1e-8);
        
        %       f1g=(1./sigm2_i(i,1)+exp(pg-tg)).^-1.*(yy(i)./exp(tg-pg) + mu_i(i,1)./sigm2_i(i,1));
        %       f2g=(1./sigm2_i(i,1)+exp(pg-tg)).^-1 + f1g.^2;
        %       m_1(i,1) = sum(sum(f1g.*cc.*nz.*fz));
        %       m_2f = sum(sum(f2g.*cc.*nz.*fz));
        
        f1g=(1./sigm2_i(i,1)+exp(pg-tg)).^-1;
        f2g=f1g.*(exp(-0.5.*pg).*yy(i)./exp(tg-pg) + mu_i(i,1)./sigm2_i(i,1));
        g2=f1g + f2g.^2;
        m_1(i,1) = sum(sum(f2g.*cc.*nz.*fz));
        m_2f = sum(sum(g2.*cc.*nz.*fz));
        
        
        
        %       i
        %       xg=linspace(minf(1),maxf(1),50);
        %       pd = @(f) dblquad(@(theta,phi) norm_pdf(yy(i),exp(0.5.*phi).*f,sqrt(exp(theta))) ...
        %         .*norm_pdf(f,mu_i(i,1),sqrt(sigm2_i(i,1))) ...
        %         .*norm_pdf(theta, mu_i(i,2), sqrt(sigm2_i(i,2))) ...
        %         .*norm_pdf(phi, mu_i(i,3),sqrt(sigm2_i(i,3))), minf(2),maxf(2),minf(3),maxf(3),1e-8);
        %       tt=arrayfun(pd,xg);
        m_1(i,:) = m_1(i,:)/m_0;
        %       m_1(i,1) = sum(xg.*tt./sum(tt));
        %       m_2f = sum(xg.^2.*tt./sum(tt));
        sigm2hati1(i,1) = m_2f/m_0 - m_1(i,1).^2;
        sigm2hati1(i,2) = m_2t/m_0 - m_1(i,2).^2;
        sigm2hati1(i,3) = m_2p/m_0 - m_1(i,3).^2;
        %       m_1(i,1) = m_1f./m_0;
        %       sigm2hati1(i,1) = m_2f./m_0 - m_1(i,1).^2;
        
        %       if any(sigm2hati1(i,:) >= sigm2_i(i,:))
        %           sigm2hati1(i,sigm2hati1(i,:) >= sigm2_i(i,:))=sigm2_i(i,sigm2hati1(i,:) >= sigm2_i(i,:));
        %       end
        
      end
    else
      m_1=zeros(size(yy,1),3);
      sigm2hati1=zeros(size(yy,1),4);
      for i=1:length(i1)
        S=[sigm2_i(i,1) sigm2_i(i,4);sigm2_i(i,4) sigm2_i(i,3)];
        [~,np]=chol(S);
        minf=mu_i(i,1:3)-7.*sqrt(sigm2_i(i,1:3));
        maxf=mu_i(i,1:3)+7.*sqrt(sigm2_i(i,1:3));
        
        if any(~isreal(minf(1:3))) || any(minf(1:3)>maxf(1:3)) || np
          logM_0(i)=0;
          m_1(i,:)=mu_i(i,1:3);
          sigm2hati1(i,:)=sigm2_i(i,:);
          continue;
        end

        s=sqrt(sigm2_i(i,1)-sigm2_i(i,4).^2./sigm2_i(i,3));
        m=@(phi) mu_i(i,1) + sigm2_i(i,4)./sigm2_i(i,3)*(phi - mu_i(i,3));
        a=@(phi) exp(-0.5.*phi).*yy(i);
        b=@(theta,phi) sqrt(exp(theta-phi));
%         Ef_thetaphi=@(theta,phi) exp(-0.5.*phi).*(s.^2+b(theta,phi).^2).^(-1) ...
%           .*(m(phi).*b(theta,phi).^2 + a(phi).*s.^2).*norm_pdf(m(phi), a(phi), sqrt(s.^2+b(theta,phi).^2)) ...
%           .*norm_pdf(phi,mu_i(i,3), sqrt(sigm2_i(i,3))) ...
%           .*norm_pdf(theta,mu_i(i,2), sqrt(sigm2_i(i,2)));
        
%         Varf_thetaphi=@(theta,phi) exp(-0.5.*phi).*(s.^2+b(theta,phi).^2).^(-2) ...
%           .*(a(phi).^2.*s.^4 + b(theta,phi).^4.*(m(phi).^2+s.^2) + b(theta,phi).^2 ...
%           .*(2*a(phi).*m(phi).*s.^2+s.^4)).*norm_pdf(m(phi), a(phi), sqrt(s.^2+b(theta,phi).^2)) ...
%           .*norm_pdf(phi,mu_i(i,3), sqrt(sigm2_i(i,3))) ...
%           .*norm_pdf(theta,mu_i(i,2), sqrt(sigm2_i(i,2)));
        
%         m_1(i,1)=integral2(Ef_thetaphi, minf(2),maxf(2), minf(3), maxf(3), ...
%           'AbsTol',1e-6,'RelTol',1e-6);
%         m_2f=integral2(Varf_thetaphi, minf(2),maxf(2), minf(3), maxf(3), ...
%           'AbsTol',1e-6,'RelTol',1e-6);

        np=51;
        z1=minf(2):(maxf(2)-minf(2))/(np-1):maxf(2);
        dz1=z1(2)-z1(1);
        z2=minf(3):(maxf(3)-minf(3))/(np-1):maxf(3);
        dz2=z2(2)-z2(1);
        tmp=dz1*dz2/9;
        c=ones(np,1);
        c(2:2:np-1)=4;
        c(3:2:np-2)=2;
        cc=bsxfun(@times, c, c').*tmp;
        [tg,pg]=meshgrid(z1,z2);
        fz1=exp(-0.5.*pg).*(s.^2+b(tg,pg).^2).^(-1) ...
          .*(m(pg).*b(tg,pg).^2 + a(pg).*s.^2).*norm_pdf(m(pg), a(pg), sqrt(s.^2+b(tg,pg).^2)) ...
          .*norm_pdf(pg,mu_i(i,3), sqrt(sigm2_i(i,3))) ...
          .*norm_pdf(tg,mu_i(i,2), sqrt(sigm2_i(i,2)));
%         fz1=exp(-0.5.*pg - log(s.^2+b(tg,pg).^2) ...
%           + log(m(pg).*b(tg,pg).^2 + a(pg).*s.^2)+norm_lpdf(m(pg), a(pg), sqrt(s.^2+b(tg,pg).^2)) ...
%           + norm_lpdf(pg,mu_i(i,3), sqrt(sigm2_i(i,3))) ...
%           + norm_lpdf(tg,mu_i(i,2), sqrt(sigm2_i(i,2))));
        fz2= exp(-0.5.*pg).*(s.^2+b(tg,pg).^2).^(-2) ...
          .*(a(pg).^2.*s.^4 + b(tg,pg).^4.*(m(pg).^2+s.^2) + b(tg,pg).^2 ...
          .*(2*a(pg).*m(pg).*s.^2+s.^4)).*norm_pdf(m(pg), a(pg), sqrt(s.^2+b(tg,pg).^2)) ...
          .*norm_pdf(pg,mu_i(i,3), sqrt(sigm2_i(i,3))) ...
          .*norm_pdf(tg,mu_i(i,2), sqrt(sigm2_i(i,2)));
        
        m_1(i,1)=sum(sum(cc.*fz1));
        m_2f=sum(sum(cc.*fz2));

%         phat_f=@(f) integral2(@(theta,phi) norm_pdf(yy(i), exp(0.5.*phi).*f, exp(0.5.*theta)) ...
%           .*arrayfun(@(p) mvnpdf([f p], [mu_i(i,1) mu_i(i,3)], S), phi) ...
%           .*norm_pdf(theta, mu_i(i,2), sqrt(sigm2_i(i,2))),...
%           minf(2),maxf(2),minf(3),maxf(3), 'AbsTol',1e-8,'RelTol',1e-8);
        
%         xg=linspace(minf(1),maxf(1),100);
%         tt=arrayfun(phat_f, xg);
%         m_1(i,1)=sum(xg.*tt./sum(tt));
%         m_2f=sum(xg.^2.*tt./sum(tt));
        
%         phat_thetaphi=@(theta,phi) norm_pdf(theta,mu_i(i,2), sqrt(sigm2_i(i,2))) ...
%           .*norm_pdf(phi,mu_i(i,3), sqrt(sigm2_i(i,3))) ...
%           .*norm_pdf(yy(i), exp(0.5.*phi).*(mu_i(i,1)+sigm2_i(i,4)./sigm2_i(i,3)*(phi-mu_i(i,3))), ...
%           sigm2_i(i,1)-sigm2_i(i,4).^2/sigm2_i(i,3));
        
%         m_0=integral2(phat_thetaphi, minf(2),maxf(2),minf(3),maxf(3), ...
%           'AbsTol',1e-8,'RelTol',1e-8);
        
%         m_1(i,2)=integral2(@(a,b) a.*phat_thetaphi(a,b), minf(2),maxf(2),minf(3),maxf(3), ...
%           'AbsTol',1e-8,'RelTol',1e-8);
%         m_1(i,3)=integral2(@(a,b) b.*phat_thetaphi(a,b), minf(2),maxf(2),minf(3),maxf(3), ...
%           'AbsTol',1e-8,'RelTol',1e-8);
        
%         m_2t=integral2(@(a,b) a.^2.*phat_thetaphi(a,b), minf(2),maxf(2),minf(3),maxf(3), ...
%           'AbsTol',1e-8,'RelTol',1e-8);
%         m_2p=integral2(@(a,b) b.^2.*phat_thetaphi(a,b), minf(2),maxf(2),minf(3),maxf(3), ...
%           'AbsTol',1e-8,'RelTol',1e-8);
        
        fz3=norm_pdf(tg,mu_i(i,2), sqrt(sigm2_i(i,2))) ...
          .*norm_pdf(pg,mu_i(i,3), sqrt(sigm2_i(i,3))) ...
          .*norm_pdf(yy(i), exp(0.5.*pg).*(mu_i(i,1)+sigm2_i(i,4)./sigm2_i(i,3)*(pg-mu_i(i,3))), ...
          sqrt((sigm2_i(i,1)-sigm2_i(i,4).^2/sigm2_i(i,3))*exp(pg) + exp(tg)));
        
        m_0=sum(sum(cc.*fz3));
        m_1(i,2)=sum(sum(tg.*cc.*fz3));
        m_1(i,3)=sum(sum(pg.*cc.*fz3));
        m_2t=sum(sum(tg.^2.*cc.*fz3));
        m_2p=sum(sum(pg.^2.*cc.*fz3));
        
        logM_0(i,:)=log(m_0);
        m_1(i,:) = m_1(i,:)/m_0;
        sigm2hati1(i,1) = m_2f/m_0 - m_1(i,1).^2;
        sigm2hati1(i,2) = m_2t/m_0 - m_1(i,2).^2;
        sigm2hati1(i,3) = m_2p/m_0 - m_1(i,3).^2;
        
        phat=@(f,theta,phi) norm_pdf(yy(i), exp(0.5.*phi).*f, exp(0.5.*theta)) ...
          .*arrayfun(@(a,b) mvnpdf([a b], [mu_i(i,1) mu_i(i,3)], S), f, phi) ...
          .*norm_pdf(theta, mu_i(i,2), sqrt(sigm2_i(i,2)));
        
%         mc=integral3(@(a,b,c) a.*c.*phat(a,b,c), minf(1),maxf(1), ...
%           minf(2),maxf(2), minf(3),maxf(3), 'AbsTol',1e-5, 'RelTol',1e-4);
        
%         mc=integral2(@(theta,phi) phi.*Ef_thetaphi(theta,phi), ...
%           minf(2),maxf(2),minf(3),maxf(3), ...
%           'AbsTol',1e-6, 'RelTol',1e-6);

        mc=sum(sum(pg.*cc.*fz1));
        
        sigm2hati1(i,4)=(mc/m_0 - m_1(i,1).*m_1(i,3));
%         sigm2hati1(i,4)=0;


      end
    end
  end
end


function [lpy, Ey, Vary] = lik_epgaussian_predy(lik, Ef, Varf, yt, zt)
%LIK_EPGAUSSIAN_PREDY  Returns the predictive mean, variance and density of y
%
%  Description         
%    [LPY] = LIK_EPGAUSSIAN_PREDY(LIK, EF, VARF YT, ZT)
%    Returns logarithm of the predictive density PY of YT, that is 
%        p(yt | y, zt) = \int p(yt | f, zt) p(f|y) df.
%    This requires also the succes counts YT, numbers of trials ZT.
%    This subfunction is needed when computing posterior predictive 
%    distributions for future observations.
%
%    [LPY, EY, VARY] = LIK_EPGAUSSIAN_PREDY(LIK, EF, VARF) takes a
%    likelihood structure LIK, posterior mean EF and posterior
%    Variance VARF of the latent variable and returns the
%    posterior predictive mean EY and variance VARY of the
%    observations related to the latent variables. This subfunction 
%    is needed when computing posterior predictive distributions for 
%    future observations.
%        
%
%  See also 
%    GPEP_PRED, GPLA_PRED, GPMC_PRED

  if isempty(zt) && size(Ef,2)==1
    error(['lik_epgaussian -> lik_epgaussian_predy: missing zt!'... 
           'EP-Gaussian likelihood needs the distribution       '...
           'q(theta), where theta=log(sigma2) as an extra input', ...
           'zt. See, for example, lik_epgaussian and gpla_e.']);
  end
  
  if isfield(lik, 'joint_mean_magnitude') && lik.joint_mean_magnitude
%     C=Ef(:,1)+Varf(:,4)./Varf(:,3).*Ef(:,3);
%     B=C.^2+Varf(:,1)-Varf(:,4).^2./Varf(:,3);
%     D=2.*Varf(:,4)./Varf(:,3).*C;
%     E=Varf(:,4).^2./Varf(:,3).^2;
%     tmp=exp(Ef(:,3)+0.5.*Varf(:,3));
%     Ey=C.*exp(0.5.*Ef(:,3)+1./8.*Varf(:,3)) ...
%       + Varf(:,4)./Varf(:,3).*exp(0.5.*Ef(:,3)+1./8.*Varf(:,3)).*tmp;
%     Vary=B.*tmp + D.*tmp.*(Ef(:,3)+Varf(:,3)) ...
%       + E.*tmp.*(Varf(:,3)+(Ef(:,3)+Varf(:,3)).^2) + exp(Ef(:,2)+Varf(:,2)./2);
%     Vary=Vary-Ey.^2;
    for i=1:size(Ef(:,1),1)
      minf=Ef(i,:)-9.*sqrt(Varf(i,1:3));
      maxf=Ef(i,:)+9.*sqrt(Varf(i,1:3));
      mf=@(phi) Ef(i,1)+Varf(i,4)./Varf(i,3).*(phi - Ef(i,3));
      s=Varf(i,1)-Varf(i,4).^2./Varf(i,3);
      Ey(i,:)=integral2(@(f,phi) exp(0.5.*phi).*f.*norm_pdf(f, mf(phi), sqrt(s)) ...
        .*norm_pdf(phi, Ef(i,3), sqrt(Varf(i,3))), minf(1),maxf(1),minf(3),maxf(3), ...
        'RelTol',1e-6, 'AbsTol',1e-10);      
      Vary(i,:)=integral2(@(f,phi) exp(phi).*f.^2.*norm_pdf(f, mf(phi), sqrt(s)) ...
        .*norm_pdf(phi, Ef(i,3), sqrt(Varf(i,3))), minf(1),maxf(1),minf(3),maxf(3), ...
        'RelTol',1e-6, 'AbsTol',1e-10);
      Vary(i,:)=Vary(i,:) + quadgk(@(theta) exp(theta).*norm_pdf(theta, Ef(i,2), sqrt(Varf(i,2))), ...
        minf(2), maxf(2), 'RelTol',1e-6, 'AbsTol',1e-10);
      Vary(i,:)=Vary(i,:) - Ey(i,:).^2;
      if ~isreal(Ey(i,:)) || Vary(i,:)<0
        Ey(i,:)=NaN;
        Vary(i,:)=NaN;
      end
    end
  else
    if ~lik.inputparam && ~lik.int_magnitude
      Sq = zt(2);
      mq = zt(1);
      %     minf=mq-9.*sqrt(Sq);
      %     maxf=mq+9.*sqrt(Sq);
      %Ey=zeros(size(yt));
      Ey=Ef;
      %Vary=zeros(size(yt));
      %   fh2=@(theta) Sq.^(-1/2).*exp(-1./(2.*Sq.^2).*(theta-mq).^2);
      %Vary=quadgk(@(theta) 1./sqrt(2*pi*Sq).*exp(-1/(2*Sq).*(theta-mq).^2 + theta), minf, maxf);
      Vary=exp(mq+Sq/2);
      Vary=Vary+Varf;
    elseif (~lik.int_likparam && lik.int_magnitude && ~lik.inputmagnitude)
      s2=lik.sigma2;
      Ey=zeros(size(Ef,1),1);
      Vary=zeros(size(Ey));
      Sq = zt(2);
      mq = zt(1);
      minf=mq-9.*sqrt(Sq);
      maxf=mq+9.*sqrt(Sq);
      Ey=Ef*quadgk(@(phi) 1./sqrt(2*pi*Sq)*exp(-1/(2*Sq)*(mq-phi).^2 + phi/2), minf, maxf);
      for i=1:size(Ef,1)
        Vary(i)=s2 + (Ef(i).^2+Varf(i)).*exp(mq+Sq/2)-Ey(i).^2;
      end
    elseif lik.inputparam && ~lik.int_magnitude
      Ey=Ef(:,1);
      Vary=zeros(size(Ey));
      for i=1:size(Ef,1)
        Sq = Varf(i,2);
        mq = Ef(i,2);
        minf=mq-9.*sqrt(Sq);
        maxf=mq+9.*sqrt(Sq);
        var=exp(Ef(i,2)+0.5.*Varf(i,2));
        %       var=quadgk(@(theta) 1./sqrt(2*pi*Sq).*exp(-1/(2*Sq).*(theta-mq).^2 + theta), minf, maxf);
        Vary(i)=Varf(i,1)+var;
        %
        %     fh1= @(theta) 1./(2*pi).*(Varf(i)+exp(theta)).^(-1/2) ...
        %       .*exp(-1./(2*(Varf(i)+exp(theta))).*(yt(i)-Ef(i)).^2);
        %     tf = @(theta) fh1(theta).*fh2(theta);
        %     [m0,Ey(i),m2]=quad_moments(tf,minf,maxf,1e-10,1e-10);
        %     Ey(i)=Ey(i);
        %
        %     tf = @(theta) (Varf(i)+exp(theta)).*norm_pdf(theta, mq, sqrt(Sq));
        %     Vary(i) = quadgk(tf, minf, maxf, 'RelTol',1e-10, 'AbsTol',1e-10);
        %
        %     xg=linspace(minf-10,maxf+10,1000);
        %     plot(xg, tf(xg));
        %     %plot(xg,fh1(xg), xg, fh2(xg), xg, tf(xg));
        %     fprintf('press enter.\n');
        %     pause
        %
      end
    elseif ~lik.int_likparam && lik.inputmagnitude
      Ey=zeros(size(Ef,1),1);
      Vary=zeros(size(Ey));
      s2=lik.sigma2;
      for i=1:size(Ef,1)
        Sq = Varf(i,2);
        mq = Ef(i,2);
        minf=mq-9.*sqrt(Sq);
        maxf=mq+9.*sqrt(Sq);
        Ey(i)=Ef(i,1).*quadgk(@(phi) ...
          1./sqrt(2*pi*Sq)*exp(-1/(2*Sq)*(mq-phi).^2 + phi/2),minf, maxf);
        var=(Ef(i,1).^2+Varf(i,1)).*exp(mq+Sq/2);
        Vary(i)=s2+var-Ey(i).^2;
      end
    elseif ~lik.inputparam && lik.int_magnitude && ~lik.inputmagnitude
      Sq(1) = zt(2);
      mq(1) = zt(1);
      Sq(2) = zt(4);
      mq(2) = zt(3);
      minf=mq-9.*sqrt(Sq);
      maxf=mq+9.*sqrt(Sq);
      Ey=Ef.*quadgk(@(phi)exp(0.5.*phi).*1./sqrt(2*pi*Sq(2)) ...
        .*exp(-1/(2*Sq(2)).*(phi-mq(2)).^2),minf(2),maxf(2));
      Vary=(Ef.^2+Varf).*exp(mq(2)+Sq(2)/2)+ exp(mq(1)+Sq(1)/2) - Ey.^2;
      %     Vary=(Ef.^2+Varf).*quadgk(@(phi)exp(phi) ...
      %       .*1./sqrt(2*pi*Sq(2)).*exp(-1/(2*Sq(2)).*(phi-mq(2)).^2),minf(2),maxf(2)) ...
      %       + quadgk(@(theta) 1./sqrt(2*pi*Sq(1)).*exp(-1/(2*Sq(1)) ...
      %         .*(theta-mq(1)).^2 + theta), minf(1),maxf(1)) - Ey.^2;
    elseif lik.inputparam && lik.int_magnitude && ~lik.inputmagnitude
      Sq(1) = zt(2);
      mq(1) = zt(1);
      minf=mq-9.*sqrt(Sq);
      maxf=mq+9.*sqrt(Sq);
      Ey=Ef(:,1).*quadgk(@(phi)exp(0.5.*phi).*1./sqrt(2*pi*Sq) ...
        .*exp(-1/(2*Sq).*(phi-mq).^2),minf,maxf);
      VarPhi=(Ef(:,1).^2+Varf(:,1)).*quadgk(@(phi)exp(phi).*1./sqrt(2*pi*Sq) ...
        .*exp(-1/(2*Sq).*(phi-mq).^2),minf,maxf);
      Vary=zeros(size(Ef,1),1);
      for i=1:size(Ef,1)
        Sq = Varf(i,2);
        mq = Ef(i,2);
        minf=mq-9.*sqrt(Sq);
        maxf=mq+9.*sqrt(Sq);
        Vary(i)=VarPhi(i)+quadgk(@(theta) 1./sqrt(2*pi*Sq).*exp(-1/(2*Sq) ...
          .*(theta-mq).^2 + theta), minf,maxf) - Ey(i).^2;
      end
    elseif lik.inputparam && lik.int_magnitude && lik.inputmagnitude
      Ey=zeros(size(Ef,1),1);
      Vary=zeros(size(Ef,1),1);
      for i=1:size(Ef,1)
        Sq(1) = Varf(i,2);
        mq(1) = Ef(i,2);
        Sq(2) = Varf(i,3);
        mq(2) = Ef(i,3);
        minf=mq-9.*sqrt(Sq);
        maxf=mq+9.*sqrt(Sq);
        Ey(i)=Ef(i,1).*exp(0.5.*(Ef(i,3)+0.25.*Varf(i,3)));
        %       Ey(i)=Ef(i,1).*quadgk(@(phi)exp(0.5.*phi).*1./sqrt(2*pi*Sq(2)) ...
        %         .*exp(-1/(2*Sq(2)).*(phi-mq(2)).^2),minf(2),maxf(2));
        VarPhi=(Ef(i,1).^2+Varf(i,1)).*exp(mq(2)+Sq(2)/2);
        Vary(i)=VarPhi+exp(mq(1)+Sq(1)/2) - Ey(i).^2;
        %       VarPhi=(Ef(i,1).^2+Varf(i,1)).*quadgk(@(phi)exp(phi).*1./sqrt(2*pi*Sq(2)) ...
        %         .*exp(-1/(2*Sq(2)).*(phi-mq(2)).^2),minf(2),maxf(2));
        %       Vary(i)=VarPhi+quadgk(@(theta) 1./sqrt(2*pi*Sq(1)).*exp(-1/(2*Sq(1)) ...
        %         .*(theta-mq(1)).^2 + theta), minf(1),maxf(1)) - Ey(i).^2;
      end
    elseif ~lik.inputparam && lik.int_magnitude && lik.inputmagnitude
      Sq = zt(2);
      mq = zt(1);
      minf=mq-7.*sqrt(Sq);
      maxf=mq+7.*sqrt(Sq);
      Ey=zeros(size(Ef,1),1);
      Vary=zeros(size(Ef,1),1);
      var=quadgk(@(theta) 1./sqrt(2*pi*Sq).*exp(-1/(2*Sq) ...
        .*(theta-mq).^2 + theta), minf,maxf);
      for i=1:size(Ef,1)
        Sq = Varf(i,2);
        mq = Ef(i,2);
        minf=mq-9.*sqrt(Sq);
        maxf=mq+9.*sqrt(Sq);
        Ey(i)=Ef(i,1).*quadgk(@(phi)exp(0.5.*phi).*1./sqrt(2*pi*Sq) ...
          .*exp(-1/(2*Sq).*(phi-mq).^2),minf,maxf);
        VarPhi=(Ef(i,1).^2+Varf(i,1)).*quadgk(@(phi)exp(phi).*1./sqrt(2*pi*Sq) ...
          .*exp(-1/(2*Sq).*(phi-mq).^2),minf,maxf);
        Vary(i)=VarPhi+var - Ey(i).^2;
      end
    end
  end
  if ~isempty(yt)
    lpy=norm_lpdf(yt, Ey, sqrt(Vary));
  else
    lpy=[];
  end
  
end

function prctys = lik_epgaussian_predprcty(lik, Ef, Varf, zt, prcty)
%LIK_EPGAUSSIAN_PREDPRCTY  Returns the percentiled of predictive density of y
%
%  Description         
%    PRCTY = LIK_EPGAUSSIAN_PREDPRCTY(LIK, EF, VARF YT, ZT)
%    Returns percentiles of the predictive density PY of YT, that is 
%    This requires also the succes counts YT, numbers of trials ZT. This
%    subfunction is needed when using function gp_predprcty.
%
%  See also 
%    GP_PREDPCTY

  if isempty(zt)
    error(['lik_epgaussian -> lik_epgaussian_predprcty: missing z!'... 
           'EP-Gaussian likelihood needs the expected number of       '...
           'occurrences as an extra input z. See, for             '...
           'example, lik_epgaussian and gpla_e.                 ']);
  end
  
  opt=optimset('TolX',.5,'Display','off');
  nt=size(Ef,1);
  prctys = zeros(nt,numel(prcty));
  prcty=prcty/100;
  for i1=1:nt
    ci = sqrt(Varf(i1));
    for i2=1:numel(prcty)
      a=floor(fminbnd(@(a) (quadgk(@(f) binocdf(a,zt(i1),logitinv(f)).*norm_pdf(f,Ef(i1),ci),Ef(i1)-6*ci,Ef(i1)+6*ci,'AbsTol',1e-4)-prcty(i2)).^2,binoinv(prcty(i2),zt(i1),logitinv(Ef(i1)-1.96*ci)),binoinv(prcty(i2),zt(i1),logitinv(Ef(i1)+1.96*ci)),opt));
      if quadgk(@(f) binocdf(a,zt(i1),logitinv(f)).*norm_pdf(f,Ef(i1),ci),Ef(i1)-6*ci,Ef(i1)+6*ci,'AbsTol',1e-4)<prcty(i2)
        a=a+1;
      end
      prctys(i1,i2)=a;
    end
  end
end

function [df,minf,maxf] = init_epgaussian_norm(yy,myy_i,sigm2_i,N)
%INIT_EPGAUSSIAN_NORM
%
%  Description
%    Return function handle to a function evaluating EP-Gaussian *
%    Gaussian which is used for evaluating (likelihood * cavity)
%    or (likelihood * posterior) Return also useful limits for
%    integration. This is private function for lik_epgaussian. This
%    subfunction is needed by subfunctions tiltedMoments and predy.
%  
% See also
%   LIK_EPGAUSSIAN_TILTEDMOMENTS, LIK_EPGAUSSIAN_PREDY
  
% avoid repetitive evaluation of constant part
  ldconst = gammaln(N+1)-gammaln(yy+1)-gammaln(N-yy+1) - log(sigm2_i)/2 - log(2*pi)/2;
%   ldconst = log(factorial(N)/(factorial(yy)*factorial(N-yy))-log(sigm2_i)/2 -log(2*pi)/2;
  
 % Create function handle for the function to be integrated
  df = @epgaussian_norm;
 % use log to avoid underflow, and derivates for faster search
  ld = @log_epgaussian_norm;
  ldg = @log_epgaussian_norm_g;
  ldg2 = @log_epgaussian_norm_g2;
  
  % Set the limits for integration
  % EP-Gaussian likelihood is log-concave so the epgaussian_norm
  % function is unimodal, which makes things easier
  if yy==0 || yy==N
    % with yy==0 or yy==N the mode of the likelihood is not defined
    % use the mode of the Gaussian (cavity or posterior) as a first guess
    modef = myy_i;
  else
    % use precision weighted mean of the Gaussian approximation of the
    % epgaussian likelihood and Gaussian
    mean_app = log(yy./(N-yy));
    ld0=1/(1+exp(-mean_app));
    ld1=(1-ld0)*ld0;
    ld2=ld0-3*ld0^2+2*ld0^3;
    var_app=inv(-( yy*(ld2*ld0-ld1^2)/ld0^2 + (N-yy)*(ld2*(ld0-1)-ld1^2)/(ld0-1)^2 ));
    
    modef = (myy_i/sigm2_i + mean_app/var_app)/(1/sigm2_i + 1/var_app);
%     sigm_app = sqrt((1/sigm2_i + 1/var_app)^-1);
  end
  % find the mode of the integrand using Newton iterations
  % few iterations is enough, since the first guess in the right direction
  niter=3;       % number of Newton iterations
  mindelta=1e-6; % tolerance in stopping Newton iterations
  for ni=1:niter
      g = ldg(modef);
      h = ldg2(modef);
      delta=-g/h;
      modef=modef+delta;
      if abs(delta)<mindelta
          break
      end
  end
  if abs(delta)>1 || isinf(delta) || isnan(delta) 
    % Newton algorithm didn't work properly so do binary search
    modef=myy_i;
    a=modef-5.*sqrt(sigm2_i); b=modef+5.*sqrt(sigm2_i); delta=1;
    while ldg(a)<0
      a=a-5.*sqrt(sigm2_i);
    end
    while ldg(b)>0
      b=b+5.*sqrt(sigm2_i);
    end
    while delta > 0.1
      modef=(a+b)/2;
      if ldg(modef) > 0
        a=modef;
      else
        b=modef;
      end
      delta=b-a;
    end
    h=ldg2(modef);
  end
  % integrand limits based on Gaussian approximation at mode
  modes=sqrt(-1/h);
  minf=modef-4*modes;
  maxf=modef+4*modes;
  modeld=ld(modef);
  iter=0;
  % check that density at end points is low enough
  lddiff=12; % min difference in log-density between mode and end-points
  minld=ld(minf);
  step=1;
  while minld>(modeld-lddiff)
    minf=minf-step*modes;
    minld=ld(minf);
    iter=iter+1;
    step=step*2;
    if iter>100
      error(['lik_negbin -> init_negbin_norm: ' ...
             'integration interval minimun not found ' ...
             'even after looking hard!'])
    end
  end
  maxld=ld(maxf);
  iter=0;
  step=1;
  while maxld>(modeld-lddiff)
    maxf=maxf+step*modes;
    maxld=ld(maxf);
    iter=iter+1;
    step=step*2;
    if iter>100
      error(['lik_negbin -> init_negbin_norm: ' ...
             'integration interval maximun not found ' ...
             'even after looking hard!'])
    end
  end
  
  
  function integrand = epgaussian_norm(f)
  % Logit * Gaussian
    integrand = exp(ldconst + yy*log(1./(1.+exp(-f)))+(N-yy)*log(1-1./(1.+exp(-f)))...
                   - 0.5 * (f-myy_i).^2./sigm2_i);
%     integrand = exp(ldconst ...
%                     +yy*log(x)+(N-yy)*log(1-x) ...
%                     -0.5*(f-myy_i).^2./sigm2_i);
    integrand(isnan(integrand)|isinf(integrand))=0;
  end
  
  function log_int = log_epgaussian_norm(f)
  % log(EP-Gaussian * Gaussian)
  % log_epgaussian_norm is used to avoid underflow when searching
  % integration interval
  
    log_int = ldconst + yy*log(1./(1.+exp(-f)))+(N-yy)*log(1-1./(1.+exp(-f)))...
                   - 0.5 * (f-myy_i).^2./sigm2_i;
%     log_int = ldconst ...
%               -log(1+exp(-yy.*f)) ...
%               -0.5*(f-myy_i).^2./sigm2_i;
    log_int(isnan(log_int)|isinf(log_int))=-Inf;
  end
  
  function g = log_epgaussian_norm_g(f)
  % d/df log(EP-Gaussian * Gaussian)
  % derivative of log_epgaussian_norm
    g = -(f-myy_i)./sigm2_i - exp(-f).*(N-yy)./((1+exp(-f)).^2.*(1-1./(1+exp(-f)))) ...
        + exp(-f).*yy./(1+exp(-f));
%     g = yy./(exp(f*yy)+1)...
%         + (myy_i - f)./sigm2_i;
  end
  
  function g2 = log_epgaussian_norm_g2(f)
  % d^2/df^2 log(EP-Gaussian * Gaussian)
  % second derivate of log_epgaussian_norm
    g2 = - (1+exp(2.*f)+exp(f).*(2+N*sigm2_i)./((1+exp(f))^2*sigm2_i));
%     a=exp(f*yy);
%     g2 = -a*(yy./(a+1)).^2 ...
%          -1/sigm2_i;
  end
  
end

function p = lik_epgaussian_invlink(lik, f, z)
%LIK_EPGAUSSIAN_INVLINK  Returns values of inverse link function
%             
%  Description 
%    P = LIK_EPGAUSSIAN_INVLINK(LIK, F) takes a likelihood structure LIK and
%    latent values F and returns the values of inverse link function P.
%    This subfunction is needed when using gp_predprctmu. 
%
%     See also
%     LIK_EPGAUSSIAN_LL, LIK_EPGAUSSIAN_PREDY
  
  p = logitinv(f);
end

function reclik = lik_epgaussian_recappend(reclik, ri, lik)
%RECAPPEND  Append the parameters to the record
%
%  Description 
%    RECLIK = GPCF_EPGAUSSIAN_RECAPPEND(RECLIK, RI, LIK) takes a
%    likelihood record structure RECLIK, record index RI and
%    likelihood structure LIK with the current MCMC samples of
%    the parameters. Returns RECLIK which contains all the old
%    samples and the current samples from LIK. This subfunction 
%    is needed when using MCMC sampling (gp_mc).
% 
%  See also
%    GP_MC
  
  if nargin == 2
    reclik.type = 'EP-Gaussian';

    % Set the function handles
    reclik.fh.pak = @lik_epgaussian_pak;
    reclik.fh.unpak = @lik_epgaussian_unpak;
    reclik.fh.ll = @lik_epgaussian_ll;
    reclik.fh.llg = @lik_epgaussian_llg;    
    reclik.fh.llg2 = @lik_epgaussian_llg2;
    reclik.fh.llg3 = @lik_epgaussian_llg3;
    reclik.fh.siteDeriv = @lik_epgaussian_siteDeriv;
    reclik.fh.tiltedMoments = @lik_epgaussian_tiltedMoments;
    reclik.fh.invlink = @lik_epgaussian_invlink;
    reclik.fh.predprcty = @lik_epgaussian_predprcty;
    reclik.fh.predy = @lik_epgaussian_predy;
    reclik.fh.recappend = @likelih_epgaussian_recappend;
    reclik.p=[];
    reclik.p.sigma2=[];
    if ~isempty(ri.p.sigma2)
      reclik.p.sigma2 = ri.p.sigma2;
    end
    return
  else
    % Append to the record
    reclik.sigma2(ri,:)=lik.sigma2;
    if ~isempty(lik.p.sigma2)
      reclik.p.sigma2 = lik.p.sigma2.fh.recappend(reclik.p.sigma2, ri, lik.p.sigma2);
    end
  end

end
