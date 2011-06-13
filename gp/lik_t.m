function lik = lik_t(varargin)
%LIK_T  Create a Student-t likelihood structure 
%
%  Description
%    LIK = LIK_T('PARAM1',VALUE1,'PARAM2,VALUE2,...)
%    creates Student-t likelihood structure in which the named
%    parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%    LIK = LIK_T(LIK,'PARAM1',VALUE1,'PARAM2,VALUE2,...)
%    modify a likelihood structure with the named parameters
%    altered with the specified values.
%  
%    Parameters for Student-t likelihood [default]
%      sigma2       - scale squared [1]
%      nu           - degrees of freedom [4]
%      sigma2_prior - prior for sigma2 [prior_logunif]
%      nu_prior     - prior for nu [prior_fixed]
%
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%    The likelihood is defined as follows:
%                  __ n
%      p(y|f, z) = || i=1 C(nu,s2) * (1 + 1/nu * (y_i - f_i)^2/s2 )^(-(nu+1)/2)
%
%      where nu is the degrees of freedom, s2 the scale and f_i the
%      latent variable defining the mean. C(nu,s2) is constant
%      depending on nu and s2.
%
%  See also
%    GP_SET, LIK_*, PRIOR_*
%
  
% Copyright (c) 2009-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_T';
  ip.addOptional('lik', [], @isstruct);
  ip.addParamValue('sigma2',0.1, @(x) isscalar(x) && x>0);
  ip.addParamValue('sigma2_prior',prior_logunif(), @(x) isstruct(x) || isempty(x));
  ip.addParamValue('nu',4, @(x) isscalar(x) && x>0);
  ip.addParamValue('nu_prior',prior_fixed, @(x) isstruct(x) || isempty(x));
  ip.addParamValue('fix_nu',[], @(x) isreal(x) && isfinite(x))
  ip.parse(varargin{:});
  lik=ip.Results.lik;
  
  if isempty(lik)
    init=true;
    lik.type = 'Student-t';
  else
    if ~isfield(lik,'type') && ~isequal(lik.type,'Student-t')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end

  % Initialize parameters
  if init || ~ismember('sigma2',ip.UsingDefaults)
    lik.sigma2 = ip.Results.sigma2;
  end
  if init || ~ismember('nu',ip.UsingDefaults)
    lik.nu = ip.Results.nu;
  end
  % Initialize prior structure
  if init
    lik.p=[];
  end
  if init || ~ismember('sigma2_prior',ip.UsingDefaults)
    lik.p.sigma2=ip.Results.sigma2_prior;
  end
  if init || ~ismember('nu_prior',ip.UsingDefaults)
    lik.p.nu=ip.Results.nu_prior;
  end
  if init || ~ismember('fix_nu',ip.UsingDefaults)
    lik.fix_nu=ip.Results.fix_nu;
  end
  
  if init      
    % Set the function handles to the subfunctions
    lik.fh.pak = @lik_t_pak;
    lik.fh.unpak = @lik_t_unpak;
    lik.fh.lp = @lik_t_lp;
    lik.fh.lpg = @lik_t_lpg;
    lik.fh.ll = @lik_t_ll;
    lik.fh.llg = @lik_t_llg;    
    lik.fh.llg2 = @lik_t_llg2;
    lik.fh.llg3 = @lik_t_llg3;
    lik.fh.tiltedMoments = @lik_t_tiltedMoments;
    lik.fh.tiltedMoments2 = @lik_t_tiltedMoments2;
    lik.fh.siteDeriv = @lik_t_siteDeriv;
    lik.fh.siteDeriv2 = @lik_t_siteDeriv2;    
    lik.fh.optimizef = @lik_t_optimizef;
    lik.fh.upfact = @lik_t_upfact;
    lik.fh.predy = @lik_t_predy;
    lik.fh.recappend = @lik_t_recappend;
  end

end

function [w, s] = lik_t_pak(lik)
%LIK_T_PAK  Combine likelihood parameters into one vector.
%
%  Description 
%    W = LIK_T_PAK(LIK) takes a likelihood structure LIK and
%    combines the parameters into a single row vector W.
%     
%       w = [ log(lik.sigma2)
%             (hyperparameters of lik.sigma2)
%             log(log(lik.nu))
%             (hyperparameters of lik.nu)]'
%
%  See also
%    LIK_T_UNPAK, GP_PAK
  
  w = []; s = {};
  if ~isempty(lik.p.sigma2)
    w = [w log(lik.sigma2)];
    s = [s; 'log(lik.sigma2)'];
    [wh sh] = feval(lik.p.sigma2.fh.pak, lik.p.sigma2);
    w = [w wh];
    s = [s; sh];
  end
  if ~isempty(lik.p.nu)
    w = [w log(log(lik.nu))];
    s = [s; 'loglog(lik.nu)'];
    [wh sh] = feval(lik.p.nu.fh.pak, lik.p.nu);
    w = [w wh];
    s = [s; sh];
  end        
end

function [lik, w] = lik_t_unpak(lik, w)
%LIK_T_UNPAK  Extract likelihood parameters from the vector.
%
%  Description
%    W = LIK_T_UNPAK(W, LIK) takes a likelihood structure LIK and
%    extracts the parameters from the vector W to the LIK
%    structure.
%     
%    Assignment is inverse of  
%       w = [ log(lik.sigma2)
%             (hyperparameters of lik.sigma2)
%             log(log(lik.nu))
%             (hyperparameters of lik.nu)]'
%
%   See also
%   LIK_T_PAK, GP_UNPAK

  if ~isempty(lik.p.sigma2)
    lik.sigma2 = exp(w(1));
    w = w(2:end);
    [p, w] = feval(lik.p.sigma2.fh.unpak, lik.p.sigma2, w);
    lik.p.sigma2 = p;
  end
  if ~isempty(lik.p.nu) 
    lik.nu = exp(exp(w(1)));
    w = w(2:end);
    [p, w] = feval(lik.p.nu.fh.unpak, lik.p.nu, w);
    lik.p.nu = p;
  end
end

function lp = lik_t_lp(lik)
%LIK_T_LP  log(prior) of the likelihood parameters
%
%  Description
%    LP = LIK_T_LP(LIK) takes a likelihood structure LIK and
%    returns log(p(th)), where th collects the parameters.
%
%  See also
%    LIK_T_LLG, LIK_T_LLG3, LIK_T_LLG2, GPLA_E
  
  v = lik.nu;
  sigma2 = lik.sigma2;
  lp = 0;
  
  if ~isempty(lik.p.sigma2) 
    lp = lp + feval(lik.p.sigma2.fh.lp, lik.sigma2, lik.p.sigma2) +log(sigma2);
  end
  if ~isempty(lik.p.nu)
    lp = lp + feval(lik.p.nu.fh.lp, lik.nu, lik.p.nu)  +log(v) +log(log(v));
  end
end

function lpg = lik_t_lpg(lik)
%LIK_T_LPG  d log(prior)/dth of the likelihood parameters th
%
%  Description
%    LPG = LIK_T_LPG(LIK) takes a likelihood structure LIK
%    and returns d log(p(th))/dth, where th collects the
%    parameters.
%
%  See also
%    LIK_T_LLG, LIK_T_LLG3, LIK_T_LLG2, GPLA_G
  
% Evaluate the gradients of log(prior)

  v = lik.nu;
  sigma2 = lik.sigma2;
  lpg = [];
  i1 = 0;
  
  if ~isempty(lik.p.sigma2) 
    i1 = i1+1;
    lpg(i1) = feval(lik.p.sigma2.fh.lpg, lik.sigma2, lik.p.sigma2).*sigma2 + 1;
  end
  if ~isempty(lik.p.nu) 
    i1 = i1+1;
    lpg(i1) = feval(lik.p.nu.fh.lpg, lik.nu, lik.p.nu).*v.*log(v) +log(v) + 1;
  end    
end

function ll = lik_t_ll(lik, y, f, z)
%LIK_T_LL  Log likelihood
%
%  Description
%    LL = LIK_T_LL(LIK, Y, F) takes a likelihood structure LIK,
%    observations Y, and latent values F. Returns the log
%    likelihood, log p(y|f,z).
%
%  See also
%    LIK_T_LLG, LIK_T_LLG3, LIK_T_LLG2, GPLA_E

  r = y-f;
  v = lik.nu;
  sigma2 = lik.sigma2;

  term = gammaln((v + 1) / 2) - gammaln(v/2) -log(v.*pi.*sigma2)/2;
  ll = term + log(1 + (r.^2)./v./sigma2) .* (-(v+1)/2);
  ll = sum(ll);
end


function llg = lik_t_llg(lik, y, f, param, z)
%LIK_T_LLG  Gradient of the log likelihood
%
%  Description
%    LOKLIKG = LIK_T_LLG(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, observations Y, and latent values F. Returns
%    the gradient of log likelihood with respect to PARAM. At the
%    moment PARAM can be 'param' or 'latent'.
%
%  See also
%    LIK_T_LL, LIK_T_LLG2, LIK_T_LLG3, GPLA_E
  
  r = y-f;
  v = lik.nu;
  sigma2 = lik.sigma2;
  
  switch param
    case 'param'
      n = length(y);

      % Derivative with respect to sigma2
      llg(1) = -n./sigma2/2 + (v+1)./2.*sum(r.^2./(v.*sigma2.^2+r.^2*sigma2));
      
      % correction for the log transformation
      llg(1) = llg(1).*sigma2;
      if ~isempty(lik.p.nu)
        % Derivative with respect to nu
        llg(2) = 0.5.* sum(psi((v+1)./2) - psi(v./2) - 1./v - log(1+r.^2./(v.*sigma2)) + (v+1).*r.^2./(v.^2.*sigma2 + v.*r.^2));
        
        % correction for the log transformation
        llg(2) = llg(2).*v.*log(v);
      end
    case 'latent'
      llg  = (v+1).*r ./ (v.*sigma2 + r.^2);            
  end
  
end


function llg2 = lik_t_llg2(lik, y, f, param, z)
%LIK_T_LLG2  Second gradients of log likelihood
%
%  Description        
%    LLG2 = LIK_T_LLG2(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, observations Y, and latent values F. Returns
%    the Hessian of log likelihood with respect to PARAM. At the
%    moment PARAM can be only 'latent'. LLG2 is a vector with
%    diagonal elements of the Hessian matrix (off diagonals are
%    zero).
%
%  See also
%    LIK_T_LL, LIK_T_LLG, LIK_T_LLG3, GPLA_E

  r = y-f;
  v = lik.nu;
  sigma2 = lik.sigma2;

  switch param
    case 'param'
      
    case 'latent'
      % The Hessian d^2 /(dfdf)
      llg2 =  (v+1).*(r.^2 - v.*sigma2) ./ (v.*sigma2 + r.^2).^2;
    case 'latent+param'
      % gradient d^2 / (dfds2)
      llg2 = -v.*(v+1).*r ./ (v.*sigma2 + r.^2).^2;
      
      % Correction for the log transformation
      llg2 = llg2.*sigma2;
      if ~isempty(lik.p.nu)
        % gradient d^2 / (dfdnu)
        llg2(:,2) = r./(v.*sigma2 + r.^2) - sigma2.*(v+1).*r./(v.*sigma2 + r.^2).^2;

        % Correction for the log transformation
        llg2(:,2) = llg2(:,2).*v.*log(v);
      end
  end
end    

function llg3 = lik_t_llg3(lik, y, f, param, z)
%LIK_T_LLG3  Third gradients of log likelihood (energy)
%
%  Description
%    LLG3 = LIK_T_LLG3(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, observations Y and latent values F and
%    returns the third gradients of log likelihood with respect
%    to PARAM. At the moment PARAM can be only 'latent'. G3 is a
%    vector with third gradients.
%
%  See also
%    LIK_T_LL, LIK_T_LLG, LIK_T_LLG2, GPLA_E, GPLA_G

  r = y-f;
  v = lik.nu;
  sigma2 = lik.sigma2;
  
  switch param
    case 'param'
      
    case 'latent'
      % Return the diagonal of W differentiated with respect to latent values / dfdfdf
      llg3 = (v+1).*(2.*r.^3 - 6.*v.*sigma2.*r) ./ (v.*sigma2 + r.^2).^3;
    case 'latent2+param'
      % Return the diagonal of W differentiated with respect to
      % likelihood parameters / dfdfds2
      llg3 = (v+1).*v.*( v.*sigma2 - 3.*r.^2) ./ (v.*sigma2 + r.^2).^3;
      llg3 = llg3.*sigma2;
      if ~isempty(lik.p.nu)
        % dfdfdnu
        llg3(:,2) = (r.^2-2.*v.*sigma2-sigma2)./(v.*sigma2 + r.^2).^2 - 2.*sigma2.*(r.^2-v.*sigma2).*(v+1)./(v.*sigma2 + r.^2).^3;
        llg3(:,2) = llg3(:,2).*v.*log(v);
      end
  end
end


function [m_0, m_1, sigm2hati1] = lik_t_tiltedMoments(lik, y, i1, sigm2_i, myy_i, z)
%LIK_T_TILTEDMOMENTS  Returns the marginal moments for EP algorithm
%
%  Description
%    [M_0, M_1, M2] = LIK_T_TILTEDMOMENTS(LIK, Y, I, S2, MYY, Z)
%    takes a likelihood structure LIK, incedence counts Y,
%    expected counts Z, index I and cavity variance S2 and mean
%    MYY. Returns the zeroth moment M_0, mean M_1 and variance
%    M_2 of the posterior marginal (see Rasmussen and Williams
%    (2006): Gaussian processes for Machine Learning, page 55).
%
%  See also
%    GPEP_E

  
  zm = @zeroth_moment;
  
  tol = 1e-8;
  yy = y(i1);
  nu = lik.nu;
  sigma2 = lik.sigma2;
  
  % Set the limits for integration and integrate with quad
  % -----------------------------------------------------
  mean_app = myy_i;
  sigm_app = sqrt(sigm2_i);


  lambdaconf(1) = mean_app - 8.*sigm_app; lambdaconf(2) = mean_app + 8.*sigm_app;
  test1 = zm((lambdaconf(2)+lambdaconf(1))/2) > zm(lambdaconf(1));
  test2 = zm((lambdaconf(2)+lambdaconf(1))/2) > zm(lambdaconf(2));
  testiter = 1;
  if test1 == 0 
    lambdaconf(1) = lambdaconf(1) - 3*sigm_app;
    test1 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(1));
    if test1 == 0
      go=true;
      while testiter<10 & go
        lambdaconf(1) = lambdaconf(1) - 2*sigm_app;
        lambdaconf(2) = lambdaconf(2) - 2*sigm_app;
        test1 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(1));
        test2 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(2));
        if test1==1&test2==1
          go=false;
        end
        testiter=testiter+1;
      end
    end
    mean_app = (lambdaconf(2)+lambdaconf(1))/2;
  elseif test2 == 0
    lambdaconf(2) = lambdaconf(2) + 3*sigm_app;
    test2 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(2));
    if test2 == 0
      go=true;
      while testiter<10 & go
        lambdaconf(1) = lambdaconf(1) + 2*sigm_app;
        lambdaconf(2) = lambdaconf(2) + 2*sigm_app;
        test1 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(1));
        test2 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(2));
        if test1==1&test2==1
          go=false;
        end
        testiter=testiter+1;
      end
    end
    mean_app = (lambdaconf(2)+lambdaconf(1))/2;
  end
  RTOL = 1.e-6;
  ATOL = 1.e-10;
  
  % Integrate with quadrature
  [m_0, m_1, m_2] = quad_moments(zm,lambdaconf(1), lambdaconf(2), RTOL, ATOL);        
  
  sigm2hati1 = m_2 - m_1.^2;
  
  function integrand = zeroth_moment(f)
    r = yy-f;
    term = gammaln((nu + 1) / 2) - gammaln(nu/2) -log(nu.*pi.*sigma2)/2;
    integrand = exp(term + log(1 + r.^2./nu./sigma2) .* (-(nu+1)/2));
    integrand = integrand.*exp(- 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2); %
  end
end

function [m_0, m_1, sigm2hati1] = lik_t_tiltedMoments2(likelih, y, i1, sigm2_i, myy_i, z, eta)
%LIKELIH_T_TILTEDMOMENTS    Returns the marginal moments for EP algorithm
%
%   Description
%   [M_0, M_1, M2] = LIKELIH_T_TILTEDMOMENTS(LIKELIH, Y, I, S2, MYY, Z)
%   takes a likelihood data structure LIKELIH, incedence counts Y,
%   expected counts Z, index I and cavity variance S2 and mean
%   MYY. Returns the zeroth moment M_0, mean M_1 and variance M_2
%   of the posterior marginal (see Rasmussen and Williams (2006):
%   Gaussian processes for Machine Learning, page 55).
%
%   See also
%   GPEP_E

  if nargin<7
    eta=1;
  end
  
  
  %tol = 1e-8;
  yy = y(i1);
  nu = likelih.nu;
  sigma2 = likelih.sigma2;
  
  % limiting distribution (nu -> infinity)
  %sigma=sqrt(sigma2);
  Vg=1/(1/sigm2_i +eta/sigma2);
  mg=Vg*(myy_i/sigm2_i +yy*eta/sigma2);
  sigm_i=sqrt(sigm2_i);
  sg=sqrt(Vg);
  
  % set integration limits and scaling
  nu_lim=1e6;
  if nu<nu_lim
    if mg>myy_i
      lambdaconf=[myy_i-6*sigm_i,max(mg+6*sg,myy_i+6*sigm_i)];
      t=linspace(myy_i,mg,10);
    else
      lambdaconf=[min(mg-6*sg,myy_i-6*sigm_i),myy_i+6*sigm_i];
      t=linspace(mg,myy_i,10);
    end
    lpt_max=max(lpt(t,0));
    C=log(1)-lpt_max;
  else
    lambdaconf=[mg-6*sg,mg+6*sg];
    C=log(1)-lpt(mg,0);
  end
  
  % set lower bound for log p(y,f)
  %C=0;
  %lpt_min=max(lpt(lambdaconf(1),C),lpt(lambdaconf(2),C));
  %C=-20-lpt_min;
  if nu>nu_lim
    m_0=exp(-0.5*log(2*pi) -0.5*log(sigm2_i+sigma2) -0.5*(yy-myy_i)^2 /(sigm2_i+sigma2));
    m_1=mg;
    sigm2hati1 = Vg;
  else
    % Integrate with quadrature
    RTOL = 1.e-12;
    ATOL = 1.e-7;
    
    [m_0, m_1, m_2] = quad_moments(@(f) exp(lpt(f,C)),lambdaconf(1), lambdaconf(2), RTOL, ATOL);
    sigm2hati1 = m_2 - m_1.^2;
    m_0=m_0*exp(-C);
  end
  
  if ~isfinite(m_0) || m_0<=0 %|| nu>nu_lim
    
    %lastwarn('')
    figure(2)
    ilim=lambdaconf;
    fh_l=@(f) t_lpdf(yy,nu,f,sqrt(sigma2)).*eta;
    fh_p=@(f)  (-0.5/sigm2_i) * (f-myy_i).^2 -0.5*log(2*pi*sigm2_i);
    t=linspace(ilim(1),ilim(2),500);
    pt=exp(lpt(t,0));
    plot(t,pt,'b',t,exp(fh_l(t)),'r',t,exp(fh_p(t)),'g',...
      t,m_0*normpdf(t,m_1,sqrt(sigm2hati1)),'k--','linewidth',2)
    ylim([0 max(pt)*1.5])
    % title(sprintf('Z=%.4f, %s=%.4f, s=%.4f',Zm,'\mu',mm,'\sigma',sqrt(Vm)))
    drawnow
    keyboard
  end
  
  function integrand = lpt(f,C)
    r = yy-f;
    if nu<nu_lim
      lpdf = gammaln((nu + 1) / 2) - gammaln(nu/2) -log(nu.*pi.*sigma2)/2;
      lpdf = lpdf + log(1 + r.^2./nu./sigma2) .* (-(nu+1)/2);
    else
      lpdf = log((nu+1)/2)/2 -log(nu.*pi.*sigma2)/2;
      %lpdf = lpdf + log1p(r.^2./nu./sigma2) .* (-(nu+1)/2);
      lpdf = lpdf + log(1+r.^2./nu./sigma2) .* (-(nu+1)/2);
    end
    %integrand = exp(lpdf*eta);
    %integrand = integrand.*exp(- 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2);
    integrand = lpdf*eta - 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2;
    integrand = integrand+C;
  end
end



function [g_i] = lik_t_siteDeriv(lik, y, i1, sigm2_i, myy_i, z)
%LIK_T_SITEDERIV  Evaluate the expectation of the gradient
%                 of the log likelihood term with respect
%                 to the likelihood parameters for EP 
%
%  Description
%    [M_0, M_1, M2] = LIK_T_TILTEDMOMENTS(LIK, Y, I, S2, MYY)
%    takes a likelihood structure LIK, observations Y, index I
%    and cavity variance S2 and mean MYY. Returns E_f [d log
%    p(y_i|f_i) /d a], where a is the likelihood parameter and
%    the expectation is over the marginal posterior. This term is
%    needed when evaluating the gradients of the marginal
%    likelihood estimate Z_EP with respect to the likelihood
%    parameters (see Seeger (2008): Expectation propagation for
%    exponential families)
%
%  See also
%    GPEP_G

  zm = @zeroth_moment;
  znu = @deriv_nu;
  zsigma2 = @deriv_sigma2;
  
  tol = 1e-8;
  yy = y(i1);
  nu = lik.nu;
  sigma2 = lik.sigma2;

  % Set the limits for integration and integrate with quad
  mean_app = myy_i;
  sigm_app = sqrt(sigm2_i);

  lambdaconf(1) = mean_app - 6.*sigm_app; lambdaconf(2) = mean_app + 6.*sigm_app;
  test1 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(1));
  test2 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(2));
  testiter = 1;
  if test1 == 0 
    lambdaconf(1) = lambdaconf(1) - 3*sigm_app;
    test1 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(1));
    if test1 == 0
      go=true;
      while testiter<10 & go
        lambdaconf(1) = lambdaconf(1) - 2*sigm_app;
        lambdaconf(2) = lambdaconf(2) - 2*sigm_app;
        test1 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(1));
        test2 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(2));
        if test1==1&test2==1
          go=false;
        end
        testiter=testiter+1;
      end
    end
    mean_app = (lambdaconf(2)+lambdaconf(1))/2;
  elseif test2 == 0
    lambdaconf(2) = lambdaconf(2) + 3*sigm_app;
    test2 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(2));
    if test2 == 0
      go=true;
      while testiter<10 & go
        lambdaconf(1) = lambdaconf(1) + 2*sigm_app;
        lambdaconf(2) = lambdaconf(2) + 2*sigm_app;
        test1 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(1));
        test2 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(2));
        if test1==1&test2==1
          go=false;
        end
        testiter=testiter+1;
      end
    end
    mean_app = (lambdaconf(2)+lambdaconf(1))/2;
  end

  % Integrate with quad
  [m_0, fhncnt] = quadgk(zm, lambdaconf(1), lambdaconf(2));
  
  %         t=linspace(lambdaconf(1),lambdaconf(2),100);
  %         plot(t,zm(t))
  %         keyboard
  
  [g_i(1), fhncnt] = quadgk( @(f) zsigma2(f).*zm(f) , lambdaconf(1), lambdaconf(2));
  g_i(1) = g_i(1)/m_0*sigma2;
  
  if ~isempty(lik.p.nu)
    [g_i(2), fhncnt] = quadgk(@(f) znu(f).*zm(f) , lambdaconf(1), lambdaconf(2));
    g_i(2) = g_i(2)/m_0.*nu.*log(nu);
  end
  
  function integrand = zeroth_moment(f)
    r = yy-f;
    term = gammaln((nu + 1) / 2) - gammaln(nu/2) -log(nu.*pi.*sigma2)/2;
    integrand = exp(term + log(1 + r.^2./nu./sigma2) .* (-(nu+1)/2));
    integrand = integrand.*exp(- 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2);
  end        

  function g = deriv_nu(f)
    r = yy-f;
    temp = 1 + r.^2./nu./sigma2;
    g = psi((nu+1)/2)./2 - psi(nu/2)./2 - 1./(2.*nu) - log(temp)./2 + (nu+1)./(2.*temp).*(r./nu).^2./sigma2;
  end

  function g = deriv_sigma2(f)
    r = yy-f;
    g  = -1/sigma2/2 + (nu+1)./2.*r.^2./(nu.*sigma2.^2 + r.^2.*sigma2);
  end

end


function [g_i] = lik_t_siteDeriv2(likelih, y, i1, sigm2_i, myy_i, z, eta, Zm)
%LIKELIH_T_SITEDERIV   Evaluate the expectation of the gradient
%                           of the log likelihood term with respect
%                           to the likelihood parameters for EP
%
%   Description
%   [M_0, M_1, M2] = LIKELIH_T_TILTEDMOMENTS(LIKELIH, Y, I, S2, MYY)
%   takes a likelihood data structure LIKELIH, observations Y, index I
%   and cavity variance S2 and mean MYY. Returns E_f [d log
%   p(y_i|f_i) /d a], where a is the likelihood parameter and the
%   expectation is over the marginal posterior. This term is
%   needed when evaluating the gradients of the marginal
%   likelihood estimate Z_EP with respect to the likelihood
%   parameters (see Seeger (2008): Expectation propagation for
%   exponential families)
%
%   See also
%   GPEP_G
    
  if nargin<7
    eta=1;
  end
    
  znu = @deriv_nu;
  zsigma2 = @deriv_sigma2;
  
  %tol = 1e-8;
  yy = y(i1);
  nu = likelih.nu;
  sigma2 = likelih.sigma2;
  %sigma = sqrt(sigma2);
  
  % limiting distribution (nu -> infinity)
  %sigma=sqrt(sigma2);
  Vg=1/(1/sigm2_i +eta/sigma2);
  mg=Vg*(myy_i/sigm2_i +yy*eta/sigma2);
  sigm_i=sqrt(sigm2_i);
  sg=sqrt(Vg);
  
  % set integration limits and scaling
  nu_lim=1e6;
  if nu<nu_lim
    if mg>myy_i
      lambdaconf=[myy_i-6*sigm_i,max(mg+6*sg,myy_i+6*sigm_i)];
      t=linspace(myy_i,mg,10);
    else
      lambdaconf=[min(mg-6*sg,myy_i-6*sigm_i),myy_i+6*sigm_i];
      t=linspace(mg,myy_i,10);
    end
    lpt_max=max(lpt(t,0));
    C=log(1)-lpt_max;
  else
    lambdaconf=[mg-6*sg,mg+6*sg];
    C=log(1)-lpt(mg,0);
  end
  
  % set lower bound for log p(y,f)
  %C=0;
  %lpt_min=max(lpt(lambdaconf(1),C),lpt(lambdaconf(2),C));
  %C=-20-lpt_min;
  
  % Integrate with quadrature
  RTOL = 1.e-12;
  ATOL = 1.e-7;
  
  zm=@(f) exp(lpt(f,C));
  
  if nu>nu_lim
    % the limiting normal observation model
    g_i(1) = (-0.5/(sigm2_i+sigma2) +0.5*(yy-myy_i)^2 /(sigm2_i+sigma2)^2 ) *sigma2;
    
    if ~likelih.fix_nu
      g_i(2) = 0;
    end
  else
    % Integrate with quad
    %[m_0, fhncnt] = quadgk(zm, lambdaconf(1), lambdaconf(2),'AbsTol',ATOL,'RelTol',RTOL)
    
    % Use the normalization determined in the likelih_t_tiltedMoments
    m_0=Zm*exp(C);
    
    [g_i(1), fhncnt] = quadgk( @(f) zsigma2(f).*zm(f) , lambdaconf(1), lambdaconf(2),'AbsTol',ATOL,'RelTol',RTOL);
    g_i(1) = g_i(1)/m_0*sigma2;
    
    if ~likelih.fix_nu
      [g_i(2), fhncnt] = quadgk(@(f) znu(f).*zm(f) , lambdaconf(1), lambdaconf(2),'AbsTol',ATOL,'RelTol',RTOL);
      g_i(2) = g_i(2)/m_0.*nu.*log(nu);
    end
    
  end
  
  if any(~isfinite(g_i)) || any(~isreal(g_i)) %|| nu > nu_lim
    t=linspace(lambdaconf(1), lambdaconf(2), 500);
    plot(t,zm(t),'b',t,zm(t).*deriv_nu(t),'r',t,zm(t).*deriv_sigma2(t),'g')
    keyboard
  end
  
  function integrand = lpt(f,C)
    r = yy-f;
    if nu<nu_lim
      lpdf = gammaln((nu + 1) / 2) - gammaln(nu/2) -log(nu.*pi.*sigma2)/2;
      lpdf = lpdf + log(1 + r.^2./nu./sigma2) .* (-(nu+1)/2);
    else
      lpdf = log((nu+1)/2)/2 -log(nu.*pi.*sigma2)/2;
      %lpdf = lpdf + log1p(r.^2./nu./sigma2) .* (-(nu+1)/2);
      lpdf = lpdf + log(1+r.^2./nu./sigma2) .* (-(nu+1)/2);
    end
    %integrand = exp(lpdf*eta);
    %integrand = integrand.*exp(- 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2);
    integrand = lpdf*eta - 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2;
    integrand = integrand+C;
  end

  function g = deriv_nu(f)
    r = yy-f;
    temp = 1 + r.^2./nu./sigma2;
    g = psi((nu+1)/2)/2 - psi(nu/2)/2 - 1/(2*nu) - log(temp)/2 + (nu+1)./(2*temp) .* (r/nu).^2 /sigma2;
  end

  function g = deriv_sigma2(f)
    r = yy-f;
    g  = -1/sigma2/2 + (nu+1)/2 * r.^2 ./ (nu*sigma2.^2 + r.^2 *sigma2);
  end

end


function [f, a] = lik_t_optimizef(gp, y, K, Lav, K_fu)
%LIK_T_OPTIMIZEF  function to optimize the latent variables
%                 with EM algorithm
%
%  Description:
%    [F, A] = LIK_T_OPTIMIZEF(GP, Y, K, Lav, K_fu) Takes Gaussian
%    process structure GP, observations Y and the covariance
%    matrix K. Solves the posterior mode of F using EM algorithm
%    and evaluates A = (K + W)\Y as a sideproduct. Lav and K_fu
%    are needed for sparse approximations. For details, see
%    Vanhatalo, Jyl�nki and Vehtari (2009): Gaussian process
%    regression with Student-t likelihood.
%
  
  iter = 1;
  sigma2 = gp.lik.sigma2;
  nu = gp.lik.nu;
  n = length(y);
  
  switch gp.type
    case 'FULL'            
      iV = ones(n,1)./sigma2;
      siV = sqrt(iV);
      B = eye(n) + siV*siV'.*K;
      L = chol(B)';
      b = iV.*y;
      a = b - siV.*(L'\(L\(siV.*(K*b))));
      f = K*a;
      while iter < 200
        fold = f;               
        iV = (nu+1) ./ (nu.*sigma2 + (y-f).^2);
        siV = sqrt(iV);
        B = eye(n) + siV*siV'.*K;
        L = chol(B)';
        b = iV.*y;
        a = b - siV.*(L'\(L\(siV.*(K*b))));
        f = K*a;
        
        if max(abs(f-fold)) < 1e-8
          break
        end
        iter = iter + 1;
      end
    case 'FIC'
      K_uu = K;
      
      Luu = chol(K_uu)';
      B=Luu\(K_fu');       % u x f

      K = diag(Lav) + B'*B;
      
      iV = ones(n,1)./sigma2;
      siV = sqrt(iV);
      B = eye(n) + siV*siV'.*K;
      L = chol(B)';
      b = iV.*y;
      a = b - siV.*(L'\(L\(siV.*(K*b))));
      f = K*a;
      while iter < 200
        fold = f;                
        iV = (nu+1) ./ (nu.*sigma2 + (y-f).^2);
        siV = sqrt(iV);
        B = eye(n) + siV*siV'.*K;
        L = chol(B)';
        b = iV.*y;
        a = b - siV.*(L'\(L\(siV.*(K*b))));
        f = K*a;
        
        if max(abs(f-fold)) < 1e-8
          break
        end
        iter = iter + 1;
      end
  end
  
end

function upfact = lik_t_upfact(gp, y, mu, ll)
  nu = gp.lik.nu;
  sigma = sqrt(gp.lik.sigma2);

  fh_e = @(f) t_pdf(f, nu, y, sigma).*norm_pdf(f, mu, ll);
  EE = quadgk(fh_e, -40, 40);
  
  
  fm = @(f) f.*t_pdf(f, nu, y, sigma).*norm_pdf(f, mu, ll)./EE;
  mm  = quadgk(fm, -40, 40);
  
  fV = @(f) (f - mm).^2.*t_pdf(f, nu, y, sigma).*norm_pdf(f, mu, ll)./EE;
  Varp = quadgk(fV, -40, 40);
  
  upfact = -(Varp - ll)./ll^2;
end

function [lpy, Ey, Vary] = lik_t_predy(lik, Ef, Varf, y, z)
%LIK_T_PREDY    Returns the predictive mean, variance and density of y
%
%  Description      
%    LPY = LIK_T_PREDY(LIK, EF, VARF YT)
%    Returns logarithm of the predictive density PY of YT, that is 
%       p(yt | zt) = \int p(yt | f, zt) p(f|y) df.
%    This requires also the observations YT.
%
%    [LPY, EY, VARY] = LIK_T_PREDY(LIK, EF, VARF) takes a likelihood
%    structure LIK, posterior mean EF and posterior Variance
%    VARF of the latent variable and returns the posterior
%    predictive mean EY and variance VARY of the observations
%    related to the latent variables
%        

%
%  See also
%    GPLA_PRED, GPEP_PRED, GPMC_PRED

  nu = lik.nu;
  sigma2 = lik.sigma2;
  sigma = sqrt(sigma2);
  
  Ey = zeros(size(Ef));
  EVary = zeros(size(Ef));
  VarEy = zeros(size(Ef)); 
  lpy = zeros(size(Ef));
  if nargout > 1
%       for i1=1:length(Ef)
%         %%% With quadrature
%         ci = sqrt(Varf(i1));
% 
%         F = @(x) x.*norm_pdf(x,Ef(i1),sqrt(Varf(i1)));
%         Ey(i1) = quadgk(F,Ef(i1)-6*ci,Ef(i1)+6*ci);
% 
%         F2 = @(x) (nu./(nu-2).*sigma2).*norm_pdf(x,Ef(i1),sqrt(Varf(i1)));
%         EVary(i1) = quadgk(F2,Ef(i1)-6*ci,Ef(i1)+6*ci);
% 
%         F3 = @(x) x.^2.*norm_pdf(x,Ef(i1),sqrt(Varf(i1)));
%         VarEy(i1) = quadgk(F3,Ef(i1)-6*ci,Ef(i1)+6*ci) - Ey(i1).^2;
%       end
%       Vary = EVary + VarEy;
      
      Ey = Ef;
      nu_p=max(2.5,nu);
      Vary=nu_p./(nu_p-2).*sigma2 +Varf;
  end
  

  for i2 = 1:length(Ef)
    mean_app = Ef(i2);
    sigm_app = sqrt(Varf(i2));
    
    pd = @(f) t_pdf(y(i2), nu, f, sigma).*norm_pdf(f,Ef(i2),sqrt(Varf(i2)));
    lpy(i2) = log(quadgk(pd, mean_app - 12*sigm_app, mean_app + 12*sigm_app));
  end

  
end

function reclik = lik_t_recappend(reclik, ri, lik)
%RECAPPEND  Record append
%  Description
%    RECCF = GPCF_SEXP_RECAPPEND(RECCF, RI, GPCF) takes old
%    covariance function record RECCF, record index RI, RECAPPEND
%    returns a structure RECCF.

% Initialize record
  if nargin == 2
    reclik.type = 'Student-t';

    % Initialize parameter
    reclik.nu = [];
    reclik.sigma2 = [];

    % Set the function handles
    reclik.fh.pak = @lik_t_pak;
    reclik.fh.unpak = @lik_t_unpak;
    reclik.fh.lp = @lik_t_lp;
    reclik.fh.lpg = @lik_t_lpg;
    reclik.fh.ll = @lik_t_ll;
    reclik.fh.llg = @lik_t_llg;    
    reclik.fh.llg2 = @lik_t_llg2;
    reclik.fh.llg3 = @lik_t_llg3;
    reclik.fh.tiltedMoments = @lik_t_tiltedMoments;
    reclik.fh.tiltedMoments2 = @lik_t_tiltedMoments2;
    reclik.fh.siteDeriv = @lik_t_siteDeriv;
    reclik.fh.siteDeriv2 = @lik_t_siteDeriv2;
    reclik.fh.optimizef = @lik_t_optimizef;
    reclik.fh.upfact = @lik_t_upfact;
    reclik.fh.predy = @lik_t_predy;
    reclik.fh.recappend = @lik_t_recappend;
    return
  end

  reclik.nu(ri,:) = lik.nu;
  reclik.sigma2(ri,:) = lik.sigma2;
end
