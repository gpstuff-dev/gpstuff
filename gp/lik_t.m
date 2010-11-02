function lik = lik_t(varargin)
%LIK_T Create a Student-t likelihood structure 
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

% allow use with or without init and set options
  if nargin<1
    do='init';
  elseif ischar(varargin{1})
    switch varargin{1}
      case 'init'
        do='init';varargin(1)=[];
      case 'set'
        do='set';varargin(1)=[];
      otherwise
        do='init';
    end
  elseif isstruct(varargin{1})
    do='set';
  else
    error('Unknown first argument');
  end

  switch do
    case 'init'
      % Initialize the likelihood structure
      lik.type = 'Student-t';
      
      % Default parameter values
      lik.nu = 4;
      lik.sigma2 = 1;
      
      % Default priors
      lik.p.sigma2 = prior_logunif;
      lik.p.nu = prior_fixed;
      
      % Set the function handles to the nested functions
      lik.fh.pak = @lik_t_pak;
      lik.fh.unpak = @lik_t_unpak;
      lik.fh.priore = @lik_t_priore;
      lik.fh.priorg = @lik_t_priorg;
      lik.fh.ll = @lik_t_ll;
      lik.fh.llg = @lik_t_llg;    
      lik.fh.llg2 = @lik_t_llg2;
      lik.fh.llg3 = @lik_t_llg3;
      lik.fh.tiltedMoments = @lik_t_tiltedMoments;
      lik.fh.siteDeriv = @lik_t_siteDeriv;
      lik.fh.optimizef = @lik_t_optimizef;
      lik.fh.upfact = @lik_t_upfact;
      lik.fh.predy = @lik_t_predy;
      lik.fh.recappend = @lik_t_recappend;

      if numel(varargin) > 0 & mod(numel(varargin),2) ~=0
        error('Wrong number of arguments')
      end
      % Loop through all the parameter values that are changed
      for i=1:2:length(varargin)-1
        switch varargin{i}
          case 'nu'
            lik.nu = varargin{i+1};
          case 'sigma2'
            lik.sigma2 = varargin{i+1};
          case 'sigma2_prior'
            lik.p.sigma2 = varargin{i+1}; 
          case 'nu_prior'
            lik.p.nu = varargin{i+1}; 
          otherwise
            error('Wrong parameter name!')
        end
      end

    case 'set'
      % Set the parameter values of covariance function
      if numel(varargin)~=1 & mod(numel(varargin),2) ~=1
        error('Wrong number of arguments')
      end
      lik = varargin{1};
      % Loop through all the parameter values that are changed
      for i=2:2:length(varargin)-1
        switch varargin{i}
          case 'nu'
            lik.nu = varargin{i+1};
          case 'sigma2'
            lik.sigma2 = varargin{i+1};
          case 'sigma2_prior'
            lik.p.sigma2 = varargin{i+1}; 
          case 'nu_prior'
            lik.p.nu = varargin{i+1}; 
          otherwise
            error('Wrong parameter name!')
        end
      end
  end
  
  
  function w = lik_t_pak(lik)
  %LIK_T_PAK  Combine likelihood parameters into one vector.
  %
  %   Description 
  %   W = LIK_T_PAK(LIK) takes a
  %   likelihood data structure LIK and combines the parameters
  %   into a single row vector W.
  %     
  %
  %   See also
  %   LIK_T_UNPAK, GP_PAK
    
    w = [];
    i1 = 0;
    if ~isempty(lik.p.sigma2)
      i1 = 1;
      w(i1) = log(lik.sigma2);
    end
    if ~isempty(lik.p.nu)
      i1 = i1+1;
      w(i1) = log(log(lik.nu));
    end        
  end


  function [lik, w] = lik_t_unpak(w, lik)
  %LIK_T_UNPAK  Extract likelihood parameters from the vector.
  %
  %   Description
  %   W = LIK_T_UNPAK(W, LIK) takes a likelihood data
  %   structure LIK and extracts the parameters from the vector W
  %   to the LIK structure.
  %     
  %
  %   See also
  %   LIK_T_PAK, GP_UNPAK

    i1 = 0;
    if ~isempty(lik.p.sigma2)
      i1 = 1;
      lik.sigma2 = exp(w(i1));
    end
    if ~isempty(lik.p.nu) 
      i1 = i1+1;
      lik.nu = exp(exp(w(i1)));
    end
  end


  function logPrior = lik_t_priore(lik)
  %LIK_T_PRIORE  log(prior) of the likelihood hyperparameters
  %
  %   Description
  %   E = LIK_T_PRIORE(LIK) takes a likelihood data 
  %   structure LIK and returns log(p(th)), where th collects 
  %   the hyperparameters.
  %
  %   See also
  %   LIK_T_LLG, LIK_T_LLG3, LIK_T_LLG2, GPLA_E
    
    v = lik.nu;
    sigma2 = lik.sigma2;
    logPrior = 0;
    
    if ~isempty(lik.p.sigma2) 
      logPrior = logPrior + feval(lik.p.sigma2.fh.e, lik.sigma2, lik.p.sigma2) -log(sigma2);
    end
    if ~isempty(lik.p.nu)
      logPrior = logPrior + feval(lik.p.nu.fh.e, lik.nu, lik.p.nu)  - log(v) - log(log(v));
    end
  end
  
  function glogPrior = lik_t_priorg(lik)
  %LIK_T_PRIORG    d log(prior)/dth of the likelihood 
  %                         hyperparameters th
  %
  %   Description
  %   E = LIK_T_PRIORG(LIK, Y, F) takes a likelihood 
  %   data structure LIK and returns d log(p(th))/dth, where 
  %   th collects the hyperparameters.
  %
  %   See also
  %   LIK_T_LLG, LIK_T_LLG3, LIK_T_LLG2, GPLA_G
    
  % Evaluate the gradients of log(prior)

    v = lik.nu;
    sigma2 = lik.sigma2;
    glogPrior = [];
    i1 = 0;
    
    if ~isempty(lik.p.sigma2) 
      i1 = i1+1;
      glogPrior(i1) = feval(lik.p.sigma2.fh.g, lik.sigma2, lik.p.sigma2).*sigma2 - 1;
    end
    if ~isempty(lik.p.nu) 
      i1 = i1+1;
      glogPrior(i1) = feval(lik.p.nu.fh.g, lik.nu, lik.p.nu).*v.*log(v) - log(v) - 1;
    end    
  end
  
  function logLik = lik_t_ll(lik, y, f, z)
  %LIK_T_LL    Log likelihood
  %
  %   Description
  %   E = LIK_T_LL(LIK, Y, F) takes a likelihood
  %   data structure LIK, observations Y, and latent values
  %   F. Returns the log likelihood, log p(y|f,z).
  %
  %   See also
  %   LIK_T_LLG, LIK_T_LLG3, LIK_T_LLG2, GPLA_E

    r = y-f;
    v = lik.nu;
    sigma2 = lik.sigma2;

    term = gammaln((v + 1) / 2) - gammaln(v/2) -log(v.*pi.*sigma2)/2;
    logLik = term + log(1 + (r.^2)./v./sigma2) .* (-(v+1)/2);
    logLik = sum(logLik);
  end

  
  function deriv = lik_t_llg(lik, y, f, param, z)
  %LIK_T_LLG    Gradient of log likelihood (energy)
  %
  %   Description
  %   G = LIK_T_LLG(LIK, Y, F, PARAM) takes a likelihood
  %   data structure LIK, observations Y, and latent values
  %   F. Returns the gradient of log likelihood with respect to
  %   PARAM. At the moment PARAM can be 'hyper' or 'latent'.
  %
  %   See also
  %   LIK_T_LL, LIK_T_LLG2, LIK_T_LLG3, GPLA_E
    
    r = y-f;
    v = lik.nu;
    sigma2 = lik.sigma2;
    
    switch param
      case 'hyper'
        n = length(y);

        % Derivative with respect to sigma2
        deriv(1) = -n./sigma2/2 + (v+1)./2.*sum(r.^2./(v.*sigma2.^2+r.^2*sigma2));
        
        % correction for the log transformation
        deriv(1) = deriv(1).*sigma2;
        if ~isempty(lik.p.nu)
          % Derivative with respect to nu
          deriv(2) = 0.5.* sum(psi((v+1)./2) - psi(v./2) - 1./v - log(1+r.^2./(v.*sigma2)) + (v+1).*r.^2./(v.^2.*sigma2 + v.*r.^2));
          
          % correction for the log transformation
          deriv(2) = deriv(2).*v.*log(v);
        end
      case 'latent'
        deriv  = (v+1).*r ./ (v.*sigma2 + r.^2);            
    end
    
  end


  function g2 = lik_t_llg2(lik, y, f, param, z)
  %LIK_T_LLG2  Second gradients of log likelihood (energy)
  %
  %   Description        
  %   G2 = LIK_T_LLG2(LIK, Y, F, PARAM) takes a likelihood
  %   data structure LIK, observations Y, and latent values
  %   F. Returns the hessian of log likelihood with respect to
  %   PARAM. At the moment PARAM can be only 'latent'. G2 is a
  %   vector with diagonal elements of the hessian matrix (off
  %   diagonals are zero).
  %
  %   See also
  %   LIK_T_LL, LIK_T_LLG, LIK_T_LLG3, GPLA_E

    r = y-f;
    v = lik.nu;
    sigma2 = lik.sigma2;

    switch param
      case 'hyper'
        
      case 'latent'
        % The Hessian d^2 /(dfdf)
        g2 =  (v+1).*(r.^2 - v.*sigma2) ./ (v.*sigma2 + r.^2).^2;
      case 'latent+hyper'
        % gradient d^2 / (dfds2)
        g2 = -v.*(v+1).*r ./ (v.*sigma2 + r.^2).^2;
        
        % Correction for the log transformation
        g2 = g2.*sigma2;
        if ~isempty(lik.p.nu)
          % gradient d^2 / (dfdnu)
          g2(:,2) = r./(v.*sigma2 + r.^2) - sigma2.*(v+1).*r./(v.*sigma2 + r.^2).^2;

          % Correction for the log transformation
          g2(:,2) = g2(:,2).*v.*log(v);
        end
    end
  end    
  
  function third_grad = lik_t_llg3(lik, y, f, param, z)
  %LIK_T_LLG3  Third gradients of log likelihood (energy)
  %
  %   Description
  %   G3 = LIK_T_LLG3(LIK, Y, F, PARAM) takes a likelihood 
  %   data structure LIK, observations Y and latent values F and
  %   returns the third gradients of log likelihood with respect to
  %   PARAM. At the moment PARAM can be only 'latent'. G3 is a
  %   vector with third gradients.
  %
  %   See also
  %   LIK_T_LL, LIK_T_LLG, LIK_T_LLG2, GPLA_E, GPLA_G

    r = y-f;
    v = lik.nu;
    sigma2 = lik.sigma2;
    
    switch param
      case 'hyper'
        
      case 'latent'
        % Return the diagonal of W differentiated with respect to latent values / dfdfdf
        third_grad = (v+1).*(2.*r.^3 - 6.*v.*sigma2.*r) ./ (v.*sigma2 + r.^2).^3;
      case 'latent2+hyper'
        % Return the diagonal of W differentiated with respect to likelihood parameters / dfdfds2
        third_grad = (v+1).*v.*( v.*sigma2 - 3.*r.^2) ./ (v.*sigma2 + r.^2).^3;
        third_grad = third_grad.*sigma2;
        if ~isempty(lik.p.nu)
          % dfdfdnu
          third_grad(:,2) = (r.^2-2.*v.*sigma2-sigma2)./(v.*sigma2 + r.^2).^2 - 2.*sigma2.*(r.^2-v.*sigma2).*(v+1)./(v.*sigma2 + r.^2).^3;
          third_grad(:,2) = third_grad(:,2).*v.*log(v);
        end
    end
  end


  function [m_0, m_1, sigm2hati1] = lik_t_tiltedMoments(lik, y, i1, sigm2_i, myy_i, z)
  %LIK_T_TILTEDMOMENTS    Returns the marginal moments for EP algorithm
  %
  %   Description
  %   [M_0, M_1, M2] = LIK_T_TILTEDMOMENTS(LIK, Y, I, S2, MYY, Z) 
  %   takes a likelihood data structure LIK, incedence counts Y, 
  %   expected counts Z, index I and cavity variance S2 and mean
  %   MYY. Returns the zeroth moment M_0, mean M_1 and variance M_2
  %   of the posterior marginal (see Rasmussen and Williams (2006):
  %   Gaussian processes for Machine Learning, page 55).
  %
  %   See also
  %   GPEP_E

    
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
  
  
  function [g_i] = lik_t_siteDeriv(lik, y, i1, sigm2_i, myy_i, z)
  %LIK_T_SITEDERIV   Evaluate the expectation of the gradient
  %                           of the log likelihood term with respect
  %                           to the likelihood parameters for EP 
  %
  %   Description
  %   [M_0, M_1, M2] = LIK_T_TILTEDMOMENTS(LIK, Y, I, S2, MYY)         
  %   takes a likelihood data structure LIK, observations Y, index I
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

  function [f, a] = lik_t_optimizef(gp, y, K, Lav, K_fu)
  %LIK_T_OPTIMIZEF   function to optimize the latent variables
  %                      with EM algorithm

  % Description:
  % [F, A] = LIK_T_OPTIMIZEF(GP, Y, K, Lav, K_fu) Takes Gaussian
  % process data structure GP, observations Y and the covariance
  % matrix K. Solves the posterior mode of F using EM algorithm and
  % evaluates A = (K + W)\Y as a sideproduct. Lav and K_fu are
  % needed for sparse approximations. For details, see Vanhatalo,
  % Jyl�nki and Vehtari (2009): Gaussian process regression with
  % Student-t likelihood.      
    
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

  function [Ey, Vary, Py] = lik_t_predy(lik, Ef, Varf, y, z)
  %LIK_T_PREDY    Returns the predictive mean, variance and density of y
  %
  %   Description         
  %   [EY, VARY] = LIK_T_PREDY(LIK, EF, VARF)
  %   takes a likelihood data structure LIK, posterior mean EF
  %   and posterior Variance VARF of the latent variable and returns
  %   the posterior predictive mean EY and variance VARY of the
  %   observations related to the latent variables
  %        
  %   [Ey, Vary, PY] = LIK_T_PREDY(LIK, EF, VARF YT)
  %   Returns also the predictive density of YT, that is 
  %        p(yt | zt) = \int p(yt | f, zt) p(f|y) df.
  %   This requires also the observations YT.
  %
  % See also:
  % la_pred, ep_pred, mc_pred

    nu = lik.nu;
    sigma2 = lik.sigma2;
    sigma = sqrt(sigma2);
    
  % $$$         sampf = gp_rnd(gp, tx, ty, x, [], [], 400);
  % $$$         r = trand(nu,size(sampf));
  % $$$         r = sampf + sqrt(sigma).*r;
  % $$$         
  % $$$         Ey = mean(r);
  % $$$         Vary = var(r, 0, 2);
    Ey = zeros(size(Ef));
    EVary = zeros(size(Ef));
    VarEy = zeros(size(Ef)); 
    Py = zeros(size(Ef));
    for i1=1:length(Ef)
      %%% With quadrature
      ci = sqrt(Varf(i1));

      F = @(x) x.*normpdf(x,Ef(i1),sqrt(Varf(i1)));
      Ey(i1) = quadgk(F,Ef(i1)-6*ci,Ef(i1)+6*ci);
      
      F2 = @(x) (nu./(nu-2).*sigma2).*normpdf(x,Ef(i1),sqrt(Varf(i1)));
      EVary(i1) = quadgk(F2,Ef(i1)-6*ci,Ef(i1)+6*ci);
      
      F3 = @(x) x.^2.*normpdf(x,Ef(i1),sqrt(Varf(i1)));
      VarEy(i1) = quadgk(F3,Ef(i1)-6*ci,Ef(i1)+6*ci) - Ey(i1).^2;
    end
    Vary = EVary + VarEy;
    
    if nargout > 2
      for i2 = 1:length(Ef)
        mean_app = Ef(i2);
        sigm_app = sqrt(Varf(i2));
        
        pd = @(f) t_pdf(y(i2), nu, f, sigma).*norm_pdf(f,Ef(i2),sqrt(Varf(i2)));
        Py(i2) = quadgk(pd, mean_app - 12*sigm_app, mean_app + 12*sigm_app);
      end
    end
    
  end

  
  function reclik = lik_t_recappend(reclik, ri, lik)
  % RECAPPEND - Record append
  %          Description
  %          RECCF = GPCF_SEXP_RECAPPEND(RECCF, RI, GPCF) takes old covariance
  %          function record RECCF, record index RI, RECAPPEND returns a
  %          structure RECCF containing following record fields:
  %          lengthHyper    =
  %          lengthHyperNu  =
  %          lengthScale    =
  %          magnSigma2     =

  % Initialize record
    if nargin == 2
      reclik.type = 'Student-t';

      % Initialize parameter
      reclik.nu = [];
      reclik.sigma2 = [];

      % Set the function handles
      reclik.fh.pak = @lik_t_pak;
      reclik.fh.unpak = @lik_t_unpak;
      reclik.fh.priore = @lik_t_priore;
      reclik.fh.priorg = @lik_t_priorg;
      reclik.fh.ll = @lik_t_ll;
      reclik.fh.llg = @lik_t_llg;    
      reclik.fh.llg2 = @lik_t_llg2;
      reclik.fh.llg3 = @lik_t_llg3;
      reclik.fh.tiltedMoments = @lik_t_tiltedMoments;
      reclik.fh.siteDeriv = @lik_t_siteDeriv;
      reclik.fh.optimizef = @lik_t_optimizef;
      reclik.fh.upfact = @lik_t_upfact;
      reclik.fh.predy = @lik_t_predy;
      reclik.fh.recappend = @lik_t_recappend;
      return
    end

    reclik.nu(ri,:) = lik.nu;
    reclik.sigma2(ri,:) = lik.sigma2;
  end
end


