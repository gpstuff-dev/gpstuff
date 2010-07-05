function likelih = likelih_t(do, varargin)
%LIKELIH_T       Create a Student-t likelihood structure 
%
%	Description
%
%	LIKELIH = LIKELIH_T('INIT') Create and initialize Student-t 
%        likelihood. 
%
%       The likelihood is defined as follows:
%                    __ n
%        p(y|f, z) = || i=1 C(nu,s2) * ( 1 + 1/nu * (y_i - f_i)^2/s2 )^(-(nu+1)/2)
%
%       where nu is the degrees of freedom, s2 the scale and f_i
%       the latent variable defining the mean. C(nu,s2) is constant
%       depending on nu and s2.
%
%	The fields in LIKELIH are:
%	  type                     = 'Student-t'
%         likelih.nu               = nu;
%         likelih.sigma2           = sigma2;
%         likelih.fix_nu           = 1 for keeping nu fixed, 0 for inferring it (1)
%         p                        = Prior structure for hyperparameters
%                                    of likelihood.
%                                    Default prior for the dispersion 
%                                    parameter is logunif.
%         likelih.fh_pak           = function handle to pak
%         likelih.fh_unpak         = function handle to unpak
%         likelih.fh_e             = function handle to the log likelihood
%         likelih.fh_g             = function handle to the gradient of 
%                                    the log likelihood
%         likelih.fh_g2            = function handle to the second gradient
%                                    of the log likelihood
%         likelih.fh_g3            = function handle to the third gradient  
%                                    of the log likelihood
%         likelih.fh_tiltedMoments = function handle to evaluate posterior
%                                    moments for EP
%         likelih.fh_siteDeriv     = function handle to help gradient evaluations
%                                    with respect to likelihood parameters in EP
%         likelih.fh_predy         = function handle to evaluate predictive 
%                                    density of y
%         likelih.fh_optimizef     = function handle to optimization of latent 
%                                    values for gpla_e
%         likelih.fh_recappend     = function handle to record append
%
%	LIKELIH = LIKELIH_T('SET', LIKELIH, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH. The fields that 
%       can be modified are:
%
%             'sigma2'             : set the sigma2
%             'nu'                 : set the degrees of freedom
%             'sigma2_prior'       : set the prior structure for sigma2
%             'nu_prior'           : set the prior structure for nu
%
%       Note! If the prior structure is set to empty matrix
%       (e.g. 'nu_prior', []) then the parameter in question is
%       considered fixed and it is not handled in optimization, grid
%       integration, MCMC etc.
%
%	See also
%       LIKELIH_LOGIT, LIKELIH_PROBIT, LIKELIH_T

% Copyright (c) 2009-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 1
        error('Not enough arguments')
    end

    % Initialize the covariance function
    if strcmp(do, 'init')
        likelih.type = 'Student-t';
        
        % Set parameters
        likelih.nu = 4;
        likelih.sigma2 = 1;
        likelih.fix_nu = 1;
        
        % Initialize prior structure
        likelih.p.sigma2 = prior_logunif('init');
        likelih.p.nu = prior_logunif('init');
        
        % Set the function handles to the nested functions
        likelih.fh_pak = @likelih_t_pak;
        likelih.fh_unpak = @likelih_t_unpak;
        likelih.fh_priore = @likelih_t_priore;
        likelih.fh_priorg = @likelih_t_priorg;
        likelih.fh_e = @likelih_t_e;
        likelih.fh_g = @likelih_t_g;    
        likelih.fh_g2 = @likelih_t_g2;
        likelih.fh_g3 = @likelih_t_g3;
        likelih.fh_tiltedMoments = @likelih_t_tiltedMoments;
        likelih.fh_siteDeriv = @likelih_t_siteDeriv;
        likelih.fh_optimizef = @likelih_t_optimizef;
        likelih.fh_upfact = @likelih_t_upfact;
        likelih.fh_predy = @likelih_t_predy;
        likelih.fh_recappend = @likelih_t_recappend;

        if nargin > 1
            if mod(nargin,2) == 0 
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=1:2:length(varargin)-1
                switch varargin{i}
                  case 'nu'
                    likelih.nu = varargin{i+1};
                  case 'sigma2'
                    likelih.sigma2 = varargin{i+1};
                  case 'fix_nu'
                    likelih.fix_nu = varargin{i+1};
                  case 'sigma2_prior'
                    likelih.p.sigma2 = varargin{i+1}; 
                  case 'nu_prior'
                    likelih.p.nu = varargin{i+1}; 
                  otherwise
                    error('Wrong parameter name!')
                end
            end
        end
    end

    % Set the parameter values of covariance function
    if strcmp(do, 'set')
        if mod(nargin,2) ~=0
            error('Wrong number of arguments')
        end
        likelih = varargin{1};
        % Loop through all the parameter values that are changed
        for i=2:2:length(varargin)-1
            switch varargin{i}
              case 'nu'
                likelih.nu = varargin{i+1};
              case 'sigma2'
                likelih.sigma2 = varargin{i+1};
              case 'fix_nu'
                likelih.fix_nu = varargin{i+1};
              case 'sigma2_prior'
                likelih.p.sigma2 = varargin{i+1}; 
              case 'nu_prior'
                likelih.p.nu = varargin{i+1}; 
              otherwise
                error('Wrong parameter name!')
            end
        end
    end


    function w = likelih_t_pak(likelih)
    %LIKELIH_T_PAK  Combine likelihood parameters into one vector.
    %
    %	Description 
    %   W = LIKELIH_T_PAK(LIKELIH) takes a
    %   likelihood data structure LIKELIH and combines the parameters
    %   into a single row vector W.
    %	  
    %
    %	See also
    %	LIKELIH_T_UNPAK, GP_PAK
        
        w = [];
        i1 = 0;
        if ~isempty(likelih.p.sigma2)
            i1 = 1;
            w(i1) = log(likelih.sigma2);
        end
        if ~isempty(likelih.p.nu) && ~likelih.fix_nu
            i1 = i1+1;
            w(i1) = log(log(likelih.nu));
        end        
    end


    function [likelih, w] = likelih_t_unpak(w, likelih)
    %LIKELIH_T_UNPAK  Extract likelihood parameters from the vector.
    %
    %	Description
    %   W = LIKELIH_T_UNPAK(W, LIKELIH) takes a likelihood data
    %   structure LIKELIH and extracts the parameters from the vector W
    %   to the LIKELIH structure.
    %	  
    %
    %	See also
    %	LIKELIH_T_PAK, GP_UNPAK

        i1 = 0;
        if ~isempty(likelih.p.sigma2)
            i1 = 1;
            likelih.sigma2 = exp(w(i1));
        end
        if ~isempty(likelih.p.nu)  && ~likelih.fix_nu
            i1 = i1+1;
            likelih.nu = exp(exp(w(i1)));
        end
    end


    function logPrior = likelih_t_priore(likelih)
    %LIKELIH_T_PRIORE  log(prior) of the likelihood hyperparameters
    %
    %   Description
    %   E = LIKELIH_T_PRIORE(LIKELIH) takes a likelihood data 
    %   structure LIKELIH and returns log(p(th)), where th collects 
    %   the hyperparameters.
    %
    %   See also
    %   LIKELIH_T_G, LIKELIH_T_G3, LIKELIH_T_G2, GPLA_E
        
        v = likelih.nu;
        sigma2 = likelih.sigma2;
        logPrior = 0;
        
        if ~isempty(likelih.p.sigma2) 
            logPrior = logPrior + feval(likelih.p.sigma2.fh_e, likelih.sigma2, likelih.p.sigma2) -log(sigma2);
        end
        if ~isempty(likelih.p.nu) && ~likelih.fix_nu
            logPrior = logPrior + feval(likelih.p.nu.fh_e, likelih.nu, likelih.p.nu)  - log(v) - log(log(v));
        end
    end
    
    function glogPrior = likelih_t_priorg(likelih)
    %LIKELIH_T_PRIORG    d log(prior)/dth of the likelihood 
    %                         hyperparameters th
    %
    %   Description
    %   E = LIKELIH_T_PRIORG(LIKELIH, Y, F) takes a likelihood 
    %   data structure LIKELIH and returns d log(p(th))/dth, where 
    %   th collects the hyperparameters.
    %
    %   See also
    %   LIKELIH_T_G, LIKELIH_T_G3, LIKELIH_T_G2, GPLA_G
        
    % Evaluate the gradients of log(prior)

        v = likelih.nu;
        sigma2 = likelih.sigma2;
        glogPrior = [];
        i1 = 0;
        
        if ~isempty(likelih.p.sigma2) 
            i1 = i1+1;
            glogPrior(i1) = feval(likelih.p.sigma2.fh_g, likelih.sigma2, likelih.p.sigma2).*sigma2 - 1;
        end
        if ~isempty(likelih.p.nu)  && ~likelih.fix_nu
            i1 = i1+1;
            glogPrior(i1) = feval(likelih.p.nu.fh_g, likelih.nu, likelih.p.nu).*v.*log(v) - log(v) - 1;
        end    
    end
    
    function logLikelih = likelih_t_e(likelih, y, f, z)
    %LIKELIH_T_E    Log likelihood
    %
    %   Description
    %   E = LIKELIH_T_E(LIKELIH, Y, F) takes a likelihood
    %   data structure LIKELIH, observations Y, and latent values
    %   F. Returns the log likelihood, log p(y|f,z).
    %
    %   See also
    %   LIKELIH_T_G, LIKELIH_T_G3, LIKELIH_T_G2, GPLA_E

        r = y-f;
        v = likelih.nu;
        sigma2 = likelih.sigma2;

        term = gammaln((v + 1) / 2) - gammaln(v/2) -log(v.*pi.*sigma2)/2;
        logLikelih = term + log(1 + (r.^2)./v./sigma2) .* (-(v+1)/2);
        logLikelih = sum(logLikelih);
    end

    
    function deriv = likelih_t_g(likelih, y, f, param, z)
    %LIKELIH_T_G    Gradient of log likelihood (energy)
    %
    %   Description
    %   G = LIKELIH_T_G(LIKELIH, Y, F, PARAM) takes a likelihood
    %   data structure LIKELIH, observations Y, and latent values
    %   F. Returns the gradient of log likelihood with respect to
    %   PARAM. At the moment PARAM can be 'hyper' or 'latent'.
    %
    %   See also
    %   LIKELIH_T_E, LIKELIH_T_G2, LIKELIH_T_G3, GPLA_E
        
        r = y-f;
        v = likelih.nu;
        sigma2 = likelih.sigma2;
        
        switch param
          case 'hyper'
            n = length(y);

                % Derivative with respect to sigma2
                deriv(1) = -n./sigma2/2 + (v+1)./2.*sum(r.^2./(v.*sigma2.^2+r.^2*sigma2));
                
                % correction for the log transformation
                deriv(1) = deriv(1).*sigma2;
            if ~likelih.fix_nu
                % Derivative with respect to nu
                deriv(2) = 0.5.* sum(psi((v+1)./2) - psi(v./2) - 1./v - log(1+r.^2./(v.*sigma2)) + (v+1).*r.^2./(v.^2.*sigma2 + v.*r.^2));
                
                % correction for the log transformation
                deriv(2) = deriv(2).*v.*log(v);
            end
          case 'latent'
            deriv  = (v+1).*r ./ (v.*sigma2 + r.^2);            
        end
        
    end


    function g2 = likelih_t_g2(likelih, y, f, param, z)
    %LIKELIH_T_G2  Second gradients of log likelihood (energy)
    %
    %   Description        
    %   G2 = LIKELIH_T_G2(LIKELIH, Y, F, PARAM) takes a likelihood
    %   data structure LIKELIH, observations Y, and latent values
    %   F. Returns the hessian of log likelihood with respect to
    %   PARAM. At the moment PARAM can be only 'latent'. G2 is a
    %   vector with diagonal elements of the hessian matrix (off
    %   diagonals are zero).
    %
    %   See also
    %   LIKELIH_T_E, LIKELIH_T_G, LIKELIH_T_G3, GPLA_E

        r = y-f;
        v = likelih.nu;
        sigma2 = likelih.sigma2;

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
            if ~likelih.fix_nu
                % gradient d^2 / (dfdnu)
                g2(:,2) = r./(v.*sigma2 + r.^2) - sigma2.*(v+1).*r./(v.*sigma2 + r.^2).^2;

                % Correction for the log transformation
                g2(:,2) = g2(:,2).*v.*log(v);
            end
        end
    end    
    
    function third_grad = likelih_t_g3(likelih, y, f, param, z)
    %LIKELIH_T_G3  Third gradients of log likelihood (energy)
    %
    %   Description
    %   G3 = LIKELIH_T_G3(LIKELIH, Y, F, PARAM) takes a likelihood 
    %   data structure LIKELIH, observations Y and latent values F and
    %   returns the third gradients of log likelihood with respect to
    %   PARAM. At the moment PARAM can be only 'latent'. G3 is a
    %   vector with third gradients.
    %
    %   See also
    %   LIKELIH_T_E, LIKELIH_T_G, LIKELIH_T_G2, GPLA_E, GPLA_G

        r = y-f;
        v = likelih.nu;
        sigma2 = likelih.sigma2;
        
        switch param
          case 'hyper'
            
          case 'latent'
            % Return the diagonal of W differentiated with respect to latent values / dfdfdf
            third_grad = (v+1).*(2.*r.^3 - 6.*v.*sigma2.*r) ./ (v.*sigma2 + r.^2).^3;
          case 'latent2+hyper'
            % Return the diagonal of W differentiated with respect to likelihood parameters / dfdfds2
            third_grad = (v+1).*v.*( v.*sigma2 - 3.*r.^2) ./ (v.*sigma2 + r.^2).^3;
            third_grad = third_grad.*sigma2;
            if ~likelih.fix_nu
              % dfdfdnu
              third_grad(:,2) = (r.^2-2.*v.*sigma2-sigma2)./(v.*sigma2 + r.^2).^2 - 2.*sigma2.*(r.^2-v.*sigma2).*(v+1)./(v.*sigma2 + r.^2).^3;
              third_grad(:,2) = third_grad(:,2).*v.*log(v);
            end
        end
    end


    function [m_0, m_1, sigm2hati1] = likelih_t_tiltedMoments(likelih, y, i1, sigm2_i, myy_i, z)
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

        
        zm = @zeroth_moment;
        
        tol = 1e-8;
        yy = y(i1);
        nu = likelih.nu;
        sigma2 = likelih.sigma2;
                
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
    
    
    function [g_i] = likelih_t_siteDeriv(likelih, y, i1, sigm2_i, myy_i, z)
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

        zm = @zeroth_moment;
        znu = @deriv_nu;
        zsigma2 = @deriv_sigma2;
        
        tol = 1e-8;
        yy = y(i1);
        nu = likelih.nu;
        sigma2 = likelih.sigma2;

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
        
        if ~likelih.fix_nu
            [g_i(2), fhncnt] = quadgk(@(f) znu(f).*zm(f) , lambdaconf(1), lambdaconf(2));
            g_i(2) = g_i(2)/m_0.*nu.*log(nu);
        end
        
        % ------------------------------------------------
        % Plot the integrand to check that integration limits are ok
% $$$         clf;ff = [lambdaconf(1):0.01:lambdaconf(2)];
% $$$         plot([lambdaconf(1) lambdaconf(2)], [0 0], 'r');hold on;plot(ff, feval(zd, ff))
% $$$         drawnow;S = sprintf('iter %d, y=%d, avgE=%.1f, sigm_a=%.2f, sigm2_i=%.2f', i1, yy, avgE, sigm_app, sigm2_i);title(S);
% $$$         pause
        % ------------------------------------------------

     
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

    function [f, a] = likelih_t_optimizef(gp, y, K, Lav, K_fu)
    %LIKELIH_T_OPTIMIZEF   function to optimize the latent variables
    %                      with EM algorithm

    % Description:
    % [F, A] = LIKELIH_T_OPTIMIZEF(GP, Y, K, Lav, K_fu) Takes Gaussian
    % process data structure GP, observations Y and the covariance
    % matrix K. Solves the posterior mode of F using EM algorithm and
    % evaluates A = (K + W)\Y as a sideproduct. Lav and K_fu are
    % needed for sparse approximations. For details, see Vanhatalo,
    % Jyl�nki and Vehtari (2009): Gaussian process regression with
    % Student-t likelihood.      
        
        iter = 1;
        sigma2 = gp.likelih.sigma2;
        nu = gp.likelih.nu;
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
    
    function upfact = likelih_t_upfact(gp, y, mu, ll)
        nu = gp.likelih.nu;
        sigma = sqrt(gp.likelih.sigma2);

        fh_e = @(f) t_pdf(f, nu, y, sigma).*norm_pdf(f, mu, ll);
        EE = quadgk(fh_e, -40, 40);
        
        
        fm = @(f) f.*t_pdf(f, nu, y, sigma).*norm_pdf(f, mu, ll)./EE;
        mm  = quadgk(fm, -40, 40);
                                
        fV = @(f) (f - mm).^2.*t_pdf(f, nu, y, sigma).*norm_pdf(f, mu, ll)./EE;
        Varp = quadgk(fV, -40, 40);
        
        upfact = -(Varp - ll)./ll^2;
    end

    function [Ey, Vary, Py] = likelih_t_predy(likelih, Ef, Varf, y, z)
    %LIKELIH_T_PREDY    Returns the predictive mean, variance and density of y
    %
    %   Description         
    %   [EY, VARY] = LIKELIH_T_PREDY(LIKELIH, EF, VARF)
    %   takes a likelihood data structure LIKELIH, posterior mean EF
    %   and posterior Variance VARF of the latent variable and returns
    %   the posterior predictive mean EY and variance VARY of the
    %   observations related to the latent variables
    %        
    %   [Ey, Vary, PY] = LIKELIH_T_PREDY(LIKELIH, EF, VARF YT)
    %   Returns also the predictive density of YT, that is 
    %        p(yt | zt) = \int p(yt | f, zt) p(f|y) df.
    %   This requires also the observations YT.
    %
    % See also:
    % la_pred, ep_pred, mc_pred

        nu = likelih.nu;
        sigma2 = likelih.sigma2;
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

    
    function reclikelih = likelih_t_recappend(reclikelih, ri, likelih)
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
            reclikelih.type = 'Student-t';

            % Initialize parameter
            reclikelih.nu = [];
            reclikelih.sigma2 = [];

            % Set the function handles
            reclikelih.fh_pak = @likelih_t_pak;
            reclikelih.fh_unpak = @likelih_t_unpak;
            reclikelih.fh_e = @likelih_t_e;
            reclikelih.fh_g = @likelih_t_g;    
            reclikelih.fh_g2 = @likelih_t_g2;
            reclikelih.fh_g3 = @likelih_t_g3;
            reclikelih.fh_tiltedMoments = @likelih_t_tiltedMoments;
            reclikelih.fh_recappend = @likelih_t_recappend;
            return
        end

        reclikelih.nu(ri,:) = likelih.nu;
        reclikelih.sigma2(ri,:) = likelih.sigma2;
    end
end


