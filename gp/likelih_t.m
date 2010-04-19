function likelih = likelih_t(do, varargin)
%likelih_t	Create a T likelihood structure for Gaussian Process
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
%         likelih.sigma            = sigma;
%         likelih.fix_nu           = 1 for keeping nu fixed, 0 for inferring it (1)
%         likelih.fh_pak           = function handle to pak
%         likelih.fh_unpak         = function handle to unpak
%         likelih.fh_e             = function handle to energy of likelihood
%         likelih.fh_g             = function handle to gradient of energy
%         likelih.fh_g2            = function handle to second derivatives of energy
%         likelih.fh_g3            = function handle to third (diagonal) gradient of energy 
%         likelih.fh_tiltedMoments = function handle to evaluate tilted moments for EP
%         likelih.fh_siteDeriv     = function handle to the derivative with respect to cite parameters
%         likelih.fh_predy         = function handle to evaluate predictive density of y
%         likelih.fh_optimizef     = function handle to optimization of latent values
%         likelih.fh_recappend     = function handle to record append
%
%	LIKELIH = LIKELIH_T('SET', LIKELIH, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH. The fields that 
%       can be modified are:
%
%             'sigma'              : set the sigma
%             'nu'                 : set the degrees of freedom
%             'sigma_prior'        : set the prior structure for sigma
%             'nu_prior'           : set the prior structure for nu
%
%	See also
%       LIKELIH_LOGIT, LIKELIH_PROBIT, LIKELIH_T
%
%

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
        likelih.sigma = 1;
        likelih.fix_nu = 1;
        
        % Initialize prior structure
        likelih.p.sigma = prior_logunif('init');
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

        if length(varargin) > 3
            if mod(nargin,2) ~=1
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=3:2:length(varargin)-1
                switch varargin{i}
                  case 'nu'
                    likelih.nu = varargin{i+1};
                  case 'sigma'
                    likelih.sigma = varargin{i+1};
                  case 'fix_nu'
                    likelih.fix_nu = varargin{i+1};
                  case 'sigma_prior'
                    likelih.p.sigma = varargin{i+1}; 
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
              case 'sigma'
                likelih.sigma = varargin{i+1};
              case 'fix_nu'
                likelih.fix_nu = varargin{i+1};
              case 'sigma_prior'
                likelih.p.sigma = varargin{i+1}; 
              case 'nu_prior'
                likelih.p.nu = varargin{i+1}; 
              otherwise
                error('Wrong parameter name!')
            end
        end
    end


    function w = likelih_t_pak(likelih)
    %LIKELIH_T_PAK      Combine likelihood parameters into one vector.
    %
    %	Description
    %	W = LIKELIH_T_PAK(GPCF, W) takes a likelihood data structure LIKELIH and
    %	combines the parameters into a single row vector W.
    %	  
    %
    %	See also
    %	LIKELIH_T_UNPAK
        
        w = [];
        i1 = 0;
        if ~isempty(likelih.p.sigma)
            i1 = 1;
            w(i1) = log(likelih.sigma);
        end
        if ~isempty(likelih.p.nu) && ~likelih.fix_nu
            i1 = i1+1;
            w(i1) = log(log(likelih.nu));
        end        
        
% $$$         if likelih.fix_nu == 1
% $$$             w(1) = log(likelih.sigma);
% $$$         else
% $$$             w(1) = log(likelih.sigma);
% $$$             w(2) = log(log(likelih.nu));
% $$$         end
    end


    function [likelih, w] = likelih_t_unpak(w, likelih)
    %LIKELIH_T_UNPAK      Combine likelihood parameters into one vector.
    %
    %	Description
    %	W = LIKELIH_T_UNPAK(GPCF, W) takes a likelihood data structure LIKELIH and
    %	combines the parameter vector W and sets the parameters in LIKELIH.
    %	  
    %
    %	See also
    %	LIKELIH_T_PAK    

        i1 = 0;
        if ~isempty(likelih.p.sigma)
            i1 = 1;
            likelih.sigma = exp(w(i1));
        end
        if ~isempty(likelih.p.nu)  && ~likelih.fix_nu
            i1 = i1+1;
            likelih.nu = exp(exp(w(i1)));
        end
    end


    function logPrior = likelih_t_priore(likelih)
    %LIKELIH_T_PRIORE    log(prior) of the likelihood hyperparameters
    %
    %   Description
    %   E = LIKELIH_T_PRIORE(LIKELIH, Y, F) takes a likelihood data structure
    %   LIKELIH
    %
    %   See also
    %   LIKELIH_T_G, LIKELIH_T_G3, LIKELIH_T_G2, GPLA_E
        
        v = likelih.nu;
        sigma = likelih.sigma;
        logPrior = 0;
        
        if ~isempty(likelih.p.sigma) 
            logPrior = feval(likelih.p.sigma.fh_e, likelih.sigma, likelih.p.sigma) - log(sigma);
        end
        if ~isempty(likelih.p.nu) && ~likelih.fix_nu
            logPrior = logPrior + feval(likelih.p.nu.fh_e, likelih.nu, likelih.p.nu)  - log(v) - log(log(v));
        end
    end
    
    function glogPrior = likelih_t_priorg(likelih)
    %LIKELIH_T_PRIORG    d log(prior)/dth of the likelihood hyperparameters th
    %
    %   Description
    %   E = LIKELIH_T_PRIORG(LIKELIH, Y, F) takes a likelihood data structure
    %   LIKELIH, 
    %
    %   See also
    %   LIKELIH_T_G, LIKELIH_T_G3, LIKELIH_T_G2, GPLA_E
        
    % Evaluate the gradients of log(prior)

        v = likelih.nu;
        sigma = likelih.sigma;
        glogPrior = [];
        i1 = 0;
        
        if ~isempty(likelih.p.sigma) 
            i1 = 1;
            glogPrior(i1) = feval(likelih.p.sigma.fh_g, likelih.sigma, likelih.p.sigma).*sigma - 1;
        end
        if ~isempty(likelih.p.nu)  && ~likelih.fix_nu
            i1 = i1+1;
            glogPrior(i1) = feval(likelih.p.nu.fh_g, likelih.nu, likelih.p.nu).*v.*log(v) - log(v) - 1;
        end    
    end
    
    function logLikelih = likelih_t_e(likelih, y, f, z)
    %LIKELIH_T_E    (Likelihood) Energy function
    %
    %   Description
    %   E = LIKELIH_T_E(LIKELIH, Y, F) takes a likelihood data structure
    %   LIKELIH, outputs Y and latent values F and returns the log likelihood.
    %
    %   See also
    %   LIKELIH_T_G, LIKELIH_T_G3, LIKELIH_T_G2, GPLA_E

        r = y-f;
        v = likelih.nu;
        sigma = likelih.sigma;

        term = gammaln((v + 1) / 2) - gammaln(v/2) -log(v.*pi)/2 - log(sigma);
        logLikelih = term + log(1 + ((r./sigma).^2)./v) .* (-(v+1)/2);
        logLikelih = sum(logLikelih);
    end

    
    function deriv = likelih_t_g(likelih, y, f, param, z)
    %LIKELIH_T_G    Gradient of (likelihood) energy function
    %
    %   Description
    %   G = LIKELIH_T_G(LIKELIH, Y, F, PARAM) takes a likelihood data structure
    %   LIKELIH, incedence counts Y and latent values F and returns the gradient of 
    %   log likelihood with respect to PARAM. At the moment PARAM can be 'hyper' or 'latent'.
    %
    %   See also
    %   LIKELIH_T_E, LIKELIH_T_G2, LIKELIH_T_G3, GPLA_E
        
        r = y-f;
        v = likelih.nu;
        sigma = likelih.sigma;
        
        switch param
          case 'hyper'
            n = length(y);

            if likelih.fix_nu
                % Derivative with respect to sigma
                deriv(1) = - n./sigma + (v+1).*sum(r.^2./(v.*sigma.^3 +sigma.*r.^2));
                
                % correction for the log transformation
                deriv(1) = deriv(1).*sigma;
            else
                % Derivative with respect to sigma
                deriv(1) = - n./sigma + (v+1).*sum(r.^2./(v.*sigma.^3 + sigma.*r.^2));
                deriv(2) = 0.5.* sum(psi((v+1)./2) - psi(v./2) - 1./v - log(1+r.^2./(v.*sigma.^2)) + (v+1).*r.^2./((v.*sigma).^2 + v.*r.^2));
                
                % correction for the log transformation
                deriv(1) = deriv(1).*sigma;
                deriv(2) = deriv(2).*v.*log(v);
            end
          case 'latent'
            deriv  = (v+1).*r ./ (v.*sigma.^2 + r.^2);            
        end
        
    end


    function g2 = likelih_t_g2(likelih, y, f, param, z)
    %LIKELIH_T_G2    Second gradients of (likelihood) energy function
    %
    %   Description
    %   G2 = LIKELIH_T_G2(LIKELIH, Y, F, PARAM) takes a likelihood data 
    %   structure LIKELIH, incedence counts Y and latent values F and returns the 
    %   hessian of log likelihood with respect to PARAM. At the moment PARAM can 
    %   be only 'latent'. G2 is a vector with diagonal elements of the hessian 
    %   matrix (off diagonals are zero).
    %
    %   See also
    %   LIKELIH_T_E, LIKELIH_T_G, LIKELIH_T_G3, GPLA_E

        r = y-f;
        v = likelih.nu;
        sigma = likelih.sigma;

        switch param
          case 'hyper'
            
          case 'latent'
            % The hessian d^2 /(dfdf)
            g2 =  (v+1).*(r.^2 - v.*sigma.^2) ./ (v.*sigma.^2 + r.^2).^2;
          case 'latent+hyper'
            if likelih.fix_nu
                % gradient d^2 / (dfds), where s is either sigma or nu            
                g2 = -2.*sigma.*v.*(v+1).*r ./ (v.*sigma.^2 + r.^2).^2;
                
                % Correction for the log transformation
                g2 = g2.*sigma;
            else
                % gradient d^2 / (dfds), where s is either sigma or nu
                
                g2(:,1) = -2.*sigma.*v.*(v+1).*r ./ (v.*sigma.^2 + r.^2).^2;
                g2(:,2) = r./(v.*sigma.^2 + r.^2) - sigma.^2.*(v+1).*r./(v.*sigma.^2 + r.^2).^2;

                % Correction for the log transformation
                g2(:,1) = g2(:,1).*sigma;
                g2(:,2) = g2(:,2).*v.*log(v);
            end
        end
    end    
    
    function third_grad = likelih_t_g3(likelih, y, f, param, z)
    %LIKELIH_T_G3    Third gradient of (likelihood) Energy function
    %
    %   Description
    %   G3 = LIKELIH_T_G3(LIKELIH, Y, F, PARAM) takes a likelihood data 
    %   structure LIKELIH, incedence counts Y and latent values F and returns the 
    %   third gradients of log likelihood with respect to PARAM. At the moment PARAM can 
    %   be only 'latent'. G3 is a vector with third gradients.
    %
    %   See also
    %   LIKELIH_T_E, LIKELIH_T_G, LIKELIH_T_G2, GPLA_E, GPLA_G

        r = y-f;
        v = likelih.nu;
        sigma = likelih.sigma;
        
        switch param
          case 'hyper'
            
          case 'latent'
            % Return the diagonal of W differentiated with respect to latent values
            third_grad = (v+1).*(2.*r.^3 - 6.*v.*sigma.^2.*r) ./ (v.*sigma.^2 + r.^2).^3;
          case 'latent2+hyper'
            if likelih.fix_nu
                % Return the diagonal of W differentiated with respect to likelihood parameters
                third_grad = 2.*(v+1).*sigma.*v.*( v.*sigma.^2 - 3.*r.^2) ./ (v.*sigma.^2 + r.^2).^3;
                third_grad = third_grad.*sigma;
            else
                third_grad(:,1) = 2.*(v+1).*sigma.*v.*( v.*sigma.^2 - 3.*r.^2) ./ (v.*sigma.^2 + r.^2).^3;
                third_grad(:,1) = third_grad(:,1).*sigma;
                
                third_grad(:,2) = (r.^2-2.*v.*sigma.^2-sigma.^2)./(v.*sigma.^2 + r.^2).^2 - 2.*sigma.^2.*(r.^2-v.*sigma.^2).*(v+1)./(v.*sigma.^2 + r.^2).^3;
                third_grad(:,2) = third_grad(:,2).*v.*log(v);
            end
        end
    end


    function [m_0, m_1, m_2] = likelih_t_tiltedMoments(likelih, y, i1, sigm2_i, myy_i, z)
    %LIKELIH_T_TILTEDMOMENTS    Returns the moments of the tilted distribution
    %
    %   Description
    %   [M_0, M_1, M2] = LIKELIH_T_TILTEDMOMENTS(LIKELIH, Y, I, S2, MYY) takes a 
    %   likelihood data structure LIKELIH, incedence counts Y, index I and cavity variance 
    %   S2 and mean MYY. Returns the zeroth moment M_0, first moment M_1 and second moment 
    %   M_2 of the tilted distribution
    %
    %   See also
    %   GPEP_E

        
        zm = @zeroth_moment;
        fm = @first_moment;
        sm = @second_moment;
        
        tol = 1e-8;
        yy = y(i1);
        nu = likelih.nu;
        sigma = likelih.sigma;
                
        % Set the limits for integration and integrate with quad
        % -----------------------------------------------------
% $$$         if nu > 2
% $$$             mean_app = (myy_i/sigm2_i + yy.*(nu+1)/(sigma.^2.*nu))/(1/sigm2_i + sigma.^2.*nu/(nu+1));
% $$$             sigm_app = sqrt( (1/sigm2_i + sigma.^2.*nu/(nu+1))^-1);
% $$$         else
            mean_app = myy_i;
            sigm_app = sqrt(sigm2_i);
% $$$         end

        lambdaconf(1) = mean_app - 6.*sigm_app; lambdaconf(2) = mean_app + 6.*sigm_app;
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
        
        % Integrate with quad
        [m_0, fhncnt] = quadgk(zm, lambdaconf(1), lambdaconf(2));
        [m_1, fhncnt] = quadgk(fm, lambdaconf(1), lambdaconf(2));
        [sigm2hati1, fhncnt] = quadgk(sm, lambdaconf(1), lambdaconf(2));

        m_2 = sigm2hati1;
        
% $$$         if ~isreal(m_2)
% $$$             ff = [lambdaconf(1):0.01:lambdaconf(2)];
% $$$             plot(ff, feval(zm, ff), 'r')
% $$$             drawnow
% $$$             pause
% $$$         end
        
        function integrand = zeroth_moment(f)
            r = yy-f;
            term = gammaln((nu + 1) / 2) - gammaln(nu/2) -log(nu.*pi)/2 - log(sigma);
            integrand = exp(term + log(1 + ((r./sigma).^2)./nu) .* (-(nu+1)/2));
            integrand = integrand.*exp(- 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2); %
        end
        
        function integrand = first_moment(f)
            r = yy-f;
            term = gammaln((nu + 1) / 2) - gammaln(nu/2) -log(nu.*pi)/2 - log(sigma);
            integrand = exp(term + log(1 + ((r./sigma).^2)./nu) .* (-(nu+1)/2));
            integrand = integrand.*exp(- 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2); %
            integrand = f.*integrand./m_0; %
        end
        function integrand = second_moment(f)
            r = yy-f;
            term = gammaln((nu + 1) / 2) - gammaln(nu/2) -log(nu.*pi)/2 - log(sigma);
            integrand = exp(term + log(1 + ((r./sigma).^2)./nu) .* (-(nu+1)/2));
            integrand = integrand.*exp(- 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2); %
            integrand = (f-m_1).^2.*integrand./m_0; %
        end
    end
    
    
    function [g_i] = likelih_t_siteDeriv(likelih, y, i1, sigm2_i, myy_i, z)
    %LIKELIH_T_SITEDERIV    Evaluate the derivative with respect to cite parameters
    %
    %

        zm = @zeroth_moment;
        znu = @deriv_nu;
        zsigma = @deriv_sigma;
        
        tol = 1e-8;
        yy = y(i1);
        nu = likelih.nu;
        sigma = likelih.sigma;

        % Set the limits for integration and integrate with quad
        % -----------------------------------------------------
% $$$         if nu > 2
% $$$             mean_app = (myy_i/sigm2_i + yy.*(nu+1)/(sigma.^2.*nu))/(1/sigm2_i + sigma.^2.*nu/(nu+1));
% $$$             sigm_app = sqrt( (1/sigm2_i + sigma.^2.*nu/(nu+1))^-1);
% $$$         else
            mean_app = myy_i;
            sigm_app = sqrt(sigm2_i);                    
% $$$         end

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
        
        if likelih.fix_nu
            [g_i(1), fhncnt] = quadgk(zsigma, lambdaconf(1), lambdaconf(2));
            g_i(1) = g_i(1).*likelih.sigma;
        else
            [g_i(1), fhncnt] = quadgk(zsigma, lambdaconf(1), lambdaconf(2));
            g_i(1) = g_i(1).*likelih.sigma;
            [g_i(2), fhncnt] = quadgk(znu, lambdaconf(1), lambdaconf(2));
            g_i(2) = g_i(2).*likelih.nu.*log(likelih.nu);
        end
                
    % $$$         ff = [lambdaconf(1):0.01:lambdaconf(2)];
    % $$$         plot(ff, feval(zm, ff), 'r')
    % $$$         hold on
    % $$$         plot(ff, feval(zd, ff))
    % $$$         drawnow

        function integrand = zeroth_moment(f); %
            integrand = exp(- 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2); %
        end        
        
        function integrand = deriv_nu(f)
            r = yy-f;
            temp = 1 + (r/sigma).^2./nu;
            g = psi((nu+1)/2)./2 - psi(nu/2)./2 - 1./(2.*nu) - log(temp)./2 + (nu+1)./(2.*temp).*(r./(nu.*sigma)).^2;            
            integrand = g.*exp(- 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2)./m_0; %
        end
        
        function integrand = deriv_sigma(f)
            r = yy-f;
            g  =-1/sigma + (nu+1).*r.^2./(nu.*sigma.^3 + sigma.* r.^2);            
            integrand = g.*exp(- 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2)./m_0; %
        end

    end

    function [f, a] = likelih_t_optimizef(gp, y, K, Lav, K_fu)
        iter = 1;
        sigma = gp.likelih.sigma;
        nu = gp.likelih.nu;
        n = length(y);

        
        switch gp.type
          case 'FULL'            
            iV = ones(n,1)./sigma.^2;
            siV = sqrt(iV);
            B = eye(n) + siV*siV'.*K;
            L = chol(B)';
            b = iV.*y;
            a = b - siV.*(L'\(L\(siV.*(K*b))));
            f = K*a;
            while iter < 200
                fold = f;               
                iV = (nu+1) ./ (nu.*sigma^2 + (y-f).^2);
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
            
            iV = ones(n,1)./sigma.^2;
            siV = sqrt(iV);
            B = eye(n) + siV*siV'.*K;
            L = chol(B)';
            b = iV.*y;
            a = b - siV.*(L'\(L\(siV.*(K*b))));
            f = K*a;
            while iter < 200
                fold = f;                
                iV = (nu+1) ./ (nu.*sigma^2 + (y-f).^2);
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
        sigma = gp.likelih.sigma;

        fh_e = @(f) t_pdf(f, nu, y, sigma).*norm_pdf(f, mu, ll);
        EE = quadgk(fh_e, -40, 40);
        
        
        fm = @(f) f.*t_pdf(f, nu, y, sigma).*norm_pdf(f, mu, ll)./EE;
        mm  = quadgk(fm, -40, 40);
                                
        fV = @(f) (f - mm).^2.*t_pdf(f, nu, y, sigma).*norm_pdf(f, mu, ll)./EE;
        Varp = quadgk(fV, -40, 40);
        
        upfact = -(Varp - ll)./ll^2;
    end

    function [Ey, Vary, Py] = likelih_t_predy(likelih, Ef, Varf, y, z)

        nu = likelih.nu;
        sigma = likelih.sigma;

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
           
           F2 = @(x) (nu./(nu-2).*sigma.^2).*normpdf(x,Ef(i1),sqrt(Varf(i1)));
           EVary(i1) = quadgk(F2,Ef(i1)-6*ci,Ef(i1)+6*ci);
           
           F3 = @(x) x.^2.*normpdf(x,Ef(i1),sqrt(Varf(i1)));
           VarEy(i1) = quadgk(F3,Ef(i1)-6*ci,Ef(i1)+6*ci) - Ey(i1).^2;
       end
        
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
            reclikelih.sigma = [];

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
        reclikelih.sigma(ri,:) = likelih.sigma;
    end
end


