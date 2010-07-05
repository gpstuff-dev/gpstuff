function likelih = likelih_negbin(do, varargin)
%LIKELIH_NEGBIN    Create a Negbin likelihood structure 
%
%	Description
%       LIKELIH = LIKELIH_NEGBIN('INIT') Create and initialize Negbin
%        likelihood.
%
%       The likelihood is defined as follows:
%                     __ n
%         p(y|f, z) = || i=1 [ (r/(r+mu_i))^r * gamma(r+y_i)/( gamma(r)*gamma(y_i+1) )
%                             * (mu/(r+mu_i))^y_i ]
%
%       where mu_i = z_i*exp(f_i) and r is the dispersion parameter.
%       z is a vector of expected mean and f the latent value vector
%       whose components are transformed to relative risk
%       exp(f_i). When using the likelihood you need to give the
%       vector z as an extra parameter to each function that requires
%       also y. For example, you should call gpla_e as follows:
%       gpla_e(w, gp, x, y, 'z', z)
%               
%
%	The fields in LIKELIH are:
%	  type                     = 'likelih_negbin'
%         disper                   = The dispersion parameter
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
%         likelih.fh_recappend     = function handle to append the record
%
%	LIKELIH = LIKELIH_NEGBIN('SET', LIKELIH, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH. The fields that 
%       can be modified are:
%
%             'disper'             : set the dispersion parameter
%             'disper_prior'       : set the prior structure for the dispersion parameter
%
%       Note! If the prior structure is set to empty matrix
%       (e.g. 'disper_prior', []) then the parameter in question
%       is considered fixed and it is not handled in optimization,
%       grid integration, MCMC etc.
%
%	See also
%       LIKELIH_LOGIT, LIKELIH_PROBIT, LIKELIH_NEGBIN
%
%

% Copyright (c) 2007-2010 Jarno Vanhatalo & Jouni Hartikainen

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 1
        error('Not enough arguments')
    end

    % Initialize the covariance function
    if strcmp(do, 'init')
        likelih.type = 'negbin';
        
        likelih.disper = 10;

        % Initialize prior structure
        likelih.p.disper = prior_logunif('init');

        % Set the function handles to the nested functions
        likelih.fh_pak = @likelih_negbin_pak;
        likelih.fh_unpak = @likelih_negbin_unpak;
        likelih.fh_priore = @likelih_negbin_priore;
        likelih.fh_priorg = @likelih_negbin_priorg;
        likelih.fh_e = @likelih_negbin_e;
        likelih.fh_g = @likelih_negbin_g;    
        likelih.fh_g2 = @likelih_negbin_g2;
        likelih.fh_g3 = @likelih_negbin_g3;
        likelih.fh_tiltedMoments = @likelih_negbin_tiltedMoments;
        likelih.fh_siteDeriv = @likelih_negbin_siteDeriv;
        likelih.fh_mcmc = @likelih_negbin_mcmc;
        likelih.fh_predy = @likelih_negbin_predy;
        likelih.fh_recappend = @likelih_negbin_recappend;

        if nargin > 1
            if mod(nargin,2) ==0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=3:2:length(varargin)-1
                switch varargin{i}
                  case 'disper'
                    likelih.disper = varargin{i+1};
                  case 'disper_prior'
                    likelih.p.disper = varargin{i+1};
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
              case 'disper'
                likelih.disper = varargin{i+1};
              case 'disper_prior'
                likelih.p.disper = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end



    function w = likelih_negbin_pak(likelih)
    %LIKELIH_NEGBIN_PAK  Combine likelihood parameters into one vector.
    %
    %	Description 
    %   W = LIKELIH_NEGBIN_PAK(LIKELIH) takes a
    %   likelihood data structure LIKELIH and combines the parameters
    %   into a single row vector W.
    %	  
    %
    %	See also
    %	LIKELIH_NEGBIN_UNPAK, GP_PAK
        
        if ~isempty(likelih.p.disper)
            w = log(likelih.disper);
        end
    end


    function [likelih, w] = likelih_negbin_unpak(w, likelih)
    %LIKELIH_NEGBIN_UNPAK  Extract likelihood parameters from the vector.
    %
    %	Description
    %   W = LIKELIH_NEGBIN_UNPAK(W, LIKELIH) takes a likelihood data
    %   structure LIKELIH and extracts the parameters from the vector W
    %   to the LIKELIH structure.
    %	  
    %
    %	See also
    %	LIKELIH_NEGBIN_PAK, GP_UNPAK

        if ~isempty(likelih.p.disper)
            i1=1;
            likelih.disper = exp(w(i1));
            %likelih.disper = w(i1);
            w = w(i1+1:end);
        end
    end


    function logPrior = likelih_negbin_priore(likelih, varargin)
    %LIKELIH_NEGBIN_PRIORE  log(prior) of the likelihood hyperparameters
    %
    %   Description
    %   E = LIKELIH_NEGBIN_PRIORE(LIKELIH) takes a likelihood data 
    %   structure LIKELIH and returns log(p(th)), where th collects 
    %   the hyperparameters.
    %
    %   See also
    %   LIKELIH_NEGBIN_G, LIKELIH_NEGBIN_G3, LIKELIH_NEGBIN_G2, GPLA_E
        

    % If prior for dispersion parameter, add its contribution
        if ~isempty(likelih.p.disper)
            logPrior = feval(likelih.p.disper.fh_e, likelih.disper, likelih.p.disper) - log(likelih.disper);
        end
    
    end

    
    function glogPrior = likelih_negbin_priorg(likelih, varargin)
    %LIKELIH_NEGBIN_PRIORG    d log(prior)/dth of the likelihood 
    %                         hyperparameters th
    %
    %   Description
    %   E = LIKELIH_NEGBIN_PRIORG(LIKELIH, Y, F) takes a likelihood 
    %   data structure LIKELIH and returns d log(p(th))/dth, where 
    %   th collects the hyperparameters.
    %
    %   See also
    %   LIKELIH_NEGBIN_G, LIKELIH_NEGBIN_G3, LIKELIH_NEGBIN_G2, GPLA_G
        
        
        if ~isempty(likelih.p.disper)            
            % Evaluate the gprior with respect to magnSigma2
            ggs = feval(likelih.p.disper.fh_g, likelih.disper, likelih.p.disper);
            glogPrior = ggs(1).*likelih.disper - 1;
            if length(ggs) > 1
                glogPrior = [glogPrior ggs(2:end)];
            end
        end
    end  
    
    function logLikelih = likelih_negbin_e(likelih, y, f, z)
    %LIKELIH_NEGBIN_E    Log likelihood
    %
    %   Description
    %   E = LIKELIH_NEGBIN_E(LIKELIH, Y, F, Z) takes a likelihood
    %   data structure LIKELIH, incedence counts Y, expected counts Z,
    %   and latent values F. Returns the log likelihood, log p(y|f,z).
    %
    %   See also
    %   LIKELIH_NEGBIN_G, LIKELIH_NEGBIN_G3, LIKELIH_NEGBIN_G2, GPLA_E
        
        if isempty(z)
            error(['likelih_negbin -> likelih_negbin_e: missing z!    '... 
                   'Negbin likelihood needs the expected number of    '...
                   'occurrences as an extra input z. See, for         '...
                   'example, likelih_negbin and gpla_e.               ']);
        end

        
        r = likelih.disper;
        mu = exp(f).*z;
        logLikelih = sum(r.*(log(r) - log(r+mu)) + gammaln(r+y) - gammaln(r) - gammaln(y+1) + y.*(log(mu) - log(r+mu)));
    end

    function g = likelih_negbin_g(likelih, y, f, param, z)
    %LIKELIH_NEGBIN_G    Gradient of log likelihood (energy)
    %
    %   Description 
    %   G = LIKELIH_NEGBIN_G(LIKELIH, Y, F, PARAM) takes a likelihood
    %   data structure LIKELIH, incedence counts Y, expected counts Z
    %   and latent values F. Returns the gradient of log likelihood 
    %   with respect to PARAM. At the moment PARAM can be 'hyper' or
    %   'latent'.
    %
    %   See also
    %   LIKELIH_NEGBIN_E, LIKELIH_NEGBIN_G2, LIKELIH_NEGBIN_G3, GPLA_E

        if isempty(z)
            error(['likelih_negbin -> likelih_negbin_g: missing z!    '... 
                   'Negbin likelihood needs the expected number of    '...
                   'occurrences as an extra input z. See, for         '...
                   'example, likelih_negbin and gpla_e.               ']);
        end

        
        mu = exp(f).*z;
        r = likelih.disper;
        switch param
          case 'hyper'      
            % Derivative using the psi function
            g = sum(1 + log(r./(r+mu)) - (r+y)./(r+mu) + psi(r + y) - psi(r));
            
            % correction for the log transformation
            g = g.*likelih.disper;
            
% $$$             % Derivative using sum formulation
% $$$             g = 0;
% $$$             for i1 = 1:length(y)
% $$$                 g = g + log(r/(r+mu(i1))) + 1 - (r+y(i1))/(r+mu(i1));
% $$$                 for i2 = 0:y(i1)-1
% $$$                     g = g + 1 / (i2 + r);
% $$$                 end
% $$$             end
% $$$             % correction for the log transformation
% $$$             g = g.*likelih.disper;
          case 'latent'
            g = y - (r+y).*mu./(r+mu);
        end
    end

    function g2 = likelih_negbin_g2(likelih, y, f, param, z)
    %LIKELIH_NEGBIN_G2  Second gradients of log likelihood (energy)
    %
    %   Description        
    %   G2 = LIKELIH_NEGBIN_G2(LIKELIH, Y, F, PARAM) takes a likelihood
    %   data structure LIKELIH, incedence counts Y, expected counts Z,
    %   and latent values F. Returns the hessian of log likelihood
    %   with respect to PARAM. At the moment PARAM can be only
    %   'latent'. G2 is a vector with diagonal elements of the hessian
    %   matrix (off diagonals are zero).
    %
    %   See also
    %   LIKELIH_NEGBIN_E, LIKELIH_NEGBIN_G, LIKELIH_NEGBIN_G3, GPLA_E

        if isempty(z)
            error(['likelih_negbin -> likelih_negbin_g2: missing z!   '... 
                   'Negbin likelihood needs the expected number of    '...
                   'occurrences as an extra input z. See, for         '...
                   'example, likelih_negbin and gpla_e.               ']);
        end

        
        mu = exp(f).*z;
        r = likelih.disper;
        switch param
          case 'hyper'
            
          case 'latent'
            g2 = - mu.*(r.^2 + y.*r)./(r+mu).^2;
          case 'latent+hyper'
            g2 = (y.*mu - mu.^2)./(r+mu).^2;
            
            % correction due to the log transformation
            g2 = g2.*likelih.disper;
        end
    end    
    
    function g3 = likelih_negbin_g3(likelih, y, f, param, z)
    %LIKELIH_NEGBIN_G3  Third gradients of log likelihood (energy)
    %
    %   Description
        
    %   G3 = LIKELIH_NEGBIN_G3(LIKELIH, Y, F, PARAM) takes a likelihood 
    %   data structure LIKELIH, incedence counts Y, expected counts Z
    %   and latent values F and returns the third gradients of log
    %   likelihood with respect to PARAM. At the moment PARAM can be
    %   only 'latent'. G3 is a vector with third gradients.
    %
    %   See also
    %   LIKELIH_NEGBIN_E, LIKELIH_NEGBIN_G, LIKELIH_NEGBIN_G2, GPLA_E, GPLA_G

        if isempty(z)
            error(['likelih_negbin -> likelih_negbin_g3: missing z!   '... 
                   'Negbin likelihood needs the expected number of    '...
                   'occurrences as an extra input z. See, for         '...
                   'example, likelih_negbin and gpla_e.               ']);
        end

        
        mu = exp(f).*z;
        r = likelih.disper;
        switch param
          case 'hyper'
            
          case 'latent'
            g3 = - mu.*(r.^2 + y.*r)./(r + mu).^2 + 2.*mu.^2.*(r.^2 + y.*r)./(r + mu).^3;
          case 'latent2+hyper'
            g3 = mu.*(y.*r - 2.*r.*mu - mu.*y)./(r+mu).^3;
            
            % correction due to the log transformation
            g3 = g3.*likelih.disper;
        end
    end
    
    function [m_0, m_1, sigm2hati1] = likelih_negbin_tiltedMoments(likelih, y, i1, sigm2_i, myy_i, z)
    %LIKELIH_NEGBIN_TILTEDMOMENTS    Returns the marginal moments for EP algorithm
    %
    %   Description
    %   [M_0, M_1, M2] = LIKELIH_NEGBIN_TILTEDMOMENTS(LIKELIH, Y, I, S2, MYY, Z) 
    %   takes a likelihood data structure LIKELIH, incedence counts Y, 
    %   expected counts Z, index I and cavity variance S2 and mean
    %   MYY. Returns the zeroth moment M_0, mean M_1 and variance M_2
    %   of the posterior marginal (see Rasmussen and Williams (2006):
    %   Gaussian processes for Machine Learning, page 55).
    %
    %   See also
    %   GPEP_E
       
        if isempty(z)
            error(['likelih_negbin -> likelih_negbin_tiltedMoments: missing z! '... 
                   'Negbin likelihood needs the expected number of             '...
                   'occurrences as an extra input z. See, for                  '...
                   'example, likelih_negbin and gpla_e.                        ']);
        end

        
        yy = y(i1);
        % Create function handle for the function to be integrated (likelihood * cavity). 

        zm = @zeroth_moment;
        gamlny = gammaln(y(i1)+1);
        avgE = z(i1);
        r = likelih.disper;

        % Set the limits for integration and integrate with quad
        % -----------------------------------------------------
        if yy > 0
            m = log(r./(yy+r)./avgE);
            s2 = r.*(yy+r)./yy;
            mean_app = (myy_i/sigm2_i + m/s2)/(1/sigm2_i + 1/s2);
            sigm_app = sqrt((1/sigm2_i + 1/s2)^-1);
        else
            mean_app = myy_i;
            sigm_app = sqrt(sigm2_i);                    
        end
        
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
        RTOL = 1.e-6;
        ATOL = 1.e-10;
                        
        % Integrate with quadrature
        [m_0, m_1, m_2] = quad_moments(zm,lambdaconf(1), lambdaconf(2), RTOL, ATOL);        
        
        sigm2hati1 = m_2 - m_1.^2;
                
        % If the second central moment is less than cavity variance integrate more
        % precisely. Theoretically should be sigm2hati1 < sigm2_i.
        if sigm2hati1 >= sigm2_i
            ATOL = ATOL.^2;
            RTOL = RTOL.^2;
            [m_0, m_1, m_2] = moments(zm, lambdaconf(1), lambdaconf(2), RTOL, ATOL);
            sigm2hati1 = m_2 - m_1.^2;
            if sigm2hati1 >= sigm2_i
                error('likelih_negbin_tilted_moments: sigm2hati1 >= sigm2_i');
            end
        end
        
        function integrand = zeroth_moment(f)
            mu = avgE.*exp(f);
            integrand = exp(-gammaln(r)-gammaln(yy+1)+yy.*(log(mu)-log(r+mu))+gammaln(r+yy)+r.*(log(r)-log(r+mu))); %
            integrand = integrand.*exp(- 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2); %
        end

    end
    
    function [g_i] = likelih_negbin_siteDeriv(likelih, y, i1, sigm2_i, myy_i, z)
    %LIKELIH_NEGBIN_SITEDERIV   Evaluate the expectation of the gradient
    %                           of the log likelihood term with respect
    %                           to the likelihood parameters for EP 
    %
    %   Description [M_0, M_1, M2] =
    %   LIKELIH_NEGBIN_TILTEDMOMENTS(LIKELIH, Y, I, S2, MYY, Z) takes
    %   a likelihood data structure LIKELIH, incedence counts Y,
    %   expected counts Z, index I and cavity variance S2 and mean
    %   MYY. Returns E_f [d log p(y_i|f_i) /d a], where a is the
    %   likelihood parameter and the expectation is over the marginal
    %   posterior. This term is needed when evaluating the gradients
    %   of the marginal likelihood estimate Z_EP with respect to the
    %   likelihood parameters (see Seeger (2008): Expectation
    %   propagation for exponential families)
    %
    %   See also
    %   GPEP_G

        if isempty(z)
            error(['likelih_negbin -> likelih_negbin_siteDeriv: missing z!'... 
                   'Negbin likelihood needs the expected number of        '...
                   'occurrences as an extra input z. See, for             '...
                   'example, likelih_negbin and gpla_e.                   ']);
        end

        
        zm = @zeroth_moment;
        zd = @deriv;
        
        tol = 1e-8;
        yy = y(i1);
        gamlny = gammaln(y(i1)+1);
        avgE = z(i1);
        r = likelih.disper;
        
        % Set the limits for integration and integrate with quad
        % -----------------------------------------------------
        if yy > 0
            m = log(r./(yy+r)./avgE);
            s2 = r.*(yy+r)./yy;
            mean_app = (myy_i/sigm2_i + m/s2)/(1/sigm2_i + 1/s2);
            sigm_app = sqrt((1/sigm2_i + 1/s2)^-1);
        else
            mean_app = myy_i;
            sigm_app = sqrt(sigm2_i);                    
        end
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
        [g_i, fhncnt] = quadgk(@(f) zd(f).*zm(f)./m_0, lambdaconf(1), lambdaconf(2));
        g_i = g_i.*r;

        % ------------------------------------------------
        % Plot the integrand to check that integration limits are ok
% $$$         clf;ff = [lambdaconf(1):0.01:lambdaconf(2)];
% $$$         plot([lambdaconf(1) lambdaconf(2)], [0 0], 'r');hold on;plot(ff, feval(zd, ff))
% $$$         drawnow;S = sprintf('iter %d, y=%d, avgE=%.1f, sigm_a=%.2f, sigm2_i=%.2f', i1, yy, avgE, sigm_app, sigm2_i);title(S);
% $$$         pause
        % ------------------------------------------------

        function integrand = zeroth_moment(f); %
            mu = avgE.*exp(f);
            integrand = exp(-gammaln(r)-gammaln(yy+1)+yy.*(log(mu)-log(r+mu))+gammaln(r+yy)+r.*(log(r)-log(r+mu))); %
            integrand = integrand.*exp(- 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2); %
        end        
        
        function g = deriv(f)
            mu = avgE.*exp(f);
            g = 0;
            g = g + log(r./(r+mu)) + 1 - (r+yy)./(r+mu);
            for i2 = 0:yy-1
                g = g + 1 ./ (i2 + r);
            end
        end
    end

    function [Ey, Vary, Py] = likelih_negbin_predy(likelih, Ef, Varf, yt, zt)
    %LIKELIH_NEGBIN_PREDY    Returns the predictive mean, variance and density of y
    %
    %   Description         
    %   [EY, VARY] = LIKELIH_NEGBIN_PREDY(LIKELIH, EF, VARF)
    %   takes a likelihood data structure LIKELIH, posterior mean EF
    %   and posterior Variance VARF of the latent variable and returns
    %   the posterior predictive mean EY and variance VARY of the
    %   observations related to the latent variables
    %        
    %   [Ey, Vary, PY] = LIKELIH_NEGBIN_PREDY(LIKELIH, EF, VARF YT, ZT)
    %   Returns also the predictive density of YT, that is 
    %        p(yt | zt) = \int p(yt | f, zt) p(f|y) df.
    %   This requires also the incedence counts YT, expected counts ZT.
    %
    % See also:
    % la_pred, ep_pred, mc_pred        
        if isempty(zt)
            error(['likelih_negbin -> likelih_negbin_predy: missing zt!'... 
                   'Negbin likelihood needs the expected number of    '...
                   'occurrences as an extra input zt. See, for         '...
                   'example, likelih_negbin and gpla_e.               ']);
        end


       avgE = zt;
       r = likelih.disper;
        
       Py = zeros(size(Ef));
       Ey = zeros(size(Ef));
       EVary = zeros(size(Ef));
       VarEy = zeros(size(Ef)); 
        
       % Evaluate Ey and Vary 
       for i1=1:length(Ef)
%            %%%% With MC 
%            % First sample f
%            f_samp = normrnd(Ef(i1),sqrt(Varf(i1)),nsamp,1);
%            la_samp = avgE(i1).*exp(f_samp);
%             
%            % Conditional mean and variance of yt (see Gelman et al. p. 23-24)
%            Ey2(i1) = mean(la_samp);
%            Vary2(i1) = mean(la_samp + la_samp.^2/r) + var(la_samp);

           %%% With quadrature
           ci = sqrt(Varf(i1));

           F = @(x) avgE(i1).*exp(x).*normpdf(x,Ef(i1),sqrt(Varf(i1)));
           Ey(i1) = quadgk(F,Ef(i1)-6*ci,Ef(i1)+6*ci);
           
           F2 = @(x) (avgE(i1).*exp(x)+((avgE(i1).*exp(x)).^2/r)).*normpdf(x,Ef(i1),sqrt(Varf(i1)));
           EVary(i1) = quadgk(F2,Ef(i1)-6*ci,Ef(i1)+6*ci);
           
           F3 = @(x) (avgE(i1).*exp(x)).^2.*normpdf(x,Ef(i1),sqrt(Varf(i1)));
           VarEy(i1) = quadgk(F3,Ef(i1)-6*ci,Ef(i1)+6*ci) - Ey(i1).^2;
       end
       Vary = EVary + VarEy;

       % Evaluate predictive density of the given observations
       if nargout > 2
           for i1=1:length(Ef)
               myy_i = Ef(i1);
               sigm2_i = Varf(i1);
               
               if yt(i1) > 0
                   m = log(r./(yt(i1)+r)./avgE(i1));
                   s2 = r.*(yt(i1)+r)./yt(i1);
                   mean_app = (myy_i/sigm2_i + m/s2)/(1/sigm2_i + 1/s2);
                   sigm_app = sqrt((1/sigm2_i + 1/s2)^-1);
               else
                   mean_app = myy_i;
                   sigm_app = sqrt(sigm2_i);
               end
               
               % Predictive density of the given observations
               pd = @(f) nbinpdf(yt(i1),r,r./(avgE(i1).*exp(f)+r)).*norm_pdf(f,myy_i,sqrt(sigm2_i));
               Py(i1) = quadgk(pd, mean_app - 12*sigm_app, mean_app + 12*sigm_app);
           end
       end
    end


    function reclikelih = likelih_negbin_recappend(reclikelih, ri, likelih)
    % RECAPPEND  Append the parameters to the record
    %
    %          Description 
    %          RECLIKELIH = GPCF_NEGBIN_RECAPPEND(RECLIKELIH, RI, LIKELIH)
    %          takes a likelihood record structure RECLIKELIH, record
    %          index RI and likelihood structure LIKELIH with the
    %          current MCMC samples of the hyperparameters. Returns
    %          RECLIKELIH which contains all the old samples and the
    %          current samples from LIKELIH.
    % 
    %  See also:
    %  gp_mc

    % Initialize record
        if nargin == 2
            reclikelih.type = 'negbin';

            % Initialize parameter
            reclikelih.disper = [];

            % Set the function handles
            reclikelih.fh_pak = @likelih_negbin_pak;
            reclikelih.fh_unpak = @likelih_negbin_unpak;
            reclikelih.fh_e = @likelih_negbin_e;
            reclikelih.fh_g = @likelih_negbin_g;    
            reclikelih.fh_g2 = @likelih_negbin_g2;
            reclikelih.fh_g3 = @likelih_negbin_g3;
            reclikelih.fh_tiltedMoments = @likelih_negbin_tiltedMoments;
            reclikelih.fh_mcmc = @likelih_negbin_mcmc;
            reclikelih.fh_predy = @likelih_negbin_predy;
            reclikelih.fh_recappend = @likelih_negbin_recappend;
            return
        end
        
        reclikelih.disper(ri,:)=likelih.disper;
    end
end


