function likelih = likelih_negbin(do, varargin)
%likelih_negbin	Create a Negbin likelihood structure for Gaussian Process
%
%	Description
%
%	LIKELIH = LIKELIH_NEGBIN('INIT', Y, YE) Create and initialize Negbin likelihood. 
%       The input argument Y contains incedence counts and YE the expected number of
%       incidences
%
%	The fields in LIKELIH are:
%	  type                     = 'likelih_negbin'
%         likelih.avgE             = YE;
%         likelih.gamlny           = gammaln(Y+1);
%         p                        = Prior structure for hyperparameters of likelihood.
%                                    Default prior for the dispersion parameter is logunif.
%         likelih.fh_pak           = function handle to pak
%         likelih.fh_unpak         = function handle to unpak
%         likelih.fh_permute       = function handle to permutation
%         likelih.fh_e             = function handle to energy of likelihood
%         likelih.fh_g             = function handle to gradient of energy
%         likelih.fh_g2            = function handle to second derivatives of energy
%         likelih.fh_g3            = function handle to third (diagonal) gradient of energy 
%         likelih.fh_tiltedMoments = function handle to evaluate tilted moments for EP
%         likelih.fh_siteDeriv     = function handle to the derivative with respect to cite parameters
%         likelih.fh_mcmc          = function handle to MCMC sampling of latent values
%         likelih.fh_recappend     = function handle to record append
%
%	LIKELIH = LIKELIH_NEGBIN('SET', LIKELIH, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH.
%
%	See also
%       LIKELIH_LOGIT, LIKELIH_PROBIT, LIKELIH_NEGBIN
%
%

% Copyright (c) 2007-2008 Jarno Vanhatalo & Jouni Hartikainen

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end

    % Initialize the covariance function
    if strcmp(do, 'init')
        y = varargin{1};
        avgE = varargin{2};
        disper = varargin{3};
        likelih.type = 'negbin';
        
        % Set parameters
        likelih.avgE = avgE;
        likelih.gamlny = gammaln(y+1);
        likelih.disper = disper;

        % Initialize prior structure
        likelih.p.disper = prior_logunif('init');

        % Set the function handles to the nested functions
        likelih.fh_pak = @likelih_negbin_pak;
        likelih.fh_unpak = @likelih_negbin_unpak;
        likelih.fh_permute = @likelih_negbin_permute;
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

        if length(varargin) > 3
            if mod(nargin,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=3:2:length(varargin)-1
                switch varargin{i}
                  case 'avgE'
                    likelih.avgE = varargin{i+1};
                  case 'gamlny'
                    likelih.gamlny = varargin{i+1};
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
              case 'avgE'
                likelih.avgE = varargin{i+1};
              case 'gamlny'
                likelih.gamlny = varargin{i+1};
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
    %LIKELIH_NEGBIN_PAK      Combine likelihood parameters into one vector.
    %
    %	Description
    %	W = LIKELIH_NEGBIN_PAK(LIKELIH, W) takes a likelihood data structure LIKELIH and
    %	combines the parameters into a single row vector W.
    %	  
    %
    %	See also
    %	LIKELIH_NEGBIN_UNPAK
        
        if ~isempty(likelih.p.disper)
            w = log(likelih.disper);
            %w = likelih.disper;
        end
    end


    function [likelih, w] = likelih_negbin_unpak(w, likelih)
    %LIKELIH_NEGBIN_UNPAK      Combine likelihood parameters into one vector.
    %
    %	Description
    %	W = LIKELIH_NEGBIN_UNPAK(LIKELIH, W) takes a likelihood data structure LIKELIH and
    %	combines the parameter vector W and sets the parameters in LIKELIH.
    %	  
    %
    %	See also
    %	LIKELIH_NEGBIN_PAK    

        if ~isempty(likelih.p.disper)
            i1=1;
            likelih.disper = exp(w(i1));
            %likelih.disper = w(i1);
            w = w(i1+1:end);
        end
    end


    function likelih = likelih_negbin_permute(likelih, p, varargin)
    %LIKELIH_NEGBIN_PERMUTE    A function to permute the ordering of parameters 
    %                           in likelihood structure
    %   Description
    %	LIKELIH = LIKELIH_NEGBIN_UNPAK(LIKELIH, P) takes a likelihood data structure
    %   LIKELIH and permutation vector P and returns LIKELIH with its parameters permuted
    %   according to P.
    %
    %   See also 
    %   GPLA_E, GPLA_G, GPEP_E, GPEP_G with CS+FIC model

        likelih.avgE = likelih.avgE(p,:);
        likelih.gamlny = likelih.gamlny(p,:);
    end

    function logPrior = likelih_negbin_priore(likelih, varargin)
    %LIKELIH_NEGBIN_PRIORE    log(prior) of the likelihood hyperparameters
    %
    %   Description
    %   E = LIKELIH_NEGBIN_PRIORE(LIKELIH, Y, F) takes a likelihood data structure
    %   LIKELIH
    %
    %   See also
    %   LIKELIH_NEGBIN_G, LIKELIH_NEGBIN_G3, LIKELIH_NEGBIN_G2, GPLA_E
        

    % If prior for dispersion parameter, add its contribution
        if ~isempty(likelih.p.disper)
            logPrior = feval(likelih.p.disper.fh_e, likelih.disper, likelih.p.disper) - log(likelih.disper);
        end
    
    end

    
    function glogPrior = likelih_negbin_priorg(likelih, varargin)
    %LIKELIH_NEGBIN_PRIORG    d log(prior)/dth of the likelihood hyperparameters th
    %
    %   Description
    %   E = LIKELIH_NEGBIN_PRIORG(LIKELIH, Y, F) takes a likelihood data structure
    %   LIKELIH, 
    %
    %   See also
    %   LIKELIH_NEGBIN_G, LIKELIH_NEGBIN_G3, LIKELIH_NEGBIN_G2, GPLA_E
        
        
        if ~isempty(likelih.p.disper)            
            % Evaluate the gprior with respect to magnSigma2
            ggs = feval(likelih.p.disper.fh_g, likelih.disper, likelih.p.disper);
            glogPrior = ggs(1).*likelih.disper - 1;
            if length(ggs) > 1
                glogPrior = [glogPrior ggs(2:end)];
            end
        end
    end  
    
    function logLikelih = likelih_negbin_e(likelih, y, f, varargin)
    %LIKELIH_NEGBIN_E    (Likelihood) Energy function
    %
    %   Description
    %   E = LIKELIH_NEGBIN_E(LIKELIH, Y, F) takes a likelihood data structure
    %   LIKELIH, incedence counts Y and latent values F and returns the log likelihood.
    %
    %   See also
    %   LIKELIH_NEGBIN_G, LIKELIH_NEGBIN_G3, LIKELIH_NEGBIN_G2, GPLA_E
            
        r = likelih.disper;
        E = likelih.avgE(:);
        mu = exp(f).*E;
        logLikelih = sum(r.*(log(r) - log(r+mu)) + gammaln(r+y) - gammaln(r) - gammaln(y+1) + y.*(log(mu) - log(r+mu)));        
    end

    function g = likelih_negbin_g(likelih, y, f, param)
    %LIKELIH_NEGBIN_G    Gradient of (likelihood) energy function
    %
    %   Description
    %   G = LIKELIH_NEGBIN_G(LIKELIH, Y, F, PARAM) takes a likelihood data structure
    %   LIKELIH, incedence counts Y and latent values F and returns the gradient of 
    %   log likelihood with respect to PARAM. At the moment PARAM can be 'hyper' or 'latent'.
    %
    %   See also
    %   LIKELIH_NEGBIN_E, LIKELIH_NEGBIN_G2, LIKELIH_NEGBIN_G3, GPLA_E
                            
        E = likelih.avgE(:);
        mu = exp(f).*E;
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

    function g2 = likelih_negbin_g2(likelih, y, f, param)
    %LIKELIH_NEGBIN_G2  Second gradients of (likelihood) energy function
    %
    %   Description
    %   G2 = LIKELIH_NEGBIN_G2(LIKELIH, Y, F, PARAM) takes a likelihood data 
    %   structure LIKELIH, incedence counts Y and latent values F and returns the 
    %   hessian of log likelihood with respect to PARAM. At the moment PARAM can 
    %   be only 'latent'. G2 is a vector with diagonal elements of the hessian 
    %   matrix (off diagonals are zero).
    %
    %   See also
    %   LIKELIH_NEGBIN_E, LIKELIH_NEGBIN_G, LIKELIH_NEGBIN_G3, GPLA_E

        E = likelih.avgE(:);
        mu = exp(f).*E;
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
    
    function g3 = likelih_negbin_g3(likelih, y, f, param)
    %LIKELIH_NEGBIN_G3  Third gradients of (likelihood) Energy function
    %
    %   Description
    %   G3 = LIKELIH_NEGBIN_G3(LIKELIH, Y, F, PARAM) takes a likelihood data 
    %   structure LIKELIH, incedence counts Y and latent values F and returns the 
    %   third gradients of log likelihood with respect to PARAM. At the moment PARAM can 
    %   be only 'latent'. G3 is a vector with third gradients.
    %
    %   See also
    %   LIKELIH_NEGBIN_E, LIKELIH_NEGBIN_G, LIKELIH_NEGBIN_G2, GPLA_E, GPLA_G

        E = likelih.avgE(:);
        mu = exp(f).*E;
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
    
    function [m_0, m_1, sigm2hati1] = likelih_negbin_tiltedMoments(likelih, y, i1, sigm2_i, myy_i)
    %LIKELIH_NEGBIN_TILTEDMOMENTS    Returns the moments of the tilted distribution
    %
    %   Description
    %   [M_0, M_1, M2] = LIKELIH_NEGBIN_TILTEDMOMENTS(LIKELIH, Y, I, S2, MYY) takes a 
    %   likelihood data structure LIKELIH, incedence counts Y, index I and cavity variance 
    %   S2 and mean MYY. Returns the zeroth moment M_0, first moment M_1 and second moment 
    %   M_2 of the tilted distribution
    %
    %   See also
    %   GPEP_E
       
        yy = y(i1);
        % Create function handle for the function to be integrated (likelihood * cavity). 

        zm = @zeroth_moment;
        gamlny = likelih.gamlny(i1);
        avgE = likelih.avgE(i1);
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
    
    function [g_i] = likelih_negbin_siteDeriv(likelih, y, i1, sigm2_i, myy_i)
    %LIKELIH_NEGBIN_SITEDERIV    Evaluate the derivative with respect to site parameters
    %
    %
        zm = @zeroth_moment;
        zd = @deriv;
        
        tol = 1e-8;
        yy = y(i1);
        gamlny = likelih.gamlny(i1);
        avgE = likelih.avgE(i1);
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
        [g_i, fhncnt] = quadgk(zd, lambdaconf(1), lambdaconf(2));
        g_i = g_i.*r;

        % ------------------------------------------------
        % Plot the integrand to check that integration limits are ok
% $$$         clf;ff = [lambdaconf(1):0.01:lambdaconf(2)];
% $$$         plot([lambdaconf(1) lambdaconf(2)], [0 0], 'r');hold on;plot(ff, feval(zd, ff))
% $$$         drawnow;S = sprintf('iter %d, y=%d, avgE=%.1f, sigm_a=%.2f, sigm2_i=%.2f', i1, yy, avgE, sigm_app, sigm2_i);title(S);
% $$$         pause
        % ------------------------------------------------

        function integrand = zeroth_moment(f); %
            integrand = exp(- 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2); %
        end        
        
        function integrand = deriv(f)
            mu = avgE.*exp(f);
            g = 0;
            g = g + log(r./(r+mu)) + 1 - (r+yy)./(r+mu);
            for i2 = 0:yy-1
                g = g + 1 ./ (i2 + r);
            end
            integrand = g.*exp(- 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2)./m_0; %
        end
    end

    function [Ey, Vary, Py] = likelih_negbin_predy(likelih, Ef, Varf, y)
    %LIKELIH_NEGBIN_PREDY    Returns the predictive mean, variance and density of y
    %
    %   Description
    %   [Ey, Vary, py] = LIKELIH_NEGBIN_PREDY(LIKELIH, EF, VARF, Y) 

       avgE = likelih.avgE;
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
%            % Conditional mean and variance of y (see Gelman et al. p. 23-24)
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
               
               if y(i1) > 0
                   m = log(r./(y(i1)+r)./avgE(i1));
                   s2 = r.*(y(i1)+r)./y(i1);
                   mean_app = (myy_i/sigm2_i + m/s2)/(1/sigm2_i + 1/s2);
                   sigm_app = sqrt((1/sigm2_i + 1/s2)^-1);
               else
                   mean_app = myy_i;
                   sigm_app = sqrt(sigm2_i);
               end
               
               % Predictive density of the given observations
               pd = @(f) nbinpdf(y(i1),r,r./(avgE(i1).*exp(f)+r)).*norm_pdf(f,myy_i,sqrt(sigm2_i));
               Py(i1) = quadgk(pd, mean_app - 12*sigm_app, mean_app + 12*sigm_app);
           end
       end
    end


    function reclikelih = likelih_negbin_recappend(reclikelih, ri, likelih)
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
            reclikelih.type = 'negbin';

            % Initialize parameter
            reclikelih.disper = [];

            % Set the function handles
            reclikelih.fh_pak = @likelih_negbin_pak;
            reclikelih.fh_unpak = @likelih_negbin_unpak;
            reclikelih.fh_permute = @likelih_negbin_permute;
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

% $$$         gpp = likelih.p;
        % record lengthScale
        if ~isempty(likelih.disper)
% $$$             if ~isempty(gpp.disper)
% $$$                 reclikelih.disperHyper(ri,:)=gpp.disper.a.s;
% $$$                 if isfield(gpp.disper,'p')
% $$$                     if isfield(gpp.disper.p,'nu')
% $$$                         reclikelih.disperHyperNu(ri,:)=gpp.disper.a.nu;
% $$$                     end
% $$$                 end
% $$$             elseif ri==1
% $$$                 reclikelih.disperHyper=[];
% $$$             end
            reclikelih.disper(ri,:)=likelih.disper;
        elseif ri==1
            reclikelih.disper=[];
        end
    end
end


