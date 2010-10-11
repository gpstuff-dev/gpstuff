function likelih = likelih_cen_t(do, varargin)
%likelih_cen_t	Create a T likelihood structure for Gaussian Process
%
%	Description
%
%	LIKELIH = LIKELIH_T('INIT', NU, SIGMA) Create and initialize Student-t likelihood. 
%       The input argument Y contains incedence counts and YE the expected number of
%       incidences
%
%	The fields in LIKELIH are:
%	  type                     = 'Student-t'
%         likelih.nu               = nu;
%         likelih.sigma            = sigma;
%         likelih.freeze_nu        = 1 for keeping nu fixed, 0 for inferring it (1)
%         likelih.fh_pak           = function handle to pak
%         likelih.fh_unpak         = function handle to unpak
%         likelih.fh_permute       = function handle to permutation
%         likelih.fh_ll             = function handle to energy of likelihood
%         likelih.fh_llg             = function handle to gradient of energy
%         likelih.fh_llg2            = function handle to second derivatives of energy
%         likelih.fh_llg3            = function handle to third (diagonal) gradient of energy 
%         likelih.fh_tiltedMoments = function handle to evaluate tilted moments for EP
%         likelih.fh_siteDeriv     = function handle to the derivative with respect to cite parameters
%         likelih.fh_mcmc          = function handle to MCMC sampling of latent values
%         likelih.fh_optimizef     = function handle to optimization of latent values
%         likelih.fh_recappend     = function handle to record append
%
%	LIKELIH = LIKELIH_T('SET', LIKELIH, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH.
%
%	See also
%       LIKELIH_LOGIT, LIKELIH_PROBIT, LIKELIH_T
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
        nu = varargin{1};
        sigma = varargin{2};
        ylim = varargin{3};
        likelih.type = 'Censored Student-t';

        % Set parameters
        likelih.nu = nu;
        likelih.sigma = sigma;
        likelih.freeze_nu = 1;
        likelih.ylim = ylim;

        % Initialize prior structure

        % Set the function handles to the nested functions
        likelih.fh_pak = @likelih_cen_t_pak;
        likelih.fh_unpak = @likelih_cen_t_unpak;
        likelih.fh_permute = @likelih_cen_t_permute;
        likelih.fh_priore = @likelih_cen_t_priore;
        likelih.fh_priorg = @likelih_cen_t_priorg;
        likelih.fh_ll = @likelih_cen_t_ll;
        likelih.fh_llg = @likelih_cen_t_llg;
        likelih.fh_llg2 = @likelih_cen_t_llg2;
        likelih.fh_llg3 = @likelih_cen_t_llg3;
        likelih.fh_tiltedMoments = @likelih_cen_t_tiltedMoments;
        likelih.fh_siteDeriv = @likelih_cen_t_siteDeriv;
        likelih.fh_mcmc = @likelih_cen_t_mcmc;
        likelih.fh_optimizef = @likelih_cen_t_optimizef;
        likelih.fh_upfact = @likelih_cen_t_upfact;
        likelih.fh_recappend = @likelih_cen_t_recappend;

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
                  case 'freeze_nu'
                    likelih.freeze_nu = varargin{i+1};
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
              case 'freeze_nu'
                likelih.freeze_nu = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end

    function w = likelih_cen_t_pak(likelih)
    %LIKELIH_T_PAK      Combine likelihood parameters into one vector.
    %
    %	Description
    %	W = LIKELIH_T_PAK(GPCF, W) takes a likelihood data structure LIKELIH and
    %	combines the parameters into a single row vector W.
    %
    %
    %	See also
    %	LIKELIH_T_UNPAK

        if likelih.freeze_nu == 1
            w(1) = log(likelih.sigma);
        else
            w(1) = log(likelih.sigma);
            w(2) = log(log(likelih.nu));
        end
    end

    function [likelih] = likelih_cen_t_unpak(w, likelih)
    %LIKELIH_T_UNPAK      Combine likelihood parameters into one vector.
    %
    %	Description
    %	W = LIKELIH_T_UNPAK(GPCF, W) takes a likelihood data structure LIKELIH and
    %	combines the parameter vector W and sets the parameters in LIKELIH.
    %
    %
    %	See also
    %	LIKELIH_T_PAK

        if likelih.freeze_nu == 1
            i1=1;
            likelih.sigma = exp(w(i1));
            w = w(i1+1:end);
        else
            i1=1;
            likelih.sigma = exp(w(i1));
            i1 = i1+1;
            likelih.nu = exp(exp(w(i1)));
            w = w(i1+1:end);
        end
    end


    function likelih = likelih_cen_t_permute(likelih, p, varargin)
    %LIKELIH_T_PERMUTE    A function to permute the ordering of parameters
    %                           in likelihood structure
    %   Description
    %	LIKELIH = LIKELIH_T_UNPAK(LIKELIH, P) takes a likelihood data structure
    %   LIKELIH and permutation vector P and returns LIKELIH with its parameters permuted
    %   according to P.
    %
    %   See also
    %   GPLA_E, GPLA_G, GPEP_E, GPEP_G with CS+FIC model

    end

    function logPrior = likelih_cen_t_priore(likelih, varargin)
    %LIKELIH_T_PRIORE    log(prior) of the likelihood hyperparameters
    %
    %   Description
    %   E = LIKELIH_T_PRIORE(LIKELIH, Y, F) takes a likelihood data structure
    %   LIKELIH
    %
    %   See also
    %   LIKELIH_T_LLG, LIKELIH_T_LLG3, LIKELIH_T_LLG2, GPLA_E

        v = likelih.nu;
        sigma = likelih.sigma;

        % Evaluate the log(prior)
        if isfield(likelih, 'p')
            % notice that nu is handled in log(log(nu)) space and sigma is handled in log(sigma) space
            gpl = likelih.p;

            if likelih.freeze_nu == 1
                logPrior =  - feval(gpl.sigma.fe, likelih.sigma, gpl.sigma.a) + log(sigma);
            else
                logPprior = - feval(gpl.nu.fe, likelih.nu, gpl.nu.a) + log(v) + log(log(v));
                logPrior =  logPprior - feval(gpl.sigma.fe, likelih.sigma, gpl.sigma.a) + log(sigma);
            end
        else
            error('likelih_cen_t -> likelih_cen_t_priore: Priors for the likelihood parameters are not defined!')
        end
    end

    function glogPrior = likelih_cen_t_priorg(likelih, varargin)
    %LIKELIH_T_PRIORG    d log(prior)/dth of the likelihood hyperparameters th
    %
    %   Description
    %   E = LIKELIH_T_PRIORG(LIKELIH, Y, F) takes a likelihood data structure
    %   LIKELIH,
    %
    %   See also
    %   LIKELIH_T_LLG, LIKELIH_T_LLG3, LIKELIH_T_LLG2, GPLA_E

    % Evaluate the log(prior)
        v = likelih.nu;
        sigma = likelih.sigma;

        if isfield(likelih, 'p')
            gpl = likelih.p;
            if likelih.freeze_nu == 1
                glogPrior(1) = - feval(gpl.sigma.fg, likelih.sigma, gpl.sigma.a).*sigma  + 1;
            else
                glogPrior(1) = - feval(gpl.sigma.fg, likelih.sigma, gpl.sigma.a).*sigma + 1;
                glogPrior(2) = - feval(gpl.nu.fg, likelih.nu, gpl.nu.a).*v.*log(v) + log(v) + 1;
            end
        else
            error('likelih_cen_t -> likelih_cen_t_priorg: Priors for the likelihood parameters are not defined!')
        end
    end

    function logLikelih = likelih_cen_t_ll(likelih, y, f, varargin)
    %LIKELIH_T_LL    (Likelihood) Energy function
    %
    %   Description
    %   E = LIKELIH_T_LL(LIKELIH, Y, F) takes a likelihood data structure
    %   LIKELIH, outputs Y and latent values F and returns the log likelihood.
    %
    %   See also
    %   LIKELIH_T_LLG, LIKELIH_T_LLG3, LIKELIH_T_LLG2, GPLA_E

        v = likelih.nu;
        sigma = likelih.sigma;
        ylim=likelih.ylim;
        
        %r = y-f;
        %term = gammaln((v + 1) / 2) - gammaln(v/2) -log(v.*pi)/2 - log(sigma);
        %logLikelih = term + log(1 + ((r./sigma).^2)./v) .* (-(v+1)/2);
        %logLikelih = sum(logLikelih);
        
        ii1=find(y<=ylim(1));
        ii2=find(y>ylim(1) & y<ylim(2));
        ii3=find(y>=ylim(2));
        
        logLikelih = sum(t_lpdf(y(ii2),v,f(ii2),sigma));
        if ~isempty(ii1)
            logLikelih = logLikelih + sum( log(tcdf( (ylim(1)-f(ii1))./sigma, v)) );
        end
        if ~isempty(ii3)
            logLikelih = logLikelih + sum( log(tcdf( (f(ii3)-ylim(2))./sigma, v)) );
        end
        
    end


    function deriv = likelih_cen_t_llg(likelih, y, f, param)
    %LIKELIH_T_LLG    Gradient of (likelihood) energy function
    %
    %   Description
    %   G = LIKELIH_T_LLG(LIKELIH, Y, F, PARAM) takes a likelihood data structure
    %   LIKELIH, incedence counts Y and latent values F and returns the gradient of
    %   log likelihood with respect to PARAM. At the moment PARAM can be 'hyper' or 'latent'.
    %
    %   See also
    %   LIKELIH_T_LL, LIKELIH_T_LLG2, LIKELIH_T_LLG3, GPLA_E
        
        v = likelih.nu;
        sigma = likelih.sigma;
        ylim = likelih.ylim;
        
        ii1=find(y<=ylim(1));
        ii2=find(y>ylim(1) & y<ylim(2));
        ii3=find(y>=ylim(2));

        switch param
          case 'hyper'
            n = length(y);

            if likelih.freeze_nu == 1
                % Derivative with respect to sigma
                r = y(ii2)-f(ii2);
                deriv(1) = sum((v+1).*r.^2./(v.*sigma.^2 +r.^2)-1)./sigma;
                
                if ~isempty(ii1)
                    z=(ylim(1)-f(ii1))./sigma;
                    deriv(1)=deriv(1) - sum(t_pdf(z,v,0,1)./tcdf(z,v).*z) ./sigma;
                end
                if ~isempty(ii3)
                    z=(f(ii3)-ylim(2))./sigma;
                    deriv(1)=deriv(1) - sum(t_pdf(z,v,0,1)./tcdf(z,v).*z) ./sigma;
                end

                % correction for the log transformation
                deriv(1) = deriv(1).*sigma;
                
            else %%%% NOT IMPLEMENTED
                r = y-f;
                % Derivative with respect to sigma and nu
                deriv(1) = - n./sigma + (v+1).*sum(r.^2./(v.*sigma.^3 +sigma.*r.^2));
                deriv(2) = 0.5.* sum(psi((v+1)./2) - psi(v./2) - 1./v - log(1+r.^2./(v.*sigma.^2)) + (v+1).*r.^2./((v.*sigma).^2 + v.*r.^2));

                % correction for the log transformation
                deriv(1) = deriv(1).*sigma;
                deriv(2) = deriv(2).*v.*log(v);
            end
          case 'latent'
            deriv=zeros(size(f));
            
            r = y(ii2)-f(ii2);
            deriv(ii2)  = (v+1).*r ./ (v.*sigma.^2 + r.^2);
            
            if ~isempty(ii1)
                z=(ylim(1)-f(ii1))./sigma;
                deriv(ii1)= -tpdf(z,v)./tcdf(z,v) ./sigma;
            end
            if ~isempty(ii3)
                z=(f(ii3)-ylim(2))./sigma;
                deriv(ii3)= tpdf(z,v)./tcdf(z,v) ./sigma;
            end
            
        end

    end


    function g2 = likelih_cen_t_llg2(likelih, y, f, param)
    %LIKELIH_T_LLG2    Second gradients of (likelihood) energy function
    %
    %
    %   NOT IMPLEMENTED!
    %
    %   Description
    %   G2 = LIKELIH_T_LLG2(LIKELIH, Y, F, PARAM) takes a likelihood data
    %   structure LIKELIH, incedence counts Y and latent values F and returns the
    %   hessian of log likelihood with respect to PARAM. At the moment PARAM can
    %   be only 'latent'. G2 is a vector with diagonal elements of the hessian
    %   matrix (off diagonals are zero).
    %
    %   See also
    %   LIKELIH_T_LL, LIKELIH_T_LLG, LIKELIH_T_LLG3, GPLA_E

        v = likelih.nu;
        sigma = likelih.sigma;
        ylim = likelih.ylim;

        ii1=find(y<=ylim(1));
        ii2=find(y>ylim(1) & y<ylim(2));
        ii3=find(y>=ylim(2));

        switch param
          case 'hyper'

          case 'latent'
            % The hessian d^2 /(dfdf)
            g2=zeros(size(f));
            
            r2 = (y(ii2)-f(ii2)).^2;
            g2(ii2) =  (v+1).*(r2 - v.*sigma.^2) ./ (v.*sigma.^2 + r2).^2;
            
            if ~isempty(ii1)
                z=(ylim(1)-f(ii1))./sigma;
                g=tpdf(z,v)./tcdf(z,v);
                g2(ii1)= -g.*( (v+1).*z./(v+z.^2) +g) ./sigma^2;
            end
            if ~isempty(ii3)
                z=(f(ii3)-ylim(2))./sigma;
                g=tpdf(z,v)./tcdf(z,v);
                g2(ii3)= -g.*( (v+1).*z./(v+z.^2) +g) ./sigma^2;
            end
            
          case 'latent+hyper'
            if likelih.freeze_nu == 1
                % gradient d^2 / (dfds), where s is either sigma or nu
                g2=zeros(size(f));
                
                r = y(ii2)-f(ii2);
                g2(ii2) = -2.*sigma.*v.*(v+1).*r ./ (v.*sigma.^2 + r.^2).^2;
                
                if ~isempty(ii1)
                    z=(ylim(1)-f(ii1))./sigma;
                    g=tpdf(z,v)./tcdf(z,v);
                    g2(ii1)= g.*(  1 -(v+1).*z.^2./(v+z.^2) -z.*g) ./sigma^2;
                end
                if ~isempty(ii3)
                    z=(f(ii3)-ylim(2))./sigma;
                    g=tpdf(z,v)./tcdf(z,v);
                    g2(ii3)= -g.*(  1 -(v+1).*z.^2./(v+z.^2) -z.*g) ./sigma^2;
                end

                % Correction for the log transformation
                g2 = g2.*sigma;
            else % NOT IMPLEMENTED !!!
                 % gradient d^2 / (dfds), where s is either sigma or nu

                g2(:,1) = -2.*sigma.*v.*(v+1).*r ./ (v.*sigma.^2 + r.^2).^2;
                g2(:,2) = r./(v.*sigma.^2 + r.^2) - sigma.^2.*(v+1).*r./(v.*sigma.^2 + r.^2).^2;

                % Correction for the log transformation
                g2(:,1) = g2(:,1).*sigma;
                g2(:,2) = g2(:,2).*v.*log(v);
            end
        end
    end

    function g3 = likelih_cen_t_llg3(likelih, y, f, param)
    %LIKELIH_T_LLG3    Third gradient of (likelihood) Energy function
    %
    %   Description
    %   G3 = LIKELIH_T_LLG3(LIKELIH, Y, F, PARAM) takes a likelihood data
    %   structure LIKELIH, incedence counts Y and latent values F and returns the
    %   third gradients of log likelihood with respect to PARAM. At the moment PARAM can
    %   be only 'latent'. G3 is a vector with third gradients.
    %
    %   See also
    %   LIKELIH_T_LL, LIKELIH_T_LLG, LIKELIH_T_LLG2, GPLA_E, GPLA_G
        
        v = likelih.nu;
        sigma = likelih.sigma;
        ylim = likelih.ylim;

        ii1=find(y<=ylim(1));
        ii2=find(y>ylim(1) & y<ylim(2));
        ii3=find(y>=ylim(2));

        switch param
          case 'hyper'

          case 'latent'
            % Return the diagonal of W differentiated with respect to latent values
            g3=zeros(size(f));
            r = y(ii2)-f(ii2);
            g3(ii2) = (v+1).*(2.*r.^3 - 6.*v.*sigma.^2.*r) ./ (v.*sigma.^2 + r.^2).^3;
            
            if ~isempty(ii1)
                z=(ylim(1)-f(ii1))./sigma;
                g=tpdf(z,v)./tcdf(z,v);
                g3(ii1)= -g.*(2.*g.^2 +3*(v+1).*z./(v+z.^2) .*g ...
                              +(nu+1).*((nu+2).*z.^2-nu)./(v+z.^2).^2  ) ./sigma^3;
            end
            if ~isempty(ii3)
                z=(f(ii3)-ylim(2))./sigma;
                g=tpdf(z,v)./tcdf(z,v);
                g3(ii3)= g.*(2.*g.^2 +3*(v+1).*z./(v+z.^2) .*g ...
                             +(nu+1).*((nu+2).*z.^2-nu)./(v+z.^2).^2  ) ./sigma^3;
            end
            
          case 'latent2+hyper'
            if likelih.freeze_nu == 1
                % Return the diagonal of W differentiated with respect to likelihood parameters
                
                g3=zeros(size(f));
                r = y(ii2)-f(ii2);
                g3(ii2) = 2.*(v+1).*sigma.*v.*( v.*sigma.^2 - 3.*r.^2) ./ (v.*sigma.^2 + r.^2).^3;
                
                if ~isempty(ii1)
                    z=(ylim(1)-f(ii1))./sigma;
                    g=tpdf(z,v)./tcdf(z,v);
                    a=(v+1).*z./(v+z.^2);
                    g3(ii1)= g.*(-2.*z.*g.^2 +(2-3.*a.*z) .*g ...
                                 +2*a -z.*(nu+1).*((nu+2).*z.^2-nu)./(v+z.^2).^2  ) ./sigma^3;
                end
                if ~isempty(ii3)
                    z=(f(ii3)-ylim(2))./sigma;
                    g=tpdf(z,v)./tcdf(z,v);
                    a=(v+1).*z./(v+z.^2);
                    g3(ii3)= g.*(-2.*z.*g.^2 +(2-3.*a.*z) .*g ...
                                 +2*a -z.*(nu+1).*((nu+2).*z.^2-nu)./(v+z.^2).^2  ) ./sigma^3;
                end

                % transformation
                g3 = g3.*sigma;
                
            else % NOT IMPLEMENTED !!!
                g3(:,1) = 2.*(v+1).*sigma.*v.*( v.*sigma.^2 - 3.*r.^2) ./ (v.*sigma.^2 + r.^2).^3;
                g3(:,1) = g3(:,1).*sigma;

                g3(:,2) = (r.^2-2.*v.*sigma.^2-sigma.^2)./(v.*sigma.^2 + r.^2).^2 - 2.*sigma.^2.*(r.^2-v.*sigma.^2).*(v+1)./(v.*sigma.^2 + r.^2).^3;
                g3(:,2) = g3(:,2).*v.*log(v);
            end
        end
    end


    function [m_0, m_1, m_2] = likelih_cen_t_tiltedMoments(likelih, y, i1, sigm2_i, myy_i)
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
        if nu > 2
            mean_app = (myy_i/sigm2_i + yy.*(nu+1)/(sigma.^2.*nu))/(1/sigm2_i + sigma.^2.*nu/(nu+1));
            sigm_app = sqrt( (1/sigm2_i + sigma.^2.*nu/(nu+1))^-1);
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
        [m_1, fhncnt] = quadgk(fm, lambdaconf(1), lambdaconf(2));
        [sigm2hati1, fhncnt] = quadgk(sm, lambdaconf(1), lambdaconf(2));

        % If the second central moment is less than cavity variance integrate more
        % precisely. Theoretically should be sigm2hati1 < sigm2_i
        if sigm2hati1 >= sigm2_i
            tol = tol.^2;
            [m_0, fhncnt] = quadgk(zm, lambdaconf(1), lambdaconf(2));
            [m_1, fhncnt] = quadgk(fm, lambdaconf(1), lambdaconf(2));
            
            [sigm2hati1, fhncnt] = quadgk(sm, lambdaconf(1), lambdaconf(2));
            if sigm2hati1 >= sigm2_i
                error('likelih_cen_t -> tiltedMoments:  sigm2hati1 >= sigm2_i ')
            end
        end
        m_2 = sigm2hati1;
        
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
    
    
    function [g_i] = likelih_cen_t_siteDeriv(likelih, y, i1, sigm2_i, myy_i)
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
        if nu > 2
            mean_app = (myy_i/sigm2_i + yy.*(nu+1)/(sigma.^2.*nu))/(1/sigm2_i + sigma.^2.*nu/(nu+1));
            sigm_app = sqrt( (1/sigm2_i + sigma.^2.*nu/(nu+1))^-1);
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
        
        if likelih.freeze_nu == 1
            [g_i(1), fhncnt] = quadgk(zsigma, lambdaconf(1), lambdaconf(2));
            g_i(1) = g_i(1).*likelih.sigma;
        else
            [g_i(1), fhncnt] = quadgk(zsigma, lambdaconf(1), lambdaconf(2));
            g_i(1) = g_i(1).*likelih.sigma;
            [g_i(2), fhncnt] = quadgk(znu, lambdaconf(1), lambdaconf(2));
            g_i(2) = g_i(2).*likelih.nu.*log(likelih.nu);
        end
        
        g_i(1) = g_i(1).*likelih.nu.*log(likelih.nu);
        g_i(2) = g_i(2).*likelih.sigma;
        
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

    function [z, energ, diagn] = likelih_cen_t_mcmc(z, opt, varargin)
    %LIKELIH_T_MCMC        Conducts the MCMC sampling of latent values
    %
    %   Description
    %   [F, ENERG, DIAG] = LIKELIH_T_MCMC(F, OPT, GP, X, Y) takes the current latent 
    %   values F, options structure OPT, Gaussian process data structure GP, inputs X and
    %   incedence counts Y. Samples new latent values and returns also energies ENERG and 
    %   diagnostics DIAG.
    %
    %   See also
    %   GP_MC
        
    % Set the state of HMC samler
        if isfield(opt, 'rstate')
            if ~isempty(opt.rstate)
                latent_rstate = opt.latent_opt.rstate;
            end
        else
            latent_rstate = sum(100*clock);
        end
        
        % Set the variables 
        gp = varargin{1};
        x = varargin{2}; 
        y = varargin{3};         
        [n,nin] = size(x);

        J = [];
        U = [];
        iJUU = [];
        Linv=[];
        L2=[];
        iLaKfuic=[];
        mincut = -300;
        
        nu = gp.likelih.nu;
        sigma = gp.likelih.sigma;
        L2 = eye(size(n,n));       
        % Evaluate the help matrices for covariance matrix
        switch gp.type
          case 'FULL'
            u = [];
            getL(z, gp, x, y);
            % Rotate z towards prior
            w = (L2\z)';    
          case 'FIC'
            error('likelih_cen_t: Latent value sampling is not (yet) implemented for FIC!');
          case {'PIC' 'PIC_BLOCK'}
            error('likelih_cen_t: Latent value sampling is not (yet) implemented for PIC!');
          case 'CS+FIC'
            error('likelih_cen_t: Latent value sampling is not (yet) implemented for CS+FIC!');
          otherwise 
            error('unknown type of GP\n')
        end
        
        
        %gradcheck(w, @lvt_er, @lvt_gr, gp, x, y, [], z)
        
        
        hmc2('state',latent_rstate)
        rej = 0;
        gradf = @lvt_gr;
        f = @lvt_er;
        for li=1:opt.repeat 
            [w, energ, diagn] = hmc2(f, w, opt, gradf, gp, x, y, u, z);
            w = w(end,:);
            if li<opt.repeat/2
                if diagn.rej
                    opt.stepadj=max(1e-5,opt.stepadj/1.4);
                else
                    opt.stepadj=min(1,opt.stepadj*1.02);
                end
            end
            rej=rej+diagn.rej/opt.repeat;
            if isfield(diagn, 'opt')
                opt=diagn.opt;
            end
        end

        w = w(end,:);
        % Rotate w to z
        w=w(:);
        switch gp.type
          case 'FULL'
            z=L2*w;
          case 'FIC'
            error('likelih_cen_t: Latent value sampling is not (yet) implemented for FIC!');
          case {'PIC' 'PIC_BLOCK'}
            error('likelih_cen_t: Latent value sampling is not (yet) implemented for PIC!');
          case 'CS+FIC'
            error('likelih_cen_t: Latent value sampling is not (yet) implemented for CS+FIC!');
        end
        opt.latent_rstate = hmc2('state');
        diagn.opt = opt;
        diagn.rej = rej;
        diagn.lvs = opt.stepadj;

        function [g, gdata, gprior] = lvt_gr(w, gp, x, y, u, varargin)
        %LVT_G	Evaluate gradient function for GP latent values with
        %               Negative-Binomial likelihood
            
        % Force z and E to be a column vector
            w=w(:);
            
            switch gp.type
              case 'FULL'
                z = L2*w;
                z = max(z,mincut);
                r = y-z;
                gdata = - (nu + 1).*r./(nu.*sigma.^2 +r.^2);
                b=Linv*z;
                gprior=Linv'*b;              
                g = (L2'*(gdata + gprior))';
              case 'FIC'
                
              case {'PIC' 'PIC_BLOCK'}

              case 'CS+FIC'

            end
        end
        

        function [e, edata, eprior] = lvt_er(w, gp, x, t, u, varargin)
        %function [e, edata, eprior] = gp_e(w, gp, x, t, varargin)
        % LVT_E     
        %
        %       E = LVT_E(X, GP, T, Z) takes.... and returns minus log from 
            
        % The field gp.likelihavgE (if given) contains the information about averige
        % expected number of cases at certain location. The target, t, is 
        % distributed as t ~ t(avgE*exp(z))
            
        % force z and E to be a column vector

            w=w(:);

            switch gp.type
              case 'FULL'
                z = L2*w;        
                z = max(z,mincut);
                B=Linv*z;
                eprior=.5*sum(B.^2);
              case 'FIC' 

              case {'PIC' 'PIC_BLOCK'}
                
              case 'CS+FIC'

            end
            r = y-z;
            term = gammaln((nu + 1) / 2) - gammaln(nu/2) -log(nu.*pi)/2 - log(sigma);
            edata = term + log(1 + ((r./sigma).^2)./nu) .* (-(nu+1)/2);
            edata = sum(-edata);
            e = edata + eprior;
        end

        function C2 = getL(w, gp, x, t, u)
        % Evaluate the cholesky decomposition if needed
            switch gp.type
              case 'FULL'
                C=gp_trcov(gp, x);
                % Evaluate a approximation for posterior covariance
                Linv = inv(chol(C)');
                
                r = y-w;
                C2 = - (nu + 1) ./ (nu.*sigma.^2 + r.^2) + 2.*(nu+1).*r.^2 ./ (nu.*sigma.^2 + r.^2).^2;
                
                L2 = chol(C)';
        % $$$                 L2 = C/chol(diag(-1./C2) + C);
        % $$$                 L2 = chol(C - L2*L2')';
              case 'FIC'
                error('likelih_cen_t: Latent value sampling is not (yet) implemented for FIC!');
              case {'PIC' 'PIC_BLOCK'}
                error('likelih_cen_t: Latent value sampling is not (yet) implemented for PIC!');
              case 'CS+FIC'
                error('likelih_cen_t: Latent value sampling is not (yet) implemented for CS+FIC!');
            end
        end
    end
    
    function [f, a] = likelih_cen_t_optimizef(gp, y, K, Lav, K_fu)
        iter = 1;
        sigma = gp.likelih.sigma;
        nu = gp.likelih.nu;
        ylim = gp.likelih.ylim;
        n = length(y);

        am=nu/2;
        bm=am*sigma^2;

        ii1=find(y<=ylim(1));
        ii2=find(y>ylim(1) & y<ylim(2));
        ii3=find(y>=ylim(2));

        iV = ones(n,1)./sigma.^2;
        siV = sqrt(iV);
        B = eye(n) + siV*siV'.*K;
        L = chol(B)';
        a=siV.*(L'\(L\(siV.*y)));
        f = K*a;
        Ey=y;
        while iter < 500
            fold = f;
            % E-step

            % noncensored observations
            iV(ii2) = (nu+1) ./ (nu.*sigma^2 + (Ey(ii2)-f(ii2)).^2);

            % censored observations
            if ~isempty(ii1) % y<ylim(1)
                n1=length(ii1);
                r=(ylim(1)-f(ii1));

                j0=0:am-1;
                c0=0.5*gamma(am) / bm^am .* (1+ r./sqrt(2*pi*bm) .* ...
                                             sum( repmat(bm./(bm+0.5*r.^2),1,am).^repmat(j0+0.5,n1,1) .* ...
                                                  repmat(gamma(j0+0.5)./gamma(j0+1),n1,1) ,2) );

                j0=0:am;
                c1=0.5*gamma(am+1) / bm^(am+1) .* (1+ r./sqrt(2*pi*bm) .* ...
                                                   sum( repmat(bm./(bm+0.5*r.^2),1,am+1).^repmat(j0+0.5,n1,1) .* ...
                                                        repmat(gamma(j0+0.5)./gamma(j0+1),n1,1) ,2) );

                iV(ii1)=c1./c0;
                d= 1/sqrt(2*pi) .* (bm+0.5*r.^2).^(-am-0.5) .* gamma(am+0.5);
                Ey(ii1) = f(ii1) - d./c1;

            end
            if ~isempty(ii3) % y>ylim(2)
                n3=length(ii3);
                r=f(ii3)-ylim(2);

                j0=0:am-1;
                c0=0.5*gamma(am) / bm^am .* (1+ r./sqrt(2*pi*bm) .* ...
                                             sum( repmat(bm./(bm+0.5*r.^2),1,am).^repmat(j0+0.5,n3,1) .* ...
                                                  repmat(gamma(j0+0.5)./gamma(j0+1),n3,1) ,2) );

                j0=0:am;
                c1=0.5*gamma(am+1) / bm^(am+1) .* (1+ r./sqrt(2*pi*bm) .* ...
                                                   sum( repmat(bm./(bm+0.5*r.^2),1,am+1).^repmat(j0+0.5,n3,1) .* ...
                                                        repmat(gamma(j0+0.5)./gamma(j0+1),n3,1) ,2) );

                iV(ii3)=c1./c0;
                d= 1/sqrt(2*pi) .* (bm+0.5*r.^2).^(-am-0.5) .* gamma(am+0.5);
                Ey(ii3) = f(ii3) + d./c1;

            end

            % M-step
            siV = sqrt(iV);
            B = eye(n) + siV*siV'.*K;
            L = chol(B)';
            a=siV.*(L'\(L\(siV.*Ey)));
            f = K*a;

            if max(abs(f-fold)) < 1e-7
                %                   figure(3)
                %                   plot(x(:,1),Ey,'rx',x(:,1),f,'b.',x(:,1),y,'ro')
                %                   title(sprintf('%d iterations',iter))
                %                   drawnow

                break
            end
            iter = iter + 1;
        end        
        
    end
    
    function upfact = likelih_cen_t_upfact(gp, y, mu, ll)
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
    
    function reclikelih = likelih_cen_t_recappend(reclikelih, ri, likelih)
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
            reclikelih.fh_pak = @likelih_cen_t_pak;
            reclikelih.fh_unpak = @likelih_cen_t_unpak;
            reclikelih.fh_permute = @likelih_cen_t_permute;
            reclikelih.fh_ll = @likelih_cen_t_ll;
            reclikelih.fh_llg = @likelih_cen_t_llg;    
            reclikelih.fh_llg2 = @likelih_cen_t_llg2;
            reclikelih.fh_llg3 = @likelih_cen_t_llg3;
            reclikelih.fh_tiltedMoments = @likelih_cen_t_tiltedMoments;
            reclikelih.fh_mcmc = @likelih_cen_t_mcmc;
            reclikelih.fh_recappend = @likelih_cen_t_recappend;
            return
        end

    % $$$         gpp = likelih.p;
    % record lengthScale
        if ~isempty(likelih.nu)
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
            reclikelih.nu(ri,:) = likelih.nu;
            reclikelih.sigma(ri,:) = likelih.sigma;
        elseif ri==1
            reclikelih.nu=[];
            reclikelih.sigma=[];
        end
    end
end
