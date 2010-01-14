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
        likelih.p.disper = [];

        % Set the function handles to the nested functions
        likelih.fh_pak = @likelih_negbin_pak;
        likelih.fh_unpak = @likelih_negbin_unpak;
        likelih.fh_permute = @likelih_negbin_permute;
        likelih.fh_e = @likelih_negbin_e;
        likelih.fh_g = @likelih_negbin_g;    
        likelih.fh_g2 = @likelih_negbin_g2;
        likelih.fh_g3 = @likelih_negbin_g3;
        likelih.fh_tiltedMoments = @likelih_negbin_tiltedMoments;
        likelih.fh_siteDeriv = @likelih_negbin_siteDeriv;
        likelih.fh_mcmc = @likelih_negbin_mcmc;
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
                    gpcf.p.disper = varargin{i+1};
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
        gpcf = varargin{1};
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
                gpcf.p.disper = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end



    function w = likelih_negbin_pak(likelih)
    %LIKELIH_NEGBIN_PAK      Combine likelihood parameters into one vector.
    %
    %	Description
    %	W = LIKELIH_NEGBIN_PAK(GPCF, W) takes a likelihood data structure LIKELIH and
    %	combines the parameters into a single row vector W.
    %	  
    %
    %	See also
    %	LIKELIH_NEGBIN_UNPAK
        
        if ~isempty(gpcf.p.disper)
            w = likelih.disper;
        end
    end


    function [likelih] = likelih_negbin_unpak(w, likelih)
    %LIKELIH_NEGBIN_UNPAK      Combine likelihood parameters into one vector.
    %
    %	Description
    %	W = LIKELIH_NEGBIN_UNPAK(GPCF, W) takes a likelihood data structure LIKELIH and
    %	combines the parameter vector W and sets the parameters in LIKELIH.
    %	  
    %
    %	See also
    %	LIKELIH_NEGBIN_PAK    

        if ~isempty(gpcf.p.disper)
            i1=1;
            likelih.disper = w(i1);
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
                        
            % Derivative using sum formulation
% $$$             g = 0;
% $$$             for i1 = 1:length(y)
% $$$                 g = g + log(r/(r+mu(i1))) + 1 - (r+y(i1))/(r+mu(i1));
% $$$                 for i2 = 0:y(i1)-1
% $$$                     g = g + 1 / (i2 + r);
% $$$                 end
% $$$             end
          case 'latent'
            g = y - (r+y).*mu./(r+mu);
        end
    end


    function g2 = likelih_negbin_g2(likelih, y, f, param)
    %LIKELIH_NEGBIN_G2    Third gradients of (likelihood) energy function
    %
    %
    %   NOT IMPLEMENTED!
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
        end
    end    
    
    function g3 = likelih_negbin_g3(likelih, y, f, param)
    %LIKELIH_NEGBIN_G3    Gradient of (likelihood) Energy function
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
        end
    end
    
    function [m_0, m_1, sigm2hati1] = likelih_negbin_tiltedMoments(likelih, y, i1, sigm2_i, myy_i)
    %LIKELIH_NEGBIN_TILTEDMOMENTS    Returns the moments of the tilted distribution
    %
    %   Description
    %   [M_0, M_1, M2] = LIKELIH_NEGBIN_TILTEDMOMENTS(LIKELIH, Y, I, S2, MYY) takes a 
    %   likelihood data structure LIKELIH, incedence counts Y, index I and cavity variance 
    %   S2 and mean MYY. Returns the zeroth moment M_0, firtst moment M_1 and second moment 
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
    %LIKELIH_NEGBIN_SITEDERIV    Evaluate the derivative with respect to cite parameters
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

    function [z, energ, diagn] = likelih_negbin_mcmc(z, opt, varargin)
    %LIKELIH_NEGBIN_MCMC        Conducts the MCMC sampling of latent values
    %
    %   Description
    %   [F, ENERG, DIAG] = LIKELIH_NEGBIN_MCMC(F, OPT, GP, X, Y) takes the current latent 
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
        switch gp.type
          case 'FULL'
            u = [];
          case 'FIC'
            u = gp.X_u;
            Lav=[];
          case 'CS+FIC'
            u = gp.X_u;
            Labl=[];
            Lp = [];            
          case {'PIC' 'PIC_BLOCK'}
            u = gp.X_u;
            ind = gp.tr_index;
            Labl=[];
            Lp = [];
        end
        n=length(y);

        J = [];
        U = [];
        iJUU = [];
        Linv=[];
        L2=[];
        iLaKfuic=[];
        mincut = -300;
        if isfield(gp.likelih,'avgE');
            E = gp.likelih.avgE(:);
        else
            E = 1;
        end     

        % Evaluate the help matrices for covariance matrix
        switch gp.type
          case 'FULL'
            getL(z, gp, x, y);
            % Rotate z towards prior
            w = (L2\z)';    
          case 'FIC'
            getL(z, gp, x, y, u);
            % Rotate z towards prior as w = (L\z)';
            % Here we take an advantage of the fact that L = chol(diag(Lav)+b'b)
            % See cholrankup.m for general case of solving a Ax=b system
            zs = z./Lp;
            w = zs + U*((J*U'-U')*zs);
          case {'PIC' 'PIC_BLOCK'}
            getL(z, gp, x, y, u);
            zs=zeros(size(z));
            for i=1:length(ind)
                zs(ind{i}) = Lp{i}\z(ind{i});
            end
            w = zs + U*((J*U'-U')*zs);
          case {'CS+FIC'}
            getL(z, gp, x, y, u);
            %zs = Lp\z;
            zs = Lp*z;
            w = zs + U*((J*U'-U')*zs);
          otherwise 
            error('unknown type of GP\n')
        end

        %gradcheck(w', @lvnegbin_er, @lvnegbin_gr, gp, x, y, u, z)
        
        hmc2('state',latent_rstate)
        rej = 0;
        gradf = @lvnegbin_gr;
        f = @lvnegbin_er;
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
            z = Lp.*(w + U*(iJUU*w));
          case  {'PIC' 'PIC_BLOCK'}
            w2 = w + U*(iJUU*w);
            for i=1:length(ind)
                z(ind{i}) = Lp{i}*w2(ind{i});
            end
          case  {'CS+FIC'}
            w2 = w + U*(iJUU*w);
            %        z = Lp*w2;
            z = Lp\w2;
        end
        opt.latent_rstate = hmc2('state');
        diagn.opt = opt;
        diagn.rej = rej;
        diagn.lvs = opt.stepadj;

        function [g, gdata, gprior] = lvnegbin_gr(w, gp, x, y, u, varargin)
        %LVNEGBIN_G	Evaluate gradient function for GP latent values with
        %               Negative-Binomial likelihood
            
        % Force z and E to be a column vector
            w=w(:);
            
            switch gp.type
              case 'FULL'
                z = L2*w;
                z = max(z,mincut);
                mu = exp(z).*E;
                %gdata = exp(z).*E - y;
                r = gp.likelih.disper;
                gdata = mu.*(r+y)./(r+mu) - y;
                b=Linv*z;
                gprior=Linv'*b;              
                g = (L2'*(gdata + gprior))';        
              case 'FIC'
                %        w(w<eps)=0;
                z = Lp.*(w + U*(iJUU*w));
                z = max(z,mincut);
                mu = exp(z).*E;
                r = gp.likelih.disper;
                gdata = mu.*(r+y)./(r+mu) - y;            
                %gdata = exp(z).*E - y;
                gprior = z./Lav - iLaKfuic*(iLaKfuic'*z);
                g = gdata +gprior;
                g = Lp.*g;
                g = g + U*(iJUU*g);
                g = g';
              case {'PIC' 'PIC_BLOCK'}
                w2= w + U*(iJUU*w);
                for i=1:length(ind)
                    z(ind{i}) = Lp{i}*w2(ind{i});
                end
                z = max(z,mincut);
                mu = exp(z).*E;
                r = gp.likelih.disper;
                gdata = mu.*(r+y)./(r+mu) - y;
                gprior = zeros(size(gdata));
                for i=1:length(ind)
                    gprior(ind{i}) = Labl{i}\z(ind{i});
                end
                gprior = gprior - iLaKfuic*(iLaKfuic'*z);
                g = gdata' + gprior';
                for i=1:length(ind)
                    g(ind{i}) = g(ind{i})*Lp{i};
                end
                g = g + g*U*(iJUU);
                %g = g';
              case 'CS+FIC'
                w2= w + U*(iJUU*w);
                %            z = Lp*w2;
                z = Lp\w2;
                z = max(z,mincut);
                r = gp.likelih.disper;
                gdata = mu.*(r+y)./(r+mu) - y;
                gprior = zeros(size(gdata));
                gprior = Labl\z;
                gprior = gprior - iLaKfuic*(iLaKfuic'*z);
                g = gdata' + gprior';
                %            g = g*Lp;
                g = g/Lp;
                g = g + g*U*(iJUU);
            end
        end

        function [e, edata, eprior] = lvnegbin_er(w, gp, x, t, u, varargin)
        %function [e, edata, eprior] = gp_e(w, gp, x, t, varargin)
        % LVNEGBIN_E     Minus log likelihood function for spatial modelling.
        %
        %       E = LVNEGBIN_E(X, GP, T, Z) takes.... and returns minus log from 
            
        % The field gp.likelihavgE (if given) contains the information about averige
        % expected number of cases at certain location. The target, t, is 
        % distributed as t ~ negbin(avgE*exp(z))
            
        % force z and E to be a column vector

            w=w(:);

            switch gp.type
              case 'FULL'
                z = L2*w;        
                z = max(z,mincut);
                B=Linv*z;
                eprior=.5*sum(B.^2);
              case 'FIC' 
                %        w(w<eps)=0;
                z = Lp.*(w + U*(iJUU*w));
                z = max(z,mincut);
                % eprior = 0.5*z'*inv(La)*z-0.5*z'*(inv(La)*K_fu*inv(K_uu+Kuf*inv(La)*K_fu)*K_fu'*inv(La))*z;
                B = z'*iLaKfuic;  % 1 x u
                eprior = 0.5*sum(z.^2./Lav)-0.5*sum(B.^2);
              case {'PIC' 'PIC_BLOCK'}
                w2= w + U*(iJUU*w);
                for i=1:length(ind)
                    z(ind{i}) = Lp{i}*w2(ind{i});
                end
                z = max(z,mincut);
                B = z'*iLaKfuic;  % 1 x u
                eprior = - 0.5*sum(B.^2);
                for i=1:length(ind)
                    eprior = eprior + 0.5*z(ind{i})'/Labl{i}*z(ind{i});
                end
              case 'CS+FIC'
                w2= w + U*(iJUU*w);
                %            z = Lp*w2;
                z = Lp\w2;
                z = max(z,mincut);
                B = z'*iLaKfuic;  % 1 x u
                eprior = - 0.5*sum(B.^2);
                eprior = eprior + 0.5*z'/Labl*z;
            end
            mu = exp(z).*E;
            %edata = sum(mu-t.*log(mu));
            r = gp.likelih.disper;
            edata = sum(gammaln(r)-y.*(log(mu)-log(r+mu))-gammaln(r+y)-r.*(log(r)-log(r+mu)));
            %        eprior = .5*sum(w.^2);
            e=edata + eprior;
        end

        function getL(w, gp, x, t, u)
        % Evaluate the cholesky decomposition if needed
            if nargin < 5
                C=gp_trcov(gp, x);
                % Evaluate a approximation for posterior variance
                % Take advantage of the matrix inversion lemma
                %        L=chol(inv(inv(C) + diag(1./gp.likelih.avgE)))';
                Linv = inv(chol(C)');
                %L2 = C/chol(diag(1./gp.likelih.avgE) + C);  %sparse(1:n, 1:n, 1./gp.likelih.avgE)
                r = gp.likelih.disper;
                
                L2 = C/chol(diag((r.^2+r.*gp.likelih.avgE+gp.likelih.avgE.^2)./(gp.likelih.avgE.*(r^2+r.*t))) + C);  %sparse(1:n, 1:n, 1./gp.likelih.avgE)
                
                L2 = chol(C - L2*L2')';
            else

        % Evaluate the Lambda (La) for specific model
                switch gp.type
                  case 'FIC'
                    [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
                    K_fu = gp_cov(gp, x, u);         % f x u
                    K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
                    K_uu = (K_uu+K_uu')/2;     % ensure the symmetry of K_uu
                    Luu = chol(K_uu)';
                    % Q_ff = K_fu*inv(K_uu)*K_fu'
                    % Here we need only the diag(Q_ff), which is evaluated below
                    b=Luu\(K_fu');       % u x f
                    Qv_ff=sum(b.^2)';
                    Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements
                                         % Lets scale Lav to ones(f,1) so that Qff+La -> sqrt(La)*Qff*sqrt(La)+I
                                         % and form iLaKfu
                    iLaKfu = zeros(size(K_fu));  % f x u,
                    for i=1:n
                        iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u 
                    end
                    c = K_uu+K_fu'*iLaKfu; 
                    c = (c+c')./2;         % ensure symmetry
                    c = chol(c)';   % u x u, 
                    ic = inv(c);
                    iLaKfuic = iLaKfu*ic';
                    %Lp = sqrt(1./(gp.likelih.avgE + 1./Lav));
                    r = gp.likelih.disper;
                    Lp = sqrt(1./((r.^2+r.*gp.likelih.avgE+gp.likelih.avgE.^2)./(gp.likelih.avgE.*(r^2+r.*t)) + 1./Lav));
                    b=b';
                    for i=1:n
                        b(i,:) = iLaKfuic(i,:).*Lp(i);
                    end        
                    [V,S2]= eig(b'*b);
                    S = sqrt(S2);
                    U = b*(V/S);
                    U(abs(U)<eps)=0;
                    %        J = diag(sqrt(diag(S2) + 0.01^2));
                    J = diag(sqrt(1-diag(S2)));   % this could be done without forming the diag matrix 
                                                  % J = diag(sqrt(2./(1+diag(S))));
                    iJUU = J\U'-U';
                    iJUU(abs(iJUU)<eps)=0;
                  case {'PIC' 'PIC_BLOCK'}
                    [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
                    K_fu = gp_cov(gp, x, u);         % f x u
                    K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
                    K_uu = (K_uu+K_uu')/2;     % ensure the symmetry of K_uu
                    Luu = chol(K_uu)';

                    % Q_ff = K_fu*inv(K_uu)*K_fu'
                    % Here we need only the diag(Q_ff), which is evaluated below
                    B=Luu\(K_fu');       % u x f
                    iLaKfu = zeros(size(K_fu));  % f x u
                    for i=1:length(ind)
                        Qbl_ff = B(:,ind{i})'*B(:,ind{i});
                        [Kbl_ff, Cbl_ff] = gp_trcov(gp, x(ind{i},:));
                        Labl{i} = Cbl_ff - Qbl_ff;
                        iLaKfu(ind{i},:) = Labl{i}\K_fu(ind{i},:);    % Check if works by changing inv(Labl{i})!!!
                    end
                    % Lets scale Lav to ones(f,1) so that Qff+La -> sqrt(La)*Qff*sqrt(La)+I
                    % and form iLaKfu
                    A = K_uu+K_fu'*iLaKfu;
                    A = (A+A')./2;            % Ensure symmetry
                    
                    % L = iLaKfu*inv(chol(A));
                    iLaKfuic = iLaKfu*inv(chol(A));
                    r = gp.likelih.disper;
                    
                    for i=1:length(ind)
                        Lp{i} = chol(inv(diag((r.^2+r.*gp.likelih.avgE(ind{i})+gp.likelih.avgE(ind{i}).^2)./(gp.likelih.avgE(ind{i}).*(r^2+r.*t(ind{i})))) + inv(Labl{i})));
                    end
                    b=zeros(size(B'));
                    
                    for i=1:length(ind)
                        b(ind{i},:) = Lp{i}*iLaKfuic(ind{i},:);
                    end   
                    
                    [V,S2]= eig(b'*b);
                    S = sqrt(S2);
                    U = b*(V/S);
                    U(abs(U)<eps)=0;
                    %        J = diag(sqrt(diag(S2) + 0.01^2));
                    J = diag(sqrt(1-diag(S2)));   % this could be done without forming the diag matrix 
                                                  % J = diag(sqrt(2./(1+diag(S))));
                    iJUU = J\U'-U';
                    iJUU(abs(iJUU)<eps)=0;
                  case 'CS+FIC'
                    % Q_ff = K_fu*inv(K_uu)*K_fu'
                    % Here we need only the diag(Q_ff), which is evaluated below
                    cf_orig = gp.cf;
                    
                    cf1 = {};
                    cf2 = {};
                    j = 1;
                    k = 1;
                    for i = 1:length(gp.cf)
                        if ~isfield(gp.cf{i},'cs')
                            cf1{j} = gp.cf{i};
                            j = j + 1;
                        else
                            cf2{k} = gp.cf{i};
                            k = k + 1;
                        end         
                    end
                    gp.cf = cf1;        

                    [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
                    K_fu = gp_cov(gp, x, u);         % f x u
                    K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
                    K_uu = (K_uu+K_uu')/2;     % ensure the symmetry of K_uu
                    Luu = chol(K_uu)';                
                    B=Luu\(K_fu');       % u x f

                    Qv_ff=sum(B.^2)';
                    Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements
                    
                    gp.cf = cf2;        
                    K_cs = gp_trcov(gp,x);
                    Labl = sparse(1:n,1:n,Lav,n,n) + K_cs;
                    gp.cf = cf_orig;
                    iLaKfu = Labl\K_fu;
                    % Lets scale Lav to ones(f,1) so that Qff+La -> sqrt(La)*Qff*sqrt(La)+I
                    % and form iLaKfu
                    A = K_uu+K_fu'*iLaKfu;
                    A = (A+A')./2;            % Ensure symmetry
                    
                    % L = iLaKfu*inv(chol(A));
                    iLaKfuic = iLaKfu*inv(chol(A));
                    
                    %Lp = chol(inv(sparse(1:n,1:n,gp.avgE,n,n) + inv(Labl)));
                    %Lp = inv(chol(sparse(1:n,1:n,gp.avgE,n,n) + inv(Labl))');
                    Lp = inv(Labl);
                    r = gp.likelih.disper;                               
                    Lp = sparse(1:n,1:n,(r.^2+r.*gp.likelih.avgE+gp.likelih.avgE.^2)./(gp.likelih.avgE.*(r^2+r.*t)),n,n) + Lp;
                    Lp = chol(Lp)';
                    %                Lp = inv(Lp);


                    b=zeros(size(B'));
                    
                    %                b = Lp*iLaKfuic;
                    b = Lp\iLaKfuic;
                    
                    [V,S2]= eig(b'*b);
                    S = sqrt(S2);
                    U = b*(V/S);
                    U(abs(U)<eps)=0;
                    %        J = diag(sqrt(diag(S2) + 0.01^2));
                    J = diag(sqrt(1-diag(S2)));   % this could be done without forming the diag matrix 
                                                  % J = diag(sqrt(2./(1+diag(S))));
                    iJUU = J\U'-U';
                    iJUU(abs(iJUU)<eps)=0;                      
                end
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


