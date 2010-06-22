function likelih = likelih_binomial(do, varargin)
%LIKELIH_BINOMIAL       Create a binomial likelihood structure 
%
%	Description%
%	LIKELIH = LIKELIH_BINOMIAL('INIT') Create and initialize 
%                 binomial likelihood. 
%
%       The likelihood is defined as follows:
%                            __ n
%                p(y|f, z) = || i=1 [ p_i^(y_i)*(1-p_i)^(z_i-y_i)) * 
%                                    gamma(z_i+1)/(gamma(y_i+1)*gamma(z_i-y_i+1))]
%
%       where p_i = exp(f_i)/ (1+exp(f_i)) is the succes probability,
%       which is a function of the latent variable f_i and z is a
%       vector of numbers of trials. When using Binomial likelihood
%       you need to give the vector z as an extra parameter to each
%       function that requires y also. For example, you should call
%       gpla_e as follows 
%            gpla_e(w, gp, x, y, 'z', z)
%
%
%	The fields in LIKELIH are:
%	  likelih.type             = 'likelih_binomial'
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
%         likelih.fh_predy         = function handle to evaluate predictive 
%                                    density of y
%         likelih.fh_recappend     = function handle to append the record
%
%	See also
%       LIKELIH_LOGIT, LIKELIH_PROBIT, LIKELIH_NEGBIN
%

% Copyright (c) 2009-2010	Jaakko Riihimäki & Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 1
        error('Not enough arguments')
    end

    % Initialize the likelihood structure
    if strcmp(do, 'init')
        likelih.type = 'binomial';
        
        % Set the function handles to the nested functions
        likelih.fh_pak = @likelih_binomial_pak;
        likelih.fh_unpak = @likelih_binomial_unpak;
        likelih.fh_e = @likelih_binomial_e;
        likelih.fh_g = @likelih_binomial_g;    
        likelih.fh_g2 = @likelih_binomial_g2;
        likelih.fh_g3 = @likelih_binomial_g3;
        likelih.fh_tiltedMoments = @likelih_binomial_tiltedMoments;
        likelih.fh_predy = @likelih_binomial_predy;
        likelih.fh_recappend = @likelih_binomial_recappend;

        if length(varargin) > 2
            if mod(nargin,2) ~=1
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=2:2:length(varargin)-1
                switch varargin{i}
                  otherwise
                    error('Wrong parameter name!')
                end
            end
        end
    end

    % Set the parameter values of likelihood
    if strcmp(do, 'set')
        if mod(nargin,2) ~=0
            error('Wrong number of arguments')
        end
        likelih = varargin{1};
        % Loop through all the parameter values that are changed
        for i=2:2:length(varargin)-1
            switch varargin{i}
              otherwise
                error('Wrong parameter name!')
            end
        end
    end



    function w = likelih_binomial_pak(likelih)
    %LIKELIH_BINOMIAL_PAK    Combine likelihood parameters into one vector.
    %
    %	Description 
    %   W = LIKELIH_BINOMIAL_PAK(LIKELIH) takes a likelihood data
    %   structure LIKELIH and returns an empty verctor W. If Binomial
    %   likelihood had hyperparameters this would combine them into a
    %   single row vector W (see e.g. likelih_negbin).
    %	  
    %
    %	See also
    %	LIKELIH_NEGBIN_UNPAK, GP_PAK

        w = [];
    end


    function [likelih, w] = likelih_binomial_unpak(w, likelih)
    %LIKELIH_BINOMIAL_UNPAK  Extract likelihood parameters from the vector.
    %
    %	Description
    %   W = LIKELIH_BINOMIAL_UNPAK(W, LIKELIH) Doesn't do anything.
    % 
    %   If Binomial likelihood had hyperparameters this would extracts
    %   them parameters from the vector W to the LIKELIH structure.
    %	  
    %
    %	See also
    %	LIKELIH_BINOMIAL_PAK, GP_UNPAK

      likelih=likelih;
      w=[];
    end



    function logLikelih = likelih_binomial_e(likelih, y, f, z)
    %LIKELIH_BINOMIAL_E    Log likelihood
    %
    %   Description
    %   E = LIKELIH_BINOMIAL_E(LIKELIH, Y, F, Z) takes a likelihood
    %   data structure LIKELIH, succes counts Y, numbers of trials Z,
    %   and latent values F. Returns the log likelihood, log p(y|f,z).
    %
    %   See also
    %   LIKELIH_BINOMIAL_G, LIKELIH_BINOMIAL_G3, LIKELIH_BINOMIAL_G2, GPLA_E
        
        if isempty(z)
            error(['likelih_binomial -> likelih_binomial_e: missing z!'... 
                   'Binomial likelihood needs the expected number of   '...
                   'occurrences as an extra input z. See, for         '...
                   'example, likelih_binomial and gpla_e.             ']);
        end

        
        expf = exp(f);
        p = expf ./ (1+expf);
        N = z;
        logLikelih =  sum(gammaln(N+1)-gammaln(y+1)-gammaln(N-y+1)+y.*log(p)+(N-y).*log(1-p));
    end


    function g = likelih_binomial_g(likelih, y, f, param, z)
    %LIKELIH_BINOMIAL_G    Gradient of log likelihood (energy)
    %
    %   Description 
    %   G = LIKELIH_BINOMIAL_G(LIKELIH, Y, F, PARAM) takes a likelihood
    %   data structure LIKELIH, succes counts Y, numbers of trials Z
    %   and latent values F. Returns the gradient of log likelihood 
    %   with respect to PARAM. At the moment PARAM can be 'hyper' or
    %   'latent'.
    %
    %   See also
    %   LIKELIH_BINOMIAL_E, LIKELIH_BINOMIAL_G2, LIKELIH_BINOMIAL_G3, GPLA_E

        if isempty(z)
            error(['likelih_binomial -> likelih_binomial_g: missing z!'... 
                   'Binomial likelihood needs the expected number of   '...
                   'occurrences as an extra input z. See, for         '...
                   'example, likelih_binomial and gpla_e.             ']);
        end

        
        
        switch param
          case 'latent'
            expf = exp(f);
            N = z;
            
            g = y./(1+expf) - (N-y).*expf./(1+expf);
        end
    end
    

    function g2 = likelih_binomial_g2(likelih, y, f, param, z)
    %LIKELIH_BINOMIAL_G2  Second gradients of log likelihood (energy)
    %
    %   Description        
    %   G2 = LIKELIH_BINOMIAL_G2(LIKELIH, Y, F, PARAM) takes a likelihood
    %   data structure LIKELIH, succes counts Y, numbers of trials Z,
    %   and latent values F. Returns the hessian of log likelihood
    %   with respect to PARAM. At the moment PARAM can be only
    %   'latent'. G2 is a vector with diagonal elements of the hessian
    %   matrix (off diagonals are zero).
    %
    %   See also
    %   LIKELIH_BINOMIAL_E, LIKELIH_BINOMIAL_G, LIKELIH_BINOMIAL_G3, GPLA_E

        if isempty(z)
            error(['likelih_binomial -> likelih_binomial_g2: missing z!'... 
                   'Binomial likelihood needs the expected number of    '...
                   'occurrences as an extra input z. See, for          '...
                   'example, likelih_binomial and gpla_e.              ']);
        end
        
        
        switch param
          case 'latent'
            expf = exp(f);
            N = z;

            g2 = -N.*expf./(1+expf).^2;
        end
    end
    
    
    function g3 = likelih_binomial_g3(likelih, y, f, param, z)
    %LIKELIH_BINOMIAL_G3  Third gradients of log likelihood (energy)
    %
    %   Description
        
    %   G3 = LIKELIH_BINOMIAL_G3(LIKELIH, Y, F, PARAM) takes a likelihood 
    %   data structure LIKELIH, succes counts Y, numbers of trials Z
    %   and latent values F and returns the third gradients of log
    %   likelihood with respect to PARAM. At the moment PARAM can be
    %   only 'latent'. G3 is a vector with third gradients.
    %
    %   See also
    %   LIKELIH_BINOMIAL_E, LIKELIH_BINOMIAL_G, LIKELIH_BINOMIAL_G2, GPLA_E, GPLA_G
    
        if isempty(z)
            error(['likelih_binomial -> likelih_binomial_g3: missing z!'... 
                   'Binomial likelihood needs the expected number of    '...
                   'occurrences as an extra input z. See, for          '...
                   'example, likelih_binomial and gpla_e.              ']);
        end

        
     switch param
          case 'latent'
            expf = exp(f);
            N = z;
            g3 = N.*(expf.*(expf-1))./(1+expf).^3;
          end
     end

    function [m_0, m_1, sigm2hati1] = likelih_binomial_tiltedMoments(likelih, y, i1, sigm2_i, myy_i, z)
    %LIKELIH_BINOMIAL_TILTEDMOMENTS    Returns the marginal moments for EP algorithm
    %
    %   Description
    %   [M_0, M_1, M2] = LIKELIH_BINOMIAL_TILTEDMOMENTS(LIKELIH, Y, I, S2, MYY, Z) 
    %   takes a likelihood data structure LIKELIH, succes counts Y, 
    %   numbers of trials Z, index I and cavity variance S2 and mean
    %   MYY. Returns the zeroth moment M_0, mean M_1 and variance M_2
    %   of the posterior marginal (see Rasmussen and Williams (2006):
    %   Gaussian processes for Machine Learning, page 55).
    %
    %   See also
    %   GPEP_E

        
        if isempty(z)
            error(['likelih_binomial -> likelih_binomial_tiltedMoments: missing z!'... 
                   'Binomial likelihood needs the expected number of               '...
                   'occurrences as an extra input z. See, for                     '...
                   'example, likelih_binomial and gpla_e.                         ']);
        end

       
        yy = y(i1);
        N = z(i1);
        
        
        % Create function handle for the function to be integrated (likelihood * cavity). 
        logbincoef=gammaln(N+1)-gammaln(yy+1)-gammaln(N-yy+1);
        zm = @(f)exp( logbincoef + yy*log(1./(1.+exp(-f)))+(N-yy)*log(1-1./(1.+exp(-f))) - 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2);
        
        % Set the integration limits (in this case based only on the prior).
        if yy > 0 && yy<N
            mean_app = log(yy./(N-yy));
            ld0=1/(1+exp(-mean_app));
            ld1=(1-ld0)*ld0;
            ld2=ld0-3*ld0^2+2*ld0^3;
            var_app=inv(-( yy*(ld2*ld0-ld1^2)/ld0^2 + (N-yy)*(ld2*(ld0-1)-ld1^2)/(ld0-1)^2 ));
            
            mean_app = (myy_i/sigm2_i + mean_app/var_app)/(1/sigm2_i + 1/var_app);
            sigm_app = sqrt((1/sigm2_i + 1/var_app)^-1);
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
            [m_0, m_1, m_2] = quad_moments(zm, lambdaconf(1), lambdaconf(2), RTOL, ATOL);
            sigm2hati1 = m_2 - m_1.^2;
            if sigm2hati1 >= sigm2_i
                error('likelih_binomial_tilted_moments: sigm2hati1 >= sigm2_i');
            end
        end
    end

    
    function [Ey, Vary, Py] = likelih_binomial_predy(likelih, Ef, Varf, yt, zt)
    %LIKELIH_BINOMIAL_PREDY    Returns the predictive mean, variance and density of y
    %
    %   Description         
    %   [EY, VARY] = LIKELIH_BINOMIAL_PREDY(LIKELIH, EF, VARF)
    %   takes a likelihood data structure LIKELIH, posterior mean EF
    %   and posterior Variance VARF of the latent variable and returns
    %   the posterior predictive mean EY and variance VARY of the
    %   observations related to the latent variables
    %        
    %   [Ey, Vary, PY] = LIKELIH_BINOMIAL_PREDY(LIKELIH, EF, VARF YT, ZT)
    %   Returns also the predictive density of YT, that is 
    %        p(yt | y, zt) = \int p(yt | f, zt) p(f|y) df.
    %   This requires also the succes counts YT, numbers of trials ZT.
    %
    %   See also 
    %   ep_pred, la_pred, mc_pred


        if isempty(zt)
            error(['likelih_binomial -> likelih_binomial_predy: missing z!'... 
                   'Binomial likelihood needs the expected number of       '...
                   'occurrences as an extra input z. See, for             '...
                   'example, likelih_binomial and gpla_e.                 ']);
        end

        
        nt=length(Ef);
        Ey=zeros(nt,1);
        EVary = zeros(nt,1);
        VarEy = zeros(nt,1);
        
        if nargout > 2
            Py=zeros(nt,1);
        end
        
        for i1=1:nt
            ci = sqrt(Varf(i1));
            F  = @(x)zt(i1)./(1+exp(-x)).*normpdf(x,Ef(i1),sqrt(Varf(i1)));
            Ey(i1) = quadgk(F,Ef(i1)-6*ci,Ef(i1)+6*ci);
            
            F2  = @(x)zt(i1)./(1+exp(-x)).*(1-1./(1+exp(-x))).*normpdf(x,Ef(i1),sqrt(Varf(i1)));
            EVary(i1) = quadgk(F2,Ef(i1)-6*ci,Ef(i1)+6*ci);
            
            F3  = @(x)(zt(i1)./(1+exp(-x))).^2.*normpdf(x,Ef(i1),sqrt(Varf(i1)));
            VarEy(i1) = quadgk(F3,Ef(i1)-6*ci,Ef(i1)+6*ci) - Ey(i1).^2;
            
            if nargout > 2
                %bin_cc=exp(gammaln(zt(i1)+1)-gammaln(yt(i1)+1)-gammaln(zt(i1)-yt(i1)+1));
                %F  = @(x)bin_cc.*(1./(1+exp(-x))).^yt(i1).*(1-(1./(1+exp(-x)))).^(zt(i1)-yt(i1)).*normpdf(x,Ef(i1),sqrt(Varf(i1)));
                F  = @(x)exp(gammaln(zt(i1)+1)-gammaln(yt(i1)+1)-gammaln(zt(i1)-yt(i1)+1) + yt(i1).*log(1./(1+exp(-x))) + (zt(i1)-yt(i1)).*log(1-(1./(1+exp(-x))))).*normpdf(x,Ef(i1),sqrt(Varf(i1)));
                Py(i1) = quadgk(F,Ef(i1)-6*ci,Ef(i1)+6*ci);
            end
        end
        Vary = EVary+VarEy;
    end
    
    
    function reclikelih = likelih_binomial_recappend(reclikelih, ri, likelih)
    % RECAPPEND  Append the parameters to the record
    %
    %          Description 
    %          RECLIKELIH = GPCF_BINOMIAL_RECAPPEND(RECLIKELIH, RI, LIKELIH)
    %          takes a likelihood record structure RECLIKELIH, record
    %          index RI and likelihood structure LIKELIH with the
    %          current MCMC samples of the hyperparameters. Returns
    %          RECLIKELIH which contains all the old samples and the
    %          current samples from LIKELIH.
    % 
    %  See also:
    %  gp_mc
        
        
        if nargin == 2
            reclikelih.type = 'binomial';

            % Set the function handles
            reclikelih.fh_pak = @likelih_binomial_pak;
            reclikelih.fh_unpak = @likelih_binomial_unpak;
            reclikelih.fh_e = @likelih_binomial_e;
            reclikelih.fh_g = @likelih_binomial_g;    
            reclikelih.fh_g2 = @likelih_binomial_g2;
            reclikelih.fh_g3 = @likelih_binomial_g3;
            reclikelih.fh_tiltedMoments = @likelih_binomial_tiltedMoments;
            reclikelih.fh_mcmc = @likelih_binomial_mcmc;
            reclikelih.fh_predy = @likelih_binomial_predy;
            reclikelih.fh_recappend = @likelih_binomial_recappend;
            return
        end

    end
end

