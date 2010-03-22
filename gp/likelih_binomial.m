function likelih = likelih_binomial(do, varargin)
%likelih_binomial	Create a binomial likelihood structure for Gaussian Process
%
%	Description
%
%	LIKELIH = LIKELIH_BINOMIAL('INIT', Y, N) Create and initialize binomial likelihood. 
%       The input argument Y contains the number 'successes', and N
%       contains the number of trials.
%
%	The fields in LIKELIH are:
%	  likelih.type             = 'likelih_binomial'
%         likelih.N                = N;
%         likelih.fh_pak           = function handle to pak
%         likelih.fh_unpak         = function handle to unpak
%         likelih.fh_permute       = function handle to permutation
%         likelih.fh_e             = function handle to energy of likelihood
%         likelih.fh_g             = function handle to gradient of energy
%         likelih.fh_g2            = function handle to second derivatives of energy
%         likelih.fh_g3            = function handle to third (diagonal) gradient of energy 
%         likelih.fh_tiltedMoments = function handle to evaluate tilted moments for EP
%         likelih.fh_predy         = function handle to evaluate predictive density of y
%         likelih.fh_recappend     = function handle to record append
%
%	LIKELIH = LIKELIH_BINOMIAL('SET', LIKELIH, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH.
%
%	See also
%       LIKELIH_LOGIT, LIKELIH_PROBIT, LIKELIH_NEGBIN
%
%

% Copyright (c) 2009-2010	Jaakko Riihimäki & Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end

    % Initialize the likelihood structure
    if strcmp(do, 'init')
        y = varargin{1};
        N = varargin{2};
        likelih.type = 'binomial';
        
        % check the arguments
        if ~isempty(find(y<0))
            error('The number of successes have to be greater or equal to zero y >= 0.')
        end     
        if ~isempty(find(N<=0))
            error('The number of trials have to be greater than zero N > 0.')
        end
        if ~isempty(find(N<y))
            error('The number of trials have to be greater or equal than the number of successes.')
        end
        
        
        % Set parameters
        likelih.N = N;

        % Initialize prior structure

        % Set the function handles to the nested functions
        likelih.fh_pak = @likelih_binomial_pak;
        likelih.fh_unpak = @likelih_binomial_unpak;
        likelih.fh_permute = @likelih_binomial_permute;
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
                  case 'N'
                    likelih.N = varargin{i+1};
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
              case 'N'
                likelih.N = varargin{i+1};
              case 'Nt'
                likelih.Nt = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end



    function w = likelih_binomial_pak(likelih, w)
    %LIKELIH_BINOMIAL_PAK      Combine likelihood parameters into one vector.
    %
    %   NO PARAMETERS!
    %
    %	Description
    %	W = LIKELIH_BINOMIAL_PAK(GPCF, W) takes a likelihood data structure LIKELIH and
    %	combines the parameters into a single row vector W.
    %	  
    %
    %	See also
    %	LIKELIH_BINOMIAL_UNPAK
        w = [];
    end


    function [likelih, w] = likelih_binomial_unpak(likelih, w)
    %LIKELIH_BINOMIAL_UNPAK      Combine likelihood parameters into one vector.
    %
    %   NO PARAMETERS!
    %
    %	Description
    %	W = LIKELIH_BINOMIAL_UNPAK(GPCF, W) takes a likelihood data structure LIKELIH and
    %	combines the parameter vector W and sets the parameters in LIKELIH.
    %	  
    %
    %	See also
    %	LIKELIH_BINOMIAL_PAK
    end



    function likelih = likelih_binomial_permute(likelih, p)
    %LIKELIH_BINOMIAL_PERMUTE    A function to permute the ordering of parameters 
    %                           in likelihood structure
    %   Description
    %	LIKELIH = LIKELIH_BINOMIAL_UNPAK(LIKELIH, P) takes a likelihood data structure
    %   LIKELIH and permutation vector P and returns LIKELIH with its parameters permuted
    %   according to P.
    %
    %   See also 
    %   GPLA_E, GPLA_G, GPEP_E, GPEP_G with CS+FIC model
        
        likelih.N = likelih.N(p,:);
    end


    function logLikelih = likelih_binomial_e(likelih, y, f)
    %LIKELIH_BINOMIAL_E    (Likelihood) Energy function
    %
    %   Description
    %   E = LIKELIH_BINOMIAL_E(LIKELIH, Y, F) takes a likelihood data structure
    %   LIKELIH, the number of 'successes' Y and latent values F and returns the log likelihood.
    %
    %   See also
    %   LIKELIH_BINOMIAL_G, LIKELIH_BINOMIAL_G3, LIKELIH_BINOMIAL_G2, GPLA_E
        
        expf = exp(f);
        p = expf ./ (1+expf);
        N = likelih.N;
        logLikelih =  sum(gammaln(N+1)-gammaln(y+1)-gammaln(N-y+1)+y.*log(p)+(N-y).*log(1-p));
    end


    function g = likelih_binomial_g(likelih, y, f, param)
    %LIKELIH_BINOMIAL_G    Gradient of (likelihood) energy function
    %
    %   Description
    %   G = LIKELIH_BINOMIAL_G(LIKELIH, Y, F, PARAM) takes a likelihood data structure
    %   LIKELIH, the number of 'successes' Y and latent values F and returns the gradient of 
    %   log likelihood with respect to PARAM. At the moment PARAM can be only 'latent'.
    %
    %   See also
    %   LIKELIH_BINOMIAL_E, LIKELIH_BINOMIAL_G2, LIKELIH_BINOMIAL_G3, GPLA_E
    
    switch param
        case 'latent'
            expf = exp(f);
            N = likelih.N;
            
            g = y./(1+expf) - (N-y).*expf./(1+expf);
        end
    end
    

    function g2 = likelih_binomial_g2(likelih, y, f, param)
    %LIKELIH_BINOMIAL_G2    Third gradients of (likelihood) energy function
    %
    %   Description
    %   G2 = LIKELIH_BINOMIAL_G2(LIKELIH, Y, F, PARAM) takes a likelihood data 
    %   structure LIKELIH, the number of 'successes' Y and latent values F and returns the 
    %   hessian of log likelihood with respect to PARAM. At the moment PARAM can 
    %   be only 'latent'. G2 is a vector with diagonal elements of the hessian 
    %   matrix (off diagonals are zero).
    %
    %   See also
    %   LIKELIH_BINOMIAL_E, LIKELIH_BINOMIAL_G, LIKELIH_BINOMIAL_G3, GPLA_E

     switch param
          case 'latent'
            expf = exp(f);
            N = likelih.N;

            g2 = -N.*expf./(1+expf).^2;
        end
    end
    
    
    function g3 = likelih_binomial_g3(likelih, y, f, param)
    %LIKELIH_BINOMIAL_G3    Gradient of (likelihood) Energy function
    %
    %   Description
    %   G3 = LIKELIH_BINOMIAL_G3(LIKELIH, Y, F, PARAM) takes a likelihood data 
    %   structure LIKELIH,  the number of 'successes' Y and latent values F and returns the 
    %   third gradients of log likelihood with respect to PARAM. At the moment PARAM can 
    %   be only 'latent'. G3 is a vector with third gradients.
    %
    %   See also
    %   LIKELIH_BINOMIAL_E, LIKELIH_BINOMIAL_G, LIKELIH_BINOMIAL_G2, GPLA_E, GPLA_G
    
     switch param
          case 'latent'
            expf = exp(f);
            N = likelih.N;
            g3 = N.*(expf.*(expf-1))./(1+expf).^3;
          end
     end

    function [m_0, m_1, sigm2hati1] = likelih_binomial_tiltedMoments(likelih, y, i1, sigm2_i, myy_i)
    %LIKELIH_BINOMIAL_TILTEDMOMENTS    Returns the moments of the tilted distribution
    %
    %   Description
    %   [M_0, M_1, M2] = LIKELIH_BINOMIAL_TILTEDMOMENTS(LIKELIH, Y, I, S2, MYY) takes a 
    %   likelihood data structure LIKELIH, the number of 'successes' Y, index I and cavity variance 
    %   S2 and mean MYY. Returns the zeroth moment M_0, first moment M_1 and second moment 
    %   M_2 of the tilted distribution
    %
    %   See also
    %   GPEP_E
       
        yy = y(i1);
        N = likelih.N(i1);
        
        
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

    
    function [Ey, Vary, Py] = likelih_binomial_predy(likelih, Ef, Varf, y)
    %LIKELIH_BINOMIAL_PREDY    Returns the predictive mean, variance and density of y
    %
    %   Description
    %   [Ey, Vary, py] = LIKELIH_BINOMIAL_PREDY(LIKELIH, EF, VARF, Y) 

        nt=length(Ef);
        Ey=zeros(nt,1);
        EVary = zeros(nt,1);
        VarEy = zeros(nt,1);
        
        if nargin > 3
            Py=zeros(nt,1);
        end
        
        for i1=1:nt
            ci = sqrt(Varf(i1));
            F  = @(x)likelih.Nt(i1)./(1+exp(-x)).*normpdf(x,Ef(i1),sqrt(Varf(i1)));
            Ey(i1) = quadgk(F,Ef(i1)-6*ci,Ef(i1)+6*ci);
            
            F2  = @(x)likelih.Nt(i1)./(1+exp(-x)).*(1-1./(1+exp(-x))).*normpdf(x,Ef(i1),sqrt(Varf(i1)));
            EVary(i1) = quadgk(F2,Ef(i1)-6*ci,Ef(i1)+6*ci);
            
            F3  = @(x)(likelih.Nt(i1)./(1+exp(-x))).^2.*normpdf(x,Ef(i1),sqrt(Varf(i1)));
            VarEy(i1) = quadgk(F3,Ef(i1)-6*ci,Ef(i1)+6*ci) - Ey(i1).^2;
            
            if nargin > 3
                bin_cc=exp(gammaln(likelih.Nt(i1)+1)-gammaln(y(i1)+1)-gammaln(likelih.Nt(i1)-y(i1)+1));
                
                F  = @(x)bin_cc.*(1./(1+exp(-x))).^y(i1).*(1-(1./(1+exp(-x)))).^(likelih.Nt(i1)-y(i1)).*normpdf(x,Ef(i1),sqrt(Varf(i1)));
                Py(i1) = quadgk(F,Ef(i1)-6*ci,Ef(i1)+6*ci);
            end
        end
        Vary = EVary+VarEy;
    end
    
    
    function reclikelih = likelih_binomial_recappend(reclikelih, ri, likelih)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_SEXP_RECAPPEND(RECCF, RI, GPCF) takes old covariance
    %          function record RECCF, record index RI, RECAPPEND returns a
    %          structure RECCF containing following record fields:
    %          lengthHyper    =
    %          lengthHyperNu  =
    %          lengthScale    =
    %          magnSigma2     =
        reclikelih = likelih;

    end
end

