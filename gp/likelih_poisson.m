function likelih = likelih_poisson(do, varargin)
%likelih_poisson	Create a Poisson likelihood structure for Gaussian Process
%
%	Description
%
%	LIKELIH = LIKELIH_POISSON('INIT') Create and initialize Poisson likelihood. 
%
%       The likelihood is defined as follows:
%                            __ n
%                p(y|f, z) = || i=1 Poisson(y_i|z_i*exp(f_i))
%
%       where z is a vector of expected mean and f the latent value vector whose 
%       components are transformed to relative risk exp(f_i). When using Poisosn 
%       likelihood you need to give the vector z as an extra parameter to each 
%       function that requires y also. For example, you should call gpla_e as follows
%           gpla_e(w, gp, x, y, 'z', z)
%
%	The fields in LIKELIH are:
%	  likelih.type             = 'likelih_poisson'
%         likelih.fh_pak           = function handle to pak
%         likelih.fh_unpak         = function handle to unpak
%         likelih.fh_e             = function handle to energy of likelihood
%         likelih.fh_g             = function handle to gradient of energy
%         likelih.fh_g2            = function handle to second derivatives of energy
%         likelih.fh_g3            = function handle to third (diagonal) gradient of energy 
%         likelih.fh_tiltedMoments = function handle to evaluate tilted moments for EP
%         likelih.fh_predy         = function handle to evaluate predictive density of y
%         likelih.fh_recappend     = function handle to record append
%
%	LIKELIH = LIKELIH_POISSON('SET', LIKELIH, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH.
%
%	See also
%       LIKELIH_LOGIT, LIKELIH_PROBIT, LIKELIH_NEGBIN

% Copyright (c) 2006      Helsinki University of Technology (author) Jarno Vanhatalo
% Copyright (c) 2007-2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 1
        error('Not enough arguments')
    end

    % Initialize the likelihood structure
    if strcmp(do, 'init')
        likelih.type = 'poisson';

        % Set the function handles to the nested functions
        likelih.fh_pak = @likelih_poisson_pak;
        likelih.fh_unpak = @likelih_poisson_unpak;
        likelih.fh_e = @likelih_poisson_e;
        likelih.fh_g = @likelih_poisson_g;    
        likelih.fh_g2 = @likelih_poisson_g2;
        likelih.fh_g3 = @likelih_poisson_g3;
        likelih.fh_tiltedMoments = @likelih_poisson_tiltedMoments;
        likelih.fh_predy = @likelih_poisson_predy;
        likelih.fh_recappend = @likelih_poisson_recappend;

        if nargin > 1
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
        gpcf = varargin{1};
        % Loop through all the parameter values that are changed
        for i=2:2:length(varargin)-1
            switch varargin{i}
              otherwise
                error('Wrong parameter name!')
            end
        end
    end
   
    function w = likelih_poisson_pak(likelih)
    %LIKELIH_POISSON_PAK      Combine likelihood parameters into one vector.
    %
    %   NO PARAMETERS
    %
    %	Description
    %	W = LIKELIH_POISSON_PAK(GPCF, W) takes a likelihood data structure LIKELIH and
    %	combines the parameters into a single row vector W.
    %	  
    %
    %	See also
    %	LIKELIH_POISSON_UNPAK
        w = [];
    end


    function [likelih, w] = likelih_poisson_unpak(w, likelih)
    %LIKELIH_POISSON_UNPAK      Combine likelihood parameters into one vector.
    %
    %   NO PARAMETERS
    %
    %	Description
    %	W = LIKELIH_POISSON_UNPAK(GPCF, W) takes a likelihood data structure LIKELIH and
    %	combines the parameter vector W and sets the parameters in LIKELIH.
    %	  
    %
    %	See also
    %	LIKELIH_POISSON_PAK
      likelih=likelih;
      w=[];
    end


    function logLikelih = likelih_poisson_e(likelih, y, f, z)
    %LIKELIH_POISSON_E    (Likelihood) Energy function
    %
    %   Description
    %   E = LIKELIH_POISSON_E(LIKELIH, Y, F) takes a likelihood data structure
    %   LIKELIH, incedence counts Y and latent values F and returns the log likelihood.
    %
    %   See also
    %   LIKELIH_POISSON_G, LIKELIH_POISSON_G3, LIKELIH_POISSON_G2, GPLA_E
        
        if isempty(z)
            error(['likelih_poisson -> likelih_poisson_e: missing z!'... 
                   'Poisson likelihood needs the expected number of '...
                   'occurrences as an extra input z. See, for       '...
                   'example, likelih_poisson and gpla_e.            ']);
        end
        
        lambda = z.*exp(f);
        gamlny = gammaln(y+1);
        logLikelih =  sum(-lambda + y.*log(lambda) - gamlny);
    end


    function deriv = likelih_poisson_g(likelih, y, f, param, z)
    %LIKELIH_POISSON_G    Gradient of (likelihood) energy function
    %
    %   Description
    %   G = LIKELIH_POISSON_G(LIKELIH, Y, F, PARAM) takes a likelihood data structure
    %   LIKELIH, incedence counts Y and latent values F and returns the gradient of 
    %   log likelihood with respect to PARAM. At the moment PARAM can be only 'latent'.
    %
    %   See also
    %   LIKELIH_POISSON_E, LIKELIH_POISSON_G2, LIKELIH_POISSON_G3, GPLA_E
        
        if isempty(z)
            error(['likelih_poisson -> likelih_poisson_g: missing z!'... 
                   'Poisson likelihood needs the expected number of '...
                   'occurrences as an extra input z. See, for       '...
                   'example, likelih_poisson and gpla_e.            ']);
        end
        
        switch param
          case 'latent'
            deriv = y - z.*exp(f);
        end
    end


    function g2 = likelih_poisson_g2(likelih, y, f, param, z)
    %LIKELIH_POISSON_G2    Third gradients of (likelihood) energy function
    %
    %   Description
    %   G2 = LIKELIH_POISSON_G2(LIKELIH, Y, F, PARAM) takes a likelihood data 
    %   structure LIKELIH, incedence counts Y and latent values F and returns the 
    %   hessian of log likelihood with respect to PARAM. At the moment PARAM can 
    %   be only 'latent'. G2 is a vector with diagonal elements of the hessian 
    %   matrix (off diagonals are zero).
    %
    %   See also
    %   LIKELIH_POISSON_E, LIKELIH_POISSON_G, LIKELIH_POISSON_G3, GPLA_E

        if isempty(z)
            error(['likelih_poisson -> likelih_poisson_g2: missing z!'... 
                   'Poisson likelihood needs the expected number of  '...
                   'occurrences as an extra input z. See, for        '...
                   'example, likelih_poisson and gpla_e.             ']);
        end
        
        switch param
          case 'latent'
            g2 = -z.*exp(f);
        end
    end    
    
    function third_grad = likelih_poisson_g3(likelih, y, f, param, z)
    %LIKELIH_POISSON_G3    Gradient of (likelihood) Energy function
    %
    %   Description
    %   G3 = LIKELIH_POISSON_G3(LIKELIH, Y, F, PARAM) takes a likelihood data 
    %   structure LIKELIH, incedence counts Y and latent values F and returns the 
    %   third gradients of log likelihood with respect to PARAM. At the moment PARAM can 
    %   be only 'latent'. G3 is a vector with third gradients.
    %
    %   See also
    %   LIKELIH_POISSON_E, LIKELIH_POISSON_G, LIKELIH_POISSON_G2, GPLA_E, GPLA_G
    
        if isempty(z)
            error(['likelih_poisson -> likelih_poisson_g3: missing z!'... 
                   'Poisson likelihood needs the expected number of  '...
                   'occurrences as an extra input z. See, for        '...
                   'example, likelih_poisson and gpla_e.             ']);
        end
        
        switch param
          case 'latent'
            third_grad = - z.*exp(f);
        end
    end

    function [m_0, m_1, sigm2hati1] = likelih_poisson_tiltedMoments(likelih, y, i1, sigm2_i, myy_i, z)
    %LIKELIH_POISSON_TILTEDMOMENTS    Returns the moments of the tilted distribution
    %
    %   Description
    %   [M_0, M_1, M2] = LIKELIH_POISSON_TILTEDMOMENTS(LIKELIH, Y, I, S2, MYY) takes a 
    %   likelihood data structure LIKELIH, incedence counts Y, index I and cavity variance 
    %   S2 and mean MYY. Returns the zeroth moment M_0, first moment M_1 and second moment 
    %   M_2 of the tilted distribution
    %
    %   See also
    %   GPEP_E
        
        if isempty(z)
            error(['likelih_poisson -> likelih_poisson_tiltedMoments: missing z!'... 
                   'Poisson likelihood needs the expected number of             '...
                   'occurrences as an extra input z. See, for                   '...
                   'example, likelih_poisson and gpla_e.                        ']);
        end
       
        yy = y(i1);
        gamlny = gammaln(y(i1)+1);
        avgE = z(i1);
        zm = @(f)exp(-avgE.*exp(f) + yy.*log(avgE.*exp(f)) - gamlny - 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2); %
        
        % Set the integration limits (in this case based only on the prior).
        if yy > 0
            mean_app = (myy_i/sigm2_i + log(yy/avgE)*yy)/(1/sigm2_i + yy);
            sigm_app = sqrt((1/sigm2_i + yy)^-1);
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
                error('likelih_poisson_tilted_moments: sigm2hati1 >= sigm2_i');
            end
        end
    end

    
    function [Ey, Vary, Py] = likelih_poisson_predy(likelih, Ef, Varf, y, z)
    %LIKELIH_POISSON_PREDY    Returns the predictive mean, variance and density of y
    %
    %   Description
    %   [Ey, Vary, py] = LIKELIH_POISSON_PREDY(LIKELIH, EF, VARF, Y) 

        if isempty(z)
            error(['likelih_poisson -> likelih_poisson_predy: missing z!'... 
                   'Poisson likelihood needs the expected number of     '...
                   'occurrences as an extra input z. See, for           '...
                   'example, likelih_poisson and gpla_e.                ']);
        end
        
        
        
        avgE = z;

        %nsamp = 10000;
        
        Py = zeros(size(Ef));
        Ey = zeros(size(Ef));
        EVary = zeros(size(Ef));
        VarEy = zeros(size(Ef)); 
        % Evaluate Ey and Vary (with MC)
        for i1=1:length(Ef)
%            %%% With MC
%            % First sample f
%            f_samp = normrnd(Ef(i1),sqrt(Varf(i1)),nsamp,1);
%            la_samp = avgE(i1).*exp(f_samp);
%  
%            % Conditional mean and variance of y (see Gelman et al. p. 23-24)
%            Ey(i1) = mean(la_samp);
%            Vary(i1) = Ey(i1) + var(la_samp);

           %%% With quadrature
           ci = sqrt(Varf(i1));

           F = @(x) avgE(i1).*exp(x).*normpdf(x,Ef(i1),sqrt(Varf(i1)));
           Ey(i1) = quadgk(F,Ef(i1)-6*ci,Ef(i1)+6*ci);
           
           EVary(i1) = Ey(i1);
           
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
                   mean_app = (myy_i/sigm2_i + log(y(i1)/avgE(i1)).*y(i1))/(1/sigm2_i + y(i1));
                   sigm_app = sqrt((1/sigm2_i + avgE(i1))^-1);
               else
                   mean_app = myy_i;
                   sigm_app = sqrt((1/sigm2_i + avgE(i1))^-1);
               end
               
               % Predictive density of the given observations
               pd = @(f) poisspdf(y(i1),avgE(i1).*exp(f)).*norm_pdf(f,myy_i,sqrt(sigm2_i));
               Py(i1) = quadgk(pd, mean_app - 12*sigm_app, mean_app + 12*sigm_app);
           end
       end
    end
    
    
    function reclikelih = likelih_poisson_recappend(reclikelih, ri, likelih)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_SEXP_RECAPPEND(RECCF, RI, GPCF) takes old covariance
    %          function record RECCF, record index RI, RECAPPEND returns a
    %          structure RECCF containing following record fields:
    %          lengthHyper    =
    %          lengthHyperNu  =
    %          lengthScale    =
    %          magnSigma2     =

        if nargin == 2
            reclikelih.type = 'poisson';

            % Set the function handles
            reclikelih.fh_pak = @likelih_poisson_pak;
            reclikelih.fh_unpak = @likelih_poisson_unpak;
            reclikelih.fh_e = @likelih_poisson_e;
            reclikelih.fh_g = @likelih_poisson_g;    
            reclikelih.fh_g2 = @likelih_poisson_g2;
            reclikelih.fh_g3 = @likelih_poisson_g3;
            reclikelih.fh_tiltedMoments = @likelih_poisson_tiltedMoments;
            reclikelih.fh_mcmc = @likelih_poisson_mcmc;
            reclikelih.fh_predy = @likelih_poisson_predy;
            reclikelih.fh_recappend = @likelih_poisson_recappend;
            return
        end
        
    end
end


