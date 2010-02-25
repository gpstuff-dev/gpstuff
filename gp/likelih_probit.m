function likelih = likelih_probit(do, varargin)
%LIKELIHOOD_PROBIT	Create a Probit likelihood structure for Gaussian Process
%
%	Description
%
%	LIKELIH = LIKELIH_PROBIT('INIT', Y, YE) Create and initialize Probit likelihood. 
%       The input argument Y contains class labels {-1,1}.
%
%	The fields in LIKELIH are:
%	  type                     = 'likelih_probit'
%         likelih.avgE             = YE;
%         likelih.gamlny           = gammaln(Y+1);
%         likelih.fh_pak           = function handle to pak
%         likelih.fh_unpak         = function handle to unpak
%         likelih.fh_permute       = function handle to permutation
%         likelih.fh_e             = function handle to energy of likelihood
%         likelih.fh_g             = function handle to gradient of energy
%         likelih.fh_g2            = function handle to second derivative of energy
%         likelih.fh_g3            = function handle to third (diagonal) gradient of energy 
%         likelih.fh_tiltedMoments = function handle to evaluate tilted moments for EP
%         likelih.fh_recappend     = function handle to record append
%
%	LIKELIH = LIKELIH_PROBIT('SET', LIKELIH, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH.
%
%	See also
%       LIKELIH_LOGIT, LIKELIH_PROBIT, LIKELIH_NEGBIN
%
%

% Copyright (c) 2007      Jaakko Riihimäki
% Copyright (c) 2007-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end

    % Initialize the covariance function
    if strcmp(do, 'init')
        likelih.type = 'probit';
        y = varargin{1};
        if ~isempty(find(y~=1 & y~= -1))
            error('The class labels have to be {-1,1}')
        end
        
        % Set the function handles to the nested functions
        likelih.fh_pak = @likelih_probit_pak;
        likelih.fh_unpak = @likelih_probit_unpak;
        likelih.fh_permute = @likelih_probit_permute;
        likelih.fh_e = @likelih_probit_e;
        likelih.fh_g = @likelih_probit_g;    
        likelih.fh_g2 = @likelih_probit_g2;
        likelih.fh_g3 = @likelih_probit_g3;
        likelih.fh_tiltedMoments = @likelih_probit_tiltedMoments;
        likelih.fh_predy = @likelih_probit_predy;
        likelih.fh_recappend = @likelih_probit_recappend;
        
        if length(varargin) > 1
            if mod(nargin,2) ~=0
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

    % Set the parameter values of likelihood function
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



    function w = likelih_probit_pak(likelih, w)
    %LIKELIH_PROBIT_PAK      Combine likelihood parameters into one vector.
    %
    %   NOT IMPLEMENTED!
    %
    %	Description
    %	W = LIKELIH_PROBIT_PAK(GPCF, W) takes a likelihood data structure LIKELIH and
    %	combines the parameters into a single row vector W.
    %	  
    %
    %	See also
    %	LIKELIH_PROBIT_UNPAK
    end


    function w = likelih_probit_unpak(likelih, w)
    %LIKELIH_PROBIT_UNPAK      Combine likelihood parameters into one vector.
    %
    %   NOT IMPLEMENTED!
    %
    %	Description
    %	W = LIKELIH_PROBIT_UNPAK(GPCF, W) takes a likelihood data structure LIKELIH and
    %	combines the parameter vector W and sets the parameters in LIKELIH.
    %	  
    %
    %	See also
    %	LIKELIH_PROBIT_PAK

    end



    function likelih = likelih_probit_permute(likelih, p)
    %LIKELIH_PROBIT_PERMUTE    A function to permute the ordering of parameters 
    %                           in likelihood structure
    %   Description
    %	LIKELIH = LIKELIH_PROBIT_UNPAK(LIKELIH, P) takes a likelihood data structure
    %   LIKELIH and permutation vector P and returns LIKELIH with its parameters permuted
    %   according to P.
    %
    %   See also 
    %   GPLA_E, GPLA_G, GPEP_E, GPEP_G with CS+FIC model

    end


    function logLikelih = likelih_probit_e(likelih, y, f)
    %LIKELIH_PROBIT_E    (Likelihood) Energy function
    %
    %   Description
    %   E = LIKELIH_PROBIT_E(LIKELIH, Y, F) takes a likelihood data structure
    %   LIKELIH, incedence counts Y and latent values F and returns the log likelihood.
    %
    %   See also
    %   LIKELIH_PROBIT_G, LIKELIH_PROBIT_G3, LIKELIH_PROBIT_G2, GPLA_E

        logLikelih = sum(log(normcdf(y.*f)));
    end


    function deriv = likelih_probit_g(likelih, y, f, param)
    %LIKELIH_PROBIT_G    G2 of (likelihood) energy function
    %
    %   Description
    %   G = LIKELIH_PROBIT_G(LIKELIH, Y, F, PARAM) takes a likelihood data structure
    %   LIKELIH, incedence counts Y and latent values F and returns the gradient of 
    %   log likelihood with respect to PARAM. At the moment PARAM can be only 'latent'.
    %
    %   See also
    %   LIKELIH_PROBIT_E, LIKELIH_PROBIT_G2, LIKELIH_PROBIT_G3, GPLA_E

        switch param
          case 'latent'
            deriv = y.*normpdf(f)./normcdf(y.*f);
        end
    end


    function g2 = likelih_probit_g2(likelih, y, f, param)
    %LIKELIH_PROBIT_G2    Third gradients of (likelihood) energy function
    %
    %   Description
    %   G2 = LIKELIH_PROBIT_G2(LIKELIH, Y, F, PARAM) takes a likelihood data 
    %   structure LIKELIH, incedence counts Y and latent values F and returns the 
    %   hessian of log likelihood with respect to PARAM. At the moment PARAM can 
    %   be only 'latent'. G2 is a vector with diagonal elements of the hessian 
    %   matrix (off diagonals are zero).
    %
    %   See also
    %   LIKELIH_PROBIT_E, LIKELIH_PROBIT_G, LIKELIH_PROBIT_G3, GPLA_E
        switch param
          case 'latent'
            z = y.*f;
            g2 = -(normpdf(f)./normcdf(z)).^2 - z.*normpdf(f)./normcdf(z);
        end
    end
    
    function thir_grad = likelih_probit_g3(likelih, y, f, param)
    %LIKELIH_PROBIT_G3    Gradient of (likelihood) Energy function
    %
    %   Description
    %   G3 = LIKELIH_PROBIT_G3(LIKELIH, Y, F, PARAM) takes a likelihood data 
    %   structure LIKELIH, incedence counts Y and latent values F and returns the 
    %   third gradients of log likelihood with respect to PARAM. At the moment PARAM can 
    %   be only 'latent'. G3 is a vector with third gradients.
    %
    %   See also
    %   LIKELIH_PROBIT_E, LIKELIH_PROBIT_G, LIKELIH_PROBIT_G2, GPLA_E, GPLA_G

        switch param
          case 'latent'
            z2 = normpdf(f)./normcdf(y.*f);
            thir_grad = 2.*y.*z2.^3 + 3.*f.*z2.^2 - z2.*(y-y.*f.^2);
        end
    end
    

    function [m_0, m_1, m_2] = likelih_probit_tiltedMoments(likelih, y, i1, sigm2_i, myy_i)
    %LIKELIH_PROBIT_TILTEDMOMENTS    Returns the moments of the tilted distribution
    %
    %   Description
    %   [M_0, M_1, M2] = LIKELIH_PROBIT_TILTEDMOMENTS(LIKELIH, Y, I, S2, MYY) takes a 
    %   likelihood data structure LIKELIH, incedence counts Y, index I and cavity variance 
    %   S2 and mean MYY. Returns the zeroth moment M_0, firtst moment M_1 and second moment 
    %   M_2 of the tilted distribution
    %
    %   See also
    %   GPEP_E

        if ~isreal(y(i1).*myy_i./sqrt(1+sigm2_i))
            error('problem with moments')
        end
        
        m_0 = normcdf(y(i1).*myy_i./sqrt(1+sigm2_i));
        zi=y(i1)*myy_i/sqrt(1+sigm2_i);
        normp_zi = normpdf(zi);
        normc_zi = normcdf(zi);
        muhati1=myy_i+(y(i1)*sigm2_i*normp_zi)/(normc_zi*sqrt(1+sigm2_i));
        sigm2hati1=sigm2_i-(sigm2_i^2*normp_zi)/((1+sigm2_i)*normc_zi)*(zi+normp_zi/normc_zi);
        m_1 = muhati1;
        m_2 = sigm2hati1;
    end

    function [Ey, Vary, py] = likelih_probit_predy(likelih, Ef, Varf, y)
    % Return the predictive probability of ty given the posterior mean Ef 
    % and variance Varf
        
        py1 = normcdf(Ef./sqrt(1+Varf));
        Ey = 2*py1 - 1;

        % This seems wrong?
        %Vary = Ey.*(1-Ey.^2);      

        Vary = 1-(2*py1-1).^2;
        
        if nargin > 3
            py = normcdf(Ef.*y./sqrt(1+Varf));    % Probability p(y_new)
        end
    end
        
    

    function reclikelih = likelih_probit_recappend(reclikelih, ri, likelih)
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


