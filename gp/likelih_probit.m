function likelih = likelih_probit(do, varargin)
%LIKELIHOOD_PROBIT	Create a Probit likelihood structure 
%
%	Description
%
%	LIKELIH = LIKELIH_PROBIT('INIT') Create and initialize Probit 
%       likelihood for classification problem with class labels {-1,1}. 
%    
%       likelihood is defined as follows:
%                            __ n
%                p(y|f, z) = || i=1 normcdf(y_i * f_i)
%    
%       where f is the latent value vector.
%
%	The fields in LIKELIH are:
%	  type                     = 'likelih_probit'
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
%       LIKELIH_PROBIT, LIKELIH_PROBIT, LIKELIH_NEGBIN
%
%

% Copyright (c) 2007      Jaakko Riihimäki
% Copyright (c) 2007-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 1
        error('Not enough arguments')
    end

    % Initialize the covariance function
    if strcmp(do, 'init')
        likelih.type = 'probit';
        
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
        likelih.fh_permute = @likelih_probit_permute;
        
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

    function likelih_probit_permute
        
    end

    function w = likelih_probit_pak(likelih)
    %LIKELIH_PROBIT_PAK    Combine likelihood parameters into one vector.
    %
    %	Description 
    %   W = LIKELIH_PROBIT_PAK(LIKELIH) takes a likelihood data
    %   structure LIKELIH and returns an empty verctor W. If Probit
    %   likelihood had hyperparameters this would combine them into a
    %   single row vector W (see e.g. likelih_negbin).
    %	  
    %
    %	See also
    %	LIKELIH_NEGBIN_UNPAK, GP_PAK

        w = [];
    end


    function [likelih, w] = likelih_probit_unpak(w, likelih)
    %LIKELIH_PROBIT_UNPAK  Extract likelihood parameters from the vector.
    %
    %	Description
    %   W = LIKELIH_PROBIT_UNPAK(W, LIKELIH) Doesn't do anything.
    % 
    %   If Probit likelihood had hyperparameters this would extracts
    %   them parameters from the vector W to the LIKELIH structure.
    %	  
    %
    %	See also
    %	LIKELIH_PROBIT_PAK, GP_UNPAK

        
      likelih=likelih;
      w=[];
    end

    function logLikelih = likelih_probit_e(likelih, y, f, z)
    %LIKELIH_PROBIT_E    Log likelihood
    %
    %   Description
    %   E = LIKELIH_PROBIT_E(LIKELIH, Y, F) takes a likelihood
    %   data structure LIKELIH, class labels Y, and latent values
    %   F. Returns the log likelihood, log p(y|f,z).
    %
    %   See also
    %   LIKELIH_PROBIT_G, LIKELIH_PROBIT_G3, LIKELIH_PROBIT_G2, GPLA_E

        if ~isempty(find(y~=1 & y~=-1))
            error('likelih_probit: The class labels have to be {-1,1}')
        end

        logLikelih = sum(log(normcdf(y.*f)));
    end


    function deriv = likelih_probit_g(likelih, y, f, param, z)
    %LIKELIH_PROBIT_G    Gradient of log likelihood (energy)
    %
    %   Description
    %   G = LIKELIH_PROBIT_G(LIKELIH, Y, F, PARAM) takes a likelihood
    %   data structure LIKELIH, class labels Y, and latent values
    %   F. Returns the gradient of log likelihood with respect to
    %   PARAM. At the moment PARAM can be 'hyper' or 'latent'.
    %
    %   See also
    %   LIKELIH_PROBIT_E, LIKELIH_PROBIT_G2, LIKELIH_PROBIT_G3, GPLA_E

        if ~isempty(find(y~=1 & y~=-1))
            error('likelih_probit: The class labels have to be {-1,1}')
        end
        
        switch param
          case 'latent'
            deriv = y.*normpdf(f)./normcdf(y.*f);
        end
    end


    function g2 = likelih_probit_g2(likelih, y, f, param, z)
    %LIKELIH_PROBIT_G2  Second gradients of log likelihood (energy)
    %
    %   Description        
    %   G2 = LIKELIH_PROBIT_G2(LIKELIH, Y, F, PARAM) takes a likelihood
    %   data structure LIKELIH, class labels Y, and latent values
    %   F. Returns the hessian of log likelihood with respect to
    %   PARAM. At the moment PARAM can be only 'latent'. G2 is a
    %   vector with diagonal elements of the hessian matrix (off
    %   diagonals are zero).
    %
    %   See also
    %   LIKELIH_PROBIT_E, LIKELIH_PROBIT_G, LIKELIH_PROBIT_G3, GPLA_E

        
        if ~isempty(find(y~=1 & y~=-1))
            error('likelih_probit: The class labels have to be {-1,1}')
        end
        
        switch param
          case 'latent'
            z = y.*f;
            g2 = -(normpdf(f)./normcdf(z)).^2 - z.*normpdf(f)./normcdf(z);
        end
    end
    
    function thir_grad = likelih_probit_g3(likelih, y, f, param, z)
    %LIKELIH_PROBIT_G3  Third gradients of log likelihood (energy)
    %
    %   Description
    %   G3 = LIKELIH_PROBIT_G3(LIKELIH, Y, F, PARAM) takes a likelihood 
    %   data structure LIKELIH, class labels Y, and latent values F
    %   and returns the third gradients of log likelihood with respect
    %   to PARAM. At the moment PARAM can be only 'latent'. G3 is a
    %   vector with third gradients.
    %
    %   See also
    %   LIKELIH_PROBIT_E, LIKELIH_PROBIT_G, LIKELIH_PROBIT_G2, GPLA_E, GPLA_G

        if ~isempty(find(y~=1 & y~=-1))
            error('likelih_probit: The class labels have to be {-1,1}')
        end
        
        switch param
          case 'latent'
            z2 = normpdf(f)./normcdf(y.*f);
            thir_grad = 2.*y.*z2.^3 + 3.*f.*z2.^2 - z2.*(y-y.*f.^2);
        end
    end
    

    function [m_0, m_1, m_2] = likelih_probit_tiltedMoments(likelih, y, i1, sigm2_i, myy_i, z)
    %LIKELIH_PROBIT_TILTEDMOMENTS    Returns the marginal moments for EP algorithm
    %
    %   Description
    %   [M_0, M_1, M2] = LIKELIH_PROBIT_TILTEDMOMENTS(LIKELIH, Y, I, S2, MYY) 
    %   takes a likelihood data structure LIKELIH, class labels Y,
    %   index I and cavity variance S2 and mean MYY. Returns the
    %   zeroth moment M_0, mean M_1 and variance M_2 of the posterior
    %   marginal (see Rasmussen and Williams (2006): Gaussian
    %   processes for Machine Learning, page 55).
    %
    %   See also
    %   GPEP_E
        
        if ~isempty(find(y~=1 & y~=-1))
            error('likelih_probit: The class labels have to be {-1,1}')
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

    function [Ey, Vary, py] = likelih_probit_predy(likelih, Ef, Varf, y, z)
    %LIKELIH_PROBIT_PREDY    Returns the predictive mean, variance and density of y
    %
    %   Description         
    %   [EY, VARY] = LIKELIH_PROBIT_PREDY(LIKELIH, EF, VARF)
    %   takes a likelihood data structure LIKELIH, posterior mean EF
    %   and posterior Variance VARF of the latent variable and returns
    %   the posterior predictive mean EY and variance VARY of the
    %   observations related to the latent variables
    %        
    %   [Ey, Vary, PY] = LIKELIH_PROBIT_PREDY(LIKELIH, EF, VARF YT)
    %   Returns also the predictive density of YT, that is 
    %        p(yt | y) = \int p(yt | f) p(f|y) df.
    %   This requires also the class labels YT.
    %
    %   See also 
    %   ep_pred, la_pred, mc_pred

        
        if ~isempty(find(y~=1 & y~=-1))
            error('likelih_probit: The class labels have to be {-1,1}')
        end

        py1 = normcdf(Ef./sqrt(1+Varf));
        Ey = 2*py1 - 1;

        Vary = 1-Ey.^2;
        
        if nargout > 2
            py = normcdf(Ef.*y./sqrt(1+Varf));    % Probability p(y_new)
        end
    end
        
    

    function reclikelih = likelih_probit_recappend(reclikelih, ri, likelih)
    % RECAPPEND  Append the parameters to the record
    %
    %          Description 
    %          RECLIKELIH = GPCF_PROBIT_RECAPPEND(RECLIKELIH, RI, LIKELIH)
    %          takes a likelihood record structure RECLIKELIH, record
    %          index RI and likelihood structure LIKELIH with the
    %          current MCMC samples of the hyperparameters. Returns
    %          RECLIKELIH which contains all the old samples and the
    %          current samples from LIKELIH.
    % 
    %  See also:
    %  gp_mc

        if nargin == 2
            reclikelih.type = 'probit';

            % Set the function handles
            reclikelih.fh_pak = @likelih_probit_pak;
            reclikelih.fh_unpak = @likelih_probit_unpak;
            reclikelih.fh_permute = @likelih_probit_permute;
            reclikelih.fh_e = @likelih_probit_e;
            reclikelih.fh_g = @likelih_probit_g;    
            reclikelih.fh_g2 = @likelih_probit_g2;
            reclikelih.fh_g3 = @likelih_probit_g3;
            reclikelih.fh_tiltedMoments = @likelih_probit_tiltedMoments;
            reclikelih.fh_mcmc = @likelih_probit_mcmc;
            reclikelih.fh_predy = @likelih_probit_predy;
            reclikelih.fh_recappend = @likelih_probit_recappend;
            return
        end

    end

end


