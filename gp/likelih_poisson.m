function likelih = likelih_poisson(do, varargin)
%LIKELIH_POISSON      Create a Poisson likelihood structure 
%
%	Description
%
%	LIKELIH = LIKELIH_POISSON('INIT') Create and initialize 
%                 Poisson likelihood. 
%
%       The likelihood is defined as follows:
%                            __ n
%                p(y|f, z) = || i=1 Poisson(y_i|z_i*exp(f_i))
%
%       where z is a vector of expected mean and f the latent value
%       vector whose components are transformed to relative risk
%       exp(f_i). When using the Poisson likelihood you need to give the
%       vector z as an extra parameter to each function that requires
%       y also. For example, you should call gpla_e as follows
%          gpla_e(w, gp, x, y, 'z', z)
%
%	The fields in LIKELIH are:
%	  likelih.type             = 'likelih_poisson'
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
%	LIKELIH = LIKELIH_POISSON('SET', LIKELIH, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in
%       LIKELIH.
%
%	See also
%       LIKELIH_LOGIT, LIKELIH_PROBIT, LIKELIH_NEGBIN

% Copyright (c) 2006-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

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
    %LIKELIH_POISSON_PAK    Combine likelihood parameters into one vector.
    %
    %	Description 
    %   W = LIKELIH_POISSON_PAK(LIKELIH) takes a likelihood data
    %   structure LIKELIH and returns an empty verctor W. If Poisson
    %   likelihood had hyperparameters this would combine them into a
    %   single row vector W (see e.g. likelih_negbin).
    %	  
    %
    %	See also
    %	LIKELIH_NEGBIN_UNPAK, GP_PAK

        w = [];
    end


    function [likelih, w] = likelih_poisson_unpak(w, likelih)
    %LIKELIH_POISSON_UNPAK  Extract likelihood parameters from the vector.
    %
    %	Description
    %   W = LIKELIH_POISSON_UNPAK(W, LIKELIH) Doesn't do anything.
    % 
    %   If Poisson likelihood had hyperparameters this would extracts
    %   them parameters from the vector W to the LIKELIH structure.
    %	  
    %
    %	See also
    %	LIKELIH_POISSON_PAK, GP_UNPAK

        likelih=likelih;
        %w=[];
    end


    function logLikelih = likelih_poisson_e(likelih, y, f, z)
    %LIKELIH_POISSON_E    Log likelihood
    %
    %   Description
    %   E = LIKELIH_POISSON_E(LIKELIH, Y, F, Z) takes a likelihood
    %   data structure LIKELIH, incedence counts Y, expected counts Z,
    %   and latent values F. Returns the log likelihood, log p(y|f,z).
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
    %LIKELIH_POISSON_G    Gradient of log likelihood (energy)
    %
    %   Description 
    %   G = LIKELIH_POISSON_G(LIKELIH, Y, F, PARAM) takes a likelihood
    %   data structure LIKELIH, incedence counts Y, expected counts Z
    %   and latent values F. Returns the gradient of log likelihood 
    %   with respect to PARAM. At the moment PARAM can be 'hyper' or
    %   'latent'.
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
    %LIKELIH_POISSON_G2  Second gradients of log likelihood (energy)
    %
    %   Description        
    %   G2 = LIKELIH_POISSON_G2(LIKELIH, Y, F, PARAM) takes a likelihood
    %   data structure LIKELIH, incedence counts Y, expected counts Z,
    %   and latent values F. Returns the hessian of log likelihood
    %   with respect to PARAM. At the moment PARAM can be only
    %   'latent'. G2 is a vector with diagonal elements of the hessian
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
    %LIKELIH_POISSON_G3  Third gradients of log likelihood (energy)
    %
    %   Description
        
    %   G3 = LIKELIH_POISSON_G3(LIKELIH, Y, F, PARAM) takes a likelihood 
    %   data structure LIKELIH, incedence counts Y, expected counts Z
    %   and latent values F and returns the third gradients of log
    %   likelihood with respect to PARAM. At the moment PARAM can be
    %   only 'latent'. G3 is a vector with third gradients.
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
    %LIKELIH_POISSON_TILTEDMOMENTS    Returns the marginal moments for EP algorithm
    %
    %   Description
    %   [M_0, M_1, M2] = LIKELIH_POISSON_TILTEDMOMENTS(LIKELIH, Y, I, S2, MYY, Z) 
    %   takes a likelihood data structure LIKELIH, incedence counts Y, 
    %   expected counts Z, index I and cavity variance S2 and mean
    %   MYY. Returns the zeroth moment M_0, mean M_1 and variance M_2
    %   of the posterior marginal (see Rasmussen and Williams (2006):
    %   Gaussian processes for Machine Learning, page 55).
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
        avgE = z(i1);
        
        % get a function handle of an unnormalized tilted distribution 
        % (likelih * cavity = Negative-binomial * Gaussian)
        % and useful integration limits
        [tf,minf,maxf]=init_poisson_norm(yy,myy_i,sigm2_i,avgE);
        
        % Integrate with quadrature
        RTOL = 1.e-6;
        ATOL = 1.e-10;
        [m_0, m_1, m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
        sigm2hati1 = m_2 - m_1.^2;
                
        % If the second central moment is less than cavity variance
        % integrate more precisely. Theoretically for log-concave
        % likelihood should be sigm2hati1 < sigm2_i.
        if sigm2hati1 >= sigm2_i
            ATOL = ATOL.^2;
            RTOL = RTOL.^2;
            [m_0, m_1, m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
            sigm2hati1 = m_2 - m_1.^2;
            if sigm2hati1 >= sigm2_i
                error('likelih_poisson_tilted_moments: sigm2hati1 >= sigm2_i');
            end
        end
    end

    
    function [Ey, Vary, Py] = likelih_poisson_predy(likelih, Ef, Varf, yt, zt)
    %LIKELIH_POISSON_PREDY    Returns the predictive mean, variance and density of y
    %
    %   Description         
    %   [EY, VARY] = LIKELIH_POISSON_PREDY(LIKELIH, EF, VARF)
    %   takes a likelihood data structure LIKELIH, posterior mean EF
    %   and posterior Variance VARF of the latent variable and returns
    %   the posterior predictive mean EY and variance VARY of the
    %   observations related to the latent variables
    %        
    %   [Ey, Vary, PY] = LIKELIH_POISSON_PREDY(LIKELIH, EF, VARF YT, ZT)
    %   Returns also the predictive density of YT, that is 
    %        p(yt | y,zt) = \int p(yt | f, zt) p(f|y) df.
    %   This requires also the incedence counts YT, expected counts ZT.
    %
    %   See also 
    %   LA_PRED, EP_PRED, MC_PRED

        if isempty(zt)
            error(['likelih_poisson -> likelih_poisson_predy: missing zt!'... 
                   'Poisson likelihood needs the expected number of     '...
                   'occurrences as an extra input zt. See, for           '...
                   'example, likelih_poisson and gpla_e.                ']);
        end
        
        avgE = zt;
        Py = zeros(size(Ef));
        Ey = zeros(size(Ef));
        EVary = zeros(size(Ef));
        VarEy = zeros(size(Ef)); 
        
        % Evaluate Ey and Vary
        for i1=1:length(Ef)
           %%% With quadrature
           myy_i = Ef(i1);
           sigm_i = sqrt(Varf(i1));
           minf=myy_i-6*sigm_i;
           maxf=myy_i+6*sigm_i;

           F = @(f) exp(log(avgE(i1))+f+norm_lpdf(f,myy_i,sigm_i));
           Ey(i1) = quadgk(F,minf,maxf);
           
           EVary(i1) = Ey(i1);
           
           F3 = @(f) exp(2*log(avgE(i1))+2*f+norm_lpdf(f,myy_i,sigm_i));
           VarEy(i1) = quadgk(F3,minf,maxf) - Ey(i1).^2;
       end
       Vary = EVary + VarEy;

       % Evaluate the posterior predictive densities of the given observations
       if nargout > 2
           for i1=1:length(Ef)
               % get a function handle of the likelihood times posterior
               % (likelihood * posterior = Poisson * Gaussian)
               % and useful integration limits
               [pdf,minf,maxf]=init_poisson_norm(...
                 yt(i1),Ef(i1),Varf(i1),avgE(i1));
               % integrate over the f to get posterior predictive distribution
               Py(i1) = quadgk(pdf, minf, maxf);
           end
       end
    end
    
    function [df,minf,maxf] = init_poisson_norm(yy,myy_i,sigm2_i,avgE)
    %INIT_POISSON_NORM
    %
    %   Description
    %    Return function handle to a function evaluating
    %    Poisson * Gaussian which is used for evaluating  
    %    (likelihood * cavity) or (likelihood * posterior) 
    %    Return also useful limits for integration.
    %    This is private function for likelih_poisson.
    %  
    %   See also
    %   LIKELIH_POISSON_TILTEDMOMENTS, LIKELIH_POISSON_PREDY
    
      % avoid repetitive evaluation of constant part
      ldconst = -gammaln(yy+1) - log(sigm2_i)/2 - log(2*pi)/2;
      
      % Create function handle for the function to be integrated
      df = @poisson_norm;
      % use log to avoid underflow, and derivates for faster search
      ld = @log_poisson_norm;
      ldg = @log_poisson_norm_g;
      ldg2 = @log_poisson_norm_g2;

      % Set the limits for integration
      % Poisson likelihood is log-concave so the poisson_norm
      % function is unimodal, which makes things easier
      if yy==0
        % with yy==0, the mode of the likelihood is not defined
        % use the mode of the Gaussian (cavity or posterior) as a first guess
        modef = myy_i;
      else
        % use precision weighted mean of the Gaussian approximation
        % of the Poisson likelihood and Gaussian
        mu=log(yy/avgE);
        s2=1./(yy+1./sigm2_i);
        modef = (myy_i/sigm2_i + mu/s2)/(1/sigm2_i + 1/s2);
      end
      % find the mode of the integrand using Newton iterations
      % few iterations is enough, since the first guess in the right direction
      niter=3;       % number of Newton iterations
      mindelta=1e-6; % tolerance in stopping Newton iterations
      for ni=1:niter
        g=ldg(modef);
        h=ldg2(modef);
        delta=-g/h;
        modef=modef+delta;
        if abs(delta)<mindelta
          break
        end
      end
      % integrand limits based on Gaussian approximation at mode
      modes=sqrt(-1/h);
      minf=modef-8*modes;
      maxf=modef+8*modes;
      modeld=ld(modef);
      iter=0;
      % check that density at end points is low enough
      lddiff=25; % min difference in log-density between mode and end-points
      minld=ld(minf);
      while minld>(modeld-lddiff)
        minf=minf-modes;
        minld=ld(minf);
        iter=iter+1;
        if iter>100
          error(['likelih_poisson -> init_poisson_norm: ' ...
                 'integration interval minimun not found ' ...
                 'even after looking hard!'])
        end
      end
      maxld=ld(maxf);
      while maxld>(modeld-lddiff)
        maxf=maxf+modes;
        maxld=ld(maxf);
        iter=iter+1;
        if iter>100
          error(['likelih_poisson -> init_poisson_norm: ' ...
                 'integration interval maximum not found ' ...
                 'even after looking hard!'])
        end
        
      end
    
      function integrand = poisson_norm(f)
      % Poisson * Gaussian
        mu = avgE.*exp(f);
        integrand = exp(ldconst ...
                        -mu+yy.*log(mu) ...
                        -0.5*(f-myy_i).^2./sigm2_i);
      end
      
      function log_int = log_poisson_norm(f)
      % log(Poisson * Gaussian)
      % log_poisson_norm is used to avoid underflow when searching
      % integration interval
        mu = avgE.*exp(f);
        log_int = ldconst ...
                  -mu+yy.*log(mu) ...
                  -0.5*(f-myy_i).^2./sigm2_i;
      end
      
      function g = log_poisson_norm_g(f)
      % d/df log(Poisson * Gaussian)
      % derivative of log_poisson_norm
        mu = avgE.*exp(f);
        g = -mu+yy...
            + (myy_i - f)./sigm2_i;
      end
      
      function g2 = log_poisson_norm_g2(f)
      % d^2/df^2 log(Poisson * Gaussian)
      % second derivate of log_poisson_norm
        mu = avgE.*exp(f);
        g2 = -mu...
             -1/sigm2_i;
      end
      
    end
    
    function reclikelih = likelih_poisson_recappend(reclikelih, ri, likelih)
    % RECAPPEND  Append the parameters to the record
    %
    %          Description 
    %          RECLIKELIH = LIKELIH_POISSON_RECAPPEND(RECLIKELIH, RI, LIKELIH)
    %          takes a likelihood record structure RECLIKELIH, record
    %          index RI and likelihood structure LIKELIH with the
    %          current MCMC samples of the hyperparameters. Returns
    %          RECLIKELIH which contains all the old samples and the
    %          current samples from LIKELIH.
    % 
    %  See also:
    %  gp_mc

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


