function likelih = likelih_logit(do, varargin)
%LIKELIH_LOGIT    Create a Logit likelihood structure 
%
%	Description
%
%	LIKELIH = LIKELIH_LOGIT('INIT') Create and initialize Logit 
%       likelihood for classification problem with class labels {-1,1}. 
%       
%       likelihood is defined as follows:
%                            __ n
%                p(y|f, z) = || i=1 1/(1 + exp(-y_i*f_i) )
%
%	The fields in LIKELIH are:
%	  type                     = 'likelih_logit'
%         likelih.fh_pak           = function handle to pak
%         likelih.fh_unpak         = function handle to unpak
%         likelih.fh_ll             = function handle to the log likelihood
%         likelih.fh_llg             = function handle to the gradient of 
%                                    the log likelihood
%         likelih.fh_llg2            = function handle to the second gradient
%                                    of the log likelihood
%         likelih.fh_llg3            = function handle to the third gradient  
%                                    of the log likelihood
%         likelih.fh_tiltedMoments = function handle to evaluate posterior
%                                    moments for EP
%         likelih.fh_predy         = function handle to evaluate predictive 
%                                    density of y
%         likelih.fh_recappend     = function handle to append the record    
%
%	See also
%       LIKELIH_LOGIT, LIKELIH_PROBIT, LIKELIH_NEGBIN

%       Copyright (c) 2008-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 1
        error('Not enough arguments')
    end

    % Initialize the likelihood structure
    if strcmp(do, 'init')
        likelih.type = 'logit';
        
        % Set the function handles to the nested functions
        likelih.fh_pak = @likelih_logit_pak;
        likelih.fh_unpak = @likelih_logit_unpak;
        likelih.fh_ll = @likelih_logit_ll;
        likelih.fh_llg = @likelih_logit_llg;    
        likelih.fh_llg2 = @likelih_logit_llg2;
        likelih.fh_llg3 = @likelih_logit_llg3;
        likelih.fh_tiltedMoments = @likelih_logit_tiltedMoments;
        likelih.fh_mcmc = @likelih_logit_mcmc;
        likelih.fh_predy = @likelih_logit_predy;
        likelih.fh_recappend = @likelih_logit_recappend;

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

    % Set the parameter values of covariance function
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



    function w = likelih_logit_pak(likelih)
    %LIKELIH_LOGIT_PAK    Combine likelihood parameters into one vector.
    %
    %	Description 
    %   W = LIKELIH_LOGIT_PAK(LIKELIH) takes a likelihood data
    %   structure LIKELIH and returns an empty verctor W. If Logit
    %   likelihood had hyperparameters this would combine them into a
    %   single row vector W (see e.g. likelih_negbin).
    %	  
    %
    %	See also
    %	LIKELIH_NEGBIN_UNPAK, GP_PAK
        
        w = [];
    end


    function [likelih, w] = likelih_logit_unpak(w, likelih)
    %LIKELIH_LOGIT_UNPAK  Extract likelihood parameters from the vector.
    %
    %	Description
    %   W = LIKELIH_LOGIT_UNPAK(W, LIKELIH) Doesn't do anything.
    % 
    %   If Logit likelihood had hyperparameters this would extracts
    %   them parameters from the vector W to the LIKELIH structure.
    %	  
    %
    %	See also
    %	LIKELIH_LOGIT_PAK, GP_UNPAK

        likelih=likelih;
        %w=[];
    end



    function logLikelih = likelih_logit_ll(likelih, y, f, z)
    %LIKELIH_LOGIT_LL    Log likelihood
    %
    %   Description
    %   E = LIKELIH_LOGIT_LL(LIKELIH, Y, F) takes a likelihood
    %   data structure LIKELIH, class labels Y, and latent values
    %   F. Returns the log likelihood, log p(y|f,z).
    %
    %   See also
    %   LIKELIH_LOGIT_LLG, LIKELIH_LOGIT_LLG3, LIKELIH_LOGIT_LLG2, GPLA_E

        if ~isempty(find(abs(y)~=1))
            error('likelih_logit: The class labels have to be {-1,1}')
        end
        
        logLikelih = sum(-log(1+exp(-y.*f)));
    end


    function deriv = likelih_logit_llg(likelih, y, f, param, z)
    %LIKELIH_LOGIT_LLG    Gradient of log likelihood (energy)
    %
    %   Description
    %   G = LIKELIH_LOGIT_LLG(LIKELIH, Y, F, PARAM) takes a likelihood
    %   data structure LIKELIH, class labels Y, and latent values
    %   F. Returns the gradient of log likelihood with respect to
    %   PARAM. At the moment PARAM can be 'hyper' or 'latent'.
    %
    %   See also
    %   LIKELIH_LOGIT_LL, LIKELIH_LOGIT_LLG2, LIKELIH_LOGIT_LLG3, GPLA_E
        
        if ~isempty(find(abs(y)~=1))
            error('likelih_logit: The class labels have to be {-1,1}')
        end

        
        t  = (y+1)/2;
        PI = 1./(1+exp(-f));
        deriv = t - PI;
        %deriv = (y+1)/2 - 1./(1+exp(-f));      
    end


    function g2 = likelih_logit_llg2(likelih, y, f, param, z)
    %LIKELIH_LOGIT_LLG2  Second gradients of log likelihood (energy)
    %
    %   Description        
    %   G2 = LIKELIH_LOGIT_LLG2(LIKELIH, Y, F, PARAM) takes a likelihood
    %   data structure LIKELIH, class labels Y, and latent values
    %   F. Returns the hessian of log likelihood with respect to
    %   PARAM. At the moment PARAM can be only 'latent'. G2 is a
    %   vector with diagonal elements of the hessian matrix (off
    %   diagonals are zero).
    %
    %   See also
    %   LIKELIH_LOGIT_LL, LIKELIH_LOGIT_LLG, LIKELIH_LOGIT_LLG3, GPLA_E

        PI = 1./(1+exp(-f));
        g2 = -PI.*(1-PI);        
    end    
    
    function third_grad = likelih_logit_llg3(likelih, y, f, param, z)
    %LIKELIH_LOGIT_LLG3  Third gradients of log likelihood (energy)
    %
    %   Description
    %   G3 = LIKELIH_LOGIT_LLG3(LIKELIH, Y, F, PARAM) takes a likelihood 
    %   data structure LIKELIH, class labels Y, and latent values F
    %   and returns the third gradients of log likelihood with respect
    %   to PARAM. At the moment PARAM can be only 'latent'. G3 is a
    %   vector with third gradients.
    %
    %   See also
    %   LIKELIH_LOGIT_LL, LIKELIH_LOGIT_LLG, LIKELIH_LOGIT_LLG2, GPLA_E, GPLA_G
        
        if ~isempty(find(abs(y)~=1))
            error('likelih_logit: The class labels have to be {-1,1}')
        end

        t  = (y+1)/2;
        PI = 1./(1+exp(-f));
        third_grad = -PI.*(1-PI).*(1-2*PI);        
    end


    function [m_0, m_1, sigm2hati1] = likelih_logit_tiltedMoments(likelih, y, i1, sigm2_i, myy_i, z)
    %LIKELIH_LOGIT_TILTEDMOMENTS    Returns the marginal moments for EP algorithm
    %
    %   Description
    %   [M_0, M_1, M2] = LIKELIH_LOGIT_TILTEDMOMENTS(LIKELIH, Y, I, S2, MYY) 
    %   takes a likelihood data structure LIKELIH, class labels Y,
    %   index I and cavity variance S2 and mean MYY. Returns the
    %   zeroth moment M_0, mean M_1 and variance M_2 of the posterior
    %   marginal (see Rasmussen and Williams (2006): Gaussian
    %   processes for Machine Learning, page 55).
    %
    %   See also
    %   GPEP_E
        
        if ~isempty(find(abs(y)~=1))
            error('likelih_logit: The class labels have to be {-1,1}')
        end
        
        yy = y(i1);
        % get a function handle of an unnormalized tilted distribution 
        % (likelih * cavity = Logit * Gaussian)
        % and useful integration limits
        [tf,minf,maxf]=init_logit_norm(yy,myy_i,sigm2_i);
        RTOL = 1.e-6;
        ATOL = 1.e-10;
        
        % Integrate with an adaptive Gauss-Kronrod quadrature
        % (Rasmussen and Nickish use in GPML interpolation between
        % a cumulative Gaussian scale mixture and linear tail
        % approximation, which could be faster, but quadrature also
        % takes only a fraction of the time EP uses overall, so no
        % need to change...)
        [m_0, m_1, m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);        
        sigm2hati1 = m_2 - m_1.^2;
        
        % If the second central moment is less than cavity variance
        % integrate more precisely. Theoretically should be
        % sigm2hati1 < sigm2_i.
        if sigm2hati1 >= sigm2_i
            ATOL = ATOL.^2;
            RTOL = RTOL.^2;
            [m_0, m_1, m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
            sigm2hati1 = m_2 - m_1.^2;
            if sigm2hati1 >= sigm2_i
                error('likelih_logit_tilted_moments: sigm2hati1 >= sigm2_i');
            end
        end
    end
    
    function [Ey, Vary, Py] = likelih_logit_predy(likelih, Ef, Varf, yt, zt)
    %LIKELIH_LOGIT_PREDY    Returns the predictive mean, variance and density of y
    %
    %   Description         
    %   [EY, VARY] = LIKELIH_LOGIT_PREDY(LIKELIH, EF, VARF)
    %   takes a likelihood data structure LIKELIH, posterior mean EF
    %   and posterior Variance VARF of the latent variable and returns
    %   the posterior predictive mean EY and variance VARY of the
    %   observations related to the latent variables
    %        
    %   [EY, VARY, PY] = LIKELIH_LOGIT_PREDY(LIKELIH, EF, VARF, YT)
    %   Returns also the predictive density of YT, that is 
    %        p(yt | y) = \int p(yt | f) p(f|y) df.
    %   This requires also the class labels YT.
    %
    %   See also 
    %   LA_PRED, EP_PRED, MC_PRED
        
        if ~isempty(find(abs(yt~=1)))
            error('likelih_logit: The class labels have to be {-1,1}')
        end

        py1 = zeros(length(Ef),1);
        for i1=1:length(Ef)
          myy_i = Ef(i1);
          sigm_i = sqrt(Varf(i1));
          minf=myy_i-6*sigm_i;
          maxf=myy_i+6*sigm_i;
          F  = @(f)1./(1+exp(-f)).*norm_pdf(f,myy_i,sigm_i);
          py1(i1) = quadgk(F,minf,maxf);
        end
        Ey = 2*py1-1;
        Vary = 1-(2*py1-1).^2;
        
        if nargout > 2
          % Quadrature integration                                    
          Py = zeros(length(Ef),1);
          for i1 = 1:length(Ef)
            % get a function handle of the likelihood times posterior
            % (likelihood * posterior = Poisson * Gaussian)
            % and useful integration limits
            [pdf,minf,maxf]=init_logit_norm(...
              yt(i1),Ef(i1),Varf(i1));
            % integrate over the f to get posterior predictive distribution
            Py(i1) = quadgk(pdf, minf, maxf);
            end
        end
    end

    function [df,minf,maxf] = init_logit_norm(yy,myy_i,sigm2_i)
    %INIT_LOGIT_NORM
    %
    %   Description
    %    Return function handle to a function evaluating
    %    Logit * Gaussian which is used for evaluating  
    %    (likelihood * cavity) or (likelihood * posterior) 
    %    Return also useful limits for integration.
    %    This is private function for likelih_logit.
    %  
    %   See also
    %   LIKELIH_LOGIT_TILTEDMOMENTS, LIKELIH_LOGIT_PREDY
    
      % avoid repetitive evaluation of constant part
      ldconst = -log(sigm2_i)/2 -log(2*pi)/2;
      
      % Create function handle for the function to be integrated
      df = @logit_norm;
      % use log to avoid underflow, and derivates for faster search
      ld = @log_logit_norm;
      ldg = @log_logit_norm_g;
      ldg2 = @log_logit_norm_g2;

      % Set the limits for integration
      % Logit likelihood is log-concave so the logit_norm
      % function is unimodal, which makes things easier
      
      % approximate guess for the location of the mode
      if sign(myy_i)==sign(yy)
        % the log-likelihood is flat on this side
        modef = myy_i;
      else
        % log-likelihood is approximately yy*f on this side
        modef=sign(myy_i)*max(abs(myy_i)-sigm2_i,0);
      end
      % find the mode of the integrand using Newton iterations
      % few iterations is enough, since the first guess in the right direction
      niter=2;       % number of Newton iterations
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
          error(['likelih_logit -> init_logit_norm: ' ...
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
          error(['likelih_logit -> init_logit_norm: ' ...
                 'integration interval maximum not found ' ...
                 'even after looking hard!'])
        end
        
      end
    
      function integrand = logit_norm(f)
      % Logit * Gaussian
        integrand = exp(ldconst ...
                        -log(1+exp(-yy.*f)) ...
                        -0.5*(f-myy_i).^2./sigm2_i);
      end
      
      function log_int = log_logit_norm(f)
      % log(Logit * Gaussian)
      % log_logit_norm is used to avoid underflow when searching
      % integration interval
        log_int = ldconst ...
                  -log(1+exp(-yy.*f)) ...
                  -0.5*(f-myy_i).^2./sigm2_i;
      end
      
      function g = log_logit_norm_g(f)
      % d/df log(Logit * Gaussian)
      % derivative of log_logit_norm
        g = yy./(exp(f*yy)+1)...
            + (myy_i - f)./sigm2_i;
      end
      
      function g2 = log_logit_norm_g2(f)
      % d^2/df^2 log(Logit * Gaussian)
      % second derivate of log_logit_norm
        a=exp(f*yy);
        g2 = -a*(yy./(a+1)).^2 ...
             -1/sigm2_i;
      end
      
    end

    function reclikelih = likelih_logit_recappend(reclikelih, ri, likelih)
    % RECAPPEND  Append the parameters to the record
    %
    %          Description 
    %          RECLIKELIH = GPCF_LOGIT_RECAPPEND(RECLIKELIH, RI, LIKELIH)
    %          takes a likelihood record structure RECLIKELIH, record
    %          index RI and likelihood structure LIKELIH with the
    %          current MCMC samples of the hyperparameters. Returns
    %          RECLIKELIH which contains all the old samples and the
    %          current samples from LIKELIH.
    % 
    %  See also:
    %  gp_mc

        if nargin == 2
            reclikelih.type = 'logit';

            % Set the function handles
            reclikelih.fh_pak = @likelih_logit_pak;
            reclikelih.fh_unpak = @likelih_logit_unpak;
            reclikelih.fh_ll = @likelih_logit_ll;
            reclikelih.fh_llg = @likelih_logit_llg;    
            reclikelih.fh_llg2 = @likelih_logit_llg2;
            reclikelih.fh_llg3 = @likelih_logit_llg3;
            reclikelih.fh_tiltedMoments = @likelih_logit_tiltedMoments;
            reclikelih.fh_mcmc = @likelih_logit_mcmc;
            reclikelih.fh_predy = @likelih_logit_predy;
            reclikelih.fh_recappend = @likelih_logit_recappend;
            return
        end
        
    end
end
