function gpcf = gpcf_matern52(do, varargin)
%GPCF_MATERN52	Create a Matern nu=5/2 covariance function for Gaussian Process
%
%	Description
%
%	GPCF = GPCF_MATERN52('INIT') Create and initialize Matern nu=3/2
%       covariance function for Gaussian process
%
%	The fields and (default values) in GPCF_MATERN32 are:
%	  type           = 'gpcf_matern52'
%	  magnSigma2     = Magnitude (squared) for exponential part. 
%                          (0.1)
%	  lengthScale    = Length scale for each input. This can be either scalar corresponding 
%                          isotropic or vector corresponding ARD. (10)
%         p              = Prior structure for covariance function parameters. 
%                          (e.g. p.lengthScale.)
%         fh_pak         = function handle to pack function
%                          (@gpcf_matern52_pak)
%         fh_unpak       = function handle to unpack function
%                          (@gpcf_matern52_unpak)
%         fh_e           = function handle to energy function
%                          (@gpcf_matern52_e)
%         fh_ghyper      = function handle to gradient of energy with respect to hyperparameters
%                          (@gpcf_matern52_ghyper)
%         fh_ginput      = function handle to gradient of function with respect to inducing inputs
%                          (@gpcf_matern52_ginput)
%         fh_cov         = function handle to covariance function
%                          (@gpcf_matern52_cov)
%         fh_trcov       = function handle to training covariance function
%                          (@gpcf_matern52_trcov)
%         fh_trvar       = function handle to training variance function
%                          (@gpcf_matern52_trvar)
%         fh_recappend   = function handle to append the record function 
%                          (gpcf_matern52_recappend)
%
%	GPCF = GPCF_MATERN52('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in GPCF. The fields that 
%       can be modified are:
%
%             'magnSigma2'         : set the magnSigma2
%             'lengthScale'        : set the lengthScale
%             'metric'             : set the metric structure into the covariance function
%             'lengthScale_prior'  : set the prior structure for lengthScale
%             'magnSigma2_prior'   ; set the prior structure for magnSigma2
%
%	See also
%       gpcf_sexp, gpcf_exp, gpcf_matern32, gpcf_ppcs2, gp_init, gp_e, gp_g, gp_trcov
%       gp_cov, gp_unpak, gp_pak
    
% Copyright (c) 2007-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 1
        error('Not enough arguments')
    end

    % Initialize the covariance function
    if strcmp(do, 'init')
        gpcf.type = 'gpcf_matern52';
        
        % Initialize parameters
        gpcf.lengthScale= 10; 
        gpcf.magnSigma2 = 0.1;
        
        % Initialize prior structure
        gpcf.p=[];
        gpcf.p.lengthScale=prior_unif('init');
        gpcf.p.magnSigma2=prior_unif('init');
        
        % Set the function handles to the nested functions
        gpcf.fh_pak = @gpcf_matern52_pak;
        gpcf.fh_unpak = @gpcf_matern52_unpak;
        gpcf.fh_e = @gpcf_matern52_e;
        gpcf.fh_ghyper = @gpcf_matern52_ghyper;
        gpcf.fh_ginput = @gpcf_matern52_ginput;
        gpcf.fh_cov = @gpcf_matern52_cov;
        gpcf.fh_trcov  = @gpcf_matern52_trcov;
        gpcf.fh_trvar  = @gpcf_matern52_trvar;
        gpcf.fh_recappend = @gpcf_matern52_recappend;
        
        if nargin > 1
            if mod(nargin,2) ~= 1
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=1:2:length(varargin)-1
                switch varargin{i}
                  case 'magnSigma2'
                    gpcf.magnSigma2 = varargin{i+1};
                  case 'lengthScale'
                    gpcf.lengthScale = varargin{i+1};
                  case 'metric'
                    gpcf.metric = varargin{i+1};
                    if isfield(gpcf, 'lengthScale')
                        gpcf = rmfield(gpcf, 'lengthScale');
                    end
                  case 'lengthScale_prior'
                    gpcf.p.lengthScale = varargin{i+1};
                  case 'magnSigma2_prior'
                    gpcf.p.magnSigma2 = varargin{i+1};
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
              case 'magnSigma2'
                gpcf.magnSigma2 = varargin{i+1};
              case 'lengthScale'
                gpcf.lengthScale = varargin{i+1};
              case 'metric'
                gpcf.metric = varargin{i+1};
                if isfield(gpcf, 'lengthScale')
                    gpcf = rmfield(gpcf, 'lengthScale');
                end
              case 'lengthScale_prior'
                gpcf.p.lengthScale = varargin{i+1};
              case 'magnSigma2_prior'
                gpcf.p.magnSigma2 = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end   
        end
    end

    function w = gpcf_matern52_pak(gpcf, w)
    %GPCF_MATERN52_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %	W = GPCF_MATERN52_PAK(GPCF, W) takes a covariance function data structure GPCF and
    %	combines the hyper-parameters into a single row vector W.
    %
    %	The ordering of the parameters in W is:
    %       w = [gpcf.magnSigma2 (hyperparameters of gpcf.lengthScale) gpcf.lengthScale]
    %	  
    %
    %	See also
    %	GPCF_MATERN52_UNPAK
        
        i1=0;i2=1;
        ww = []; w = [];
        
        if ~isempty(gpcf.p.magnSigma2)
            i1 = i1+1;
            w(i1) = log(gpcf.magnSigma2);
            
            % Hyperparameters of magnSigma2
            ww = feval(gpcf.p.magnSigma2.fh_pak, gpcf.p.magnSigma2);
        end        
        
        if isfield(gpcf,'metric')
            
            w = [w feval(gpcf.metric.pak, gpcf.metric)];
        else
            if ~isempty(gpcf.p.lengthScale)
                w = [w log(gpcf.lengthScale)];
                            
                % Hyperparameters of lengthScale
                w = [w feval(gpcf.p.lengthScale.fh_pak, gpcf.p.lengthScale)];
            end
        end
        w = [w ww];
    end

    
    function [gpcf, w] = gpcf_matern52_unpak(gpcf, w)
    %GPCF_MATERN52_UNPAK  Separate covariance function hyper-parameter vector into components.
    %
    %	Description
    %	[GPCF, W] = GPCF_MATERN52_UNPAK(GPCF, W) takes a covariance function data structure GPCF
    %	and  a hyper-parameter vector W, and returns a covariance function data
    %	structure  identical to the input, except that the covariance hyper-parameters 
    %   has been set to the values in W. Deletes the values set to GPCF from W and returns 
    %   the modeified W. 
    %
    %	See also
    %	GPCF_MATERN52_PAK
        
        gpp=gpcf.p;
        if ~isempty(gpp.magnSigma2)
            i1=1;
            gpcf.magnSigma2 = exp(w(i1));
            w = w(i1+1:end);
        end

        if isfield(gpcf,'metric')
            [metric, w] = feval(gpcf.metric.unpak, gpcf.metric, w);
            gpcf.metric = metric;
        else            
            if ~isempty(gpp.lengthScale)
                i2=length(gpcf.lengthScale);
                i1=1;
                gpcf.lengthScale = exp(w(i1:i2));
                w = w(i2+1:end);
                                
                % Hyperparameters of lengthScale
                [p, w] = feval(gpcf.p.lengthScale.fh_unpak, gpcf.p.lengthScale, w);
                gpcf.p.lengthScale = p;
            end
        end
        
        if ~isempty(gpp.magnSigma2)
            % Hyperparameters of magnSigma2
            [p, w] = feval(gpcf.p.magnSigma2.fh_unpak, gpcf.p.magnSigma2, w);
            gpcf.p.magnSigma2 = p;
        end
    end

    function eprior =gpcf_matern52_e(gpcf, x, t)
    %GPCF_MATERN52_E     Evaluate the energy of prior of MATERN52 parameters
    %
    %	Description
    %	E = GPCF_MATERN52_E(GPCF, X, T) takes a covariance function data structure 
    %   GPCF together with a matrix X of input vectors and a matrix T of target 
    %   vectors and evaluates log p(th) x J, where th is a vector of MATERN52 parameters 
    %   and J is the Jakobian of transformation exp(w) = th. (Note that the parameters 
    %   are log transformed, when packed.)
    %
    %	See also
    %	GPCF_MATERN52_PAK, GPCF_MATERN52_UNPAK, GPCF_MATERN52_G, GP_E

        eprior = 0;
        gpp=gpcf.p;
        
        [n, m] =size(x);

        if isfield(gpcf,'metric')            
            if ~isempty(gpcf.p.magnSigma2)
                eprior=eprior + feval(gpp.magnSigma2.fh_e, gpcf.magnSigma2, gpp.magnSigma2) - log(gpcf.magnSigma2);
            end
            eprior = eprior + feval(gpcf.metric.e, gpcf.metric, x, t);
            
        else
            % Evaluate the prior contribution to the error. The parameters that
            % are sampled are from space W = log(w) where w is all the "real" samples.
            % On the other hand errors are evaluated in the W-space so we need take
            % into account also the  Jacobian of transformation W -> w = exp(W).
            % See Gelman et.all., 2004, Bayesian data Analysis, second edition, p24.

            if ~isempty(gpcf.p.magnSigma2)
                eprior = feval(gpp.magnSigma2.fh_e, gpcf.magnSigma2, gpp.magnSigma2) - log(gpcf.magnSigma2);
            end
            if ~isempty(gpp.lengthScale)
                eprior = eprior + feval(gpp.lengthScale.fh_e, gpcf.lengthScale, gpp.lengthScale) - sum(log(gpcf.lengthScale));
            end
        end
    end
    
    function [DKff, gprior]  = gpcf_matern52_ghyper(gpcf, x, x2, mask)
    %GPCF_MATERN52_GHYPER     Evaluate gradient of covariance function and hyper-prior with 
    %                     respect to the hyperparameters.
    %
    %	Descriptioni
    %	[GPRIOR, DKff, DKuu, DKuf] = GPCF_MATERN52_GHYPER(GPCF, X, T, G, GDATA, GPRIOR, VARARGIN) 
    %   takes a covariance function data structure GPCF, a matrix X of input vectors, a
    %   matrix T of target vectors and vectors GDATA and GPRIOR. Returns:
    %      GPRIOR  = d log(p(th))/dth, where th is the vector of hyperparameters 
    %      DKff    = gradients of covariance matrix Kff with respect to th (cell array with matrix elements)
    %      DKuu    = gradients of covariance matrix Kuu with respect to th (cell array with matrix elements)
    %      DKuf    = gradients of covariance matrix Kuf with respect to th (cell array with matrix elements)
    %
    %   Here f refers to latent values and u to inducing varianble (e.g. Kuf is the covariance 
    %   between u and f). See Vanhatalo and Vehtari (2007) for details.
    %
    %	See also
    %   GPCF_MATERN52_PAK, GPCF_MATERN52_UNPAK, GPCF_MATERN52_E, GP_G
        
        gpp=gpcf.p;
        [n, m] =size(x);
        
        i1=0;i2=1;
        DKff = {};
        gprior = [];

        % Evaluate: DKff{1} = d Kff / d magnSigma2
        %           DKff{2} = d Kff / d lengthScale
        % NOTE! Here we have already taken into account that the parameters are transformed
        % through log() and thus dK/dlog(p) = p * dK/dp

        % evaluate the gradient for training covariance
        if nargin == 2
            Cdm = gpcf_matern52_trcov(gpcf, x);
            
            ii1=0;
            if ~isempty(gpcf.p.magnSigma2)
                ii1 = ii1 +1;
                DKff{ii1} = Cdm;
            end

            if isfield(gpcf,'metric')
                dist = feval(gpcf.metric.distance, gpcf.metric, x);
                [gdist, gprior_dist] = feval(gpcf.metric.ghyper, gpcf.metric, x);
                ma2 = gpcf.magnSigma2;
                for i=1:length(gdist)
                    ii1 = ii1+1;
                    DKff{ii1} = ma2.*(sqrt(5) + 10.*dist./3).*gdist{i}.*exp(-sqrt(5).*dist);
                    DKff{ii1} = DKff{ii1} - ma2.*(1+sqrt(5).*dist+5.*dist.^2./3).*exp(-sqrt(5).*dist).*sqrt(5).*gdist{i};
                end
            else
                if ~isempty(gpcf.p.lengthScale)
                    ma2 = gpcf.magnSigma2;
                    % loop over all the lengthScales
                    if length(gpcf.lengthScale) == 1
                        % In the case of isotropic MATERN52
                        s = 1./gpcf.lengthScale;
                        dist = 0;
                        for i=1:m
                            dist = dist + gminus(x(:,i),x(:,i)').^2;
                        end
                        D = ma2./3.*(5.*dist.*s^2 + 5.*sqrt(5.*dist).*dist.*s.^3).*exp(-sqrt(5.*dist).*s);
                        ii1 = ii1+1;
                        DKff{ii1} = D;
                    else
                        % In the case ARD is used
                        s = 1./gpcf.lengthScale.^2;
                        dist = 0;
                        for i=1:m
                            dist = dist + s(i).*(gminus(x(:,i),x(:,i)')).^2;
                        end
                        dist=sqrt(dist);
                        for i=1:m
                            D = ma2.*s(i).*((5+5.*sqrt(5).*dist)/3).*(gminus(x(:,i),x(:,i)')).^2.*exp(-sqrt(5).*dist);
                            
                            ii1 = ii1+1;
                            DKff{ii1} = D;
                        end
                    end
                end
            end
            % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
        elseif nargin == 3
            if size(x,2) ~= size(x2,2)
                error('gpcf_matern52 -> _ghyper: The number of columns in x and x2 has to be the same. ')
            end

            ii1=0;
            K = feval(gpcf.fh_cov, gpcf, x, x2);
            
            if ~isempty(gpcf.p.magnSigma2)
                ii1 = ii1 +1;
                DKff{ii1} = K;
            end
            
            if isfield(gpcf,'metric')                
                dist = feval(gpcf.metric.distance, gpcf.metric, x, x2);
                [gdist, gprior_dist] = feval(gpcf.metric.ghyper, gpcf.metric, x, x2);
                for i=1:length(gdist)
                    ii1 = ii1+1;
                    ma2 = gpcf.magnSigma2;
                    DKff{ii1} = ma2.*(sqrt(5) + 10.*dist./3).*gdist{i}.*exp(-sqrt(5).*dist);
                    DKff{ii1} = DKff{ii1} - ma2.*(1+sqrt(5).*dist+5.*dist.^2./3).*exp(-sqrt(5).*dist).*sqrt(5).*gdist{i};
                end
            else
                if ~isempty(gpcf.p.lengthScale)
                    % Evaluate help matrix for calculations of derivatives with respect to the lengthScale
                    if length(gpcf.lengthScale) == 1
                        % In the case of isotropic MATERN52
                        s = 1./gpcf.lengthScale;
                        ma2 = gpcf.magnSigma2;
                        dist = 0; 
                        for i=1:m
                            dist = dist + gminus(x(:,i),x2(:,i)').^2;
                        end
                        DK = ma2./3.*(5.*dist.*s^2 + 5.*sqrt(5.*dist).*dist.*s.^3).*exp(-sqrt(5.*dist).*s);
                        ii1 = ii1+1;
                        DKff{ii1} = DK;
                    else
                        % In the case ARD is used
                        s = 1./gpcf.lengthScale.^2;
                        ma2 = gpcf.magnSigma2;
                        dist = 0;
                        for i=1:m
                            dist = dist + s(i).*(gminus(x(:,i),x2(:,i)')).^2;
                        end
                        for i=1:m
                            D1 = ma2.*exp(-sqrt(5.*dist)).*s(i).*(gminus(x(:,i),x2(:,i)')).^2;;
                            DK = (5./3 + 5.*sqrt(5.*dist)/3).*D1;
                            ii1=ii1+1;
                            DKff{ii1} = DK;
                        end     
                    end
                end
            end
            % Evaluate: DKff{1}    = d mask(Kff,I) / d magnSigma2
            %           DKff{2...} = d mask(Kff,I) / d lengthScale
        elseif nargin == 4
                        
            ii1=0;            
            if ~isempty(gpcf.p.magnSigma2)
                ii1 = ii1+1;
                DKff{ii1} = feval(gpcf.fh_trvar, gpcf, x);   % d mask(Kff,I) / d magnSigma2
            end
            if isfield(gpcf,'metric')
                dist = 0;
                [gdist, gprior_dist] = feval(gpcf.metric.ghyper, gpcf.metric, x, [], 1);
                for i=1:length(gdist)
                    ii1 = ii1+1;
                    DKff{ii1} = 0;
                end
            else
                if ~isempty(gpcf.p.lengthScale)
                    for i2=1:length(gpcf.lengthScale)
                        ii1 = ii1+1;
                        DKff{ii1}  = 0;                          % d mask(Kff,I) / d lengthScale
                    end
                end
            end
        end
        if nargout > 1            
            ggs = [];
            if ~isempty(gpcf.p.magnSigma2)            
                % Evaluate the gprior with respect to magnSigma2
                i1 = 1;
                ggs = feval(gpp.magnSigma2.fh_g, gpcf.magnSigma2, gpp.magnSigma2);
                gprior = ggs(i1).*gpcf.magnSigma2 - 1;
            end
            
            if isfield(gpcf,'metric')
                % Evaluate the data contribution of gradient with respect to lengthScale
                for i2=1:length(gprior_dist)
                    i1 = i1+1;                    
                    gprior(i1)=gprior_dist(i2);
                end
            else
                if ~isempty(gpcf.p.lengthScale)
                    i1=i1+1; 
                    lll = length(gpcf.lengthScale);
                    gg = feval(gpp.lengthScale.fh_g, gpcf.lengthScale, gpp.lengthScale);
                    gprior(i1:i1-1+lll) = gg(1:lll).*gpcf.lengthScale - 1;
                    gprior = [gprior gg(lll+1:end)];
                end
            end
            if length(ggs) > 1
                gprior = [gprior ggs(2:end)];
            end
        end
    end
    
    function [DKff, gprior]  = gpcf_matern52_ginput(gpcf, x, x2)
    %GPCF_MATERN52_GIND     Evaluate gradient of covariance function with 
    %                       respect to the inducing inputs.
    %
    %	Descriptioni
    %	DKff = GPCF_MATERN52_GIND(GPCF, X, T, G, GDATA_IND, GPRIOR_IND, VARARGIN) 
    %   takes a covariance function data structure GPCF, a matrix X of input vectors, a
    %   matrix T of target vectors and vectors GDATA_IND and GPRIOR_IND. Returns:
    %      DKff    = gradients of covariance matrix Kuf with respect to Xu (cell array with matrix elements)
    %
    %   Here f refers to latent values and u to inducing varianble (e.g. Kuf is the covariance 
    %   between u and f). See Vanhatalo and Vehtari (2007) for details.
    %
    %	See also
    %   GPCF_MATERN52_PAK, GPCF_MATERN52_UNPAK, GPCF_MATERN52_E, GP_G
        
        [n, m] =size(x);
        ma2 = gpcf.magnSigma2;

        if nargin == 2
            if isfield(gpcf,'metric')
                K = feval(gpcf.fh_trcov, gpcf, x);
                dist = feval(gpcf.metric.distance, gpcf.metric, x);
                gdist = feval(gpcf.metric.ginput, gpcf.metric, x);
                ii1 = 0;
                for i=1:length(gdist)
                    ii1 = ii1+1;
                    ma2 = gpcf.magnSigma2;
                    DKff{ii1} = ma2.*(sqrt(5) + 10.*dist./3).*gdist{i}.*exp(-sqrt(5).*dist);
                    DKff{ii1} = DKff{ii1} - ma2.*(1+sqrt(5).*dist+5.*dist.^2./3).*exp(-sqrt(5).*dist).*sqrt(5).*gdist{i};
                end
            else
                if length(gpcf.lengthScale) == 1
                    s = repmat(1./gpcf.lengthScale.^2, 1, m);
                else
                    s = 1./gpcf.lengthScale.^2;
                end
                dist=0; 
                for i2=1:m
                    dist = dist + s(i2).*(gminus(x(:,i2),x(:,i2)')).^2;
                end
                dist=sqrt(dist);
                ii1 = 0;
                for i=1:m
                    for j = 1:n
                        D1 = zeros(n,n);
                        D1(j,:) = sqrt(s(i)).*gminus(x(j,i),x(:,i)');
                        D1 = D1 + D1';
                        
                        DK = ma2.*(10/3 - 5 - 5.*sqrt(5).*dist./3).*exp(-sqrt(5).*dist).*D1;                    
                        
                        ii1 = ii1 + 1;
                        DKff{ii1} = DK;
                    end
                end
            end
        elseif nargin == 3
            if isfield(gpcf,'metric')
                K = feval(gpcf.fh_cov, gpcf, x, x2);
                dist = feval(gpcf.metric.distance, gpcf.metric, x, x2);
                gdist = feval(gpcf.metric.ginput, gpcf.metric, x, x2);
                ii1 = 0;
                ma2 = gpcf.magnSigma2;
                for i=1:length(gdist)
                    ii1 = ii1+1;
                    DKff{ii1} = ma2.*(sqrt(5) + 10.*dist./3).*gdist{i}.*exp(-sqrt(5).*dist);
                    DKff{ii1} = DKff{ii1} - ma2.*(1+sqrt(5).*dist+5.*dist.^2./3).*exp(-sqrt(5).*dist).*sqrt(5).*gdist{i};
                end
            else
                [n2, m2] =size(x2);
                if length(gpcf.lengthScale) == 1
                    s = repmat(1./gpcf.lengthScale.^2, 1, m);
                else
                    s = 1./gpcf.lengthScale.^2;
                end
                dist=0; 
                for i2=1:m
                    dist = dist + s(i2).*(gminus(x(:,i2),x2(:,i2)')).^2;
                end
                dist=sqrt(dist);
                ii1 = 0;
                for i=1:m
                    for j = 1:n
                        D1 = zeros(n,n2);
                        D1(j,:) = sqrt(s(i)).*gminus(x(j,i),x2(:,i)');
                        
                        DK = ma2.*(10/3 - 5 - 5.*sqrt(5).*dist./3).*exp(-sqrt(5).*dist).*D1;                    
                        
                        ii1 = ii1 + 1;
                        DKff{ii1} = DK;
                    end
                end            
            end
        end
    end


    function C = gpcf_matern52_cov(gpcf, x1, x2)
    % GP_MATERN52_COV     Evaluate covariance matrix between two input vectors.
    %
    %         Description
    %         C = GP_MATERN52_COV(GP, TX, X) takes in covariance function of a Gaussian
    %         process GP and two matrixes TX and X that contain input vectors to
    %         GP. Returns covariance matrix C. Every element ij of C contains
    %         covariance between inputs i in TX and j in X.
    %
    %
    %         See also
    %         GPCF_MATERN52_TRCOV, GPCF_MATERN52_TRVAR, GP_COV, GP_TRCOV        
        
        if isempty(x2)
            x2=x1;
        end
        [n1,m1]=size(x1);
        [n2,m2]=size(x2);

        if m1~=m2
            error('the number of columns of X1 and X2 has to be same')
        end

        if isfield(gpcf,'metric')
            ma2 = gpcf.magnSigma2;
            dist = sqrt(5).*feval(gpcf.metric.distance, gpcf.metric, x1, x2);
            dist(dist<eps) = 0;
            C = ma2.*(1 + dist + dist.^2./3).*exp(-dist);
            C(C<eps)=0;
        else
            
            C=zeros(n1,n2);
            ma2 = gpcf.magnSigma2;
            
            % Evaluate the covariance
            if ~isempty(gpcf.lengthScale)  
                s2 = 1./gpcf.lengthScale.^2;
                % If ARD is not used make s a vector of 
                % equal elements 
                if size(s2)==1
                    s2 = repmat(s2,1,m1);
                end
                dist2=zeros(n1,n2);
                for j=1:m1
                    dist2 = dist2 + s2(:,j).*(gminus(x1(:,j),x2(:,j)')).^2;
                end
                dist = sqrt(5.*dist2);
                C = ma2.*(1 + dist + 5.*dist2./3).*exp(-dist);
            end
            C(C<eps)=0;
        end
    end

    function C = gpcf_matern52_trcov(gpcf, x)
    % GP_MATERN52_TRCOV     Evaluate training covariance matrix of inputs.
    %
    %         Description
    %         C = GP_MATERN52_TRCOV(GP, TX) takes in covariance function of a Gaussian
    %         process GP and matrix TX that contains training input vectors. 
    %         Returns covariance matrix C. Every element ij of C contains covariance 
    %         between inputs i and j in TX
    %
    %
    %         See also
    %         GPCF_MATERN52_COV, GPCF_MATERN52_TRVAR, GP_COV, GP_TRCOV

        if isfield(gpcf,'metric')
            ma2 = gpcf.magnSigma2;
            dist = sqrt(5).*feval(gpcf.metric.distance, gpcf.metric, x);
            C = ma2.*(1 + dist + dist.^2./3).*exp(-dist);
        else
            C = trcov(gpcf,x);
            
            if isnan(C)
                [n, m] =size(x);
                
                s2 = 1./(gpcf.lengthScale).^2;
                if size(s2)==1
                    s2 = repmat(s2,1,m);
                end
                ma2 = gpcf.magnSigma2;
                
                % Here we take advantage of the 
                % symmetry of covariance matrix
                C=zeros(n,n);
                for i1=2:n
                    i1n=(i1-1)*n;
                    for i2=1:i1-1
                        ii=i1+(i2-1)*n;
                        for i3=1:m
                            C(ii)=C(ii)+s2(i3).*(x(i1,i3)-x(i2,i3)).^2;       % the covariance function
                        end
                        C(i1n+i2)=C(ii); 
                    end
                end
                dist = sqrt(5.*C);
                C = ma2.*(1 + dist + 5.*C./3).*exp(-dist);
                C(C<eps)=0;
            end
        end
    end
    
    function C = gpcf_matern52_trvar(gpcf, x)
    % GP_MATERN52_TRVAR     Evaluate training variance vector
    %
    %         Description
    %         C = GP_MATERN52_TRVAR(GPCF, TX) takes in covariance function of a Gaussian
    %         process GPCF and matrix TX that contains training inputs. Returns variance 
    %         vector C. Every element i of C contains variance of input i in TX
    %
    %
    %         See also
    %         GPCF_MATERN52_COV, GP_COV, GP_TRCOV
        
        [n, m] =size(x);

        C = ones(n,1)*gpcf.magnSigma2;
        C(C<eps)=0;
    end

    function reccf = gpcf_matern52_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_MATERN52_RECAPPEND(RECCF, RI, GPCF) takes old covariance
    %          function record RECCF, record index RI and covariance function structure. 
    %          Appends the parameters of GPCF to the RECCF in the ri'th place.
    %
    %          RECAPPEND returns a structure RECCF containing following record fields:
    %          lengthHyper    
    %          lengthHyperNu  
    %          lengthScale    
    %          magnSigma2     
    %
    %          See also
    %          GP_MC  and GP_MC -> RECAPPEND
        
    % Initialize record
        if nargin == 2
            reccf.type = 'gpcf_matern52';
            
            % Initialize parameters
            reccf.lengthScale= [];
            reccf.magnSigma2 = [];
            
            % Set the function handles
            reccf.fh_pak = @gpcf_matern52_pak;
            reccf.fh_unpak = @gpcf_matern52_unpak;
            reccf.fh_e = @gpcf_matern52_e;
            reccf.fh_g = @gpcf_matern52_g;
            reccf.fh_cov = @gpcf_matern52_cov;
            reccf.fh_trcov  = @gpcf_matern52_trcov;
            reccf.fh_trvar  = @gpcf_matern52_trvar;
            %  gpcf.fh_sampling = @hmc2;
            %  reccf.sampling_opt = hmc2_opt;
            reccf.fh_recappend = @gpcf_matern52_recappend;  
            reccf.p=[];
            reccf.p.lengthScale=[];
            reccf.p.magnSigma2=[];
            if ~isempty(ri.p.lengthScale)
                reccf.p.lengthScale = ri.p.lengthScale;
            end
            if ~isempty(ri.p.magnSigma2)
                reccf.p.magnSigma2 = ri.p.magnSigma2;
            end
            return
        end
        
        gpp = gpcf.p;
                
        if ~isfield(gpcf,'metric')
            % record lengthScale
            if ~isempty(gpcf.lengthScale)
                reccf.lengthScale(ri,:)=gpcf.lengthScale;
                reccf.p.lengthScale = feval(gpp.lengthScale.fh_recappend, reccf.p.lengthScale, ri, gpcf.p.lengthScale);
            elseif ri==1
                reccf.lengthScale=[];
            end
        end
        % record magnSigma2
        if ~isempty(gpcf.magnSigma2)
            reccf.magnSigma2(ri,:)=gpcf.magnSigma2;
        elseif ri==1
            reccf.magnSigma2=[];
        end
    end
end
