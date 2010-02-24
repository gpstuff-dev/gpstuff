function gpcf = gpcf_rq(do, varargin)
%GPCF_RQ	Create an rational quadratic covariance function for Gaussian Process
%
%	Description
%
%	GPCF = GPCF_RQ('INIT', NIN) Create and initialize rational quaratic
%       covariance function for Gaussian process
%
%	The fields and (default values) in GPCF_RQ are:
%	  type           = 'gpcf_rq'
%	  nin            = Number of inputs. (NIN)
%	  nout           = Number of outputs. (always 1)
%	  magnSigma2     = Magnitude (squared) for exponential part. 
%                          (0.1)
%	  lengthScale    = Length scale for each input. This can be either scalar corresponding 
%                          isotropic or vector corresponding ARD. 
%                          (repmat(10, 1, nin))
%         p              = Prior structure for covariance function parameters. 
%                          (e.g. p.lengthScale.)
%         fh_pak         = function handle to pack function
%                          (@gpcf_rq_pak)
%         fh_unpak       = function handle to unpack function
%                          (@gpcf_rq_unpak)
%         fh_e           = function handle to energy function
%                          (@gpcf_rq_e)
%         fh_ghyper      = function handle to gradient of energy with respect to hyperparameters
%                          (@gpcf_rq_ghyper)
%         fh_ginput      = function handle to gradient of function with respect to inducing inputs
%                          (@gpcf_rq_ginput)
%         fh_cov         = function handle to covariance function
%                          (@gpcf_rq_cov)
%         fh_trcov       = function handle to training covariance function
%                          (@gpcf_rq_trcov)
%         fh_trvar       = function handle to training variance function
%                          (@gpcf_rq_trvar)
%         fh_recappend   = function handle to append the record function 
%                          (gpcf_rq_recappend)
%
%	GPCF = GPCF_RQ('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in GPCF.
%       The fields that can be modified are:
%
%             'magnSigma2'         : set the magnSigma2
%             'alpha'              : set the alpha
%             'lengthScale'        : set the lengthScale
%             'metric'             : set the metric structure into the covariance function
%             'magnSigma2_prior'   ; set the prior structure for magnSigma2
%             'alpha_prior'        ; set the prior structure for alpha
%             'lengthScale_prior'  : set the prior structure for lengthScale
%
%	See also
%       gpcf_sexp, gpcf_matern32, gpcf_matern52, gpcf_ppcs2, gp_init, gp_e, gp_g, gp_trcov
%       gp_cov, gp_unpak, gp_pak
    
% Copyright (c) 2000-2001,2010 Aki Vehtari
% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010 Tuomas Nikoskinen

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end

    % Initialize the covariance function
    if strcmp(do, 'init')
        nin = varargin{1};
        gpcf.type = 'gpcf_rq';
        gpcf.nin = nin;
        gpcf.nout = 1;

        % Initialize parameters
        gpcf.lengthScale= repmat(10, 1, nin);
        gpcf.magnSigma2 = 0.1;
        gpcf.alpha = 20;  % Value for the exponent in the rq covariance function
        
        % Initialize prior structure
        gpcf.p = [];
        gpcf.p.lengthScale = prior_unif('init');
        gpcf.p.magnSigma2 = prior_unif('init');
        gpcf.p.alpha = prior_unif('init');
        
        % Set the function handles
        gpcf.fh_pak = @gpcf_rq_pak;
        gpcf.fh_unpak = @gpcf_rq_unpak;
        gpcf.fh_e = @gpcf_rq_e;
        gpcf.fh_ghyper = @gpcf_rq_ghyper;
        gpcf.fh_ginput = @gpcf_rq_ginput;
        gpcf.fh_cov = @gpcf_rq_cov;
        gpcf.fh_trcov  = @gpcf_rq_trcov;
        gpcf.fh_trvar  = @gpcf_rq_trvar;
        gpcf.fh_recappend = @gpcf_rq_recappend;

        if length(varargin) > 1
            if mod(nargin,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=2:2:length(varargin)-1
                switch varargin{i}
                  case 'magnSigma2'
                    gpcf.magnSigma2 = varargin{i+1};
                  case 'lengthScale'
                    gpcf.lengthScale = varargin{i+1};
                  case 'alpha'
                    gpcf.alpha = varargin{i+1};
                  case 'metric'
                    gpcf.metric = varargin{i+1};
                    if isfield(gpcf, 'lengthScale')
                        gpcf = rmfield(gpcf, 'lengthScale');
                    end
                    if isfield(gpcf.p, 'lengthScale')
                        gpcf.p = rmfield(gpcf.p, 'lengthScale');
                    end
                  case 'lengthScale_prior'
                    gpcf.p.lengthScale = varargin{i+1};
                  case 'magnSigma2_prior'
                    gpcf.p.magnSigma2 = varargin{i+1};
                  case 'alpha_prior'
                    gpcf.p.alpha = varargin{i+1};
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
              case 'alpha'
                gpcf.alpha = varargin{i+1};                
              case 'metric'
                gpcf.metric = varargin{i+1};
                if isfield(gpcf, 'lengthScale')
                    gpcf = rmfield(gpcf, 'lengthScale');
                end
                if isfield(gpcf.p, 'lengthScale')
                    gpcf.p = rmfield(gpcf.p, 'lengthScale');
                end
              case 'lengthScale_prior'
                gpcf.p.lengthScale = varargin{i+1};
              case 'magnSigma2_prior'
                gpcf.p.magnSigma2 = varargin{i+1};
              case 'alpha_prior'
                gpcf.p.alpha = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end

    function w = gpcf_rq_pak(gpcf)
    %GPCF_RQ_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %	W = GPCF_RQ_PAK(GPCF) takes a covariance function data structure GPCF and
    %	combines the hyper-parameters into a single row vector W.
    %
    %	The ordering of the parameters in W is:
    %       w = [gpcf.magnSigma2 gpcf.alpha (hyperparameters of gpcf.lengthScale) gpcf.lengthScale ]
    %	  
    %
    %	See also
    %	GPCF_RQ_UNPAK

        i1=0;i2=1;
        ww = []; w = [];
        
        if ~isempty(gpcf.p.magnSigma2)
            i1 = i1+1;
            w(i1) = log(gpcf.magnSigma2);
            
            % Hyperparameters of magnSigma2
            ww = feval(gpcf.p.magnSigma2.fh_pak, gpcf.p.magnSigma2);
        end        

        if ~isempty(gpcf.p.alpha)
            i1=i1+1;
            w(i1)= log(log(gpcf.alpha));

            % Hyperparameters of alpha
            ww = [ww feval(gpcf.p.alpha.fh_pak, gpcf.p.alpha)];
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


    function [gpcf, w] = gpcf_rq_unpak(gpcf, w)
    %GPCF_RQ_UNPAK  Separate covariance function hyper-parameter vector into components.
    %
    %	Description
    %	[GPCF, W] = GPCF_RQ_UNPAK(GPCF, W) takes a covariance function data structure GPCF
    %	and  a hyper-parameter vector W, and returns a covariance function data
    %	structure  identical to the input, except that the covariance hyper-parameters 
    %   has been set to the values in W. Deletes the values set to GPCF from W and returns 
    %   the modeified W. 
    %
    %	See also
    %	GPCF_RQ_PAK

        gpp=gpcf.p;
        if ~isempty(gpp.magnSigma2)
            i1=1;
            gpcf.magnSigma2 = exp(w(i1));
            w = w(i1+1:end);
        end
        
        if ~isempty(gpp.alpha)
            i1=1;
            gpcf.alpha = exp(exp(w(i1)));
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

        if ~isempty(gpp.alpha)
            % Hyperparameters of alpha
            [p, w] = feval(gpcf.p.alpha.fh_unpak, gpcf.p.alpha, w);
            gpcf.p.alpha = p;
        end
        
    end

    function eprior =gpcf_rq_e(gpcf, x, t)
    %GPCF_RQ_E     Evaluate the energy of prior of RQ parameters
    %
    %	Description
    %	E = GPCF_RQ_E(GPCF, X, T) takes a covariance function data structure 
    %   GPCF together with a matrix X of input vectors and a matrix T of target 
    %   vectors and evaluates log p(th) x J, where th is a vector of SEXP parameters 
    %   and J is the Jacobian of transformation exp(w) = th. (Note that the parameters 
    %   are log transformed, when packed.)
    %
    %	See also
    %	GPCF_RQ_PAK, GPCF_RQ_UNPAK, GPCF_RQ_G, GP_E

        eprior = 0;
        gpp=gpcf.p;
        
        [n, m] =size(x);

        % Evaluate the prior contribution to the error. The parameters that
        % are sampled are from space W = log(w) where w is all the "real" samples.
        % On the other hand errors are evaluated in the W-space so we need take
        % into account also the  Jacobian of transformation W -> w = exp(W).
        % See Gelman et.all., 2004, Bayesian data Analysis, second edition, p24.
        if ~isempty(gpcf.p.magnSigma2)
                eprior = feval(gpp.magnSigma2.fh_e, gpcf.magnSigma2, gpp.magnSigma2) - log(gpcf.magnSigma2);
        end
        
        if ~isempty(gpcf.p.alpha)
          eprior = eprior + feval(gpp.alpha.fh_e, gpcf.alpha, gpp.alpha) -log(gpcf.alpha) -log(log(gpcf.alpha));
        end        

        if isfield(gpcf,'metric')
            eprior = eprior + feval(gpcf.metric.e, gpcf.metric, x, t);
        else

          if ~isempty(gpp.lengthScale)
            eprior = eprior + feval(gpp.lengthScale.fh_e, gpcf.lengthScale, gpp.lengthScale) - sum(log(gpcf.lengthScale));
          end
        end
    end

    function [DKff, gprior]  = gpcf_rq_ghyper(gpcf, x, x2, mask)
    %GPCF_RQ_GHYPER     Evaluate gradient of covariance function and hyper-prior with 
    %                    respect to the hyperparameters.
    %
    %	Description
    %	[GPRIOR, DKff, DKuu, DKuf] = GPCF_RQ_GHYPER(GPCF, X, T, G, GDATA, GPRIOR, VARARGIN) 
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
    %   GPCF_RQ_PAK, GPCF_RQ_UNPAK, GPCF_RQ_E, GP_G
        
        gpp=gpcf.p;
        [n, m] =size(x);
        a=(gpcf.alpha+1)/gpcf.alpha;

        i1=0;i2=1;
        DKff = {};
        gprior = [];

        % Evaluate: DKff{1} = d Kff / d magnSigma2
        %           DKff{2} = d Kff / d alpha
        %           DKff{3} = d Kff / d lengthscale
        % NOTE! Here we have already taken into account that the parameters are transformed
        % through log() and thus dK/dlog(p) = p * dK/dp

        % evaluate the gradient for training covariance
        if nargin == 2
            Cdm = gpcf_rq_trcov(gpcf, x);

            ii1=0;

            if ~isempty(gpcf.p.magnSigma2)
                ii1=ii1+1;
                DKff{ii1} = Cdm;
            end
            
            ma2=gpcf.magnSigma2;
            
            if isfield(gpcf,'metric')
                dist = feval(gpcf.metric.distance, gpcf.metric, x);
                [gdist, gprior_dist] = feval(gpcf.metric.ghyper, gpcf.metric, x);
                % dalpha
                ii1=ii1+1;
                DKff{ii1} = (ma2.^(1-a).*.5.*dist.^2.*Cdm.^a - gpcf.alpha.*log(Cdm.^(-1/gpcf.alpha)./ma2.^(-1/gpcf.alpha)).*Cdm).*log(gpcf.alpha);

                % dlengthscale
                for i=1:length(gdist)
                    ii1=ii1+1;
                    DKff{ii1} = Cdm.*-dist./(1+dist.^2./(2*gpcf.alpha)).*gdist{i};
                end
            else
                % loop over all the lengthScales
                if length(gpcf.lengthScale) == 1
                    % Isotropic = no ARD
                    s = 1./(gpcf.lengthScale^2);
                    dist2 = 0;
                    for i=1:m
                        dist2 = dist2 + (gminus(x(:,i),x(:,i)')).^2;
                    end
                    % dalpha
                    ii1=ii1+1;
                    DKff{ii1} = (ma2^(1-a).*.5.*dist2.*s.*Cdm.^a - gpcf.alpha.*log(Cdm.^(-1/gpcf.alpha)./ma2^(-1/gpcf.alpha)).*Cdm).*log(gpcf.alpha);
                    % dlengthscale
                    ii1 = ii1+1;
                    DKff{ii1} = Cdm.^a.*s.*dist2.*gpcf.magnSigma2^(-a+1);
                else
                    % ARD
                    s = 1./(gpcf.lengthScale.^2);
                    % skip dalpha for a moment
                    ii1=ii1+1;
                    iialpha=ii1; 
                    D=zeros(size(Cdm));
                    for i=1:m
                        dist2 =(gminus(x(:,i),x(:,i)')).^2;
                        % sum distance for the dalpha
                        D=D+dist2.*s(i); 
                        % dlengthscale
                        ii1 = ii1+1;
                        DKff{ii1}=Cdm.^a.*s(i).*dist2.*gpcf.magnSigma2.^(-a+1);
                    end
                    % dalpha
                    DKff{iialpha} = (ma2^(1-a).*.5.*D.*Cdm.^a - gpcf.alpha.*log(Cdm.^(-1/gpcf.alpha)./ma2^(-1/gpcf.alpha)).*Cdm).*log(gpcf.alpha);
                end
            end
            % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
        elseif nargin == 3
            if size(x,2) ~= size(x2,2)
                error('gpcf_rq -> _ghyper: The number of columns in x and x2 has to be the same. ')
            end
            
            ii1=1;
            K = feval(gpcf.fh_cov, gpcf, x, x2);
            DKff{ii1} = K;
            
            if isfield(gpcf,'metric')                
                dist = feval(gpcf.metric.distance, gpcf.metric, x, x2);
                [gdist, gprior_dist] = feval(gpcf.metric.ghyper, gpcf.metric, x, x2);
                for i=1:length(gdist)
                    ii1 = ii1+1;                    
                    DKff{ii1} = -K.*gdist{i};                    
                end
            else
                % Evaluate help matrix for calculations of derivatives with respect to the lengthScale
                if length(gpcf.lengthScale) == 1
                    % In the case of an isotropic EXP
                    s = 1/gpcf.lengthScale^2;
                    dist = 0;
                    for i=1:m
                        dist = dist + (gminus(x(:,i),x2(:,i)')).^2;
                    end
                    DK_l = s.*K.^a.*dist.*gpcf.magnSigma2^(1-a);
                    ii1=ii1+1;
                    DKff{ii1} = DK_l;
                else
                    % In the case ARD is used
                    s = 1./gpcf.lengthScale.^2;        % set the length
                    for i=1:m
                        D1 = s(i).*K.^a.*gminus(x(:,i),x2(:,i)').^2.*gpcf.magnSigma2^(1-a);
                        ii1=ii1+1;
                        DKff{ii1} = D1;
                    end
                end
            end
            % Evaluate: DKff{1}    = d mask(Kff,I) / d magnSigma2
            %           DKff{2...} = d mask(Kff,I) / d lengthScale
        elseif nargin == 4
            if isfield(gpcf,'metric')
                ii1=1;
                DKff{ii1} = feval(gpcf.fh_trvar, gpcf, x);   % d mask(Kff,I) / d magnSigma2
                
                dist = 0;
                [gdist, gprior_dist] = feval(gpcf.metric.ghyper, gpcf.metric, x, [], 1);
                for i=1:length(gdist)
                    ii1 = ii1+1;
                    DKff{ii1} = 0;
                end
            else
                ii1=1;
                DKff{ii1} = feval(gpcf.fh_trvar, gpcf, x);   % d mask(Kff,I) / d magnSigma2
                for i2=1:length(gpcf.lengthScale)
                    ii1 = ii1+1;
                    DKff{ii1}  = 0;                          % d mask(Kff,I) / d lengthScale
                end
            end
        end
        if nargout > 1
            ggs = [];
            if ~isempty(gpcf.p.magnSigma2)            
                % Evaluate the gprior with respect to magnSigma2
                i1 = 1;
                ggs = feval(gpp.magnSigma2.fh_g, gpcf.magnSigma2, gpp.magnSigma2);
                gprior(i1) = ggs(1).*gpcf.magnSigma2 - 1;
            end

            ggs2=[];
            if ~isempty(gpcf.p.alpha)
                % Evaluate the gprior with respect to alpha
                i1 = i1 + 1;
                ggs2 = feval(gpp.alpha.fh_g, gpcf.alpha, gpp.alpha);
                gprior(i1) = ggs2(1).*gpcf.alpha.*log(gpcf.alpha) -log(gpcf.alpha) - 1;
                %gprior(i1) = ggs2(1).*gpcf.alpha - 1;
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

            if length(ggs2) > 1
                gprior = [gprior ggs2(2:end)];
            end


        end
    end

    function [DKff, gprior]  = gpcf_rq_ginput(gpcf, x, x2)
    %GPCF_RQ_GIND     Evaluate gradient of covariance function with 
    %                  respect to the inducing inputs.
    %
    %	Descriptioni
    %	[GPRIOR_IND, DKuu, DKuf] = GPCF_RQ_GIND(GPCF, X, T, G, GDATA_IND, GPRIOR_IND, VARARGIN) 
    %   takes a covariance function data structure GPCF, a matrix X of input vectors, a
    %   matrix T of target vectors and vectors GDATA_IND and GPRIOR_IND. Returns:
    %      GPRIOR  = d log(p(th))/dth, where th is the vector of hyperparameters 
    %      DKuu    = gradients of covariance matrix Kuu with respect to Xu (cell array with matrix elements)
    %      DKuf    = gradients of covariance matrix Kuf with respect to Xu (cell array with matrix elements)
    %
    %   Here f refers to latent values and u to inducing varianble (e.g. Kuf is the covariance 
    %   between u and f). See Vanhatalo and Vehtari (2007) for details.
    %
    %	See also
    %   GPCF_RQ_PAK, GPCF_RQ_UNPAK, GPCF_RQ_E, GP_G
        a=(gpcf.alpha+1)/gpcf.alpha;
        [n, m] =size(x);
               
        if nargin == 2
            K = feval(gpcf.fh_trcov, gpcf, x);
            ii1 = 0;
            if isfield(gpcf,'metric')
                dist = feval(gpcf.metric.distance, gpcf.metric, x);
                [gdist, gprior_dist] = feval(gpcf.metric.ginput, gpcf.metric, x);
                for i=1:length(gdist)
                    ii1 = ii1+1;
                    DKff{ii1} = -K.*gdist{ii1};
                    gprior(ii1) = gprior_dist(ii1);
                end
            else
                if length(gpcf.lengthScale) == 1
                    % In the case of an isotropic EXP
                    s = repmat(1./gpcf.lengthScale.^2, 1, m);
                else
                    s = 1./gpcf.lengthScale.^2;
                end
                
                for i=1:m
                    for j = 1:n
                        DK = zeros(size(K));
                        DK(j,:) = -s(i).*gminus(x(j,i),x(:,i)');
                        DK = DK + DK';    
                        
                        DK = DK.*K.^a.*gpcf.magnSigma2^(1-a);      
                        
                        ii1 = ii1 + 1;
                        DKff{ii1} = DK;
                        gprior(ii1) = 0; 
                    end
                end
            end        
        elseif nargin == 3
            [n2, m2] =size(x2);
            K = feval(gpcf.fh_cov, gpcf, x, x2);
            ii1 = 0;
            if isfield(gpcf,'metric')
                dist = feval(gpcf.metric.distance, gpcf.metric, x, x2);
                [gdist, gprior_dist] = feval(gpcf.metric.ginput, gpcf.metric, x, x2);
                for i=1:length(gdist)
                    ii1 = ii1+1;
                    DKff{ii1}   = -K.*gdist{ii1};
                    gprior(ii1) = gprior_dist(ii1);
                end
            else 
                if length(gpcf.lengthScale) == 1
                    % In the case of an isotropic EXP
                    s = repmat(1./gpcf.lengthScale.^2, 1, m);
                else
                    s = 1./gpcf.lengthScale.^2;
                end
                
                ii1 = 0;
                for i=1:m
                    for j = 1:n
                        DK= zeros(size(K));
                        DK(j,:) = -s(i).*gminus(x(j,i),x2(:,i)');
                        
                        DK = DK.*K.^a.*gpcf.magnSigma2^(1-a);
                        
                        ii1 = ii1 + 1;
                        DKff{ii1} = DK;
                        gprior(ii1) = 0; 
                    end
                end
            end
        end
    end
    
    function C = gpcf_rq_cov(gpcf, x1, x2)
    % GP_RQ_COV     Evaluate covariance matrix between two input vectors.
    %
    %         Description
    %         C = GP_RQ_COV(GP, TX, X) takes in covariance function of a Gaussian
    %         process GP and two matrixes TX and X that contain input vectors to
    %         GP. Returns covariance matrix C. Every element ij of C contains
    %         covariance between inputs i in TX and j in X.
    %
    %
    %         See also
    %         GPCF_RQ_TRCOV, GPCF_RQ_TRVAR, GP_COV, GP_TRCOV
        
        if isempty(x2)
            x2=x1;
        end
        [n1,m1]=size(x1);
        [n2,m2]=size(x2);
        
        if m1~=m2
            error('the number of columns of X1 and X2 has to be same')
        end
        
        if isfield(gpcf,'metric')
            dist = feval(gpcf.metric.distance, gpcf.metric, x1, x2).^2;
            dist(dist<eps) = 0;
            C = gpcf.magnSigma2.*(1+dist./(2*gpcf.alpha)).^(-gpcf.alpha);
        else
            C=zeros(n1,n2);
            ma2 = gpcf.magnSigma2;
           
            
            % Evaluate the covariance
            if ~isempty(gpcf.lengthScale)  
                s2 = 1./(2.*gpcf.alpha.*gpcf.lengthScale.^2);      
                % If ARD is not used make s a vector of 
                % equal elements 
                if size(s2)==1
                    s2 = repmat(s2,1,m1);
                end
                dist=zeros(n1,n2);
                for j=1:m1
                    dist = dist + s2(j).*(gminus(x1(:,j),x2(:,j)')).^2;
                end
                dist(dist<eps) = 0;
                C = ma2.*(1+dist).^(-gpcf.alpha);
            end
        end
    end

    function C = gpcf_rq_trcov(gpcf, x)
    % GP_RQ_TRCOV     Evaluate training covariance matrix of inputs.
    %
    %         Description
    %         C = GP_RQ_TRCOV(GP, TX) takes in covariance function of a Gaussian
    %         process GP and matrix TX that contains training input vectors. 
    %         Returns covariance matrix C. Every element ij of C contains covariance 
    %         between inputs i and j in TX
    %
    %
    %         See also
    %         GPCF_RQ_COV, GPCF_RQ_TRVAR, GP_COV, GP_TRCOV
        if isfield(gpcf,'metric')
            % If other than scaled euclidean metric
            [n, m] =size(x);            
            ma = gpcf.magnSigma2;
            
            C = zeros(n,n);
            for ii1=1:n-1
                d = zeros(n-ii1,1);
                col_ind = ii1+1:n;
                d = feval(gpcf.metric.distance, gpcf.metric, x(col_ind,:), x(ii1,:)).^2;                
                C(col_ind,ii1) = d;
            end
            C(C<eps) = 0;
            C = C+C';
            C = ma.*(1+C./(2*gpcf.alpha)).^(-gpcf.alpha);     
        else
            % If scaled euclidean metric
            % Try to use the C-implementation            
            %C = trcov(gpcf, x);
            C=NaN;

            if isnan(C)
                % If there wasn't C-implementation do here
                [n, m] =size(x);
                
                s2 = 1./(2*gpcf.alpha.*gpcf.lengthScale.^2);
                if size(s2)==1
                    s2 = repmat(s2,1,m);
                end
                ma = gpcf.magnSigma2;
                
                C = zeros(n,n);
                for ii1=1:n-1
                    d = zeros(n-ii1,1);
                    col_ind = ii1+1:n;
                    for ii2=1:m
                        d = d+s2(ii2).*(x(col_ind,ii2)-x(ii1,ii2)).^2;
                    end
                    C(col_ind,ii1) = d;
                end
                C(C<eps) = 0;
                C = C+C';
                C = ma.*(1+C).^(-gpcf.alpha);
            end
        end
    end
    
    function C = gpcf_rq_trvar(gpcf, x)
    % GP_RQ_TRVAR     Evaluate training variance vector
    %
    %         Description
    %         C = GP_RQ_TRVAR(GPCF, TX) takes in covariance function of a Gaussian
    %         process GPCF and matrix TX that contains training inputs. Returns variance 
    %         vector C. Every element i of C contains variance of input i in TX
    %
    %
    %         See also
    %         GPCF_RQ_COV, GP_COV, GP_TRCOV

        [n, m] =size(x);

        C = ones(n,1).*gpcf.magnSigma2;
        C(C<eps)=0;
    end

    function reccf = gpcf_rq_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_RQ_RECAPPEND(RECCF, RI, GPCF) takes old covariance
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
    %          GP_MC and GP_MC -> RECAPPEND
        
    % Initialize record
        if nargin == 2
            reccf.type = 'gpcf_rq';
            reccf.nin = ri;
            gpcf.nout = 1;
            
            % Initialize parameters
            reccf.lengthScale= [];
            reccf.magnSigma2 = [];
            reccf.gpcf.alpha = [];
            
            % Set the function handles
            reccf.fh_pak = @gpcf_rq_pak;
            reccf.fh_unpak = @gpcf_rq_unpak;
            reccf.fh_e = @gpcf_rq_e;
            reccf.fh_g = @gpcf_rq_g;
            reccf.fh_cov = @gpcf_rq_cov;
            reccf.fh_trcov  = @gpcf_rq_trcov;
            reccf.fh_trvar  = @gpcf_rq_trvar;
            reccf.fh_recappend = @gpcf_rq_recappend;  
            reccf.p=[];
            reccf.p.lengthScale=[];
            reccf.p.magnSigma2=[];
            if ~isempty(ri.p.lengthScale)
                reccf.p.lengthScale = ri.p.lengthScale;
            end
            if ~isempty(ri.p.magnSigma2)
                reccf.p.magnSigma2 = ri.p.magnSigma2;
            end
            if ~isempty(ri.p.alpha)
                reccf.p.alpha = ri.p.alpha;
            end
            return
        end

        gpp = gpcf.p;
        
        if ~isfield(gpcf,'metric')
            % record lengthScale
            if ~isempty(gpcf.lengthScale)
                reccf.lengthScale(ri,:)=gpcf.lengthScale;
                if ~isempty(ri.p.lengthScale)
                      reccf.p.lengthScale = feval(gpp.lengthScale.fh_recappend, reccf.p.lengthScale, ri, gpcf.p.lengthScale);
                end
            elseif ri==1
                reccf.lengthScale=[];
            end
        end
        % record magnSigma2
        if ~isempty(gpcf.magnSigma2)
            reccf.magnSigma2(ri,:)=gpcf.magnSigma2;
            if ~isempty(ri.p.magnSigma2)
	          reccf.p.magnSigma2 = feval(gpp.magnSigma2.fh_recappend, reccf.p.magnSigma2, ri, gpcf.p.magnSigma2);
            end
        elseif ri==1
            reccf.magnSigma2=[];
        end

        % record alpha
        if ~isempty(gpcf.alpha)
            reccf.alpha(ri,:)=gpcf.alpha;
            if ~isempty(ri.p.alpha)
		 reccf.p.alpha = feval(gpp.alpha.fh_recappend, reccf.p.alpha, ri, gpcf.p.alpha);
            end
        elseif ri==1
            reccf.alpha=[];
        end
    end

end
