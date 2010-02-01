function gpcf = gpcf_constant(do, varargin)
%GPCF_CONSTANT	Create a constant covariance function for Gaussian Process
%
%	Description
%
%	GPCF = GPCF_CONSTANT('INIT', NIN) Create and initialize constant
%       covariance function for Gaussian process
%
%	The fields and (default values) in GPCF_CONSTANT are:
%	  type           = 'gpcf_constant'
%	  nin            = Number of inputs. (NIN)
%	  nout           = Number of outputs. (always 1)
%	  constSigma2     = Prior variance on the constant term.
%                          (0.1)
%         p              = Prior structure for covariance function parameters. 
%                          (e.g. p.constSigma2.)
%         fh_pak         = function handle to pack function
%                          (@gpcf_constant_pak)
%         fh_unpak       = function handle to unpack function
%                          (@gpcf_constant_unpak)
%         fh_e           = function handle to energy function
%                          (@gpcf_constant_e)
%         fh_ghyper      = function handle to gradient of energy with respect to hyperparameters
%                          (@gpcf_constant_ghyper)
%         fh_ginput      = function handle to gradient of function with respect to inducing inputs
%                          (@gpcf_constant_ginput)
%         fh_cov         = function handle to covariance function
%                          (@gpcf_constant_cov)
%         fh_trcov       = function handle to training covariance function
%                          (@gpcf_constant_trcov)
%         fh_trvar       = function handle to training variance function
%                          (@gpcf_constant_trvar)
%         fh_recappend   = function handle to append the record function 
%                          (gpcf_constant_recappend)
%
%	GPCF = GPCF_CONSTANT('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in GPCF.
%
%	See also
%       gpcf_exp, gpcf_matern32, gpcf_matern52, gpcf_ppcs2, gp_init, gp_e, gp_g, gp_trcov
%       gp_cov, gp_unpak, gp_pak
    
% Copyright (c) 2000-2001 Aki Vehtari
% Copyright (c) 2007-2009 Jarno Vanhatalo
% Copyright (c) 2010      Jaakko Riihimaki

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end

    % Initialize the covariance function
    if strcmp(do, 'init')
        nin = varargin{1};
        gpcf.type = 'gpcf_constant';
        gpcf.nin = nin;
        gpcf.nout = 1;

        % Initialize parameters
        gpcf.constSigma2 = 0.1;

        % Initialize prior structure
        gpcf.p=[];
        gpcf.p.constSigma2=prior_unif('init');

        % Set the function handles to the nested functions
        gpcf.fh_pak = @gpcf_constant_pak;
        gpcf.fh_unpak = @gpcf_constant_unpak;
        gpcf.fh_e = @gpcf_constant_e;
        gpcf.fh_ghyper = @gpcf_constant_ghyper;
        gpcf.fh_ginput = @gpcf_constant_ginput;
        gpcf.fh_cov = @gpcf_constant_cov;
        gpcf.fh_trcov  = @gpcf_constant_trcov;
        gpcf.fh_trvar  = @gpcf_constant_trvar;
        gpcf.fh_recappend = @gpcf_constant_recappend;

        if length(varargin) > 1
            if mod(nargin,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=2:2:length(varargin)-1
                switch varargin{i}
                  case 'constSigma2'
                    gpcf.constSigma2 = varargin{i+1};
                  case 'fh_sampling'
                    gpcf.fh_sampling = varargin{i+1};
                  case 'constSigma2_prior'
                    gpcf.p.constSigma2 = varargin{i+1};
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
              case 'constSigma2'
                gpcf.constSigma2 = varargin{i+1};
              case 'fh_sampling'
                gpcf.fh_sampling = varargin{i+1};
              case 'constSigma2_prior'
                gpcf.p.constSigma2 = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end

    function w = gpcf_constant_pak(gpcf, w)
    %GPCF_CONSTANT_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %	W = GPCF_CONSTANT_PAK(GPCF, W) takes a covariance function data structure GPCF and
    %	combines the hyper-parameters into a single row vector W.
    %
    %	The ordering of the parameters in W is:
    %       w = [gpcf.constSigma2 (hyperparameters of gpcf.constSigma2)]
    %	  
    %
    %	See also
    %	GPCF_CONSTANT_UNPAK
        
        w = [];
        
        if ~isempty(gpcf.p.constSigma2)
            w = log(gpcf.constSigma2);
            
            % Hyperparameters of constSigma2
            w = [w feval(gpcf.p.constSigma2.fh_pak, gpcf.p.constSigma2)];
        end        
     end

    function [gpcf, w] = gpcf_constant_unpak(gpcf, w)
    %GPCF_CONSTANT_UNPAK  Separate covariance function hyper-parameter vector into components.
    %
    %	Description
    %	[GPCF, W] = GPCF_CONSTANT_UNPAK(GPCF, W) takes a covariance function data structure GPCF
    %	and  a hyper-parameter vector W, and returns a covariance function data
    %	structure  identical to the input, except that the covariance hyper-parameters 
    %   has been set to the values in W. Deletes the values set to GPCF from W and returns 
    %   the modeified W. 
    %
    %	See also
    %	GPCF_CONSTANT_PAK
    %
        gpp=gpcf.p;
        if ~isempty(gpp.constSigma2)
            gpcf.constSigma2 = exp(w(1));
            w = w(2:end);

            % Hyperparameters of magnSigma2
            [p, w] = feval(gpcf.p.constSigma2.fh_unpak, gpcf.p.constSigma2, w);
            gpcf.p.constSigma2 = p;
        end
    end

    function eprior =gpcf_constant_e(gpcf, x, t)
    %GPCF_CONSTANT_E     Evaluate the energy of prior of CONSTANT parameters
    %
    %	Description
    %	E = GPCF_CONSTANT_E(GPCF, X, T) takes a covariance function data structure 
    %   GPCF together with a matrix X of input vectors and a matrix T of target 
    %   vectors and evaluates log p(th) x J, where th is a vector of CONSTANT parameters 
    %   and J is the Jakobian of transformation exp(w) = th. (Note that the parameters 
    %   are log transformed, when packed.)
    %
    %	See also
    %	GPCF_CONSTANT_PAK, GPCF_CONSTANT_UNPAK, GPCF_CONSTANT_G, GP_E
    %

        % Evaluate the prior contribution to the error. The parameters that
        % are sampled are from space W = log(w) where w is all the "real" samples.
        % On the other hand errors are evaluated in the W-space so we need take
        % into account also the  Jakobian of transformation W -> w = exp(W).
        % See Gelman et.all., 2004, Bayesian data Analysis, second edition, p24.
        eprior = 0;
        gpp=gpcf.p;

        if ~isempty(gpp.constSigma2)
            eprior = feval(gpp.constSigma2.fh_e, gpcf.constSigma2, gpp.constSigma2) - log(gpcf.constSigma2);
        end
    end

    function [DKff, gprior]  = gpcf_constant_ghyper(gpcf, x, x2, mask)  % , t, g, gdata, gprior, varargin
    %GPCF_CONSTANT_GHYPER     Evaluate gradient of covariance function and hyper-prior with 
    %                     respect to the hyperparameters.
    %
    %	Description
    %	[GPRIOR, DKff, DKuu, DKuf] = GPCF_CONSTANT_GHYPER(GPCF, X, T, G, GDATA, GPRIOR, VARARGIN) 
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
    %   GPCF_CONSTANT_PAK, GPCF_CONSTANT_UNPAK, GPCF_CONSTANT_E, GP_G

        gpp=gpcf.p;
        [n, m] =size(x);

        i1=0;
        DKff = {};
        gprior = [];
        
        % Evaluate: DKff{1} = d Kff / d constSigma2
        %           DKff{2} = d Kff / d coeffSigma2
        % NOTE! Here we have already taken into account that the parameters are transformed
        % through log() and thus dK/dlog(p) = p * dK/dp

        
        % evaluate the gradient for training covariance
        if nargin == 2
            
            if ~isempty(gpcf.p.constSigma2)
                DKff{1}=ones(n)*gpcf.constSigma2;
            end
            
        % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
        elseif nargin == 3
            if size(x,2) ~= size(x2,2)
                error('gpcf_constant -> _ghyper: The number of columns in x and x2 has to be the same. ')
            end

            if ~isempty(gpcf.p.constSigma2)
                DKff{1}=ones([n size(x2,1)])*gpcf.constSigma2;
            end
            
            % Evaluate: DKff{1}    = d mask(Kff,I) / d constSigma2
            %           DKff{2...} = d mask(Kff,I) / d coeffSigma2
        elseif nargin == 4

            if ~isempty(gpcf.p.constSigma2)
                DKff{1}=ones(n,1)*gpcf.constSigma2; % d mask(Kff,I) / d constSigma2
            end
        end

        if nargout > 1
            ggs = [];
            if ~isempty(gpcf.p.constSigma2)
                % Evaluate the gprior with respect to magnSigma2
                ggs = feval(gpp.constSigma2.fh_g, gpcf.constSigma2, gpp.constSigma2);
                gprior = ggs(1).*gpcf.constSigma2 - 1;
            end

            if length(ggs) > 1
                gprior = [gprior ggs(2:end)];
            end
        end
    end


    function [DKff, gprior]  = gpcf_constant_ginput(gpcf, x, x2)
    %GPCF_CONSTANT_GIND     Evaluate gradient of covariance function with 
    %                   respect to x.
    %
    %	Description
    %	[GPRIOR_IND, DKuu, DKuf] = GPCF_CONSTANT_GIND(GPCF, X, T, G, GDATA_IND, GPRIOR_IND, VARARGIN) 
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
    %   GPCF_CONSTANT_PAK, GPCF_CONSTANT_UNPAK, GPCF_CONSTANT_E, GP_G
        
        [n, m] =size(x);
        
        if nargin == 2
            
            %K = feval(gpcf.fh_trcov, gpcf, x);
            
%             if length(gpcf.coeffSigma2) == 1
%                 % In the case of an isotropic CONSTANT
%                 s = repmat(gpcf.coeffSigma2, 1, m);
%             else
%                 s = gpcf.coeffSigma2;
%             end
            
            ii1 = 0;
            for i=1:m
                for j = 1:n
                    ii1 = ii1 + 1;
                    DKff{ii1} = zeros(n);
                    gprior(ii1) = 0;
                end
            end
            
        elseif nargin == 3
            %K = feval(gpcf.fh_cov, gpcf, x, x2);
            
            ii1 = 0;
            for i=1:m
                for j = 1:n
                    ii1 = ii1 + 1;
                    DKff{ii1} = zeros(n, size(x2,1));
                    gprior(ii1) = 0; 
                end
            end
        end
    end


    function C = gpcf_constant_cov(gpcf, x1, x2, varargin)
    % GP_CONSTANT_COV     Evaluate covariance matrix between two input vectors.
    %
    %         Description
    %         C = GP_CONSTANT_COV(GP, TX, X) takes in covariance function of a Gaussian
    %         process GP and two matrixes TX and X that contain input vectors to
    %         GP. Returns covariance matrix C. Every element ij of C contains
    %         covariance between inputs i in TX and j in X.
    %
    %
    %         See also
    %         GPCF_CONSTANT_TRCOV, GPCF_CONSTANT_TRVAR, GP_COV, GP_TRCOV
        
        if isempty(x2)
            x2=x1;
        end
        [n1,m1]=size(x1);
        [n2,m2]=size(x2);

        if m1~=m2
            error('the number of columns of X1 and X2 has to be same')
        end

        C = ones(n1,n2)*gpcf.constSigma2;
    end

    function C = gpcf_constant_trcov(gpcf, x)
    % GP_CONSTANT_TRCOV     Evaluate training covariance matrix of inputs.
    %
    %         Description
    %         C = GP_CONSTANT_TRCOV(GP, TX) takes in covariance function of a Gaussian
    %         process GP and matrix TX that contains training input vectors. 
    %         Returns covariance matrix C. Every element ij of C contains covariance 
    %         between inputs i and j in TX
    %
    %
    %         See also
    %         GPCF_CONSTANT_COV, GPCF_CONSTANT_TRVAR, GP_COV, GP_TRCOV

        n =size(x,1);
        C = ones(n,n)*gpcf.constSigma2;

    end


    function C = gpcf_constant_trvar(gpcf, x)
    % GP_CONSTANT_TRVAR     Evaluate training variance vector
    %
    %         Description
    %         C = GP_CONSTANT_TRVAR(GPCF, TX) takes in covariance function of a Gaussian
    %         process GPCF and matrix TX that contains training inputs. Returns variance 
    %         vector C. Every element i of C contains variance of input i in TX
    %
    %
    %         See also
    %         GPCF_CONSTANT_COV, GPCF_CONSTANT_COVVEC, GP_COV, GP_TRCOV
                

        n =size(x,1);
        C = ones(n,1)*gpcf.constSigma2;
        
    end

    function reccf = gpcf_constant_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_CONSTANT_RECAPPEND(RECCF, RI, GPCF) takes old covariance
    %          function record RECCF, record index RI and covariance function structure. 
    %          Appends the parameters of GPCF to the RECCF in the ri'th place.
    %
    %          RECAPPEND returns a structure RECCF containing following record fields:
    %          lengthHyper    
    %          lengthHyperNu  
    %          coeffSigma2    
    %          constSigma2     
    %
    %          See also
    %          GP_MC and GP_MC -> RECAPPEND

    % Initialize record
        if nargin == 2
            reccf.type = 'gpcf_constant';
            reccf.nin = ri;
            reccf.nout = 1;

            % Initialize parameters
            reccf.constSigma2 = [];

            % Set the function handles
            reccf.fh_pak = @gpcf_constant_pak;
            reccf.fh_unpak = @gpcf_constant_unpak;
            reccf.fh_e = @gpcf_constant_e;
            reccf.fh_g = @gpcf_constant_g;
            reccf.fh_cov = @gpcf_constant_cov;
            reccf.fh_trcov  = @gpcf_constant_trcov;
            reccf.fh_trvar  = @gpcf_constant_trvar;
            reccf.fh_recappend = @gpcf_constant_recappend;
            gpcf.p=[];
            gpcf.p.constSigma2=[];
            if ~isempty(ri.p.constSigma2)
                reccf.p.constSigma2 = ri.p.constSigma2;
            end

            return
        end

        % record constSigma2
        if ~isempty(gpcf.constSigma2)
            reccf.constSigma2(ri,:)=gpcf.constSigma2;
        elseif ri==1
            reccf.constSigma2=[];
        end
    end
end