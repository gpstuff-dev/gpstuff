function gpcf = gpcf_dotproduct(do, varargin)
%GPCF_DOTPRODUCT	Create a dot product covariance function for Gaussian Process
%
%	Description
%
%	GPCF = GPCF_DOTPRODUCT('INIT', NIN) Create and initialize dot product
%       covariance function for Gaussian process
%
%	The fields and (default values) in GPCF_DOTPRODUCT are:
%	  type           = 'gpcf_dotproduct'
%	  nin            = Number of inputs. (NIN)
%	  nout           = Number of outputs. (always 1)
%	  constSigma2     = Prior variance on the constant term.
%                          (0.1)
%	  coeffSigma2    = Prior variances on the linear coefficients. This can be
%                      either scalar corresponding 
%                      isotropic or vector corresponding ARD. 
%                      (repmat(10, 1, nin))
%         p              = Prior structure for covariance function parameters. 
%                          (e.g. p.coeffSigma2.)
%         fh_pak         = function handle to pack function
%                          (@gpcf_dotproduct_pak)
%         fh_unpak       = function handle to unpack function
%                          (@gpcf_dotproduct_unpak)
%         fh_e           = function handle to energy function
%                          (@gpcf_dotproduct_e)
%         fh_ghyper      = function handle to gradient of energy with respect to hyperparameters
%                          (@gpcf_dotproduct_ghyper)
%         fh_ginput      = function handle to gradient of function with respect to inducing inputs
%                          (@gpcf_dotproduct_ginput)
%         fh_cov         = function handle to covariance function
%                          (@gpcf_dotproduct_cov)
%         fh_trcov       = function handle to training covariance function
%                          (@gpcf_dotproduct_trcov)
%         fh_trvar       = function handle to training variance function
%                          (@gpcf_dotproduct_trvar)
%         fh_recappend   = function handle to append the record function 
%                          (gpcf_dotproduct_recappend)
%
%	GPCF = GPCF_DOTPRODUCT('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in GPCF.
%
%	See also
%       gpcf_exp, gpcf_matern32, gpcf_matern52, gpcf_ppcs2, gp_init, gp_e, gp_g, gp_trcov
%       gp_cov, gp_unpak, gp_pak
    
% Copyright (c) 2000-2001 Aki Vehtari
% Copyright (c) 2007-2008 Jarno Vanhatalo
% Copyright (c) 2008      Jaakko Riihimaki

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end

    % Initialize the covariance function
    if strcmp(do, 'init')
        nin = varargin{1};
        gpcf.type = 'gpcf_dotproduct';
        gpcf.nin = nin;
        gpcf.nout = 1;

        % Initialize parameters
        gpcf.coeffSigma2= repmat(10, 1, nin);
        gpcf.constSigma2 = 0.1;

        % Initialize prior structure
        gpcf.p=[];
        gpcf.p.coeffSigma2=[];
        gpcf.p.constSigma2=[];

        % Set the function handles to the nested functions
        gpcf.fh_pak = @gpcf_dotproduct_pak;
        gpcf.fh_unpak = @gpcf_dotproduct_unpak;
        gpcf.fh_e = @gpcf_dotproduct_e;
        gpcf.fh_ghyper = @gpcf_dotproduct_ghyper;
        gpcf.fh_ginput = @gpcf_dotproduct_ginput;
        gpcf.fh_cov = @gpcf_dotproduct_cov;
        gpcf.fh_covvec = @gpcf_dotproduct_covvec;
        gpcf.fh_trcov  = @gpcf_dotproduct_trcov;
        gpcf.fh_trvar  = @gpcf_dotproduct_trvar;
        gpcf.fh_recappend = @gpcf_dotproduct_recappend;

        if length(varargin) > 1
            if mod(nargin,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=2:2:length(varargin)-1
                switch varargin{i}
                  case 'constSigma2'
                    gpcf.constSigma2 = varargin{i+1};
                  case 'coeffSigma2'
                    gpcf.coeffSigma2 = varargin{i+1};
                  case 'fh_sampling'
                    gpcf.fh_sampling = varargin{i+1};
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
              case 'coeffSigma2'
                gpcf.coeffSigma2 = varargin{i+1};
              case 'fh_sampling'
                gpcf.fh_sampling = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end

    function w = gpcf_dotproduct_pak(gpcf, w)
    %GPCF_DOTPRODUCT_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %	W = GPCF_DOTPRODUCT_PAK(GPCF, W) takes a covariance function data structure GPCF and
    %	combines the hyper-parameters into a single row vector W.
    %
    %	The ordering of the parameters in W is:
    %       w = [gpcf.constSigma2 (hyperparameters of gpcf.coeffSigma2) gpcf.coeffSigma2]
    %	  
    %
    %	See also
    %	GPCF_DOTPRODUCT_UNPAK
        gpp=gpcf.p;

        i1=0;i2=1;
        if ~isempty(w)
            i1 = length(w);
        end
        i1 = i1+1;
        w(i1) = gpcf.constSigma2;
        i2=i1+length(gpcf.coeffSigma2);
        i1=i1+1;
        w(i1:i2)=gpcf.coeffSigma2;
        i1=i2;
        
        % Hyperparameters of coeffSigma2
        if isfield(gpp.coeffSigma2, 'p') && ~isempty(gpp.coeffSigma2.p)
            i1=i1+1;
            w(i1)=gpp.coeffSigma2.a.s;
            if any(strcmp(fieldnames(gpp.coeffSigma2.p),'nu'))
                i1=i1+1;
                w(i1)=gpp.coeffSigma2.a.nu;
            end
        end
    end




    function [gpcf, w] = gpcf_dotproduct_unpak(gpcf, w)
    %GPCF_DOTPRODUCT_UNPAK  Separate covariance function hyper-parameter vector into components.
    %
    %	Description
    %	[GPCF, W] = GPCF_DOTPRODUCT_UNPAK(GPCF, W) takes a covariance function data structure GPCF
    %	and  a hyper-parameter vector W, and returns a covariance function data
    %	structure  identical to the input, except that the covariance hyper-parameters 
    %   has been set to the values in W. Deletes the values set to GPCF from W and returns 
    %   the modeified W. 
    %
    %	See also
    %	GPCF_DOTPRODUCT_PAK
    %
        gpp=gpcf.p;
        i1=0;i2=1;
        i1=i1+1;
        gpcf.constSigma2=w(i1);
        i2=i1+length(gpcf.coeffSigma2);
        i1=i1+1;
        gpcf.coeffSigma2=w(i1:i2);
        i1=i2;
        % Hyperparameters of coeffSigma2
        if isfield(gpp.coeffSigma2, 'p') && ~isempty(gpp.coeffSigma2.p)
            i1=i1+1;
            gpcf.p.coeffSigma2.a.s=w(i1);
            if any(strcmp(fieldnames(gpp.coeffSigma2.p),'nu'))
                i1=i1+1;
                gpcf.p.coeffSigma2.a.nu=w(i1);
            end
        end        
        w = w(i1+1:end);
    end

    function eprior =gpcf_dotproduct_e(gpcf, x, t)
    %GPCF_DOTPRODUCT_E     Evaluate the energy of prior of DOTPRODUCT parameters
    %
    %	Description
    %	E = GPCF_DOTPRODUCT_E(GPCF, X, T) takes a covariance function data structure 
    %   GPCF together with a matrix X of input vectors and a matrix T of target 
    %   vectors and evaluates log p(th) x J, where th is a vector of DOTPRODUCT parameters 
    %   and J is the Jakobian of transformation exp(w) = th. (Note that the parameters 
    %   are log transformed, when packed.)
    %
    %	See also
    %	GPCF_DOTPRODUCT_PAK, GPCF_DOTPRODUCT_UNPAK, GPCF_DOTPRODUCT_G, GP_E
    %
        [n, m] =size(x);

        % Evaluate the prior contribution to the error. The parameters that
        % are sampled are from space W = log(w) where w is all the "real" samples.
        % On the other hand errors are evaluated in the W-space so we need take
        % into account also the  Jakobian of transformation W -> w = exp(W).
        % See Gelman et.all., 2004, Bayesian data Analysis, second edition, p24.
        eprior = 0;
        gpp=gpcf.p;

        eprior=eprior...
               +feval(gpp.constSigma2.fe, ...
                      gpcf.constSigma2, gpp.constSigma2.a)...
               -log(gpcf.constSigma2);
        if isfield(gpp.coeffSigma2, 'p') && ~isempty(gpp.coeffSigma2.p)
            eprior=eprior...
                   +feval(gpp.coeffSigma2.p.s.fe, ...
                          gpp.coeffSigma2.a.s, gpp.coeffSigma2.p.s.a)...
                   -log(gpp.coeffSigma2.a.s);
            if any(strcmp(fieldnames(gpp.coeffSigma2.p),'nu'))
                eprior=eprior...
                       +feval(gpp.p.coeffSigma2.nu.fe, ...
                              gpp.coeffSigma2.a.nu, gpp.coeffSigma2.p.nu.a)...
                       -log(gpp.coeffSigma2.a.nu);
            end
        end
        eprior=eprior...
               +feval(gpp.coeffSigma2.fe, ...
                      gpcf.coeffSigma2, gpp.coeffSigma2.a)...
               -sum(log(gpcf.coeffSigma2));

    end

    function [DKff, gprior]  = gpcf_dotproduct_ghyper(gpcf, x, x2, mask)  % , t, g, gdata, gprior, varargin
    %GPCF_DOTPRODUCT_GHYPER     Evaluate gradient of covariance function and hyper-prior with 
    %                     respect to the hyperparameters.
    %
    %	Description
    %	[GPRIOR, DKff, DKuu, DKuf] = GPCF_DOTPRODUCT_GHYPER(GPCF, X, T, G, GDATA, GPRIOR, VARARGIN) 
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
    %   GPCF_DOTPRODUCT_PAK, GPCF_DOTPRODUCT_UNPAK, GPCF_DOTPRODUCT_E, GP_G

        gpp=gpcf.p;
        [n, m] =size(x);

        i1=0;i2=1;
        
        % Evaluate: DKff{1} = d Kff / d constSigma2
        %           DKff{2} = d Kff / d coeffSigma2
        % NOTE! Here we have already taken into account that the parameters are transformed
        % through log() and thus dK/dlog(p) = p * dK/dp

        
        % evaluate the gradient for training covariance
        if nargin == 2
            
            DKff{1}=ones(n)*gpcf.constSigma2;
            if length(gpcf.coeffSigma2) == 1
                DKff{2}=gpcf.coeffSigma2*x*(x');
            else
                for i=1:m
                    DD = gpcf.coeffSigma2(i)*x(:,i)*(x(:,i)');
                    DD(abs(DD)<=eps) = 0;
                    DKff{1+i}= (DD+DD')./2;
                end
            end
            
            
        % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
        elseif nargin == 3
            if size(x,2) ~= size(x2,2)
                error('gpcf_dotproduct -> _ghyper: The number of columns in x and x2 has to be the same. ')
            end
            
            DKff{1}=ones([n size(x2,1)])*gpcf.constSigma2;
            if length(gpcf.coeffSigma2) == 1
                DKff{2}=gpcf.coeffSigma2*x*(x2');
            else
                for i=1:m
                    DKff{1+i}=gpcf.coeffSigma2(i)*x(:,i)*(x2(:,i)');
                end
            end
            
            
            % Evaluate: DKff{1}    = d mask(Kff,I) / d constSigma2
            %           DKff{2...} = d mask(Kff,I) / d coeffSigma2
        elseif nargin == 4

            DKff{1}=ones(n,1)*gpcf.constSigma2; % d mask(Kff,I) / d constSigma2
            if length(gpcf.coeffSigma2) == 1
                DKff{2}=gpcf.coeffSigma2*sum(x.^2,2); % d mask(Kff,I) / d coeffSigma2
            else
                for i=1:m
                    DKff{1+i}=gpcf.coeffSigma2(i)*(x(:,i).^2); % d mask(Kff,I) / d coeffSigma2
                end
            end
            

        end

        if nargout > 1
            % Evaluate the gprior with respect to constSigma2
            i1 = i1+1;
            gprior(i1)=feval(gpp.constSigma2.fg, ...
                             gpcf.constSigma2, ...
                             gpp.constSigma2.a, 'x').*gpcf.constSigma2 - 1;
            % Evaluate the data contribution of gradient with respect to coeffSigma2
            if length(gpcf.coeffSigma2)>1
                for i2=1:gpcf.nin
                    i1=i1+1;
                    gprior(i1)=feval(gpp.coeffSigma2.fg, ...
                                     gpcf.coeffSigma2(i2), ...
                                     gpp.coeffSigma2.a, 'x').*gpcf.coeffSigma2(i2) - 1;
                end
            else
                i1=i1+1;
                gprior(i1)=feval(gpp.coeffSigma2.fg, ...
                                 gpcf.coeffSigma2, ...
                                 gpp.coeffSigma2.a, 'x').*gpcf.coeffSigma2 -1;
            end
            % Evaluate the prior contribution of gradient with respect to coeffSigma2.p.s (and coeffSigma2.p.nu)
            if isfield(gpp.coeffSigma2, 'p') && ~isempty(gpp.coeffSigma2.p)
                i1=i1+1;
                gprior(i1)=...
                    feval(gpp.coeffSigma2.p.s.fg, ...
                          gpp.coeffSigma2.a.s,...
                          gpp.coeffSigma2.p.s.a, 'x').*gpp.coeffSigma2.a.s - 1 ...
                    +feval(gpp.coeffSigma2.fg, ...
                           gpcf.coeffSigma2, ...
                           gpp.coeffSigma2.a, 's').*gpp.coeffSigma2.a.s;
                if any(strcmp(fieldnames(gpp.coeffSigma2.p),'nu'))
                    i1=i1+1;
                    gprior(i1)=...
                        feval(gpp.coeffSigma2.p.nu.fg, ...
                              gpp.coeffSigma2.a.nu,...
                              gpp.coeffSigma2.p.nu.a, 'x').*gpp.coeffSigma2.a.nu -1 ...
                        +feval(gpp.coeffSigma2.fg, ...
                               gpcf.coeffSigma2, ...
                               gpp.coeffSigma2.a, 'nu').*gpp.coeffSigma2.a.nu;
                end
            end
        end
    end


    function [DKff, gprior]  = gpcf_dotproduct_ginput(gpcf, x, x2)
    %GPCF_DOTPRODUCT_GIND     Evaluate gradient of covariance function with 
    %                   respect to x.
    %
    %	Description
    %	[GPRIOR_IND, DKuu, DKuf] = GPCF_DOTPRODUCT_GIND(GPCF, X, T, G, GDATA_IND, GPRIOR_IND, VARARGIN) 
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
    %   GPCF_DOTPRODUCT_PAK, GPCF_DOTPRODUCT_UNPAK, GPCF_DOTPRODUCT_E, GP_G
        
        [n, m] =size(x);
        
        if nargin == 2
            
            %K = feval(gpcf.fh_trcov, gpcf, x);
            
            if length(gpcf.coeffSigma2) == 1
                % In the case of an isotropic DOTPRODUCT
                s = repmat(gpcf.coeffSigma2, 1, m);
            else
                s = gpcf.coeffSigma2;
            end
            
            ii1 = 0;
            for i=1:m
                for j = 1:n

                    DK = zeros(n);
                    DK(j,:)=s(i)*x(:,i)';
                    
                    DK = DK + DK';
                    
                    ii1 = ii1 + 1;
                    DKff{ii1} = DK;
                    gprior(ii1) = 0;
                end
            end
            
        elseif nargin == 3
            %K = feval(gpcf.fh_cov, gpcf, x, x2);
            
            if length(gpcf.coeffSigma2) == 1
                % In the case of an isotropic DOTPRODUCT
                s = repmat(gpcf.coeffSigma2, 1, m);
            else
                s = gpcf.coeffSigma2;
            end
            
            ii1 = 0;
            for i=1:m
                for j = 1:n
                    
                    DK = zeros(n, size(x2,1));
                    DK(j,:)=s(i)*x2(:,i)';
                    
                    ii1 = ii1 + 1;
                    DKff{ii1} = DK;
                    gprior(ii1) = 0; 
                end
            end
        end
    end


    function C = gpcf_dotproduct_cov(gpcf, x1, x2, varargin)
    % GP_DOTPRODUCT_COV     Evaluate covariance matrix between two input vectors.
    %
    %         Description
    %         C = GP_DOTPRODUCT_COV(GP, TX, X) takes in covariance function of a Gaussian
    %         process GP and two matrixes TX and X that contain input vectors to
    %         GP. Returns covariance matrix C. Every element ij of C contains
    %         covariance between inputs i in TX and j in X.
    %
    %
    %         See also
    %         GPCF_DOTPRODUCT_TRCOV, GPCF_DOTPRODUCT_TRVAR, GP_COV, GP_TRCOV
        
        if isempty(x2)
            x2=x1;
        end
        [n1,m1]=size(x1);
        [n2,m2]=size(x2);

        if m1~=m2
            error('the number of columns of X1 and X2 has to be same')
        end

        C = x1*diag(gpcf.coeffSigma2)*(x2');
        C(abs(C)<=eps) = 0;
        C = gpcf.constSigma2+C;
    end

    function C = gpcf_dotproduct_trcov(gpcf, x)
    % GP_DOTPRODUCT_TRCOV     Evaluate training covariance matrix of inputs.
    %
    %         Description
    %         C = GP_DOTPRODUCT_TRCOV(GP, TX) takes in covariance function of a Gaussian
    %         process GP and matrix TX that contains training input vectors. 
    %         Returns covariance matrix C. Every element ij of C contains covariance 
    %         between inputs i and j in TX
    %
    %
    %         See also
    %         GPCF_DOTPRODUCT_COV, GPCF_DOTPRODUCT_TRVAR, GP_COV, GP_TRCOV

        C = x*diag(gpcf.coeffSigma2)*(x');
        C(abs(C)<=eps) = 0;
        C = gpcf.constSigma2+C;
        C = (C+C')./2;

    end

    function C = gpcf_dotproduct_covvec(gpcf, x1, x2, varargin)
    % GPCF_DOTPRODUCT_COVVEC     Evaluate covariance vector between two input vectors.
    %
    %         Description
    %         C = GPCF_DOTPRODUCT_COVVEC(GP, TX, X) takes in Gaussian process GP and two
    %         matrixes TX and X that contain input vectors to GP. Returns
    %         covariance vector C, where every element i of C contains covariance
    %         between input i in TX and i in X.
    %
    %
    %         See also
    %         GPCF_DOTPRODUCT_COV, GPCF_DOTPRODUCT_TRVAR, GP_COV, GP_TRCOV

    error('Should not end up here! Not implemented...')
        
    end

    function C = gpcf_dotproduct_trvar(gpcf, x)
    % GP_DOTPRODUCT_TRVAR     Evaluate training variance vector
    %
    %         Description
    %         C = GP_DOTPRODUCT_TRVAR(GPCF, TX) takes in covariance function of a Gaussian
    %         process GPCF and matrix TX that contains training inputs. Returns variance 
    %         vector C. Every element i of C contains variance of input i in TX
    %
    %
    %         See also
    %         GPCF_DOTPRODUCT_COV, GPCF_DOTPRODUCT_COVVEC, GP_COV, GP_TRCOV
                

        if length(gpcf.coeffSigma2) == 1
            C=gpcf.coeffSigma2.*sum(x.^2,2)+gpcf.constSigma2;
        else
            C=sum(repmat(gpcf.coeffSigma2, size(x,1), 1).*x.^2,2)+gpcf.constSigma2;
        end
        C(abs(C)<eps)=0;
  
    end

    function reccf = gpcf_dotproduct_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_DOTPRODUCT_RECAPPEND(RECCF, RI, GPCF) takes old covariance
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
            reccf.type = 'gpcf_dotproduct';
            reccf.nin = ri;
            reccf.nout = 1;

            % Initialize parameters
            reccf.coeffSigma2= [];
            reccf.constSigma2 = [];

            % Set the function handles
            reccf.fh_pak = @gpcf_dotproduct_pak;
            reccf.fh_unpak = @gpcf_dotproduct_unpak;
            reccf.fh_e = @gpcf_dotproduct_e;
            reccf.fh_g = @gpcf_dotproduct_g;
            reccf.fh_cov = @gpcf_dotproduct_cov;
            reccf.fh_trcov  = @gpcf_dotproduct_trcov;
            reccf.fh_trvar  = @gpcf_dotproduct_trvar;
            reccf.fh_recappend = @gpcf_dotproduct_recappend;
            return
        end

        gpp = gpcf.p;
        % record coeffSigma2
        if ~isempty(gpcf.coeffSigma2)
            if ~isempty(gpp.coeffSigma2)
                reccf.lengthHyper(ri,:)=gpp.coeffSigma2.a.s;
                if isfield(gpp.coeffSigma2,'p')
                    if isfield(gpp.coeffSigma2.p,'nu')
                        reccf.lengthHyperNu(ri,:)=gpp.coeffSigma2.a.nu;
                    end
                end
            elseif ri==1
                reccf.lengthHyper=[];
            end
            reccf.coeffSigma2(ri,:)=gpcf.coeffSigma2;
        elseif ri==1
            reccf.coeffSigma2=[];
        end
        % record constSigma2
        if ~isempty(gpcf.constSigma2)
            reccf.constSigma2(ri,:)=gpcf.constSigma2;
        elseif ri==1
            reccf.constSigma2=[];
        end
    end
end