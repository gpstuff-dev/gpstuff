function gpcf = gpcf_prod(do, varargin)
%GPCF_PROD	Create a product form covariance function for Gaussian Process
%
%	Description
%
%	GPCF = GPCF_PROD('INIT', NIN, 'functions', {GPCF_1, GPCF_2, ...}) Create and 
%       initialize product form covariance function for Gaussian process. The covariance 
%       will be 
%          GPCF = GPCF_1 .* GPCF_2 .* ... .* GPCF_N
%
%	The fields and (default values) in GPCF_PROD are:
%	  type           = 'gpcf_prod'
%	  nin            = Number of inputs. (NIN)
%	  nout           = Number of outputs. (always 1)
%         functions      = cell array containing the covariance functions to be multiplied
%         fh_pak         = function handle to pack function
%                          (@gpcf_prod_pak)
%         fh_unpak       = function handle to unpack function
%                          (@gpcf_prod_unpak)
%         fh_e           = function handle to energy function
%                          (@gpcf_prod_e)
%         fh_ghyper      = function handle to gradient of energy with respect to hyperparameters
%                          (@gpcf_prod_ghyper)
%         fh_ginput      = function handle to gradient of function with respect to inducing inputs
%                          (@gpcf_prod_ginput)
%         fh_cov         = function handle to covariance function
%                          (@gpcf_prod_cov)
%         fh_trcov       = function handle to training covariance function
%                          (@gpcf_prod_trcov)
%         fh_trvar       = function handle to training variance function
%                          (@gpcf_prod_trvar)
%         fh_recappend   = function handle to append the record function 
%                          (gpcf_prod_recappend)
%
%	GPCF = GPCF_PROD('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in GPCF.
%
%	See also
%       gpcf_exp, gpcf_matern32, gpcf_matern52, gpcf_ppcs2, gp_init, gp_e, gp_g, gp_trcov
%       gp_cov, gp_unpak, gp_pak
    
% Copyright (c) 2000-2001 Aki Vehtari
% Copyright (c) 2007-2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end

    % Initialize the covariance function
    if strcmp(do, 'init')
        nin = varargin{1};
        gpcf.type = 'gpcf_prod';
        gpcf.nin = nin;
        gpcf.nout = 1;

        % Initialize parameters
        gpcf.functions = {};
        
        % Set the function handles to the nested functions
        gpcf.fh_pak = @gpcf_prod_pak;
        gpcf.fh_unpak = @gpcf_prod_unpak;
        gpcf.fh_e = @gpcf_prod_e;
        gpcf.fh_ghyper = @gpcf_prod_ghyper;
        gpcf.fh_ginput = @gpcf_prod_ginput;
        gpcf.fh_cov = @gpcf_prod_cov;
        gpcf.fh_trcov  = @gpcf_prod_trcov;
        gpcf.fh_trvar  = @gpcf_prod_trvar;
        gpcf.fh_recappend = @gpcf_prod_recappend;

        if length(varargin) > 1
            if mod(nargin,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=2:2:length(varargin)-1
                switch varargin{i}
                  case 'functions'
                    % Set covariance functions into gpcf
                    cfs = varargin{i+1};
                    for i = 1:length(cfs)
                        gpcf.functions{i} = cfs{i};
                    end
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
              case 'functions'
                % Set covariance functions into gpcf
                cfs = varargin{i+1};
                for i = 1:length(cfs)
                    gpcf.functions{i} = cfs{i};
                end
              otherwise
                error('Wrong parameter name!')
            end
        end
    end

    
    function w = gpcf_prod_pak(gpcf, w)
    %GPCF_PROD_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %	W = GPCF_PROD_PAK(GPCF, W) takes a covariance function data structure GPCF and
    %	combines the hyper-parameters into a single row vector W.
    %
    %	The ordering of the parameters in W is:
    %       w = [gpcf.magnSigma2 (hyperparameters of gpcf.lengthScale) gpcf.lengthScale]
    %	  
    %
    %	See also
    %	GPCF_PROD_UNPAK
        
        ncf = length(gpcf.functions);
        
        for i=1:ncf
            cf = gpcf.functions{i};
            w = feval(cf.fh_pak, cf, w);
        end

        
    end




    function [gpcf, w] = gpcf_prod_unpak(gpcf, w)
    %GPCF_PROD_UNPAK  Separate covariance function hyper-parameter vector into components.
    %
    %	Description
    %	[GPCF, W] = GPCF_PROD_UNPAK(GPCF, W) takes a covariance function data structure GPCF
    %	and  a hyper-parameter vector W, and returns a covariance function data
    %	structure  identical to the input, except that the covariance hyper-parameters 
    %   has been set to the values in W. Deletes the values set to GPCF from W and returns 
    %   the modeified W. 
    %
    %	See also
    %	GPCF_PROD_PAK
    %
        ncf = length(gpcf.functions);
        
        for i=1:ncf
            cf = gpcf.functions{i};
            [cf, w] = feval(cf.fh_unpak, cf, w);
            gpcf.functions{i} = cf;
        end

    end
    
    function eprior =gpcf_prod_e(gpcf, x, t)
    %GPCF_PROD_E     Evaluate the energy of prior of PROD parameters
    %
    %	Description
    %	E = GPCF_PROD_E(GPCF, X, T) takes a covariance function data structure 
    %   GPCF together with a matrix X of input vectors and a matrix T of target 
    %   vectors and evaluates log p(th) x J, where th is a vector of PROD parameters 
    %   and J is the Jakobian of transformation exp(w) = th. (Note that the parameters 
    %   are log transformed, when packed.)
    %
    %	See also
    %	GPCF_PROD_PAK, GPCF_PROD_UNPAK, GPCF_PROD_G, GP_E
    %
        
        eprior = 0;
        ncf = length(gpcf.functions);
        for i=1:ncf
            cf = gpcf.functions{i};
            eprior = eprior + feval(cf.fh_e, cf, x, t);
        end
        
    end

    function [DKff, gprior]  = gpcf_prod_ghyper(gpcf, x, x2, mask)  % , t, g, gdata, gprior, varargin
    %GPCF_PROD_GHYPER     Evaluate gradient of covariance function and hyper-prior with 
    %                     respect to the hyperparameters.
    %
    %	Description
    %	[GPRIOR, DKff, DKuu, DKuf] = GPCF_PROD_GHYPER(GPCF, X, T, G, GDATA, GPRIOR, VARARGIN) 
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
    %   GPCF_PROD_PAK, GPCF_PROD_UNPAK, GPCF_PROD_E, GP_G

        [n, m] =size(x);

        gprior = [];
        % Evaluate: DKff{1} = d Kff / d magnSigma2
        %           DKff{2} = d Kff / d lengthScale
        % NOTE! Here we have already taken into account that the parameters are transformed
        % through log() and thus dK/dlog(p) = p * dK/dp

        % evaluate the gradient for training covariance
        if nargin == 2
            
            ncf = length(gpcf.functions);
            
            % evaluate the individual covariance functions
            for i=1:ncf
                cf = gpcf.functions{i};
                C{i} = feval(cf.fh_trcov, cf, x);
            end
            
            % Evaluate the gradients
            ind = 1:ncf;
            DKff = {};
            for i=1:ncf
                cf = gpcf.functions{i};
                [DK, gpr] = feval(cf.fh_ghyper, cf, x);
                gprior = [gprior gpr];
                
                CC = 1;
                for kk = ind(ind~=i)
                    CC = CC.*C{kk};
                end
                
                for j = 1:length(DK)
                    DKff{end+1} = DK{j}.*CC;
                end
            end
            
            % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
        elseif nargin == 3
            if size(x,2) ~= size(x2,2)
                error('gpcf_prod -> _ghyper: The number of columns in x and x2 has to be the same. ')
            end
            
            ncf = length(gpcf.functions);
            
            % evaluate the individual covariance functions
            for i=1:ncf
                cf = gpcf.functions{i};
                C{i} = feval(cf.fh_cov, cf, x, x2);
            end
            
            % Evaluate the gradients
            ind = 1:ncf;
            DKff = {};
            for i=1:ncf
                cf = gpcf.functions{i};
                [DK, gpr] = feval(cf.fh_ghyper, cf, x, x2);
                gprior = [gprior gpr];
                
                CC = 1;
                for kk = ind(ind~=i)
                    CC = CC.*C{kk};
                end
                
                for j = 1:length(DK)
                    DKff{end+1} = DK{j}.*CC;
                end
            end

            
            
            % Evaluate: DKff{1}    = d mask(Kff,I) / d magnSigma2
            %           DKff{2...} = d mask(Kff,I) / d lengthScale
        elseif nargin == 4
            ncf = length(gpcf.functions);
            
            % evaluate the individual covariance functions
            for i=1:ncf
                cf = gpcf.functions{i};
                C{i} = feval(cf.fh_trvar, cf, x);
            end
            
            % Evaluate the gradients
            ind = 1:ncf;
            DKff = {};
            for i=1:ncf
                cf = gpcf.functions{i};
                [DK, gpr] = feval(cf.fh_ghyper, cf, [], 1);
                gprior = [gprior gpr;]
                
                CC = 1;
                for kk = ind(ind~=i)
                    CC = CC.*C{kk};
                end
                
                for j = 1:length(DK)
                    DKff{end+1} = DK{j}.*CC;
                end
            end
        end
    end


    function [DKff, gprior]  = gpcf_prod_ginput(gpcf, x, x2)
    %GPCF_PROD_GIND     Evaluate gradient of covariance function with 
    %                   respect to x.
    %
    %	Descriptioni
    %	[GPRIOR_IND, DKuu, DKuf] = GPCF_PROD_GIND(GPCF, X, T, G, GDATA_IND, GPRIOR_IND, VARARGIN) 
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
    %   GPCF_PROD_PAK, GPCF_PROD_UNPAK, GPCF_PROD_E, GP_G
        
        [n, m] =size(x);
        gprior = [];
        % Evaluate: DKff{1} = d Kff / d magnSigma2
        %           DKff{2} = d Kff / d lengthScale
        % NOTE! Here we have already taken into account that the parameters are transformed
        % through log() and thus dK/dlog(p) = p * dK/dp

        % evaluate the gradient for training covariance
        if nargin == 2
            
            ncf = length(gpcf.functions);
            
            % evaluate the individual covariance functions
            for i=1:ncf
                cf = gpcf.functions{i};
                C{i} = feval(cf.fh_trcov, cf, x);
            end
            
            % Evaluate the gradients
            ind = 1:ncf;
            for i=1:ncf
                cf = gpcf.functions{i};
                [DK, gpr] = feval(cf.fh_g, cf, x);
                gprior = [gprior gpr];
                
                CC = 1;
                for kk = ind(ind~=i)
                    CC = CC.*C{kk};
                end
                
                for j = 1:length(DK)
                    DKff{i+j-1} = DK{j}.*CC;
                end
            end

            % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
        elseif nargin == 3
            if size(x,2) ~= size(x2,2)
                error('gpcf_prod -> _ghyper: The number of columns in x and x2 has to be the same. ')
            end
            
            ncf = length(gpcf.functions);
            
            % evaluate the individual covariance functions
            for i=1:ncf
                cf = gpcf.functions{i};
                C{i} = feval(cf.fh_cov, cf, x, x2);
            end
            
            % Evaluate the gradients
            ind = 1:ncf;
            for i=1:ncf
                cf = gpcf.functions{i};
                [DK, gpr] = feval(cf.fh_g, cf, x, x2);
                gprior = [gprior gpr];
                
                CC = 1;
                for kk = ind(ind~=i)
                    CC = CC.*C{kk};
                end
                
                for j = 1:length(DK)
                    DKff{i+j-1} = DK{j}.*CC;
                end
            end
        end
        
    end


    function C = gpcf_prod_cov(gpcf, x1, x2)
    % GP_PROD_COV     Evaluate covariance matrix between two input vectors.
    %
    %         Description
    %         C = GP_PROD_COV(GP, TX, X) takes in covariance function of a Gaussian
    %         process GP and two matrixes TX and X that contain input vectors to
    %         GP. Returns covariance matrix C. Every element ij of C contains
    %         covariance between inputs i in TX and j in X.
    %
    %
    %         See also
    %         GPCF_PROD_TRCOV, GPCF_PROD_TRVAR, GP_COV, GP_TRCOV
        
        if isempty(x2)
            x2=x1;
        end
        [n1,m1]=size(x1);
        [n2,m2]=size(x2);

        if m1~=m2
            error('the number of columns of X1 and X2 has to be same')
        end

        ncf = length(gpcf.functions);
        
        % evaluate the individual covariance functions
        C = 1;
        for i=1:ncf
            cf = gpcf.functions{i};
            C = C.*feval(cf.fh_cov, cf, x1, x2);
        end        
    end

    function C = gpcf_prod_trcov(gpcf, x)
    % GP_PROD_TRCOV     Evaluate training covariance matrix of inputs.
    %
    %         Description
    %         C = GP_PROD_TRCOV(GP, TX) takes in covariance function of a Gaussian
    %         process GP and matrix TX that contains training input vectors. 
    %         Returns covariance matrix C. Every element ij of C contains covariance 
    %         between inputs i and j in TX
    %
    %
    %         See also
    %         GPCF_PROD_COV, GPCF_PROD_TRVAR, GP_COV, GP_TRCOV

        ncf = length(gpcf.functions);
        
        % evaluate the individual covariance functions
        C = 1;
        for i=1:ncf
            cf = gpcf.functions{i};
            C = C.*feval(cf.fh_trcov, cf, x);
        end
    end

    function C = gpcf_prod_trvar(gpcf, x)
    % GP_PROD_TRVAR     Evaluate training variance vector
    %
    %         Description
    %         C = GP_PROD_TRVAR(GPCF, TX) takes in covariance function of a Gaussian
    %         process GPCF and matrix TX that contains training inputs. Returns variance 
    %         vector C. Every element i of C contains variance of input i in TX
    %
    %
    %         See also
    %         GPCF_PROD_COV, GP_COV, GP_TRCOV


        ncf = length(gpcf.functions);
        
        % evaluate the individual covariance functions
        C = 1;
        for i=1:ncf
            cf = gpcf.functions{i};
            C = C.*feval(cf.fh_trvar, cf, x);
        end
    end

    function reccf = gpcf_prod_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_PROD_RECAPPEND(RECCF, RI, GPCF) takes old covariance
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
            reccf.type = 'gpcf_prod';
            reccf.nin = ri.nin;
            reccf.nout = 1;

            % Initialize parameters
            ncf = length(ri.functions);
            for i=1:ncf
                cf = ri.functions{i};
                reccf.functions{i} = feval(cf.fh_recappend, [], ri.functions{i});
            end
            
            % Set the function handles
            reccf.fh_pak = @gpcf_prod_pak;
            reccf.fh_unpak = @gpcf_prod_unpak;
            reccf.fh_e = @gpcf_prod_e;
            reccf.fh_g = @gpcf_prod_g;
            reccf.fh_cov = @gpcf_prod_cov;
            reccf.fh_trcov  = @gpcf_prod_trcov;
            reccf.fh_trvar  = @gpcf_prod_trvar;
            reccf.fh_recappend = @gpcf_prod_recappend;
            return
        end
        
        %loop over all of the covariance functions
        ncf = length(gpcf.functions);
        for i=1:ncf
            cf = gpcf.functions{i};
            reccf.functions{i} = feval(cf.fh_recappend, reccf.functions{i}, ri, cf);
        end
    end
end