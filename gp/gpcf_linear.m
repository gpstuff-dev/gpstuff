function gpcf = gpcf_linear(do, varargin)
%GPCF_LINEAR	Create a linear covariance function
%
%	Description
%        GPCF = GPCF_LINEAR('init', OPTIONS) Create and initialize
%        linear covariance function for Gaussian process. OPTIONS is
%        optional parameter-value pair used as described below by
%        GPCF_LINEAR('set',...
%
%        GPCF = GPCF_LINEAR('SET', GPCF, OPTIONS) Set the fields of GPCF
%        as described by the parameter-value pairs ('FIELD', VALUE) in
%        the OPTIONS. The fields that can be modified are:
%
%             'coeffSigma2'        : variance (squared) for regressor 
%                                    coefficient (default 10) (can
%                                    also be vector)
%             'coeffSigma2_prior'  : prior structure for coeffSigma2
%             'selectedVariables'  : vector defining which inputs are 
%                                    active
%
%       Note! If the prior structure is set to empty matrix
%       (e.g. 'coeffSigma2_prior', []) then the parameter in question
%       is considered fixed and it is not handled in optimization,
%       grid integration, MCMC etc.
%
%	See also
%       gpcf_exp, gp_init, gp_e, gp_g, gp_trcov, gp_cov, gp_unpak, gp_pak

% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2008-2010 Jaakko Riihim�ki

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    ip=inputParser;
    ip.FunctionName = 'GPCF_LINEAR';
    ip.addRequired('do', @(x) ismember(x, {'init','set'}));
    ip.addOptional('gpcf', [], @isstruct);
    ip.addParamValue('coeffSigma2',[], @(x) isvector(x) && all(x>0));
    ip.addParamValue('coeffSigma2_prior',[], @(x) isstruct(x) || isempty(x));
    ip.addParamValue('selectedVariables',[], @(x) isvector(x) && all(x>0));
    ip.parse(do, varargin{:});
    do=ip.Results.do;
    gpcf=ip.Results.gpcf;
    coeffSigma2=ip.Results.coeffSigma2;
    coeffSigma2_prior=ip.Results.coeffSigma2_prior;
    selectedVariables=ip.Results.selectedVariables;

    switch do
        case 'init'
            gpcf.type = 'gpcf_linear';

            % Initialize parameters
            if isempty(coeffSigma2)
                gpcf.coeffSigma2 = 10;
            else
                gpcf.coeffSigma2 = coeffSigma2;
            end
            if ~isempty(selectedVariables)
                gpcf.selectedVariables = selectedVariables;
                if ~sum(strcmp(varargin, 'coeffSigma2'))
                    gpcf.coeffSigma2 = repmat(10, 1, length(gpcf.selectedVariables));
                end
            end

            % Initialize prior structure
            gpcf.p=[];
            if isempty(coeffSigma2_prior)
                gpcf.p.coeffSigma2=prior_unif('init');
            else
                gpcf.p.coeffSigma2=coeffSigma2_prior;
            end

            % Set the function handles to the nested functions
            gpcf.fh_pak = @gpcf_linear_pak;
            gpcf.fh_unpak = @gpcf_linear_unpak;
            gpcf.fh_e = @gpcf_linear_e;
            gpcf.fh_ghyper = @gpcf_linear_ghyper;
            gpcf.fh_ginput = @gpcf_linear_ginput;
            gpcf.fh_cov = @gpcf_linear_cov;
            gpcf.fh_trcov  = @gpcf_linear_trcov;
            gpcf.fh_trvar  = @gpcf_linear_trvar;
            gpcf.fh_recappend = @gpcf_linear_recappend;

        case 'set'
            % Set the parameter values of covariance function
            % go through all the parameter values that are changed
            if isempty(gpcf)
                error('with set option you have to provide the old covariance structure.')
            end
            if ~isempty(coeffSigma2);
                gpcf.coeffSigma2=coeffSigma2;
            end
            if ~isempty(selectedVariables)
                gpcf.selectedVariables=selectedVariables;
            end
            if ~isempty(coeffSigma2_prior);
                gpcf.p.coeffSigma2=coeffSigma2_prior;
            end
    end
    

    function w = gpcf_linear_pak(gpcf, w)
    %GPCF_LINEAR_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %   W = GPCF_LINEAR_PAK(GPCF) takes a covariance function data
    %   structure GPCF and combines the covariance function parameters
    %   and their hyperparameters into a single row vector W and takes
    %   a logarithm of the covariance function parameters.
    %
    %       w = [ log(gpcf.coeffSigma2)
    %             (hyperparameters of gpcf.coeffSigma2)]'
    %	  
    %
    %	See also
    %	GPCF_LINEAR_UNPAK
        
        w = [];
        if ~isempty(gpcf.p.coeffSigma2)
            w = log(gpcf.coeffSigma2);
            
            % Hyperparameters of coeffSigma2
            w = [w feval(gpcf.p.coeffSigma2.fh_pak, gpcf.p.coeffSigma2)];
        end
    end

    function [gpcf, w] = gpcf_linear_unpak(gpcf, w)
    %GPCF_LINEAR_UNPAK  Sets the covariance function parameters pack into the structure
    %
    %	Description
    %   [GPCF, W] = GPCF_LINEAR_UNPAK(GPCF, W) takes a covariance
    %   function data structure GPCF and a hyper-parameter vector W,
    %   and returns a covariance function data structure identical to
    %   the input, except that the covariance hyper-parameters have
    %   been set to the values in W. Deletes the values set to GPCF
    %   from W and returns the modeified W.
    %
    %   The covariance function parameters are transformed via exp
    %   before setting them into the structure.
    %
    %	See also
    %	GPCF_LINEAR_PAK

        
        gpp=gpcf.p;

        if ~isempty(gpp.coeffSigma2)
            i2=length(gpcf.coeffSigma2);
            i1=1;
            gpcf.coeffSigma2 = exp(w(i1:i2));
            w = w(i2+1:end);
            
            % Hyperparameters of coeffSigma2
            [p, w] = feval(gpcf.p.coeffSigma2.fh_unpak, gpcf.p.coeffSigma2, w);
            gpcf.p.coeffSigma2 = p;
        end
    end

    function eprior =gpcf_linear_e(gpcf, x, t)
    %GPCF_LINEAR_E     Evaluate the energy of prior of LINEAR parameters
    %
    %	Description
    %   E = GPCF_LINEAR_E(GPCF, X, T) takes a covariance function data
    %   structure GPCF together with a matrix X of input vectors and a
    %   vector T of target vectors and evaluates log p(th) x J, where
    %   th is a vector of LINEAR parameters and J is the Jacobian of
    %   transformation exp(w) = th. (Note that the parameters are log
    %   transformed, when packed.) 
    %
    %   Also the log prior of the hyperparameters of the covariance
    %   function parameters is added to E if hyper-hyperprior is
    %   defined.
    %
    %	See also
    %	GPCF_LINEAR_PAK, GPCF_LINEAR_UNPAK, GPCF_LINEAR_G, GP_E

        [n, m] =size(x);

        % Evaluate the prior contribution to the error. The parameters that
        % are sampled are from space W = log(w) where w is all the "real" samples.
        % On the other hand errors are evaluated in the W-space so we need take
        % into account also the  Jakobian of transformation W -> w = exp(W).
        % See Gelman et.all., 2004, Bayesian data Analysis, second edition, p24.
        eprior = 0;
        gpp=gpcf.p;

        if ~isempty(gpp.coeffSigma2)
            eprior = feval(gpp.coeffSigma2.fh_e, gpcf.coeffSigma2, gpp.coeffSigma2) - sum(log(gpcf.coeffSigma2));
        end
    end

    function [DKff, gprior]  = gpcf_linear_ghyper(gpcf, x, x2, mask)  % , t, g, gdata, gprior, varargin
    %GPCF_LINEAR_GHYPER     Evaluate gradient of covariance function and hyper-prior with 
    %                     respect to the hyperparameters.
    %
    %	Description
    %	[DKff, GPRIOR] = GPCF_LINEAR_GHYPER(GPCF, X) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X) with respect to th (cell array with matrix
    %   elements), and GPRIOR = d log (p(th))/dth, where th is the
    %   vector of hyperparameters
    %
    %	[DKff, GPRIOR] = GPCF_LINEAR_GHYPER(GPCF, X, X2) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X2) with respect to th (cell array with matrix
    %   elements), and GPRIOR = d log (p(th))/dth, where th is the
    %   vector of hyperparameters
    %
    %	[DKff, GPRIOR] = GPCF_LINEAR_GHYPER(GPCF, X, [], MASK) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the diagonal of gradients of
    %   covariance matrix Kff = k(X,X2) with respect to th (cell array
    %   with matrix elements), and GPRIOR = d log (p(th))/dth, where
    %   th is the vector of hyperparameters. This is needed for
    %   example with FIC sparse approximation.
    %
    %	See also
    %   GPCF_LINEAR_PAK, GPCF_LINEAR_UNPAK, GPCF_LINEAR_E, GP_G


        gpp=gpcf.p;
        [n, m] =size(x);

        i1=0;
        DKff = {};
        gprior = [];
        
        % Evaluate: DKff{1} = d Kff / d coeffSigma2
        % NOTE! Here we have already taken into account that the parameters are transformed
        % through log() and thus dK/dlog(p) = p * dK/dp

        
        % evaluate the gradient for training covariance
        if nargin == 2
            
            if isfield(gpcf, 'selectedVariables')
                if ~isempty(gpcf.p.coeffSigma2)
                    if length(gpcf.coeffSigma2) == 1
                        DKff{1}=gpcf.coeffSigma2*x(:,gpcf.selectedVariables)*(x(:,gpcf.selectedVariables)');
                    else
                        for i=1:length(gpcf.coeffSigma2)
                            DD = gpcf.coeffSigma2(i)*x(:,gpcf.selectedVariables(i))*(x(:,gpcf.selectedVariables(i))');
                            DD(abs(DD)<=eps) = 0;
                            DKff{i}= (DD+DD')./2;
                        end
                    end
                end
            else
                if ~isempty(gpcf.p.coeffSigma2)
                    if length(gpcf.coeffSigma2) == 1
                        DKff{1}=gpcf.coeffSigma2*x*(x');
                    else
                        for i=1:m
                            DD = gpcf.coeffSigma2(i)*x(:,i)*(x(:,i)');
                            DD(abs(DD)<=eps) = 0;
                            DKff{i}= (DD+DD')./2;
                        end
                    end
                end
            end
            
            
        % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
        elseif nargin == 3
            if size(x,2) ~= size(x2,2)
                error('gpcf_linear -> _ghyper: The number of columns in x and x2 has to be the same. ')
            end

            if isfield(gpcf, 'selectedVariables')
                if ~isempty(gpcf.p.coeffSigma2)
                    if length(gpcf.coeffSigma2) == 1
                        DKff{1}=gpcf.coeffSigma2*x(:,gpcf.selectedVariables)*(x2(:,gpcf.selectedVariables)');
                    else
                        for i=1:length(gpcf.coeffSigma2)
                            DKff{i}=gpcf.coeffSigma2(i)*x(:,gpcf.selectedVariables(i))*(x2(:,gpcf.selectedVariables(i))');
                        end
                    end
                end
            else
                if ~isempty(gpcf.p.coeffSigma2)
                    if length(gpcf.coeffSigma2) == 1
                        DKff{1}=gpcf.coeffSigma2*x*(x2');
                    else
                        for i=1:m
                            DKff{i}=gpcf.coeffSigma2(i)*x(:,i)*(x2(:,i)');
                        end
                    end
                end
            end
            % Evaluate: DKff{1}    = d mask(Kff,I) / d constSigma2
            %           DKff{2...} = d mask(Kff,I) / d coeffSigma2
        elseif nargin == 4
            
            if isfield(gpcf, 'selectedVariables')
                if ~isempty(gpcf.p.coeffSigma2)
                    if length(gpcf.coeffSigma2) == 1
                        DKff{1}=gpcf.coeffSigma2*sum(x(:,gpcf.selectedVariables).^2,2); % d mask(Kff,I) / d coeffSigma2
                    else
                        for i=1:length(gpcf.coeffSigma2)
                            DKff{i}=gpcf.coeffSigma2(i)*(x(:,gpcf.selectedVariables(i)).^2); % d mask(Kff,I) / d coeffSigma2
                        end
                    end
                end
            else
                if ~isempty(gpcf.p.coeffSigma2)
                    if length(gpcf.coeffSigma2) == 1
                        DKff{1}=gpcf.coeffSigma2*sum(x.^2,2); % d mask(Kff,I) / d coeffSigma2
                    else
                        for i=1:m
                            DKff{i}=gpcf.coeffSigma2(i)*(x(:,i).^2); % d mask(Kff,I) / d coeffSigma2
                        end
                    end
                end
            end
        end
        
        if nargout > 1
            
            if ~isempty(gpcf.p.coeffSigma2)
                lll = length(gpcf.coeffSigma2);
                gg = feval(gpp.coeffSigma2.fh_g, gpcf.coeffSigma2, gpp.coeffSigma2);
                gprior = gg(1:lll).*gpcf.coeffSigma2 - 1;
                gprior = [gprior gg(lll+1:end)];
            end
        end
    end


    function [DKff, gprior]  = gpcf_linear_ginput(gpcf, x, x2)
    %GPCF_LINEAR_GINPUT     Evaluate gradient of covariance function with 
    %                     respect to x.
    %
    %	Description
    %	DKff = GPCF_LINEAR_GHYPER(GPCF, X) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X) with respect to X (cell array with matrix
    %   elements)
    %
    %	DKff = GPCF_LINEAR_GHYPER(GPCF, X, X2) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X2) with respect to X (cell array with matrix
    %   elements).
    %
    %	See also
    %   GPCF_LINEAR_PAK, GPCF_LINEAR_UNPAK, GPCF_LINEAR_E, GP_G        
        [n, m] =size(x);
        
        if nargin == 2
            
            %K = feval(gpcf.fh_trcov, gpcf, x);
            
            if length(gpcf.coeffSigma2) == 1
                % In the case of an isotropic LINEAR
                s = repmat(gpcf.coeffSigma2, 1, m);
            else
                s = gpcf.coeffSigma2;
            end
            
            ii1 = 0;
            if isfield(gpcf, 'selectedVariables')
                for i=1:length(gpcf.selectedVariables)
                    for j = 1:n
                        
                        DK = zeros(n);
                        DK(j,:)=s(i)*x(:,gpcf.selectedVariables(i))';
                        
                        DK = DK + DK';
                        
                        ii1 = ii1 + 1;
                        DKff{ii1} = DK;
                        gprior(ii1) = 0;
                    end
                end
            else
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
            end
            
            
            
        elseif nargin == 3
            %K = feval(gpcf.fh_cov, gpcf, x, x2);
            
            if length(gpcf.coeffSigma2) == 1
                % In the case of an isotropic LINEAR
                s = repmat(gpcf.coeffSigma2, 1, m);
            else
                s = gpcf.coeffSigma2;
            end
            
            ii1 = 0;
            if isfield(gpcf, 'selectedVariables')
                for i=1:length(gpcf.selectedVariables)
                    for j = 1:n
                        
                        DK = zeros(n, size(x2,1));
                        DK(j,:)=s(i)*x2(:,gpcf.selectedVariables(i))';
                        
                        ii1 = ii1 + 1;
                        DKff{ii1} = DK;
                        gprior(ii1) = 0;
                    end
                end
            else
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
    end


    function C = gpcf_linear_cov(gpcf, x1, x2, varargin)
    % GP_LINEAR_COV     Evaluate covariance matrix between two input vectors.
    %
    %         Description         
    %         C = GP_LINEAR_COV(GP, TX, X) takes in covariance function of a
    %         Gaussian process GP and two matrixes TX and X that
    %         contain input vectors to GP. Returns covariance matrix
    %         C. Every element ij of C contains covariance between
    %         inputs i in TX and j in X.
    %
    %
    %         See also
    %         GPCF_LINEAR_TRCOV, GPCF_LINEAR_TRVAR, GP_COV, GP_TRCOV
        
        if isempty(x2)
            x2=x1;
        end
        [n1,m1]=size(x1);
        [n2,m2]=size(x2);

        if m1~=m2
            error('the number of columns of X1 and X2 has to be same')
        end
        
        if isfield(gpcf, 'selectedVariables')
            C = x1(:,gpcf.selectedVariables)*diag(gpcf.coeffSigma2)*(x2(:,gpcf.selectedVariables)');
        else
            C = x1*diag(gpcf.coeffSigma2)*(x2');
        end
        C(abs(C)<=eps) = 0;
    end

    function C = gpcf_linear_trcov(gpcf, x)
    % GP_LINEAR_TRCOV     Evaluate training covariance matrix of inputs.
    %
    %         Description
    %         C = GP_LINEAR_TRCOV(GP, TX) takes in covariance function of a
    %         Gaussian process GP and matrix TX that contains training
    %         input vectors. Returns covariance matrix C. Every
    %         element ij of C contains covariance between inputs i and
    %         j in TX
    %
    %         See also
    %         GPCF_LINEAR_COV, GPCF_LINEAR_TRVAR, GP_COV, GP_TRCOV

        if isfield(gpcf, 'selectedVariables')
            C = x(:,gpcf.selectedVariables)*diag(gpcf.coeffSigma2)*(x(:,gpcf.selectedVariables)');
        else
            C = x*diag(gpcf.coeffSigma2)*(x');
        end
        C(abs(C)<=eps) = 0;
        C = (C+C')./2;

    end


    function C = gpcf_linear_trvar(gpcf, x)
    % GP_LINEAR_TRVAR     Evaluate training variance vector
    %
    %         Description
    %         C = GP_LINEAR_TRVAR(GPCF, TX) takes in covariance function 
    %         of a Gaussian process GPCF and matrix TX that contains
    %         training inputs. Returns variance vector C. Every
    %         element i of C contains variance of input i in TX
    %
    %
    %         See also
    %         GPCF_LINEAR_COV, GP_COV, GP_TRCOV

                

        if length(gpcf.coeffSigma2) == 1
            if isfield(gpcf, 'selectedVariables')
                C=gpcf.coeffSigma2.*sum(x(:,gpcf.selectedVariables).^2,2);
            else
                C=gpcf.coeffSigma2.*sum(x.^2,2);
            end
        else
            if isfield(gpcf, 'selectedVariables')
                C=sum(repmat(gpcf.coeffSigma2, size(x,1), 1).*x(:,gpcf.selectedVariables).^2,2);
            else
                C=sum(repmat(gpcf.coeffSigma2, size(x,1), 1).*x.^2,2);
            end
        end
        C(abs(C)<eps)=0;
  
    end

    function reccf = gpcf_linear_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %
    %          Description
    %          RECCF = GPCF_LINEAR_RECAPPEND(RECCF, RI, GPCF)
    %          takes a likelihood record structure RECCF, record
    %          index RI and likelihood structure GPCF with the
    %          current MCMC samples of the hyperparameters. Returns
    %          RECCF which contains all the old samples and the
    %          current samples from GPCF .
    %
    %          See also
    %          GP_MC and GP_MC -> RECAPPEND


    % Initialize record
        if nargin == 2
            reccf.type = 'gpcf_linear';

            % Initialize parameters
            reccf.coeffSigma2= [];

            % Set the function handles
            reccf.fh_pak = @gpcf_linear_pak;
            reccf.fh_unpak = @gpcf_linear_unpak;
            reccf.fh_e = @gpcf_linear_e;
            reccf.fh_g = @gpcf_linear_g;
            reccf.fh_cov = @gpcf_linear_cov;
            reccf.fh_trcov  = @gpcf_linear_trcov;
            reccf.fh_trvar  = @gpcf_linear_trvar;
            reccf.fh_recappend = @gpcf_linear_recappend;
            gpcf.p=[];
            gpcf.p.coeffSigma2=[];
            if ~isempty(ri.p.coeffSigma2)
                reccf.p.coeffSigma2 = ri.p.coeffSigma2;
            end

            return
        end

        gpp = gpcf.p;
        % record coeffSigma2
        if ~isempty(gpcf.coeffSigma2)
            reccf.coeffSigma2(ri,:)=gpcf.coeffSigma2;
            reccf.p.coeffSigma2 = feval(gpp.coeffSigma2.fh_recappend, reccf.p.coeffSigma2, ri, gpcf.p.coeffSigma2);
        elseif ri==1
            reccf.coeffSigma2=[];
        end
        
        if isfield(gpcf, 'selectedVariables')
        	reccf.selectedVariables = gpcf.selectedVariables;
        end
        
    end
end