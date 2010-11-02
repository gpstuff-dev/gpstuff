function gpcf = gpcf_noise(varargin)
%GPCF_NOISE  Create a independent noise covariance function
%
%  Description
%    GPCF = GPCF_NOISE('PARAM1',VALUE1,'PARAM2,VALUE2,...) creates
%    independent noise covariance function structure in which the
%    named parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%    GPCF = GPCF_NOISE(GPCF,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a covariance function structure with the named
%    parameters altered with the specified values.
%
%    Parameters for independent noise covariance function [default]
%      noiseSigma2       - variance of the independent noise [0.1]
%      noiseSigma2_prior - prior for noiseSigma2 [prior_logunif]
%
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%  See also
%    GP_SET, GPCF_*, PRIOR_*

% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

  % allow use with or without init and set options
  if nargin<1
    do='init';
  elseif ischar(varargin{1})
    switch varargin{1}
      case 'init'
        do='init';varargin(1)=[];
      case 'set'
        do='set';varargin(1)=[];
      otherwise
        do='init';
    end
  elseif isstruct(varargin{1})
    do='set';
  else
    error('Unknown first argument');
  end
  
    ip=inputParser;
    ip.FunctionName = 'GPCF_NOISE';
    ip.addOptional('gpcf', [], @isstruct);
    ip.addParamValue('noiseSigma2',[], @(x) isscalar(x) && x>0);
    ip.addParamValue('noiseSigma2_prior',NaN, @(x) isstruct(x) || isempty(x));
    ip.parse(varargin{:});
    gpcf=ip.Results.gpcf;
    noiseSigma2=ip.Results.noiseSigma2;
    noiseSigma2_prior=ip.Results.noiseSigma2_prior;

    switch do
        case 'init'
            gpcf.type = 'gpcf_noise';

            % Initialize parameters
            if isempty(noiseSigma2)
                gpcf.noiseSigma2 = 0.1;
            else
                gpcf.noiseSigma2 = noiseSigma2;
            end

            % Initialize prior structure
            gpcf.p=[];
            if ~isstruct(noiseSigma2_prior)&&isnan(noiseSigma2_prior)
                gpcf.p.noiseSigma2 = prior_logunif;
            else
                gpcf.p.noiseSigma2 = noiseSigma2_prior;
            end

            % Set the function handles to the nested functions
            gpcf.fh.pak = @gpcf_noise_pak;
            gpcf.fh.unpak = @gpcf_noise_unpak;
            gpcf.fh.e = @gpcf_noise_e;
            gpcf.fh.ghyper = @gpcf_noise_ghyper;
            gpcf.fh.gind = @gpcf_noise_ginput;
            gpcf.fh.cov = @gpcf_noise_cov;
            gpcf.fh.trcov  = @gpcf_noise_trcov;
            gpcf.fh.trvar  = @gpcf_noise_trvar;
            gpcf.fh.sampling = @hmc2;
            gpcf.sampling_opt = hmc2_opt;
            gpcf.fh.recappend = @gpcf_noise_recappend;

        case 'set'
            % Set the parameter values of covariance function
            % go through all the parameter values that are changed
            if isempty(gpcf)
                error('with set option you have to provide the old covariance structure.')
            end
            if ~isempty(noiseSigma2);
                gpcf.noiseSigma2=noiseSigma2;
            end
            if ~isstruct(noiseSigma2_prior)&& isnan(noiseSigma2_prior)
            else
                gpcf.p.noiseSigma2=noiseSigma2_prior;
            end
    end


    function w = gpcf_noise_pak(gpcf)
    %GPCF_NOISE_PAK      Combine GP covariance function hyper-parameters into one vector.
    %
    %  Description
    %   W = GPCF_NOISE_PAK(GPCF) takes a covariance function data
    %   structure GPCF and combines the covariance function parameters
    %   and their hyperparameters into a single row vector W and takes
    %   a logarithm of the covariance function parameters.
    %
    %       w = [ log(gpcf.noiseSigma2)
    %             (hyperparameters of gpcf.magnSigma2)]'
    %     
    %
    %  See also
    %   GPCF_NOISE_UNPAK


        w = [];    
        if ~isempty(gpcf.p.noiseSigma2)
            w(1) = log(gpcf.noiseSigma2);
            
            % Hyperparameters of noiseSigma2
            w = [w feval(gpcf.p.noiseSigma2.fh.pak, gpcf.p.noiseSigma2)];
        end    

    end

    function [gpcf, w] = gpcf_noise_unpak(gpcf, w)
    %GPCF_NOISE_UNPAK  Sets the covariance function parameters pack into the structure
    %
    %  Description
    %   [GPCF, W] = GPCF_NOISE_UNPAK(GPCF, W) takes a covariance
    %   function data structure GPCF and a hyper-parameter vector W,
    %   and returns a covariance function data structure identical to
    %   the input, except that the covariance hyper-parameters have
    %   been set to the values in W. Deletes the values set to GPCF
    %   from W and returns the modeified W.
    %
    %   The covariance function parameters are transformed via exp
    %   before setting them into the structure.
    %
    %  See also
    %   GPCF_NOISE_PAK

    
        if ~isempty(gpcf.p.noiseSigma2)
                gpcf.noiseSigma2 = exp(w(1));
                w = w(2:end);
                                
                % Hyperparameters of lengthScale
                [p, w] = feval(gpcf.p.noiseSigma2.fh.unpak, gpcf.p.noiseSigma2, w);
                gpcf.p.noiseSigma2 = p;
        end
    end


    function eprior =gpcf_noise_e(gpcf, p, t)
    %GPCF_NOISE_E     Evaluate the energy of prior of NOISE parameters
    %
    %  Description
    %   E = GPCF_NOISE_E(GPCF, X, T) takes a covariance function data
    %   structure GPCF together with a matrix X of input vectors and a
    %   vector T of target vectors and evaluates log p(th) x J, where
    %   th is a vector of NOISE parameters and J is the Jacobian of
    %   transformation exp(w) = th. (Note that the parameters are log
    %   transformed, when packed.) 
    %
    %   Also the log prior of the hyperparameters of the covariance
    %   function parameters is added to E if hyper-hyperprior is
    %   defined.
    %
    %  See also
    %   GPCF_NOISE_PAK, GPCF_NOISE_UNPAK, GPCF_NOISE_G, GP_E

        eprior = 0;

        if ~isempty(gpcf.p.noiseSigma2)
            % Evaluate the prior contribution to the error.
            gpp=gpcf.p;
            eprior = feval(gpp.noiseSigma2.fh.e, gpcf.noiseSigma2, gpp.noiseSigma2) - log(gpcf.noiseSigma2);
        end
    end

    function [D,gprior]  = gpcf_noise_ghyper(gpcf, x, x2) %g, gdata, gprior
    %GPCF_NOISE_GHYPER     Evaluate gradient of covariance function and hyper-prior with 
    %                     respect to the hyperparameters.
    %
    %  Description
    %   [DKff, GPRIOR] = GPCF_NOISE_GHYPER(GPCF, X) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X) with respect to th (cell array with matrix
    %   elements), and GPRIOR = d log (p(th))/dth, where th is the
    %   vector of hyperparameters
    %
    %   [DKff, GPRIOR] = GPCF_NOISE_GHYPER(GPCF, X, X2) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X2) with respect to th (cell array with matrix
    %   elements), and GPRIOR = d log (p(th))/dth, where th is the
    %   vector of hyperparameters
    %
    %   [DKff, GPRIOR] = GPCF_NOISE_GHYPER(GPCF, X, [], MASK) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the diagonal of gradients of
    %   covariance matrix Kff = k(X,X2) with respect to th (cell array
    %   with matrix elements), and GPRIOR = d log (p(th))/dth, where
    %   th is the vector of hyperparameters. This is needed for
    %   example with FIC sparse approximation.
    %
    %  See also
    %   GPCF_NOISE_PAK, GPCF_NOISE_UNPAK, GPCF_NOISE_E, GP_G

        D = {};
        gprior = [];

        if ~isempty(gpcf.p.noiseSigma2)
            gpp=gpcf.p;
            
            D{1}=gpcf.noiseSigma2;
            
            ggs = feval(gpp.noiseSigma2.fh.g, gpcf.noiseSigma2, gpp.noiseSigma2);
            gprior = ggs(1).*gpcf.noiseSigma2 - 1;
            if length(ggs) > 1
                gprior = [gprior ggs(2:end)];
            end            
        end
    end

    function DKff  = gpcf_noise_ginput(gpcf, x, t, g_ind, gdata_ind, gprior_ind, varargin)
    %GPCF_NOISE_GINPUT     Evaluate gradient of covariance function with 
    %                     respect to x.
    %
    %  Description
    %   DKff = GPCF_NOISE_GHYPER(GPCF, X) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X) with respect to X (cell array with matrix
    %   elements)
    %
    %   DKff = GPCF_NOISE_GHYPER(GPCF, X, X2) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X2) with respect to X (cell array with matrix
    %   elements).
    %
    %  See also
    %   GPCF_NOISE_PAK, GPCF_NOISE_UNPAK, GPCF_NOISE_E, GP_G
        

    end

    function C = gpcf_noise_cov(gpcf, x1, x2)
    % GP_NOISE_COV     Evaluate covariance matrix between two input vectors.
    %
    %         Description         
    %         C = GP_NOISE_COV(GP, TX, X) takes in covariance function of a
    %         Gaussian process GP and two matrixes TX and X that
    %         contain input vectors to GP. Returns covariance matrix
    %         C. Every element ij of C contains covariance between
    %         inputs i in TX and j in X.
    %
    %
    %         See also
    %         GPCF_NOISE_TRCOV, GPCF_NOISE_TRVAR, GP_COV, GP_TRCOV


        if isempty(x2)
            x2=x1;
        end
        [n1,m1]=size(x1);
        [n2,m2]=size(x2);

        if m1~=m2
            error('the number of columns of X1 and X2 has to be same')
        end

        C = sparse([],[],[],n1,n2,0);
    end

    function C = gpcf_noise_trcov(gpcf, x)
    % GP_NOISE_TRCOV     Evaluate training covariance matrix of inputs.
    %
    %         Description
    %         C = GP_NOISE_TRCOV(GP, TX) takes in covariance function of a
    %         Gaussian process GP and matrix TX that contains training
    %         input vectors. Returns covariance matrix C. Every
    %         element ij of C contains covariance between inputs i and
    %         j in TX
    %
    %         See also
    %         GPCF_NOISE_COV, GPCF_NOISE_TRVAR, GP_COV, GP_TRCOV

        [n, m] =size(x);
        n1=n+1;

        C = sparse([],[],[],n,n,0);
        C(1:n1:end)=C(1:n1:end)+gpcf.noiseSigma2;

    end

    function C = gpcf_noise_trvar(gpcf, x)
    % GP_NOISE_TRVAR     Evaluate training variance vector
    %
    %         Description
    %         C = GP_NOISE_TRVAR(GPCF, TX) takes in covariance function 
    %         of a Gaussian process GPCF and matrix TX that contains
    %         training inputs. Returns variance vector C. Every
    %         element i of C contains variance of input i in TX
    %
    %
    %         See also
    %         GPCF_NOISE_COV, GP_COV, GP_TRCOV



        [n, m] =size(x);
        C=ones(n,1)*gpcf.noiseSigma2;

    end

    function reccf = gpcf_noise_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %
    %          Description
    %          RECCF = GPCF_NOISE_RECAPPEND(RECCF, RI, GPCF)
    %          takes a covariance function record structure RECCF, record
    %          index RI and covariance function structure GPCF with the
    %          current MCMC samples of the hyperparameters. Returns
    %          RECCF which contains all the old samples and the
    %          current samples from GPCF .
    %
    %          See also
    %          GP_MC and GP_MC -> RECAPPEND

    % Initialize record
        if nargin == 2
            reccf.type = 'gpcf_noise';
            
            % Initialize parameters
            reccf.noiseSigma2 = []; 
            
            % Set the function handles
            reccf.fh.pak = @gpcf_noise_pak;
            reccf.fh.unpak = @gpcf_noise_unpak;
            reccf.fh.e = @gpcf_noise_e;
            reccf.fh.g = @gpcf_noise_g;
            reccf.fh.cov = @gpcf_noise_cov;
            reccf.fh.trcov  = @gpcf_noise_trcov;
            reccf.fh.trvar  = @gpcf_noise_trvar;
            %  gpcf.fh.sampling = @hmc2;
            reccf.sampling_opt = hmc2_opt;
            reccf.fh.recappend = @gpcf_noise_recappend;  
            reccf.p=[];
            reccf.p.noiseSigma2=[];
            if ~isempty(ri.p.noiseSigma2)
                reccf.p.noiseSigma2 = ri.p.noiseSigma2;
            end
            return
        end

        gpp = gpcf.p;

        % record noiseSigma
        if ~isempty(gpcf.noiseSigma2)
            reccf.noiseSigma2(ri,:)=gpcf.noiseSigma2;
            reccf.p.noiseSigma2 = feval(gpp.noiseSigma2.fh.recappend, reccf.p.noiseSigma2, ri, gpcf.p.noiseSigma2);
        elseif ri==1
            reccf.noiseSigma2=[];
        end
    end

end
