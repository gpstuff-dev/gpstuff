function gpcf = gpcf_periodic(varargin) 
%GPCF_PERIODIC  Create a periodic covariance function for Gaussian Process
%
%  Description
%    GPCF = GPCF_PERIODIC('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates periodic covariance function structure in which the
%    named parameters have the specified values. Any unspecified
%    parameters are set to default values.
%  
%    GPCF = GPCF_PERIODIC(GPCF,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a covariance function structure with the named
%    parameters altered with the specified values.
%
%    Periodic covariance function with squared exponential decay
%    part as in Rasmussen & Williams (2006) Gaussian processes for
%    Machine Learning.
%  
%    Parameters for periodic covariance function [default]
%      magnSigma2             - magnitude (squared) [0.1] 
%      lengthScale            - length scale for each input [10]
%                               This can be either scalar
%                               corresponding isotropic or vector
%                               corresponding ARD
%      period                 - length of the periodic component(s) [1]
%      lengthScale_sexp       - length scale for the squared exponential 
%                               component [10] This can be either scalar
%                               corresponding isotropic or vector
%                               corresponding ARD. 
%      decay                  - determines whether the squared exponential 
%                               decay term is used (1) or not (0). 
%                               Not a hyperparameter for the function.
%      magnSigma2_prior       - prior structure for magnSigma2 [prior_sqrtunif]
%      lengthScale_prior      - prior structure for lengthScale [prior_unif]
%      lengthScale_sexp_prior - prior structure for lengthScale_sexp 
%                               [prior_fixed]
%      period_prior           - prior structure for period [prior_fixed]
%
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%  See also
%    GP_SET, GPCF_*, PRIOR_*
    
% Copyright (c) 2009-2010 Heikki Peura
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
    ip.FunctionName = 'GPCF_PERIODIC';
    ip.addOptional('gpcf', [], @isstruct);
    ip.addParamValue('magnSigma2',[], @(x) isscalar(x) && x>0);
    ip.addParamValue('lengthScale',[], @(x) isvector(x) && all(x>0));
    ip.addParamValue('period',[], @(x) isscalar(x) && x>0 && mod(x,1)==0);
    ip.addParamValue('lengthScale_sexp',[], @(x) isvector(x) && all(x>0));
    ip.addParamValue('optimPeriod',[], @(x) isscalar(x) && (x==0||x==1));
    ip.addParamValue('decay',[], @(x) isscalar(x) && (x==0||x==1));
    ip.addParamValue('magnSigma2_prior',NaN, @(x) isstruct(x) || isempty(x));
    ip.addParamValue('lengthScale_prior',NaN, @(x) isstruct(x) || isempty(x));
    ip.addParamValue('lengthScale_sexp_prior',NaN, @(x) isstruct(x) || isempty(x));
    ip.addParamValue('period_prior',NaN, @(x) isstruct(x) || isempty(x));
    ip.parse(varargin{:});
    gpcf=ip.Results.gpcf;
    magnSigma2=ip.Results.magnSigma2;
    lengthScale=ip.Results.lengthScale;
    period=ip.Results.period;
    optimPeriod=ip.Results.optimPeriod;
    lengthScale_sexp=ip.Results.lengthScale_sexp;
    decay=ip.Results.decay;
    magnSigma2_prior=ip.Results.magnSigma2_prior;
    lengthScale_prior=ip.Results.lengthScale_prior;
    lengthScale_sexp_prior=ip.Results.lengthScale_sexp_prior;
    period_prior=ip.Results.period_prior;
    
    switch do
        case 'init'
            % Initialize the covariance function
            
            % Initialize parameters
            if isempty(lengthScale)
                gpcf.lengthScale = 10;
            else
                gpcf.lengthScale=lengthScale;
            end
            if isempty(magnSigma2)
                gpcf.magnSigma2 = 0.1;
            else
                gpcf.magnSigma2=magnSigma2;
            end
            if isempty(lengthScale_sexp)
                gpcf.lengthScale_sexp = 10;
            else
                gpcf.lengthScale_sexp=lengthScale_sexp;
            end
            if isempty(period)
                gpcf.period = 1;
            else
                gpcf.period=period;
            end
            if isempty(optimPeriod)                
                gpcf.optimPeriod = 0;
            else
                gpcf.optimPeriod=optimPeriod;
            end
            if isempty(decay)
                gpcf.decay = 0;
            else
                gpcf.decay=decay;
            end
            
            % Initialize prior structure
            gpcf.p=[];
            if ~isstruct(lengthScale_prior)&isnan(lengthScale_prior)
                gpcf.p.lengthScale=prior_unif;
            else
                gpcf.p.lengthScale=lengthScale_prior;
            end
            if ~isstruct(magnSigma2_prior)&isnan(magnSigma2_prior)
                gpcf.p.magnSigma2=prior_sqrtunif;
            else
                gpcf.p.magnSigma2=magnSigma2_prior;
            end
            if ~isstruct(period_prior)&isnan(period_prior);
                gpcf.p.period=[];
            else
                gpcf.p.period=period_prior;
            end
            if isnan(lengthScale_sexp_prior)
                gpcf.p.lengthScale_sexp=[];
            else
                gpcf.p.lengthScale_sexp=lengthScale_sexp_prior;
            end
            
            % Set the function handles to the nested functions
            gpcf.fh.pak = @gpcf_periodic_pak;
            gpcf.fh.unpak = @gpcf_periodic_unpak;
            gpcf.fh.e = @gpcf_periodic_e;
            gpcf.fh.ghyper = @gpcf_periodic_ghyper;
            gpcf.fh.ginput = @gpcf_periodic_ginput;
            gpcf.fh.cov = @gpcf_periodic_cov;
            gpcf.fh.covvec = @gpcf_periodic_covvec;
            gpcf.fh.trcov  = @gpcf_periodic_trcov;
            gpcf.fh.trvar  = @gpcf_periodic_trvar;
            gpcf.fh.recappend = @gpcf_periodic_recappend;
            
        case 'set'
            % Set the parameter values of covariance function
            % go through all the parameter values that are changed
            if isempty(gpcf)
                error('with set option you have to provide the old covariance structure.')
            end
            if ~isempty(magnSigma2);gpcf.magnSigma2=magnSigma2;end
            if ~isempty(lengthScale);gpcf.lengthScale=lengthScale;end
            if ~isempty(period);gpcf.period=period;end
            if ~isempty(optimPeriod);gpcf.optimPeriod=optimPeriod;end
            if ~isempty(lengthScale_sexp);gpcf.lengthScale_sexp=lengthScale_sexp;end
            if ~isempty(decay);gpcf.decay=decay;end
            if ~isstruct(period_prior)&isnan(period_prior);else;gpcf.p.period=period_prior;end
            if ~isstruct(lengthScale_sexp_prior)&isnan(lengthScale_sexp_prior);else;gpcf.p.lengthScale_sexp=lengthScale_sexp_prior;end
            if ~isstruct(magnSigma2_prior)&isnan(magnSigma2_prior);else;gpcf.p.magnSigma2=magnSigma2_prior;end
            if ~isstruct(lengthScale_prior)&isnan(lengthScale_prior);else;gpcf.p.lengthScale=lengthScale_prior;end
    end
    

    function [w, s] = gpcf_periodic_pak(gpcf)
    %GPCF_PERIODIC_PAK   Combine GP covariance function hyper-parameters into one vector.
    %
    %  Description
    %   W = GPCF_PERIODIC_PAK(GPCF) takes a covariance function data
    %   structure GPCF and combines the covariance function parameters
    %   and their hyperparameters into a single row vector W and takes
    %   a logarithm of the covariance function parameters.
    %
    %       w = [ log(gpcf.magnSigma2)
    %             (hyperparameters of gpcf.magnSigma2) 
    %             log(gpcf.lengthScale(:))
    %             (hyperparameters of gpcf.lengthScale)
    %             log(gpcf.lengthScale_sexp)
    %             (hyperparameters of gpcf.lengthScale_sexp)
    %             log(gpcf.period)
    %             (hyperparameters of gpcf.period)]'
    %     
    %
    %  See also
    %   GPCF_PERIODIC_UNPAK

        
        if isfield(gpcf,'metric')
           error('Periodic covariance function not compatible with metrics.');
        else
            i1=0;i2=1;
            w = []; s = {};
            
            if ~isempty(gpcf.p.magnSigma2)
                w = [w log(gpcf.magnSigma2)];
                s = [s; 'log(periodic.magnSigma2)'];
                
                % Hyperparameters of magnSigma2
                [wh sh] = feval(gpcf.p.magnSigma2.fh.pak, gpcf.p.magnSigma2);
                w = [w wh];
                s = [s; sh];
            end
            
            if ~isempty(gpcf.p.lengthScale)
                w = [w log(gpcf.lengthScale)];
                s = [s; 'log(periodic.lengthScale)'];
                            
                % Hyperparameters of lengthScale
                [wh  sh] = feval(gpcf.p.lengthScale.fh.pak, gpcf.p.lengthScale);
                w = [w wh];
                s = [s; sh];
            end
            
            if ~isempty(gpcf.p.lengthScale_sexp)  && gpcf.decay == 1
                w = [w log(gpcf.lengthScale_sexp)];
                s = [s; 'log(periodic.lengthScale_sexp)'];
                            
                % Hyperparameters of lengthScale_sexp
                [wh sh] = feval(gpcf.p.lengthScale_sexp.fh.pak, gpcf.p.lengthScale_sexp);
                w = [w wh];
                s = [s; sh];
            end
            
            if ~isempty(gpcf.p.period) && gpcf.optimPeriod == 1
                w = [w log(gpcf.period)];
                s = [s; 'log(periodic.period)'];
                
                % Hyperparameters of period
                [wh sh] = feval(gpcf.p.period.fh.pak, gpcf.p.period);
                w = [w wh];
                s = [s; sh];
            end
        end
    end




    function [gpcf, w] = gpcf_periodic_unpak(gpcf, w)
    %GPCF_PERIODIC_UNPAK  Sets the covariance function parameters pack into the structure
    %
    %  Description
    %   [GPCF, W] = GPCF_PERIODIC_UNPAK(GPCF, W) takes a covariance
    %   function data structure GPCF and a hyper-parameter vector W,
    %   and returns a covariance function data structure identical to
    %   the input, except that the covariance hyper-parameters have
    %   been set to the values in W. Deletes the values set to GPCF
    %   from W and returns the modified W.
    %
    %   The covariance function parameters are transformed via exp
    %   before setting them into the structure.
    %
    %  See also
    %   GPCF_PERIODIC_PAK

        if isfield(gpcf,'metric')
           error('Covariance function not compatible with metrics');
        else
            gpp=gpcf.p;
            if ~isempty(gpp.magnSigma2)
                i1=1;
                gpcf.magnSigma2 = exp(w(i1));
                w = w(i1+1:end);
            end
            if ~isempty(gpp.lengthScale)
                i2=length(gpcf.lengthScale);
                i1=1;
                gpcf.lengthScale = exp(w(i1:i2));
                w = w(i2+1:end);
            end
            if ~isempty(gpp.lengthScale_sexp) && gpcf.decay == 1
                i2=length(gpcf.lengthScale_sexp);
                i1=1;
                gpcf.lengthScale_sexp = exp(w(i1:i2));
                w = w(i2+1:end);
            end
            if ~isempty(gpp.period) && gpcf.optimPeriod == 1
                i2=length(gpcf.period);
                i1=1;
                gpcf.period = exp(w(i1:i2));
                w = w(i2+1:end);
            end
            % hyperparameters
            if ~isempty(gpp.magnSigma2)
                [p, w] = feval(gpcf.p.magnSigma2.fh.unpak, gpcf.p.magnSigma2, w);
                gpcf.p.magnSigma2 = p;
            end
            if ~isempty(gpp.lengthScale)
                [p, w] = feval(gpcf.p.lengthScale.fh.unpak, gpcf.p.lengthScale, w);
                gpcf.p.lengthScale = p;
            end
            if ~isempty(gpp.lengthScale_sexp)
                [p, w] = feval(gpcf.p.lengthScale_sexp.fh.unpak, gpcf.p.lengthScale_sexp, w);
                gpcf.p.lengthScale_sexp = p;
            end
            if ~isempty(gpp.period)  && gpcf.optimPeriod == 1
                [p, w] = feval(gpcf.p.period.fh.unpak, gpcf.p.period, w);
                gpcf.p.period = p;
            end
            
        end
    end

    function eprior =gpcf_periodic_e(gpcf, x, t) 
    %GPCF_PERIODIC_E     Evaluate the energy of prior of PERIODIC parameters
    %
    %  Description
    %   E = GPCF_PERIODIC_E(GPCF, X, T) takes a covariance function data
    %   structure GPCF together with a matrix X of input vectors and a
    %   vector T of target vectors and evaluates log p(th) x J, where
    %   th is a vector of PERIODIC parameters and J is the Jacobian of
    %   transformation exp(w) = th. (Note that the parameters are log
    %   transformed, when packed.) 
    %
    %   Also the log prior of the hyperparameters of the covariance
    %   function parameters is added to E if hyper-hyperprior is
    %   defined.
    %
    %  See also
    %   GPCF_PERIODIC_PAK, GPCF_PERIODIC_UNPAK, GPCF_PERIODIC_G, GP_E

        eprior = 0;
        gpp=gpcf.p;
        
        [n, m] =size(x);

        if isfield(gpcf,'metric')
           error('Covariance function not compatible with metrics');
        else
            % Evaluate the prior contribution to the error. The parameters that
            % are sampled are from space W = log(w) where w is all the "real" samples.
            % On the other hand errors are evaluated in the W-space so we need take
            % into account also the  Jacobian of transformation W -> w = exp(W).
            % See Gelman et.all., 2004, Bayesian data Analysis, second edition, p24.
            
            if ~isempty(gpcf.p.magnSigma2)
                eprior = feval(gpp.magnSigma2.fh.e, gpcf.magnSigma2, gpp.magnSigma2) - log(gpcf.magnSigma2);
            end
            if ~isempty(gpp.lengthScale)
                eprior = eprior + feval(gpp.lengthScale.fh.e, gpcf.lengthScale, gpp.lengthScale) - sum(log(gpcf.lengthScale));
            end
          
            if ~isempty(gpp.lengthScale_sexp) && gpcf.decay == 1
                eprior = eprior + feval(gpp.lengthScale_sexp.fh.e, gpcf.lengthScale_sexp, gpp.lengthScale_sexp) - sum(log(gpcf.lengthScale_sexp));
            end
            if ~isempty(gpcf.p.period) && gpcf.optimPeriod == 1
                eprior = feval(gpp.period.fh.e, gpcf.period, gpp.period) - sum(log(gpcf.period));
            end
        end

    end
    
    function [DKff, gprior]  = gpcf_periodic_ghyper(gpcf, x, x2, mask)
    %GPCF_PERIODIC_GHYPER     Evaluate gradient of covariance function and hyper-prior with 
    %                     respect to the hyperparameters.
    %
    %  Description
    %   [DKff, GPRIOR] = GPCF_PERIODIC_GHYPER(GPCF, X) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X) with respect to th (cell array with matrix
    %   elements), and GPRIOR = d log (p(th))/dth, where th is the
    %   vector of hyperparameters
    %
    %   [DKff, GPRIOR] = GPCF_PERIODIC_GHYPER(GPCF, X, X2) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X2) with respect to th (cell array with matrix
    %   elements), and GPRIOR = d log (p(th))/dth, where th is the
    %   vector of hyperparameters
    %
    %   [DKff, GPRIOR] = GPCF_PERIODIC_GHYPER(GPCF, X, [], MASK) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the diagonal of gradients of
    %   covariance matrix Kff = k(X,X2) with respect to th (cell array
    %   with matrix elements), and GPRIOR = d log (p(th))/dth, where
    %   th is the vector of hyperparameters. This is needed for
    %   example with FIC sparse approximation.
    %
    %  See also
    %   GPCF_PERIODIC_PAK, GPCF_PERIODIC_UNPAK, GPCF_PERIODIC_E, GP_G

        gpp=gpcf.p;
        [n, m] =size(x);

        i1=0;i2=1;
        gp_period=gpcf.period;
        DKff={};
        gprior=[];
        
        % Evaluate: DKff{1} = d Kff / d magnSigma2
        %           DKff{2} = d Kff / d lengthScale
        % NOTE! Here we have already taken into account that the parameters are transformed
        % through log() and thus dK/dlog(p) = p * dK/dp

        % evaluate the gradient for training covariance
        if nargin == 2
            Cdm = gpcf_periodic_trcov(gpcf, x);
            
            ii1=1;
            DKff{ii1} = Cdm;

            if isfield(gpcf,'metric')
                error('Covariance function not compatible with metrics');
            else
                % loop over all the lengthScales
                if length(gpcf.lengthScale) == 1
                    % In the case of isotropic PERIODIC
                    s = 2./gpcf.lengthScale.^2;
                    dist = 0;
                    for i=1:m
                        D = sin(pi.*bsxfun(@minus,x(:,i),x(:,i)')./gp_period);
                        dist = dist + 2.*D.^2;
                    end
                    D = Cdm.*s.*dist;
                    
                    ii1 = ii1+1;
                    DKff{ii1} = D;
                else
                    % In the case ARD is used
                    for i=1:m
                        s = 2./gpcf.lengthScale(i).^2;
                        dist = sin(pi.*bsxfun(@minus,x(:,i),x(:,i)')./gp_period);
                        D = Cdm.*s.*2.*dist.^2;
                        
                        ii1 = ii1+1;
                        DKff{ii1} = D;
                    end
                end
                
                if gpcf.decay == 1
                    if length(gpcf.lengthScale_sexp) == 1
                        % In the case of isotropic PERIODIC
                        s = 1./gpcf.lengthScale_sexp.^2;
                        dist = 0;
                        for i=1:m
                            D = bsxfun(@minus,x(:,i),x(:,i)');
                            dist = dist + D.^2;
                        end
                        D = Cdm.*s.*dist;
                    
                        ii1 = ii1+1;
                        DKff{ii1} = D;
                    else
                        % In the case ARD is used
                        for i=1:m
                            s = 1./gpcf.lengthScale_sexp(i).^2;
                            dist = bsxfun(@minus,x(:,i),x(:,i)');
                            D = Cdm.*s.*dist.^2;
                        
                            ii1 = ii1+1;
                            DKff{ii1} = D;
                        end
                    end
                end
                
                if gpcf.optimPeriod == 1
                    % Evaluate help matrix for calculations of derivatives
                    % with respect to the period
                    if length(gpcf.lengthScale) == 1
                    % In the case of an isotropic PERIODIC
                        s = repmat(1./gpcf.lengthScale.^2, 1, m);
                    
                 
                    dist = 0;
                    for i=1:m
                        dist = dist + 2.*pi./gp_period.*sin(2.*pi.*bsxfun(@minus,x(:,i),x(:,i)')./gp_period).*bsxfun(@minus,x(:,i),x(:,i)').*s(i);                        
                    end
                    D = Cdm.*dist;
                    ii1=ii1+1;
                    DKff{ii1} = D;
                    else
                    % In the case ARD is used
                    for i=1:m
                        s = 1./gpcf.lengthScale(i).^2;        % set the length
                        dist = 2.*pi./gp_period.*sin(2.*pi.*bsxfun(@minus,x(:,i),x(:,i)')./gp_period).*bsxfun(@minus,x(:,i),x(:,i)');
                        D = Cdm.*s.*dist;
                        
                        ii1=ii1+1;
                        DKff{ii1} = D;
                    end
                    end
                end
                
            end
            % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
        elseif nargin == 3
            if size(x,2) ~= size(x2,2)
                error('gpcf_periodic -> _ghyper: The number of columns in x and x2 has to be the same. ')
            end
            
            ii1=1;
            K = feval(gpcf.fh.cov, gpcf, x, x2);
            DKff{ii1} = K;
            
            if isfield(gpcf,'metric')                
               error('Covariance function not compatible with metrics');
            else 
                % Evaluate help matrix for calculations of derivatives with respect to the lengthScale
                if length(gpcf.lengthScale) == 1
                    % In the case of an isotropic PERIODIC
                    s = 1./gpcf.lengthScale.^2;
                    dist = 0; dist2 = 0;
                    for i=1:m
                        dist = dist + 2.*sin(pi.*bsxfun(@minus,x(:,i),x2(:,i)')./gp_period).^2;                        
                    end
                    DK_l = 2.*s.*K.*dist;
                    
                    ii1=ii1+1;
                    DKff{ii1} = DK_l;
                else
                    % In the case ARD is used
                    for i=1:m
                        s = 1./gpcf.lengthScale(i).^2;        % set the length
                        dist = 2.*sin(pi.*bsxfun(@minus,x(:,i),x2(:,i)')./gp_period);
                        DK_l = 2.*s.*K.*dist.^2;
                        
                        ii1=ii1+1;
                        DKff{ii1} = DK_l;
                    end
                end
                
                if gpcf.decay == 1
                    % Evaluate help matrix for calculations of derivatives with
                    % respect to the lengthScale_sexp
                    if length(gpcf.lengthScale_sexp) == 1
                        % In the case of an isotropic PERIODIC
                        s = 1./gpcf.lengthScale_sexp.^2;
                        dist = 0; dist2 = 0;
                        for i=1:m
                            dist = dist + bsxfun(@minus,x(:,i),x2(:,i)').^2;                        
                        end
                        DK_l = s.*K.*dist;
                    
                        ii1=ii1+1;
                        DKff{ii1} = DK_l;
                    else
                        % In the case ARD is used
                        for i=1:m
                            s = 1./gpcf.lengthScale_sexp(i).^2;        % set the length
                            dist = bsxfun(@minus,x(:,i),x2(:,i)');
                            DK_l = s.*K.*dist.^2;
                        
                            ii1=ii1+1;
                            DKff{ii1} = DK_l;
                        end
                    end
                end
                
                if gpcf.optimPeriod == 1
                    % Evaluate help matrix for calculations of derivatives
                    % with respect to the period
                    if length(gpcf.lengthScale) == 1
                    % In the case of an isotropic PERIODIC
                        s = repmat(1./gpcf.lengthScale.^2, 1, m);
                    else
                        s = 1./gpcf.lengthScale.^2;
                    end
                        dist = 0; dist2 = 0;
                        for i=1:m
                            dist = dist + 2.*pi./gp_period.*sin(2.*pi.*bsxfun(@minus,x(:,i),x2(:,i)')./gp_period).*bsxfun(@minus,x(:,i),x2(:,i)').*s(i);                        
                        end
                        DK_l = K.*dist;
                    
                        ii1=ii1+1;
                        DKff{ii1} = DK_l;
                    
                    % In the case ARD is used
                    for i=1:m
                        s = 1./gpcf.lengthScale(i).^2;        % set the length
                        dist = 2.*pi./gp_period.*sin(2.*pi.*bsxfun(@minus,x(:,i),x2(:,i)')./gp_period).*bsxfun(@minus,x(:,i),x2(:,i)');
                        DK_l = s.*K.*dist;
                       
                        ii1=ii1+1;
                        DKff{ii1} = DK_l;
                    end
                    
                end
                
            end
            % Evaluate: DKff{1}    = d mask(Kff,I) / d magnSigma2
            %           DKff{2...} = d mask(Kff,I) / d lengthScale etc.
        elseif nargin == 4
            if isfield(gpcf,'metric')
               error('Covariance function not compatible with metrics');
            else
                ii1=1;
                DKff{ii1} = feval(gpcf.fh.trvar, gpcf, x);   % d mask(Kff,I) / d magnSigma2
                for i2=1:length(gpcf.lengthScale)
                    ii1 = ii1+1;
                    DKff{ii1}  = 0;                          % d mask(Kff,I) / d lengthScale
                end
                if gpcf.decay == 1
                    for i2=1:length(gpcf.lengthScale_sexp)
                        ii1 = ii1+1;
                        DKff{ii1}  = 0;                      % d mask(Kff,I) / d lengthScale_sexp
                    end
                end
                if gpcf.optimPeriod == 1
                    ii1 = ii1+1;                             % d mask(Kff,I) / d period
                    DKff{ii1}  = 0;
                end
            end
        end
       
        if nargout > 1
            if isfield(gpcf,'metric')
               error('Covariance function not compatible with metrics');
            else
                if ~isempty(gpcf.p.magnSigma2)            
                % Evaluate the gprior with respect to magnSigma2
                i1 = 1;
                ggs = feval(gpp.magnSigma2.fh.g, gpcf.magnSigma2, gpp.magnSigma2);
                gprior = ggs(i1).*gpcf.magnSigma2 - 1;
                end
                if ~isempty(gpcf.p.lengthScale)
                    i1=i1+1; 
                    lll = length(gpcf.lengthScale);
                    gg = feval(gpp.lengthScale.fh.g, gpcf.lengthScale, gpp.lengthScale);
                    gprior(i1:i1-1+lll) = gg(1:lll).*gpcf.lengthScale - 1;
                    gprior = [gprior gg(lll+1:end)];
                end
                if gpcf.decay == 1
                    i1=i1+1; 
                    lll = length(gpcf.lengthScale_sexp);
                    gg = feval(gpp.lengthScale_sexp.fh.g, gpcf.lengthScale_sexp, gpp.lengthScale_sexp);
                    gprior(i1:i1-1+lll) = gg(1:lll).*gpcf.lengthScale_sexp - 1;
                    gprior = [gprior gg(lll+1:end)];
                end
                if ~isempty(gpcf.p.period) && gpcf.optimPeriod == 1
                    i1=i1+1; 
                    lll = length(gpcf.period);
                    gg = feval(gpp.period.fh.g, gpcf.period, gpp.period);
                    gprior(i1:i1-1+lll) = gg(1:lll).*gpcf.period - 1;
                    gprior = [gprior gg(lll+1:end)];
                end
            end
        end
    end


    function [DKff, gprior]  = gpcf_periodic_ginput(gpcf, x, x2)
    %GPCF_PERIODIC_GINPUT     Evaluate gradient of covariance function with 
    %                     respect to x.
    %
    %  Description
    %   DKff = GPCF_PERIODIC_GHYPER(GPCF, X) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X) with respect to X (cell array with matrix
    %   elements)
    %
    %   DKff = GPCF_PERIODIC_GHYPER(GPCF, X, X2) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X2) with respect to X (cell array with matrix
    %   elements).
    %
    %  See also
    %   GPCF_PERIODIC_PAK, GPCF_PERIODIC_UNPAK, GPCF_PERIODIC_E, GP_G
        
        [n, m] =size(x);
        gp_period=gpcf.period;
        ii1 = 0;
        if length(gpcf.lengthScale) == 1
            % In the case of an isotropic PERIODIC
            s = repmat(1./gpcf.lengthScale.^2, 1, m);
            gp_period = repmat(1./gp_period, 1, m);
        else
            s = 1./gpcf.lengthScale.^2;
        end
        if gpcf.decay == 1
            if length(gpcf.lengthScale_sexp) == 1
                % In the case of an isotropic PERIODIC
                s_sexp = repmat(1./gpcf.lengthScale_sexp.^2, 1, m);
            else
                s_sexp = 1./gpcf.lengthScale_sexp.^2;
            end
        end

        if nargin == 2
            K = feval(gpcf.fh.trcov, gpcf, x);
            if isfield(gpcf,'metric')
                error('Covariance function not compatible with metrics');
            else


                for i=1:m
                    for j = 1:n
                        DK = zeros(size(K));
                        DK(j,:) = -s(i).*2.*pi./gp_period.*sin(2.*pi.*bsxfun(@minus,x(j,i),x(:,i)')./gp_period);
                        if gpcf.decay == 1
                            DK(j,:) = DK(j,:)-s_sexp(i).*bsxfun(@minus,x(j,i),x(:,i)');
                        end
                        DK = DK + DK';

                        DK = DK.*K;      % dist2 = dist2 + dist2' - diag(diag(dist2));

                        ii1 = ii1 + 1;
                        DKff{ii1} = DK;
                        gprior(ii1) = 0;
                    end
                end
            end
            
        elseif nargin == 3
            K = feval(gpcf.fh.cov, gpcf, x, x2);

            if isfield(gpcf,'metric')
                error('Covariance function not compatible with metrics');
            else
                ii1 = 0;
                for i=1:m
                    for j = 1:n
                        DK= zeros(size(K));
                        if gpcf.decay == 1
                            DK(j,:) = -s(i).*2.*pi./gp_period.*sin(2.*pi.*bsxfun(@minus,x(j,i),x2(:,i)')./gp_period)-s_sexp(i).*bsxfun(@minus,x(j,i),x2(:,i)');
                        else
                            DK(j,:) = -s(i).*2.*pi./gp_period.*sin(2.*pi.*bsxfun(@minus,x(j,i),x2(:,i)')./gp_period);
                        end
                        DK = DK.*K;

                        ii1 = ii1 + 1;
                        DKff{ii1} = DK;
                        gprior(ii1) = 0;
                    end
                end
            end
        end
    end


    function C = gpcf_periodic_cov(gpcf, x1, x2)
    % GP_PERIODIC_COV     Evaluate covariance matrix between two input vectors.
    %
    %         Description         
    %         C = GP_PERIODIC_COV(GP, TX, X) takes in covariance function of a
    %         Gaussian process GP and two matrixes TX and X that
    %         contain input vectors to GP. Returns covariance matrix
    %         C. Every element ij of C contains covariance between
    %         inputs i in TX and j in X.
    %
    %
    %         See also
    %         GPCF_PERIODIC_TRCOV, GPCF_PERIODIC_TRVAR, GP_COV, GP_TRCOV
        
        if isempty(x2)
            x2=x1;
        end
        [n1,m1]=size(x1);
        [n2,m2]=size(x2);
        gp_period=gpcf.period;

        if m1~=m2
            error('the number of columns of X1 and X2 has to be same')
        end
        
        if isfield(gpcf,'metric')
            error('Covariance function not compatible with metrics');
        else

            C=zeros(n1,n2);
            ma2 = gpcf.magnSigma2;

            % Evaluate the covariance
            if ~isempty(gpcf.lengthScale)
                s = 1./gpcf.lengthScale.^2;
                if gpcf.decay == 1
                    s_sexp = 1./gpcf.lengthScale_sexp.^2;
                end
                if m1==1 && m2==1
                    dd = bsxfun(@minus,x1,x2');
                    dist=2.*sin(pi.*dd./gp_period).^2.*s;
                    if gpcf.decay == 1
                        dist = dist + dd.^2.*s_sexp./2;
                    end
                else
                    % If ARD is not used make s a vector of
                    % equal elements
                    if size(s)==1
                        s = repmat(s,1,m1);
                    end
                    if gpcf.decay == 1
                        if size(s_sexp)==1
                            s_sexp = repmat(s_sexp,1,m1);
                        end
                    end

                    dist=zeros(n1,n2);
                    for j=1:m1
                        dd = bsxfun(@minus,x1(:,j),x2(:,j)');
                        dist = dist + 2.*sin(pi.*dd./gp_period).^2.*s(:,j);
                        if gpcf.decay == 1
                            dist = dist +dd.^2.*s_sexp(:,j)./2;
                        end
                    end
                end
                dist(dist<eps) = 0;
                C = ma2.*exp(-dist);
            end

        end
    end
    

    function C = gpcf_periodic_trcov(gpcf, x)
    % GP_PERIODIC_TRCOV     Evaluate training covariance matrix of inputs.
    %
    %         Description
    %         C = GP_PERIODIC_TRCOV(GP, TX) takes in covariance function of a
    %         Gaussian process GP and matrix TX that contains training
    %         input vectors. Returns covariance matrix C. Every
    %         element ij of C contains covariance between inputs i and
    %         j in TX
    %
    %         See also
    %         GPCF_PERIODIC_COV, GPCF_PERIODIC_TRVAR, GP_COV, GP_TRCOV

        
        if isfield(gpcf,'metric')
            error('Covariance function not compatible with metrics'); 
        else

            %C = trcov(gpcf, x);
        
            [n, m] =size(x);
            gp_period=gpcf.period;
 
            s = 1./(gpcf.lengthScale);
            s2 = s.^2;
            if size(s)==1
                s2 = repmat(s2,1,m);
                gp_period = repmat(gp_period,1,m);
            end
            if gpcf.decay == 1
                s_sexp = 1./(gpcf.lengthScale_sexp);
                s_sexp2 = s_sexp.^2;
                if size(s_sexp)==1
                    s_sexp2 = repmat(s_sexp2,1,m);
                end
            end
            
            ma = gpcf.magnSigma2;
 
            C = zeros(n,n);
            for ii1=1:n-1
                d = zeros(n-ii1,1);
                col_ind = ii1+1:n;
                for ii2=1:m
                    d = d+2.*s2(ii2).*sin(pi.*(x(col_ind,ii2)-x(ii1,ii2))./gp_period(ii2)).^2;
                    if gpcf.decay == 1
                        d=d+s_sexp2(ii2)./2.*(x(col_ind,ii2)-x(ii1,ii2)).^2;
                    end
                end
                C(col_ind,ii1) = d;

            end
            C(C<eps) = 0;
            C = C+C';
            C = ma.*exp(-C);
        end
    end

    function C = gpcf_periodic_trvar(gpcf, x)
    % GP_PERIODIC_TRVAR     Evaluate training variance vector
    %
    %         Description
    %         C = GP_PERIODIC_TRVAR(GPCF, TX) takes in covariance function 
    %         of a Gaussian process GPCF and matrix TX that contains
    %         training inputs. Returns variance vector C. Every
    %         element i of C contains variance of input i in TX
    %
    %
    %         See also
    %         GPCF_PERIODIC_COV, GP_COV, GP_TRCOV


        [n, m] =size(x);

        C = ones(n,1)*gpcf.magnSigma2;
        C(C<eps)=0;
    end

    function reccf = gpcf_periodic_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %
    %          Description
    %          RECCF = GPCF_PERIODIC_RECAPPEND(RECCF, RI, GPCF)
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
            reccf.type = 'gpcf_periodic';

            % Initialize parameters
            reccf.lengthScale= [];
            reccf.magnSigma2 = [];
            reccf.optimPeriod=[];
            reccf.lengthScale_sexp = [];
            reccf.period = [];
           

            % Set the function handles
            reccf.fh.pak = @gpcf_periodic_pak;
            reccf.fh.unpak = @gpcf_periodic_unpak;
            reccf.fh.e = @gpcf_periodic_e;
            reccf.fh.g = @gpcf_periodic_g;
            reccf.fh.cov = @gpcf_periodic_cov;
            reccf.fh.trcov  = @gpcf_periodic_trcov;
            reccf.fh.trvar  = @gpcf_periodic_trvar;
            reccf.fh.recappend = @gpcf_periodic_recappend;
             reccf.p=[];
            reccf.p.lengthScale=[];
            reccf.p.magnSigma2=[];
            if gpcf.decay == 1
                reccf.p.lengthScale_sexp=[];
                if ~isempty(ri.p.lengthScale_sexp)
                reccf.p.lengthScale_sexp = ri.p.lengthScale_sexp;
                end
            end
            
            if gpcf.optimPeriod == 1
                reccf.p.period=[];
                if ~isempty(ri.p.period)
                    reccf.p.period= ri.p.period;
                end
            end
            if isfield(ri.p,'lengthScale') && ~isempty(ri.p.lengthScale)
                reccf.p.lengthScale = ri.p.lengthScale;
            end
            if ~isempty(ri.p.magnSigma2)
                reccf.p.magnSigma2 = ri.p.magnSigma2;
            end
            return
        end

        gpp = gpcf.p;
        
        % record lengthScale
            if ~isempty(gpcf.lengthScale)
                reccf.lengthScale(ri,:)=gpcf.lengthScale;
                reccf.p.lengthScale = feval(gpp.lengthScale.fh.recappend, reccf.p.lengthScale, ri, gpcf.p.lengthScale);
            elseif ri==1
                reccf.lengthScale=[];
            end
        
        % record magnSigma2
        if ~isempty(gpcf.magnSigma2)
            reccf.magnSigma2(ri,:)=gpcf.magnSigma2;
        elseif ri==1
            reccf.magnSigma2=[];
        end
        
        % record lengthScale_sexp
        if ~isempty(gpcf.lengthScale_sexp) && gpcf.decay == 1
            reccf.lengthScale_sexp(ri,:)=gpcf.lengthScale_sexp;
            reccf.p.lengthScale_sexp = feval(gpp.lengthScale_sexp.fh.recappend, reccf.p.lengthScale_sexp, ri, gpcf.p.lengthScale_sexp);

        elseif ri==1
            reccf.lengthScale_sexp=[];
        end
        
            % record period
            if ~isempty(gpcf.period)
                reccf.period(ri,:)=gpcf.period;
                reccf.p.period = feval(gpp.period.fh.recappend, reccf.p.period, ri, gpcf.p.period);

            elseif ri==1
                reccf.period=[];
            end
            
       % record optimPeriod
            if ~isempty(gpcf.optimPeriod)
                reccf.optimPeriod(ri,:)=gpcf.optimPeriod;
            elseif ri==1
                reccf.optimPeriod=[];
            end
       % record decay
            if ~isempty(gpcf.decay)
                reccf.decay(ri,:)=gpcf.decay;
            elseif ri==1
                reccf.decay=[];
            end
      

    end
end