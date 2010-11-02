function gpcf = gpcf_sexp(varargin)
%GPCF_SEXP  Create a squared exponential covariance function
%
%  Description
%    GPCF = GPCF_SEXP('PARAM1',VALUE1,'PARAM2,VALUE2,...) creates
%    squared exponential covariance function structure in which the
%    named parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%    GPCF = GPCF_SEXP(GPCF,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a covariance function structure with the named
%    parameters altered with the specified values.
%  
%    Parameters for squared exponential covariance function [default]
%      magnSigma2        - magnitude (squared) [0.1]
%      lengthScale       - length scale for each input. [10]
%                          This can be either scalar corresponding
%                          to an isotropic function or vector
%                          defining own length-scale for each input
%                          direction.
%      magnSigma2_prior  - prior for magnSigma2  [prior_unif]
%      lengthScale_prior - prior for lengthScale [prior_unif]
%      metric            - metric structure used by the covariance function []
%
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%  See also
%    GP_SET, GPCF_*, PRIOR_*, METRIC_*

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
    ip.FunctionName = 'GPCF_SEXP';
    ip.addOptional('gpcf', [], @isstruct);
    ip.addParamValue('magnSigma2',[], @(x) isscalar(x) && x>0);
    ip.addParamValue('lengthScale',[], @(x) isvector(x) && all(x>0));
    ip.addParamValue('metric',[], @isstruct);
    ip.addParamValue('magnSigma2_prior',NaN, @(x) isstruct(x) || isempty(x));
    ip.addParamValue('lengthScale_prior',NaN, @(x) isstruct(x) || isempty(x));
    ip.parse(varargin{:});
    gpcf=ip.Results.gpcf;
    magnSigma2=ip.Results.magnSigma2;
    lengthScale=ip.Results.lengthScale;
    metric=ip.Results.metric;
    magnSigma2_prior=ip.Results.magnSigma2_prior;
    lengthScale_prior=ip.Results.lengthScale_prior;

    switch do
        case 'init'
            gpcf.type = 'gpcf_sexp';

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

            % Initialize prior structure
            gpcf.p=[];
            if ~isstruct(lengthScale_prior)&isnan(lengthScale_prior)
                gpcf.p.lengthScale=prior_unif;
            else
                gpcf.p.lengthScale=lengthScale_prior;
            end
            if ~isstruct(magnSigma2_prior)&isnan(magnSigma2_prior)
                gpcf.p.magnSigma2=prior_unif;
            else
                gpcf.p.magnSigma2=magnSigma2_prior;
            end


            %Initialize metric
            if ~isempty(metric)
                gpcf.metric = metric;
                gpcf = rmfield(gpcf, 'lengthScale');
            end

            % Set the function handles to the nested functions
            gpcf.fh.pak = @gpcf_sexp_pak;
            gpcf.fh.unpak = @gpcf_sexp_unpak;
            gpcf.fh.e = @gpcf_sexp_e;
            gpcf.fh.ghyper = @gpcf_sexp_ghyper;
            gpcf.fh.ghypergrad = @gpcf_sexp_ghypergrad;
            gpcf.fh.ghypergrad2 = @gpcf_sexp_ghypergrad2;
            gpcf.fh.ginput = @gpcf_sexp_ginput;
            gpcf.fh.ginput2 = @gpcf_sexp_ginput2;
            gpcf.fh.ginput3 = @gpcf_sexp_ginput3;
            gpcf.fh.ginput4 = @gpcf_sexp_ginput4;
            gpcf.fh.cov = @gpcf_sexp_cov;
            gpcf.fh.trcov  = @gpcf_sexp_trcov;
            gpcf.fh.trvar  = @gpcf_sexp_trvar;
            gpcf.fh.recappend = @gpcf_sexp_recappend;

        case 'set'
            % Set the parameter values of covariance function
            % go through all the parameter values that are changed
            if isempty(gpcf)
                error('with set option you have to provide the old covariance structure.')
            end
            if ~isempty(magnSigma2);gpcf.magnSigma2=magnSigma2;end
            if ~isempty(lengthScale);gpcf.lengthScale=lengthScale;end
            if ~isempty(metric)
                gpcf.metric=metric;
                if isfield(gpcf, 'lengthScale')
                    gpcf = rmfield(gpcf, 'lengthScale');
                end
            end
            if ~isstruct(magnSigma2_prior)&isnan(magnSigma2_prior);else;gpcf.p.magnSigma2=magnSigma2_prior;end
            if ~isstruct(lengthScale_prior)&isnan(lengthScale_prior);else;gpcf.p.lengthScale=lengthScale_prior;end
    end

    
    function w = gpcf_sexp_pak(gpcf)
    %GPCF_SEXP_PAK       Combine GP covariance function hyper-parameters into one vector.
    %
    %  Description
    %   W = GPCF_SEXP_PAK(GPCF) takes a covariance function data
    %   structure GPCF and combines the covariance function parameters
    %   and their hyperparameters into a single row vector W and takes
    %   a logarithm of the covariance function parameters.
    %
    %       w = [ log(gpcf.magnSigma2)
    %             (hyperparameters of gpcf.magnSigma2) 
    %             log(gpcf.lengthScale(:))
    %             (hyperparameters of gpcf.lengthScale)]'
    %     
    %
    %  See also
    %   GPCF_SEXP_UNPAK
        
        i1=0;i2=1;
        ww = []; w = [];
        
        if ~isempty(gpcf.p.magnSigma2)
            i1 = i1+1;
            w(i1) = log(gpcf.magnSigma2);
            
            % Hyperparameters of magnSigma2
            ww = feval(gpcf.p.magnSigma2.fh.pak, gpcf.p.magnSigma2);
        end        
        
        if isfield(gpcf,'metric')
            
            w = [w feval(gpcf.metric.pak, gpcf.metric)];
        else
            if ~isempty(gpcf.p.lengthScale)
                w = [w log(gpcf.lengthScale)];
                            
                % Hyperparameters of lengthScale
                w = [w feval(gpcf.p.lengthScale.fh.pak, gpcf.p.lengthScale)];
            end
        end
        w = [w ww];
    end




    function [gpcf, w] = gpcf_sexp_unpak(gpcf, w)
    %GPCF_SEXP_UNPAK  Sets the covariance function parameters pack into the structure
    %
    %  Description
    %   [GPCF, W] = GPCF_SEXP_UNPAK(GPCF, W) takes a covariance
    %   function data structure GPCF and a hyper-parameter vector W,
    %   and returns a covariance function data structure identical to
    %   the input, except that the covariance hyper-parameters have
    %   been set to the values in W. Deletes the values set to GPCF
    %   from W and returns the modified W.
    %
    %   Assignment is inverse of  
    %       w = [ log(gpcf.magnSigma2)
    %             (hyperparameters of gpcf.magnSigma2) 
    %             log(gpcf.lengthScale(:))
    %             (hyperparameters of gpcf.lengthScale)]'
    %
    %  See also
    %   GPCF_SEXP_PAK
    
        gpp=gpcf.p;
        if ~isempty(gpp.magnSigma2)
            i1=1;
            gpcf.magnSigma2 = exp(w(i1));
            assert(gpcf.magnSigma2>0 && ~isinf(gpcf.magnSigma2))
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
                assert(all(gpcf.lengthScale>0 & isfinite(gpcf.lengthScale)))
                w = w(i2+1:end);
                                
                % Hyperparameters of lengthScale
                [p, w] = feval(gpcf.p.lengthScale.fh.unpak, gpcf.p.lengthScale, w);
                gpcf.p.lengthScale = p;
            end
        end
        
        if ~isempty(gpp.magnSigma2)
            % Hyperparameters of magnSigma2
            [p, w] = feval(gpcf.p.magnSigma2.fh.unpak, gpcf.p.magnSigma2, w);
            gpcf.p.magnSigma2 = p;
        end
    end
    
    function eprior =gpcf_sexp_e(gpcf, x, t)
    %GPCF_SEXP_E     Evaluate the energy of prior of SEXP parameters
    %
    %  Description
    %   E = GPCF_SEXP_E(GPCF, X, T) takes a covariance function data
    %   structure GPCF together with a matrix X of input vectors and a
    %   vector T of target vectors and evaluates log p(th) x J, where
    %   th is a vector of SEXP parameters and J is the Jacobian of
    %   transformation exp(w) = th. (Note that the parameters are log
    %   transformed, when packed.) 
    %
    %   Also the log prior of the hyperparameters of the covariance
    %   function parameters is added to E if hyper-hyperprior is
    %   defined.
    %
    %  See also
    %   GPCF_SEXP_PAK, GPCF_SEXP_UNPAK, GPCF_SEXP_G, GP_E
    
        eprior = 0;
        gpp=gpcf.p;
        
        [n, m] =size(x);
        if ~isempty(gpcf.p.magnSigma2)
                eprior = feval(gpp.magnSigma2.fh.e, gpcf.magnSigma2, gpp.magnSigma2) - log(gpcf.magnSigma2);
        end

        if isfield(gpcf,'metric')
            eprior = eprior + feval(gpcf.metric.e, gpcf.metric, x, t);
        else
            % Evaluate the prior contribution to the error. The parameters that
            % are sampled are from space W = log(w) where w is all the "real" samples.
            % On the other hand errors are evaluated in the W-space so we need take
            % into account also the  Jacobian of transformation W -> w = exp(W).
            % See Gelman et.all., 2004, Bayesian data Analysis, second edition, p24.

            if ~isempty(gpp.lengthScale)
                eprior = eprior + feval(gpp.lengthScale.fh.e, gpcf.lengthScale, gpp.lengthScale) - sum(log(gpcf.lengthScale));
            end
        end
    end

    function [DKff, gprior]  = gpcf_sexp_ghyper(gpcf, x, x2, mask)  % , t, g, gdata, gprior, varargin
    %GPCF_SEXP_GHYPER     Evaluate gradient of covariance function and hyper-prior with 
    %                     respect to the hyperparameters.
    %
    %  Description
    %   [DKff, GPRIOR] = GPCF_SEXP_GHYPER(GPCF, X) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X) with respect to th (cell array with matrix
    %   elements), and GPRIOR = d log (p(th))/dth, where th is the
    %   vector of hyperparameters
    %
    %   [DKff, GPRIOR] = GPCF_SEXP_GHYPER(GPCF, X, X2) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X2) with respect to th (cell array with matrix
    %   elements), and GPRIOR = d log (p(th))/dth, where th is the
    %   vector of hyperparameters
    %
    %   [DKff, GPRIOR] = GPCF_SEXP_GHYPER(GPCF, X, [], MASK) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the diagonal of gradients of
    %   covariance matrix Kff = k(X,X2) with respect to th (cell array
    %   with matrix elements), and GPRIOR = d log (p(th))/dth, where
    %   th is the vector of hyperparameters. This is needed for
    %   example with FIC sparse approximation.
    %
    %  See also
    %   GPCF_SEXP_PAK, GPCF_SEXP_UNPAK, GPCF_SEXP_E, GP_G

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
            Cdm = gpcf_sexp_trcov(gpcf, x);
            ii1=0;

            if ~isempty(gpcf.p.magnSigma2)
                ii1 = ii1 +1;
                DKff{ii1} = Cdm;
            end

            if isfield(gpcf,'metric')
                dist = feval(gpcf.metric.distance, gpcf.metric, x);
                [gdist, gprior_dist] = feval(gpcf.metric.ghyper, gpcf.metric, x);
                for i=1:length(gdist)
                    ii1 = ii1+1;
                    DKff{ii1} = -Cdm.*dist.*gdist{i};
                end
            else
                if ~isempty(gpcf.p.lengthScale)
                    % loop over all the lengthScales
                    if length(gpcf.lengthScale) == 1
                        % In the case of isotropic SEXP
                        s = 2./gpcf.lengthScale.^2;
                        dist = 0;
                        for i=1:m
                            D = bsxfun(@minus,x(:,i),x(:,i)');
                            dist = dist + D.^2;
                        end
                        D = Cdm.*s.*dist./2;
                        
                        ii1 = ii1+1;
                        DKff{ii1} = D;
                    else
                        % In the case ARD is used
                        for i=1:m
                            s = 2./gpcf.lengthScale(i).^2;
                            dist = bsxfun(@minus,x(:,i),x(:,i)');
                            D = Cdm.*s.*dist.^2./2;
                            
                            ii1 = ii1+1;
                            DKff{ii1} = D;
                        end
                    end
                end
            end
            % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
        elseif nargin == 3
            if size(x,2) ~= size(x2,2)
                error('gpcf_sexp -> _ghyper: The number of columns in x and x2 has to be the same. ')
            end
            
            ii1=0;
            K = feval(gpcf.fh.cov, gpcf, x, x2);
            
            if ~isempty(gpcf.p.magnSigma2)
                ii1 = ii1 +1;
                DKff{ii1} = K;
            end
            
            if isfield(gpcf,'metric')                
                dist = feval(gpcf.metric.distance, gpcf.metric, x, x2);
                [gdist, gprior_dist] = feval(gpcf.metric.ghyper, gpcf.metric, x, x2);
                for i=1:length(gdist)
                    ii1 = ii1+1;                    
                    DKff{ii1} = -K.*dist.*gdist{i};                    
                end
            else
                if ~isempty(gpcf.p.lengthScale)
                    % Evaluate help matrix for calculations of derivatives with respect to the lengthScale
                    if length(gpcf.lengthScale) == 1
                        % In the case of an isotropic SEXP
                        s = 1./gpcf.lengthScale.^2;
                        dist = 0; dist2 = 0;
                        for i=1:m
                            dist = dist + (bsxfun(@minus,x(:,i),x2(:,i)')).^2;                        
                        end
                        DK_l = s.*K.*dist;
                        
                        ii1=ii1+1;
                        DKff{ii1} = DK_l;
                    else
                        % In the case ARD is used
                        for i=1:m
                            s = 1./gpcf.lengthScale(i).^2;        % set the length
                            dist = bsxfun(@minus,x(:,i),x2(:,i)');
                            DK_l = s.*K.*dist.^2;
                            
                            ii1=ii1+1;
                            DKff{ii1} = DK_l;
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
                DKff{ii1} = feval(gpcf.fh.trvar, gpcf, x);   % d mask(Kff,I) / d magnSigma2
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
            i1=0;
            if ~isempty(gpcf.p.magnSigma2)            
                % Evaluate the gprior with respect to magnSigma2
                i1 = i1+1;
                ggs = feval(gpp.magnSigma2.fh.g, gpcf.magnSigma2, gpp.magnSigma2);
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
                    gg = feval(gpp.lengthScale.fh.g, gpcf.lengthScale, gpp.lengthScale);
                    gprior(i1:i1-1+lll) = gg(1:lll).*gpcf.lengthScale - 1;
                    gprior = [gprior gg(lll+1:end)];
                end
            end
            if length(ggs) > 1
                gprior = [gprior ggs(2:end)];
            end
        end
    end

function DKff  = gpcf_sexp_ghypergrad(gpcf, x)
    %GPCF_SEXP_GHYPERGRAD     Evaluate gradient of covariance function, of which has been taken
    %                         partial derivative with respect to x, with respect to hyperparameters. 
    %
    %  Description
    %   DKff = GPCF_SEXP_GHYPERGRAD(GPCF, X) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of derivatived covariance
    %   matrix dK(df,f)/dhyp = d(d k(X,X)/dx)/dhyp, with respect to the hyperparameters
    %
    %   Evaluate: DKff{1:m} = d Kff / d magnSigma2
    %             DKff{m+1:2m} = d Kff / d lengthScale_m
    %   m is the dimension of inputs. If ARD is used, then multiple
    %   lengthScales.
    %
    %  See also
    %   GPCF_SEXP_GINPUT
    
        [n, m] =size(x);
        ii1=0;
        Cdm = feval(gpcf.fh.ginput4, gpcf, x);
            
        % grad with respect to MAGNSIGMA
        if ~isempty(gpcf.p.magnSigma2)
            if m==1
                ii1 = ii1 +1;
                DKff{ii1} = Cdm{1};
            else
                DKffapu = cat(1,Cdm{1:m});
                ii1=ii1+1;
                DKff{ii1}=DKffapu;
            end
        end
        
        % grad with respect to LENGTHSCALE
        if isfield(gpcf,'metric')
            error('Metric doesnt work with grad.obs')
        else
            if ~isempty(gpcf.p.lengthScale)
                % loop over all the lengthScales
                if length(gpcf.lengthScale) == 1
                    % In the case of isotropic SEXP
                    s = 1./gpcf.lengthScale.^2;
                    dist = 0;
                    for i=1:m
                        D = bsxfun(@minus,x(:,i),x(:,i)');
                        dist = dist + D.^2;
                    end
                    % input dimension is 1
                    if m==1
                        G = Cdm{1}.*(dist.*s - 2); 
                        ii1 = ii1+1;
                        DKff{ii1} = G;
                    % input dimension is >1    
                    else
                        for i=1:m
                            G{i} = 2.*Cdm{i}.*(dist.*s./2 - 1);
                        end
                        DKffapu=cat(1,G{1:m});
                        ii1 = ii1+1;
                        DKff{ii1} = DKffapu;
                    end
                else
                     % In the case ARD is used
                     if m~=length(gpcf.lengthScale)
                         error('Amount of lengtscales dont match input dimension')
                     end
                     %Preparing
                    for i=1:m
                        dist{i} = bsxfun(@minus,x(:,i),x(:,i)').^2;
                        s(i) = 1./gpcf.lengthScale(i);
                    end

                    for i=1:m
                        for j=1:m
                            % if structure is to check: is x derivative different from lengthscale
                            % derivative
                            if j~=i
                                D{j}= Cdm{j}.*dist{i}.*s(i).^2;
                            else
                                D{j} = Cdm{i}.*(dist{i}.*s(i).^2 - 2);
                            end
                        end
                        ii1=ii1+1;
                        DKffapu2{i}=cat(1,D{1:m});
                        DKff{ii1}=DKffapu2{i};
                    end
                end
            end
        end
    end

    function DKff  = gpcf_sexp_ghypergrad2(gpcf, x)
    %GPCF_SEXP_GHYPERGRAD2     Evaluate gradient of covariance function, of which has been taken
    %                          partial derivatives with respect to both input variables x, with respect to hyperparameters. 
    %
    %  Description
    %   DKff = GPCF_SEXP_GHYPERGRAD2(GPCF, X) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of derivatived covariance
    %   matrix dK(df,df)/dhyp = d(d² k(X1,X2)/dX1dX2)/dhyp with respect to the hyperparameters
    %
    %   Evaluate: DKff{1-m} = d Kff / d magnSigma2
    %             DKff{m+1-2m} = d Kff / d lengthScale_m
    %   m is the dimension of inputs. If ARD is used, then multiple
    %   lengthScales.
    %
    %  See also
    %   GPCF_SEXP_GINPUT, GPCF_SEXP_GINPUT2 
    
        [n, m] =size(x);
        DKff = {};
        [DKdd, DKdd3, DKdd4] = feval(gpcf.fh.ginput2, gpcf, x, x);
        ii1=0;

        if m>1
            % Cross derivative matrices (non-diagonal).
            DKdda=feval(gpcf.fh.ginput3, gpcf, x,x);

            %MAGNSIGMA 
            %add matrices to the diagonal of help matrix, size (m*n,m*n)
             DKffapu=blkdiag(DKdd{:});
             
            % add non-diagonal matrices 
            if m==2
                DKffapund=[zeros(n,n) DKdda{1};DKdda{1} zeros(n,n)];
            else
                t1=1;
                DKffapund=zeros(m*n,m*n);
                for i=1:m-1
                    aa=zeros(m-1,m);
                    t2=t1+m-2-(i-1);
                    aa(m-1,i)=1;
                    k=kron(aa,cat(1,zeros((i)*n,n),DKdda{t1:t2}));
                    k(1:n*(m),:)=[];
                    k=k+k';
                    DKffapund = DKffapund + k;
                    t1=t2+1;
                end
            end
            
            DKffapu=DKffapu+DKffapund;
        end
            
         % grad with respect to MAGNSIGMA
         if ~isempty(gpcf.p.magnSigma2)
             if m==1
                 ii1 = ii1 +1;
                 DKff{ii1} = DKdd{1};
             else
                 ii1=ii1+1;
                 DKff{ii1}=DKffapu;
             end
         else
             error('no prior set to magnSigma')
         end  
         
        % grad with respect to LENGTHSCALE
        % metric doesn't work with grad obs
        if isfield(gpcf,'metric')
            error('metric doesnt work with grad.obs')
        else
            if ~isempty(gpcf.p.lengthScale)
                % loop over all the lengthScales
                if length(gpcf.lengthScale) == 1
                    % In the case of isotropic SEXP
                    s = 1./gpcf.lengthScale;
                    dist = 0;
                    for i=1:m
                        D = bsxfun(@minus,x(:,i),x(:,i)');
                        dist = dist + D.^2;
                    end
                        if m==1
                            %diagonal matrices
                            ii1 = ii1+1;
                            DKff{ii1} = DKdd3{1}.*(dist.*s.^2 - 2)-DKdd4{1}.*(dist.*s.^2 - 4);
                        else
                            %diagonal matrices
                            for i=1:m
                                DKffdiag{i} = DKdd3{i}.*(dist.*s.^2 - 2) - DKdd4{i}.*(dist.*s.^2 - 4);
                            end
                            
                            %nondiag.matrices
                            %how many pairs = num, m=2 -> 1 pair, m=3 -> 3pairs
                            % m=4 -> 6 pairs
                            num=1;
                            if m>2
                                for i=2:m-1
                                    num=num+i;
                                end
                            end
                            for i=1:num    
                                DKffnondiag{i} = DKdda{i}.*(dist.*s.^2-4);
                            end
                            
                            % Gather matrices to diagonal
                            DKffapu2=blkdiag(DKffdiag{:});

                            % non-diagonal matrices   
                            if m==2
                                DKffapu2nd=[zeros(n,n) DKffnondiag{1};DKffnondiag{1} zeros(n,n)];
                            else
                                t1=1;
                                DKffapu2nd=zeros(m*n,m*n);
                                for i=1:m-1
                                    aa=zeros(m-1,m);
                                    t2=t1+m-2-(i-1);
                                    aa(m-1,i)=1;
                                    k=kron(aa,cat(1,zeros((i)*n,n),DKffnondiag{t1:t2}));
                                    k(1:n*(m),:)=[];
                                    k=k+k';
                                    DKffapu2nd = DKffapu2nd + k;
                                    t1=t2+1;
                                end
                            end
                            ii1=ii1+1;
                            DKff{ii1}=DKffapu2+DKffapu2nd;
                        end
                else
                % In the case ARD is used
                % Now lengthScale derivatives differ from the case where
                % there's only one lengthScale, so here we take that to account
            
                    %Preparing, Di is diagonal help matrix and NDi
                    %is non-diagonal help matrix
                    for i=1:m
                        Di2{i}=zeros(n,n);
                        NDi{i}=zeros(m*n,m*n);
                        s(i) = 1./gpcf.lengthScale(i);
                        D = bsxfun(@minus,x(:,i),x(:,i)').*s(i);
                        dist{i} = D.^2;                        
                    end
                    
                    % diagonal matrices for each lengthScale

                    for j=1:m
                        for i=1:m
                            % same x and lengthscale derivative
                            if i==j
                                  Di2{i} = DKdd3{i}.*(dist{i} - 2) - DKdd4{i}.*(dist{i} - 4);
                            end
                            % different x and lengthscale derivative
                            if i~=j
                                  Di2{i}=DKdd3{i}.*dist{j} - DKdd4{i}.*dist{j};
                            end
                        end
                        Di{j}=blkdiag(Di2{:});
                    end 
                    
                    %Non-diagonal matrices
                    if m==2
                        for k=1:2
                            Dnondiag=DKdda{1}.*(dist{k}-2);
                            NDi{k}=[zeros(n,n) Dnondiag;Dnondiag zeros(n,n)];
                        end
                    else
                        for k=1:m
                            ii3=0;
                            NDi{k}=zeros(m*n,m*n);
                            for j=0:m-2
                                for i=1+j:m-1
                                     ii3=ii3+1;
                                     sar=j*1+1;
                                     riv=i+1;
                                     % if lengthscale and either x derivate dimensions
                                     % are same, else if not.
                                     if sar==k || riv==k
                                        Dnondiag{i}=DKdda{ii3}.*(dist{k}-2);
                                     else
                                        Dnondiag{i}=DKdda{ii3}.*dist{k};
                                     end
                                end
                                aa=zeros(m-1,m);
                                aa(m-1,j+1)=1;
                                kk=kron(aa,cat(1,zeros((j+1)*n,n),Dnondiag{1+j:m-1}));
                                kk(1:n*(m),:)=[];
                                kk=kk+kk';
                                NDi{k} = NDi{k} + kk;
                            end
                        end
                    end

                    %and the final matrix is diag. + non-diag matrices
                    for i=1:m
                        ii1=ii1+1;
                        DKff{ii1}=NDi{i}+Di{i};        
                    end
                end
            end
        end
    
    end


    function DKff  = gpcf_sexp_ginput(gpcf, x, x2)
    %GPCF_SEXP_GINPUT     Evaluate gradient of covariance function with 
    %                     respect to x.
    %
    %  Description
    %   DKff = GPCF_SEXP_GHYPER(GPCF, X) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X) with respect to X (cell array with matrix
    %   elements)
    %
    %   DKff = GPCF_SEXP_GHYPER(GPCF, X, X2) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X2) with respect to X (cell array with matrix
    %   elements).
    %
    %  See also
    %   GPCF_SEXP_PAK, GPCF_SEXP_UNPAK, GPCF_SEXP_E, GP_G
        
        [n, m] =size(x);
        ii1 = 0;
        if nargin == 2
            K = feval(gpcf.fh.trcov, gpcf, x);
            if isfield(gpcf,'metric')
                dist = feval(gpcf.metric.distance, gpcf.metric, x);
                gdist = feval(gpcf.metric.ginput, gpcf.metric, x);
                for i=1:length(gdist)
                    ii1 = ii1+1;
                    DKff{ii1} = -K.*dist.*gdist{ii1};
                end
            else
                if length(gpcf.lengthScale) == 1
                    % In the case of an isotropic SEXP
                    s = repmat(1./gpcf.lengthScale.^2, 1, m);
                else
                    s = 1./gpcf.lengthScale.^2;
                end
                for i=1:m
                    for j = 1:n
                        DK = zeros(size(K));
                        DK(j,:) = -s(i).*bsxfun(@minus,x(j,i),x(:,i)');
                        DK = DK + DK';
                        
                        DK = DK.*K;      % dist2 = dist2 + dist2' - diag(diag(dist2));
                        
                        ii1 = ii1 + 1;
                        DKff{ii1} = DK;
                    end
                end
            end
            
        elseif nargin == 3
            K = feval(gpcf.fh.cov, gpcf, x, x2);
            
            if isfield(gpcf,'metric')
                dist = feval(gpcf.metric.distance, gpcf.metric, x, x2);
                gdist = feval(gpcf.metric.ginput, gpcf.metric, x, x2);
                for i=1:length(gdist)
                    ii1 = ii1+1;
                    DKff{ii1}   = -K.*dist.*gdist{ii1};
                end
            else
                
                if length(gpcf.lengthScale) == 1
                    % In the case of an isotropic SEXP
                    s = repmat(1./gpcf.lengthScale.^2, 1, m);
                else
                    s = 1./gpcf.lengthScale.^2;
                end
                
                ii1 = 0;
                for i=1:m
                    for j = 1:n
                        DK= zeros(size(K));
                        DK(j,:) = -s(i).*bsxfun(@minus,x(j,i),x2(:,i)');
                        
                        DK = DK.*K;
                        
                        ii1 = ii1 + 1;
                        DKff{ii1} = DK;
                    end
                end
            end
        end
    end

    function [DKff, DKff1, DKff2]  = gpcf_sexp_ginput2(gpcf, x, x2)
    %GPCF_SEXP_GINPUT2     Evaluate gradient of covariance function with
    %                      respect to both input variables x and x2 (in
    %                      same dimension).
    %
    %  Description
    %   DKff = GPCF_SEXP_GINPUT2(GPCF, X, X2) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of twice derivatived covariance
    %   matrix K(df,df) = dk(X1,X2)/dX1dX2 (cell array with matrix elements).
    %   Input variable's dimensions are expected to be same.
    %   The function returns also DKff1 and DKff2 which are parts
    %   of DKff and needed with GHYPERGRAD2. DKff = DKff1 - DKff2.
    %   
    %  See also
    %   GPCF_SEXP_GINPUT, GPCF_SEXP_GINPUT2, GPCF_SEXP_GHYPERGRAD2 
    
        [n, m] =size(x);
        [n2,m2] =size(x2);
        ii1 = 0;
        if nargin ~= 3
            error('Needs 3 input arguments')
        end
   
        if isequal(x,x2)
           K = feval(gpcf.fh.trcov, gpcf, x); 
        else
           K = feval(gpcf.fh.cov, gpcf, x, x2);
        end

        %metric doesn't work with grad.obs on
        if isfield(gpcf,'metric')
            error('Metric doesnt work with grad.obs')
        else
            if length(gpcf.lengthScale) == 1
                % In the case of an isotropic SEXP
                s = repmat(1./gpcf.lengthScale.^2, 1, m);
            else
                s = 1./gpcf.lengthScale.^2;
            end

            for i=1:m
                DK2 = s(i).^2.*bsxfun(@minus,x(:,i),x2(:,i)').^2.*K;
                DK = s(i).*K;     
                ii1 = ii1 + 1;
                DKff1{ii1} = DK;
                DKff2{ii1} = DK2;
                DKff{ii1} = DK - DK2;
            end
        end
    end

    function DKff  = gpcf_sexp_ginput3(gpcf, x, x2)
    %GPCF_SEXP_GINPUT3     Evaluate gradient of covariance function with
    %                      respect to both input variables x and x2 (in different dimensions). 
    %
    %  Description
    %   DKff = GPCF_SEXP_GINPUT3(GPCF, X, X2) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of twice derivatived covariance
    %   matrix K(df,df) = dk(X1,X2)/dX1dX2 (cell array with matrix elements). 
    %   The derivative is calculated in multidimensional problem between
    %   input's observation dimensions which are not same .
    %   
    %  See also
    %   GPCF_SEXP_GINPUT, GPCF_SEXP_GINPUT2, GPCF_SEXP_GHYPERGRAD2 
        
        [n, m] =size(x);
        [n2,m2] =size(x2);
        if nargin ~= 3
            error('Needs 3 input arguments')
        end

        if isequal(x,x2)
           K = feval(gpcf.fh.trcov, gpcf, x); 
        else
           K = feval(gpcf.fh.cov, gpcf, x, x2);
        end
        
        % Derivative the cov.function with respect to both input variables
        % but in different dimensions. Resulting matrices are for the cov. matrix
        % k(df/dx,df/dx) non-diagonal part. Matrices are added to DKff in columnwise order
        % for ex. dim=3: k(df/dx1,df/dx2),(..dx1,dx3..),(..dx2,dx3..)
        
        if isfield(gpcf,'metric')
            error('Metric doesnt work with ginput3')
        else
            if length(gpcf.lengthScale) == 1
                % In the case of an isotropic SEXP
                s = repmat(1./gpcf.lengthScale.^2, 1, m);
            else
                s = 1./gpcf.lengthScale.^2;
            end
            ii3=0;
            for i=1:m-1
                for j=i+1:m
                    ii3=ii3+1;
                    DKff{ii3} = s(j).*bsxfun(@minus,x(:,j),x2(:,j)').*(-s(i).*bsxfun(@minus,x(:,i),x2(:,i)').*K);
                end
            end
        end
    end

    function DKff  = gpcf_sexp_ginput4(gpcf, x, x2)
    %GPCF_SEXP_GINPUT     Evaluate gradient of covariance function with 
    %                     respect to x. Simplified and faster version of
    %                     sexp_ginput, returns full matrices.
    %
    %  Description
    %   DKff = GPCF_SEXP_GHYPER(GPCF, X) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X) with respect to X (whole matrix)
    %
    %   DKff = GPCF_SEXP_GHYPER(GPCF, X, X2) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X2) with respect to X (whole matrix).
    %
    %  See also
    %   GPCF_SEXP_PAK, GPCF_SEXP_UNPAK, GPCF_SEXP_E, GP_G
    
        [n, m] =size(x);
        ii1 = 0;
        if nargin==2
           flag=1;
           K = feval(gpcf.fh.trcov, gpcf, x); 
        else
           flag=0;
           K = feval(gpcf.fh.cov, gpcf, x, x2);
           if isequal(x,x2)
               error('ginput4 fuktio saa vaaran inputin')
           end
        end
        
        if isfield(gpcf,'metric')
            error('no metric implemented')
        else
            if length(gpcf.lengthScale) == 1
                % In the case of an isotropic SEXP
                s = repmat(1./gpcf.lengthScale.^2, 1, m);
            else
                s = 1./gpcf.lengthScale.^2;
            end
            for i=1:m
                DK = zeros(size(K));
                if flag==1
                   DK = -s(i).*bsxfun(@minus,x(:,i),x(:,i)');
                else
                   DK = -s(i).*bsxfun(@minus,x(:,i),x2(:,i)');
                end
                DK = DK.*K;
                ii1 = ii1 + 1;
                DKff{ii1} = DK;
            end
        end
    end

    function C = gpcf_sexp_cov(gpcf, x1, x2)
    % GP_SEXP_COV     Evaluate covariance matrix between two input vectors.
    %
    %         Description         
    %         C = GP_SEXP_COV(GP, TX, X) takes in covariance function of a
    %         Gaussian process GP and two matrixes TX and X that
    %         contain input vectors to GP. Returns covariance matrix
    %         C. Every element ij of C contains covariance between
    %         inputs i in TX and j in X.
    %
    %
    %         See also
    %         GPCF_SEXP_TRCOV, GPCF_SEXP_TRVAR, GP_COV, GP_TRCOV
        
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
            C = gpcf.magnSigma2.*exp(-dist./2);            
        else
            C=zeros(n1,n2);
            ma2 = gpcf.magnSigma2;
            
            % Evaluate the covariance
            if ~isempty(gpcf.lengthScale)
                s2 = 1./gpcf.lengthScale.^2;
                if m1==1 && m2==1
                    dd = bsxfun(@minus,x1,x2');
                    dist=dd.^2*s2;
                else
                    % If ARD is not used make s a vector of
                    % equal elements
                    if size(s2)==1
                        s2 = repmat(s2,1,m1);
                    end
                    dist=zeros(n1,n2);
                    for j=1:m1
                        dd = bsxfun(@minus,x1(:,j),x2(:,j)');
                        dist = dist + dd.^2.*s2(:,j);
                    end
                end
                dist(dist<eps) = 0;
                C = ma2.*exp(-dist./2);
            end
        end
    end

    function C = gpcf_sexp_trcov(gpcf, x)
    % GP_SEXP_TRCOV     Evaluate training covariance matrix of inputs.
    %
    %         Description
    %         C = GP_SEXP_TRCOV(GP, TX) takes in covariance function of a
    %         Gaussian process GP and matrix TX that contains training
    %         input vectors. Returns covariance matrix C. Every
    %         element ij of C contains covariance between inputs i and
    %         j in TX
    %
    %         See also
    %         GPCF_SEXP_COV, GPCF_SEXP_TRVAR, GP_COV, GP_TRCOV

        if isfield(gpcf,'metric')
            % If other than scaled euclidean metric
            [n, m] =size(x);            
            ma2 = gpcf.magnSigma2;
            
            C = zeros(n,n);
            for ii1=1:n-1
                d = zeros(n-ii1,1);
                col_ind = ii1+1:n;
                d = feval(gpcf.metric.distance, gpcf.metric, x(col_ind,:), x(ii1,:)).^2;                
                C(col_ind,ii1) = d./2;
            end
            C(C<eps) = 0;
            C = C+C';
            C = ma2.*exp(-C);            
        else
            % If scaled euclidean metric
            % Try to use the C-implementation
            C = trcov(gpcf, x);
            if isnan(C)
                % If there wasn't C-implementation do here
                [n, m] =size(x);
                
                s = 1./(gpcf.lengthScale);
                s2 = s.^2;
                if size(s)==1
                    s2 = repmat(s2,1,m);
                end
                ma2 = gpcf.magnSigma2;
                
                C = zeros(n,n);
                for ii1=1:n-1
                    d = zeros(n-ii1,1);
                    col_ind = ii1+1:n;
                    for ii2=1:m
                        d = d+s2(ii2).*(x(col_ind,ii2)-x(ii1,ii2)).^2;
                    end
                    C(col_ind,ii1) = d./2;
                end
                C(C<eps)=0;
                C = C+C';
                C = ma2.*exp(-C);
            end
        end
    end

    function C = gpcf_sexp_trvar(gpcf, x)
    % GP_SEXP_TRVAR     Evaluate training variance vector
    %
    %         Description
    %         C = GP_SEXP_TRVAR(GPCF, TX) takes in covariance function 
    %         of a Gaussian process GPCF and matrix TX that contains
    %         training inputs. Returns variance vector C. Every
    %         element i of C contains variance of input i in TX
    %
    %
    %         See also
    %         GPCF_SEXP_COV, GP_COV, GP_TRCOV


        [n, m] =size(x);

        C = ones(n,1)*gpcf.magnSigma2;
        C(C<eps)=0;
    end

    function reccf = gpcf_sexp_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %
    %          Description
    %          RECCF = GPCF_SEXP_RECAPPEND(RECCF, RI, GPCF)
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
            reccf.type = 'gpcf_sexp';

            % Initialize parameters
            reccf.lengthScale= [];
            reccf.magnSigma2 = [];

            % Set the function handles
            reccf.fh.pak = @gpcf_sexp_pak;
            reccf.fh.unpak = @gpcf_sexp_unpak;
            reccf.fh.e = @gpcf_sexp_e;
            reccf.fh.g = @gpcf_sexp_g;
            reccf.fh.cov = @gpcf_sexp_cov;
            reccf.fh.trcov  = @gpcf_sexp_trcov;
            reccf.fh.trvar  = @gpcf_sexp_trvar;
            reccf.fh.recappend = @gpcf_sexp_recappend;
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
                reccf.p.lengthScale = feval(gpp.lengthScale.fh.recappend, reccf.p.lengthScale, ri, gpcf.p.lengthScale);
            elseif ri==1
                reccf.lengthScale=[];
            end
        end
        % record magnSigma2
        if ~isempty(gpcf.magnSigma2)
            reccf.magnSigma2(ri,:)=gpcf.magnSigma2;
            reccf.p.magnSigma2 = feval(gpp.magnSigma2.fh.recappend, reccf.p.magnSigma2, ri, gpcf.p.magnSigma2);
        elseif ri==1
            reccf.magnSigma2=[];
        end
    end
end
