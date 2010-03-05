function gpcf = gpcf_ppcs3(do, varargin)
%GPCF_PPCS3	Create a piece wise polynomial (q=3) covariance function for Gaussian Process
%
%	Description
%
%	GPCF = GPCF_PPCS3('INIT', NIN) Create and initialize piece wise polynomial 
%       covariance function for Gaussian process
%
%	The fields and (default values) in GPCF_ppcs3 are:
%	  type           = 'gpcf_PPCS3'
%	  nin            = Number of inputs. (NIN)
%	  nout           = Number of outputs. (always 1)
%         cs             = 1. Tells that gpcf_ppcs3 is compact support function.
%         l              = floor(nin/2) + 4. This parameter defines the order of the polynomial.
%                          You can change the order by settin field 'l_nin' to a value greater than nin.
%	  magnSigma2     = Magnitude (squared) for exponential part. 
%                          (0.1)
%	  lengthScale    = Length scale for each input. This can be either scalar corresponding 
%                          isotropic or vector corresponding ARD. 
%                          (repmat(10, 1, nin))
%         p              = Prior structure for covariance function parameters. 
%                          (e.g. p.lengthScale.)
%         fh_pak         = function handle to pack function
%                          (@gpcf_ppcs3_pak)
%         fh_unpak       = function handle to unpack function
%                          (@gpcf_ppcs3_unpak)
%         fh_e           = function handle to energy function
%                          (@gpcf_ppcs3_e)
%         fh_ghyper      = function handle to gradient of energy with respect to hyperparameters
%                          (@gpcf_ppcs3_ghyper)
%         fh_ginput      = function handle to gradient of function with respect to inducing inputs
%                          (@gpcf_ppcs3_ginput)
%         fh_cov         = function handle to covariance function
%                          (@gpcf_ppcs3_cov)
%         fh_trcov       = function handle to training covariance function
%                          (@gpcf_ppcs3_trcov)
%         fh_trvar       = function handle to training variance function
%                          (@gpcf_ppcs3_trvar)
%         fh_recappend   = function handle to append the record function 
%                          (gpcf_ppcs3_recappend)
%
%	GPCF = GPCF_PPCS3('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in GPCF.  The fields that 
%       can be modified are:
%
%             'magnSigma2'         : set the magnSigma2
%             'lengthScale'        : set the lengthScale
%             'metric'             : set the metric structure into the covariance function
%             'lengthScale_prior'  : set the prior structure for lengthScale
%             'magnSigma2_prior'   ; set the prior structure for magnSigma2
%             'l_nin'              : set gpcf.l = floor(l_nin/2) + 1
%
%
%       The piecewise polynomial function is the following:
%
%           k(x_i, x_j) = ma.*cs.^(l+3).*((l^3 + 9*l^2 + 23*l + 15).*r.^3 + ...
%                          (6*l^2 + 36*l + 45).*r.^2 + (15*l + 45).*r + 15)/15;
%
%       where r = sum( (x_i,d - x_j,d).^2./l^2_d )
%             l = floor(l_nin/2) + 4     
%             cs = max(0,1-r);
%       and l_nin must be greater or equal to gpcf.nin
%       
%
%	See also
%       gpcf_sexp, gpcf_exp, gpcf_matern32, gpcf_matern52, gp_init, gp_e, gp_g, gp_trcov
%       gp_cov, gp_unpak, gp_pak
    
% Copyright (c) 2006-2007 Jouni Hartikainen
% Copyright (c) 2006-2009 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end
    
    % Initialize the covariance function
    if strcmp(do, 'init')
        nin = varargin{1};
        gpcf.type = 'gpcf_ppcs3';
        gpcf.nin = nin;
        gpcf.nout = 1;
        gpcf.l = floor(nin/2) + 4;
        
        % cf is compactly supported
        gpcf.cs = 1;
        
        % Initialize parameters
        gpcf.lengthScale= repmat(1, 1, nin); 
        gpcf.magnSigma2 = 0.1;
        
        % Initialize prior structure
        gpcf.p=[];
        gpcf.p.lengthScale=prior_unif('init');
        gpcf.p.magnSigma2=prior_unif('init');

        
        % Set the function handles to the nested functions
        gpcf.fh_pak = @gpcf_ppcs3_pak;
        gpcf.fh_unpak = @gpcf_ppcs3_unpak;
        gpcf.fh_e = @gpcf_ppcs3_e;
        gpcf.fh_ghyper = @gpcf_ppcs3_ghyper;
        gpcf.fh_ginput = @gpcf_ppcs3_ginput;
        gpcf.fh_cov = @gpcf_ppcs3_cov;
        gpcf.fh_trcov  = @gpcf_ppcs3_trcov;
        gpcf.fh_trvar  = @gpcf_ppcs3_trvar;
        gpcf.fh_recappend = @gpcf_ppcs3_recappend;
        
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
                  case 'metric'
                    gpcf.metric = varargin{i+1};
                    gpcf = rmfield(gpcf, 'lengthScale');
                  case 'lengthScale_prior'
                    gpcf.p.lengthScale = varargin{i+1};
                  case 'magnSigma2_prior'
                    gpcf.p.magnSigma2 = varargin{i+1};
                  case 'l_nin'
                    if varargin{i+1} < gpcf.nin
                        error('The l_nin has to be greater than egual to the number of inputs!')
                    end
                    gpcf.l = floor(varargin{i+1}/2) + 4;
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
              case 'l_nin'
                if varargin{i+1} < gpcf.nin
                    error('The l_nin has to be greater than egual to the number of inputs!')
                end
                gpcf.l = floor(varargin{i+1}/2) + 4;
              otherwise
                error('Wrong parameter name!')
            end    
        end
    end
    
    function w = gpcf_ppcs3_pak(gpcf, w)
    %GPCF_ppcs3_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %	W = GPCF_ppcs3_PAK(GPCF, W) takes a covariance function data structure GPCF and
    %	combines the hyper-parameters into a single row vector W.
    %
    %	The ordering of the parameters in W is:
    %       w = [gpcf.magnSigma2 (hyperparameters of gpcf.lengthScale) gpcf.lengthScale]
    %	  
    %
    %	See also
    %	GPCF_ppcs3_UNPAK        
        
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




    function [gpcf, w] = gpcf_ppcs3_unpak(gpcf, w)
    %GPCF_ppcs3_UNPAK  Separate covariance function hyper-parameter vector into components.
    %
    %	Description
    %	[GPCF, W] = GPCF_ppcs3_UNPAK(GPCF, W) takes a covariance function data structure GPCF
    %	and  a hyper-parameter vector W, and returns a covariance function data
    %	structure  identical to the input, except that the covariance hyper-parameters 
    %   has been set to the values in W. Deletes the values set to GPCF from W and returns 
    %   the modeified W. 
    %
    %	See also
    %	GPCF_ppcs3_PAK
        
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
    
    function eprior =gpcf_ppcs3_e(gpcf, x, t)
    %GPCF_ppcs3_E     Evaluate the energy of prior of ppcs3 parameters
    %
    %	Description
    %	E = GPCF_ppcs3_E(GPCF, X, T) takes a covariance function data structure 
    %   GPCF together with a matrix X of input vectors and a matrix T of target 
    %   vectors and evaluates log p(th) x J, where th is a vector of PPCS3 parameters 
    %   and J is the Jakobian of transformation exp(w) = th. (Note that the parameters 
    %   are log transformed, when packed.)
    %
    %	See also
    %	GPCF_ppcs3_PAK, GPCF_ppcs3_UNPAK, GPCF_ppcs3_G, GP_E

        eprior = 0;
        gpp=gpcf.p;
        
        [n, m] =size(x);

        if isfield(gpcf,'metric')
            
            if ~isempty(gpcf.p.magnSigma2)
                eprior=eprior...
                       +feval(gpp.magnSigma2.fe, ...
                              gpcf.magnSigma2, gpp.magnSigma2.a)...
                       -log(gpcf.magnSigma2);
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
    
    function [DKff, gprior]  = gpcf_ppcs3_ghyper(gpcf, x, x2, mask)
    %GPCF_ppcs3_GHYPER     Evaluate gradient of covariance function and hyper-prior with 
    %                     respect to the hyperparameters.
    %
    %	Descriptioni
    %	[GPRIOR, DKff, DKuu, DKuf] = GPCF_ppcs3_GHYPER(GPCF, X, T, G, GDATA, GPRIOR, VARARGIN) 
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
    %   GPCF_ppcs3_PAK, GPCF_ppcs3_UNPAK, GPCF_ppcs3_E, GP_G
        
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
            Cdm = gpcf_ppcs3_trcov(gpcf, x);

            ii1=0;
            if ~isempty(gpcf.p.magnSigma2)
                ii1 = ii1 +1;
                DKff{ii1} = Cdm;
            end
            
            l = gpcf.l;
            [I,J] = find(Cdm);
            
            if isfield(gpcf,'metric')
                % Compute the sparse distance matrix and its gradient.
                ntriplets = (nnz(Cdm)-n)./2;
                I = zeros(ntriplets,1);
                J = zeros(ntriplets,1);
                dist = zeros(ntriplets,1);
                for jj = 1:length(gpcf.metric.components)
                    gdist{jj} = zeros(ntriplets,1);
                end
                ntriplets = 0;                
                for ii=1:n-1
                    col_ind = ii + find(Cdm(ii+1:n,ii));
                    d = zeros(length(col_ind),1);
                    d = feval(gpcf.metric.distance, gpcf.metric, x(col_ind,:), x(ii,:));
                    
                    [gd, gprior_dist] = feval(gpcf.metric.ghyper, gpcf.metric, x(col_ind,:), x(ii,:));

                    ntrip_prev = ntriplets;
                    ntriplets = ntriplets + length(d);
                    
                    ind_tr = ntrip_prev+1:ntriplets;
                    I(ind_tr) = col_ind;
                    J(ind_tr) = ii;
                    dist(ind_tr) = d;
                    for jj = 1:length(gd)
                        gdist{jj}(ind_tr) = gd{jj};
                    end
                end
                
                ma2 = gpcf.magnSigma2;
                    
                cs = 1-dist;
                
                const1 = l^3 + 9*l^2 + 23*l + 15;
                const2 = 6*l^2 + 36*l + 45;
                const3 = 15*l + 45;
                                      
                Dd = -(l+3).*cs.^(l+2).*(const1.*d.^3 + const2.*d.^2 + const3.*d + 15)/15;
                Dd = Dd + cs.^(l+3).*(3.*const1.*d.^2 + 2.*const2.*d + const3)./15;
                Dd = ma2.*Dd;
                
                for i=1:length(gdist)
                    ii1 = ii1+1;
                    D = Dd.*gdist{i};
                    D = sparse(I,J,D,n,n);
                    DKff{ii1} = D + D';
                end
                
            else
                if ~isempty(gpcf.p.lengthScale)
                    % loop over all the lengthScales
                    if length(gpcf.lengthScale) == 1
                        % In the case of isotropic ppcs3
                        s2 = 1./gpcf.lengthScale.^2;
                        ma2 = gpcf.magnSigma2;
                        
                        % Calculate the sparse distance (lower triangle) matrix
                        d2 = 0;
                        for i = 1:m
                            d2 = d2 + s2.*(x(I,i) - x(J,i)).^2;
                        end
                        d = sqrt(d2);
                        
                        % Create the 'compact support' matrix, that is, (1-R)_+,
                        % where ()_+ truncates all non-positive inputs to zero.
                        cs = 1-d;
                        
                        % Calculate the gradient matrix
                        const1 = l^3 + 9*l^2 + 23*l + 15;
                        const2 = 6*l^2 + 36*l + 45;
                        const3 = 15*l + 45;
                        
                        D = -(l+3).*cs.^(l+2).*(const1.*d.^3 + const2.*d.^2 + const3.*d + 15)/15;
                        D = D + cs.^(l+3).*(3.*const1.*d.^2 + 2.*const2.*d + const3)./15;
                        D = -d.*ma2.*D;
                        D = sparse(I,J,D,n,n);
                        
                        ii1 = ii1+1;
                        DKff{ii1} = D;
                    else
                        % In the case ARD is used
                        s2 = 1./gpcf.lengthScale.^2;
                        ma2 = gpcf.magnSigma2;
                        
                        % Calculate the sparse distance (lower triangle) matrix
                        % and the distance matrix for each component
                        d2 = 0;
                        d_l2 = [];
                        for i = 1:m
                            d_l2(:,i) = s2(i).*(x(I,i) - x(J,i)).^2;
                            d2 = d2 + d_l2(:,i);
                        end
                        d = sqrt(d2);
                        d_l = d_l2;
                        
                        % Create the 'compact support' matrix, that is, (1-R)_+,
                        % where ()_+ truncates all non-positive inputs to zero.
                        cs = 1-d;
                                            
                        const1 = l^3 + 9*l^2 + 23*l + 15;
                        const2 = 6*l^2 + 36*l + 45;
                        const3 = 15*l + 45;
                        
                        Dd = -(l+3).*cs.^(l+2).*(const1.*d.^3 + const2.*d.^2 + const3.*d + 15)/15;
                        Dd = Dd + cs.^(l+3).*(3.*const1.*d.^2 + 2.*const2.*d + const3)./15;
                        Dd = -ma2.*Dd;
                        int = d ~= 0;
                        
                        
                        for i = 1:m
                            % Calculate the gradient matrix
                            D = d_l(:,i).*Dd;
                            % Divide by r in cases where r is non-zero
                            D(int) = D(int)./d(int);
                            D = sparse(I,J,D,n,n);
                            
                            ii1 = ii1+1;
                            DKff{ii1} = D;
                        end
                    end
                end
            end
            % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
        elseif nargin == 3
            if size(x,2) ~= size(x2,2)
                error('gpcf_ppcs -> _ghyper: The number of columns in x and x2 has to be the same. ')
            end
            
            ii1=0;
            K = feval(gpcf.fh_cov, gpcf, x, x2);
            
            if ~isempty(gpcf.p.magnSigma2)
                ii1 = ii1 +1;
                DKff{ii1} = K;
            end

            l = gpcf.l;
            
            if isfield(gpcf,'metric')
                % If other than scaled euclidean metric
                [n1,m1]=size(x);
                [n2,m2]=size(x2);
                
                ma = gpcf.magnSigma2;
                
                % Compute the sparse distance matrix.
                ntriplets = nnz(K);
                I = zeros(ntriplets,1);
                J = zeros(ntriplets,1);
                R = zeros(ntriplets,1);
                dist = zeros(ntriplets,1);
                for jj = 1:length(gpcf.metric.components)
                    gdist{jj} = zeros(ntriplets,1);
                end
                ntriplets = 0;
                for ii=1:n2
                    d = zeros(n1,1);
                    d = feval(gpcf.metric.distance, gpcf.metric, x, x2(ii,:));
                    [gd, gprior_dist] = feval(gpcf.metric.ghyper, gpcf.metric, x, x2(ii,:));
                    
                    I0t = find(d==0);
                    d(d >= 1) = 0;
                    [I2,J2,R2] = find(d);
                    len = length(R);
                    ntrip_prev = ntriplets;
                    ntriplets = ntriplets + length(R2);

                    ind_tr = ntrip_prev+1:ntriplets;
                    I(ind_tr) = I2;
                    J(ind_tr) = ii;
                    dist(ind_tr) = R2;
                    for jj = 1:length(gd)
                        gdist{jj}(ind_tr) = gd{jj}(I2);
                    end
                end

                
                ma2 = gpcf.magnSigma2;
                    
                cs = 1-dist;
                    
                const1 = l^3 + 9*l^2 + 23*l + 15;
                const2 = 6*l^2 + 36*l + 45;
                const3 = 15*l + 45;
                                      
                Dd = -(l+3).*cs.^(l+2).*(const1.*d.^3 + const2.*d.^2 + const3.*d + 15)/15;
                Dd = Dd + cs.^(l+3).*(3.*const1.*d.^2 + 2.*const2.*d + const3)./15;
                Dd = ma2.*Dd;
                
                for i=1:length(gdist)
                    ii1 = ii1+1;
                    D = Dd.*gdist{i};
                    D = sparse(I,J,D,n1,n2);
                    DKff{ii1} = D;
                end

            else
                if ~isempty(gpcf.p.lengthScale)
                    % loop over all the lengthScales
                    if length(gpcf.lengthScale) == 1
                        % In the case of isotropic ppcs3
                        s2 = 1./gpcf.lengthScale.^2;
                        ma2 = gpcf.magnSigma2;
                        
                        % Calculate the sparse distance (lower triangle) matrix
                        dist1 = 0;
                        for i=1:m
                            dist1 = dist1 + s2.*(gminus(x(:,i),x2(:,i)')).^2;
                        end
                        d1 = sqrt(dist1); 
                        cs1 = max(1-d1,0);
                    
                        const1 = l^3 + 9*l^2 + 23*l + 15;
                        const2 = 6*l^2 + 36*l + 45;
                        const3 = 15*l + 45;
                        
                        D = -(l+3).*cs1.^(l+2).*(const1.*d1.^3 + const2.*d1.^2 + const3.*d1 + 15)/15;
                        D = D + cs1.^(l+3).*(3.*const1.*d1.^2 + 2.*const2.*d1 + const3)./15;
                        DK_l = -d1.*ma2.*D;
                        
                        ii1=ii1+1;
                        DKff{ii1} = DK_l;
                    else
                        % In the case ARD is used
                        s2 = 1./gpcf.lengthScale.^2;
                        ma2 = gpcf.magnSigma2;
                        
                        % Calculate the sparse distance (lower triangle) matrix
                        % and the distance matrix for each component
                        dist1 = 0; 
                        d_l1 = [];
                        for i = 1:m
                            dist1 = dist1 + s2(i).*gminus(x(:,i),x2(:,i)').^2;
                            d_l1{i} = s2(i).*(gminus(x(:,i),x2(:,i)')).^2;
                        end
                        d1 = sqrt(dist1); 
                        cs1 = max(1-d1,0);
                        
                        const1 = l^3 + 9*l^2 + 23*l + 15;
                        const2 = 6*l^2 + 36*l + 45;
                        const3 = 15*l + 45;
                        
                        D = -(l+3).*cs1.^(l+2).*(const1.*d1.^3 + const2.*d1.^2 + const3.*d1 + 15)/15;
                        D = D + cs1.^(l+3).*(3.*const1.*d1.^2 + 2.*const2.*d1 + const3)./15;
                        
                        for i = 1:m
                            % Calculate the gradient matrix
                            DK_l = -D.*ma2.*d_l1{i};
                            % Divide by r in cases where r is non-zero
                            DK_l(d1 ~= 0) = DK_l(d1 ~= 0)./d1(d1 ~= 0);
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
    
    function [gprior,DKff]  = gpcf_ppcs3_ginput(gpcf, x, x2)
    %GPCF_PPCS3_GINPUT     Evaluate gradient of covariance function with 
    %                   respect to the inducing inputs.
    %
    %	Descriptioni
    %	[GPRIOR_IND, DKuu, DKuf] = GPCF_PPCS3_GINPUT(GPCF, X, T, G, GDATA_IND, GPRIOR_IND, VARARGIN) 
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
    %   GPCF_PPCS3_PAK, GPCF_PPCS3_UNPAK, GPCF_PPCS3_E, GP_G
        
        DKuu={};
        DKff={};
    end
    
    
    function C = gpcf_ppcs3_cov(gpcf, x1, x2, varargin)
    % GP_ppcs3_COV     Evaluate covariance matrix between two input vectors.
    %
    %         Description
    %         C = GP_ppcs3_COV(GP, TX, X) takes in covariance function of a Gaussian
    %         process GP and two matrixes TX and X that contain input vectors to
    %         GP. Returns covariance matrix C. Every element ij of C contains
    %         covariance between inputs i in TX and j in X.
    %
    %
    %         See also
    %         GPCF_ppcs3_TRCOV, GPCF_ppcs3_TRVAR, GP_COV, GP_TRCOV

        if isfield(gpcf,'metric')
            % If other than scaled euclidean metric
            [n1,m1]=size(x1);
            [n2,m2]=size(x2);
            
            ma = gpcf.magnSigma2;
            l = gpcf.l;
            
            % Compute the sparse distance matrix.
            ntriplets = max(1,floor(0.03*n1*n2));
            I = zeros(ntriplets,1);
            J = zeros(ntriplets,1);
            R = zeros(ntriplets,1);
            ntriplets = 0;
            I0=zeros(ntriplets,1);
            J0=zeros(ntriplets,1);
            nn0=0;
            for ii1=1:n2
                d = zeros(n1,1);
                d = feval(gpcf.metric.distance, gpcf.metric, x1, x2(ii1,:));
                I0t = find(d==0);
                d(d >= 1) = 0;
                [I2,J2,R2] = find(d);
                len = length(R);
                ntrip_prev = ntriplets;
                ntriplets = ntriplets + length(R2);

                I(ntrip_prev+1:ntriplets) = I2;
                J(ntrip_prev+1:ntriplets) = ii1;
                R(ntrip_prev+1:ntriplets) = R2;
                I0(nn0+1:nn0+length(I0t)) = I0t;
                J0(nn0+1:nn0+length(I0t)) = ii1;
                nn0 = nn0+length(I0t);
            end
            r = sparse(I(1:ntriplets),J(1:ntriplets),R(1:ntriplets));
            [I,J,r] = find(r);
            cs = full(sparse(max(0, 1-r)));
            
            const1 = l^3 + 9*l^2 + 23*l + 15;
            const2 = 6*l^2 + 36*l + 45;
            const3 = 15*l + 45;
            C = ma.*cs.^(l+3).*(const1.*r.^3 + const2.*r.^2 + const3.*r + 15)/15;
            C = sparse(I,J,C,n1,n2) + sparse(I0,J0,ma,n1,n2);
        else
            % If scaled euclidean metric
            
            [n1,m1]=size(x1);
            [n2,m2]=size(x2);
            
            s = 1./(gpcf.lengthScale);
            s2 = s.^2;
            if size(s)==1
                s2 = repmat(s2,1,m1);
            end
            ma = gpcf.magnSigma2;
            l = gpcf.l;
            
            % Compute the sparse distance matrix.
            ntriplets = max(1,floor(0.03*n1*n2));
            I = zeros(ntriplets,1);
            J = zeros(ntriplets,1);
            R = zeros(ntriplets,1);
            ntriplets = 0;
            I0=zeros(ntriplets,1);
            J0=zeros(ntriplets,1);
            nn0=0;
            for ii1=1:n2
                d = zeros(n1,1);
                for j=1:m1
                    d = d + s2(j).*(x1(:,j)-x2(ii1,j)).^2;
                end
                d = sqrt(d);
                I0t = find(d==0);
                d(d >= 1) = 0;
                [I2,J2,R2] = find(d);
                len = length(R);
                ntrip_prev = ntriplets;
                ntriplets = ntriplets + length(R2);

                I(ntrip_prev+1:ntriplets) = I2;
                J(ntrip_prev+1:ntriplets) = ii1;
                R(ntrip_prev+1:ntriplets) = R2;
                I0(nn0+1:nn0+length(I0t)) = I0t;
                J0(nn0+1:nn0+length(I0t)) = ii1;
                nn0 = nn0+length(I0t);
            end
            r = sparse(I(1:ntriplets),J(1:ntriplets),R(1:ntriplets));
            [I,J,r] = find(r);
            cs = full(sparse(max(0, 1-r)));
            
            const1 = l^3 + 9*l^2 + 23*l + 15;
            const2 = 6*l^2 + 36*l + 45;
            const3 = 15*l + 45;
            C = ma.*cs.^(l+3).*(const1.*r.^3 + const2.*r.^2 + const3.*r + 15)/15;
            
            C = sparse(I,J,C,n1,n2) + sparse(I0,J0,ma,n1,n2);
        end
    end
    
    function C = gpcf_ppcs3_trcov(gpcf, x)
    % GP_PPCS3_TRCOV     Evaluate training covariance matrix of inputs.
    %
    %         Description
    %         C = GP_PPCS3_TRCOV(GP, TX) takes in covariance function of a Gaussian
    %         process GP and matrix TX that contains training input vectors. 
    %         Returns covariance matrix C. Every element ij of C contains covariance 
    %         between inputs i and j in TX
    %
    %
    %         See also
    %         GPCF_ppcs3_TRCOV, GPCF_ppcs3_TRVAR, GP_COV, GP_TRCOV
        
        if isfield(gpcf,'metric')
            % If other than scaled euclidean metric
            
            [n, m] =size(x);            
            ma = gpcf.magnSigma2;
            l = gpcf.l;
            
            % Compute the sparse distance matrix.
            ntriplets = max(1,floor(0.03*n*n));
            I = zeros(ntriplets,1);
            J = zeros(ntriplets,1);
            R = zeros(ntriplets,1);
            ntriplets = 0;
            for ii1=1:n-1
                d = zeros(n-ii1,1);
                col_ind = ii1+1:n;
                d = feval(gpcf.metric.distance, gpcf.metric, x(col_ind,:), x(ii1,:));
                d(d >= 1) = 0;
                [I2,J2,R2] = find(d);
                len = length(R);
                ntrip_prev = ntriplets;
                ntriplets = ntriplets + length(R2);
                if (ntriplets > len)
                    I(2*len) = 0;
                    J(2*len) = 0;
                    R(2*len) = 0;
                end
                ind_tr = ntrip_prev+1:ntriplets;
                I(ind_tr) = ii1+I2;
                J(ind_tr) = ii1;
                R(ind_tr) = R2;
            end
            R = sparse(I(1:ntriplets),J(1:ntriplets),R(1:ntriplets),n,n);
            
            % Find the non-zero elements of R.
            [I,J,rn] = find(R);
            const1 = l^3 + 9*l^2 + 23*l + 15;
            const2 = 6*l^2 + 36*l + 45;
            const3 = 15*l + 45;
            cs = max(0,1-rn);
            C = ma.*cs.^(l+3).*(const1.*rn.^3 + const2.*rn.^2 + const3.*rn + 15)/15;
            C = sparse(I,J,C,n,n);
            C = C + C' + sparse(1:n,1:n,ma,n,n);
            
        else
            % If a scaled euclidean metric try first mex-implementation 
            % and if there is not such... 
            C = trcov(gpcf,x);
            % ... evaluate the covariance here.
            if isnan(C)
                [n, m] =size(x);
                
                s = 1./(gpcf.lengthScale);
                s2 = s.^2;
                if size(s)==1
                    s2 = repmat(s2,1,m);
                end
                ma = gpcf.magnSigma2;
                l = gpcf.l;
                
                % Compute the sparse distance matrix.
                ntriplets = max(1,floor(0.03*n*n));
                I = zeros(ntriplets,1);
                J = zeros(ntriplets,1);
                R = zeros(ntriplets,1);
                ntriplets = 0;
                for ii1=1:n-1
                    d = zeros(n-ii1,1);
                    col_ind = ii1+1:n;
                    for ii2=1:m
                        d = d+s2(ii2).*(x(col_ind,ii2)-x(ii1,ii2)).^2;
                    end
                    %d = sqrt(d);
                    d(d >= 1) = 0;
                    [I2,J2,R2] = find(d);
                    len = length(R);
                    ntrip_prev = ntriplets;
                    ntriplets = ntriplets + length(R2);
                    if (ntriplets > len)
                        I(2*len) = 0;
                        J(2*len) = 0;
                        R(2*len) = 0;
                    end
                    ind_tr = ntrip_prev+1:ntriplets;
                    I(ind_tr) = ii1+I2;
                    J(ind_tr) = ii1;
                    R(ind_tr) = sqrt(R2);
                end
                R = sparse(I(1:ntriplets),J(1:ntriplets),R(1:ntriplets),n,n);
                
                % Find the non-zero elements of R.
                [I,J,rn] = find(R);
                const1 = l^3 + 9*l^2 + 23*l + 15;
                const2 = 6*l^2 + 36*l + 45;
                const3 = 15*l + 45;
                cs = max(0,1-rn);
                C = ma.*cs.^(l+3).*(const1.*rn.^3 + const2.*rn.^2 + const3.*rn + 15)/15;
                C = sparse(I,J,C,n,n);
                C = C + C' + sparse(1:n,1:n,ma,n,n);
            end
        end
    end    
    
    function C = gpcf_ppcs3_trvar(gpcf, x)
    % GP_ppcs3_TRVAR     Evaluate training variance vector
    %
    %         Description
    %         C = GP_ppcs3_TRVAR(GPCF, TX) takes in covariance function of a Gaussian
    %         process GPCF and matrix TX that contains training inputs. Returns variance 
    %         vector C. Every element i of C contains variance of input i in TX
    %
    %
    %         See also
    %         GPCF_ppcs3_COV, GP_COV, GP_TRCOV
        
        [n, m] =size(x);
        
        C = ones(n,1)*gpcf.magnSigma2;
        C(C<eps)=0;
    end

    function reccf = gpcf_ppcs3_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_ppcs3_RECAPPEND(RECCF, RI, GPCF) takes old covariance
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
            reccf.type = 'gpcf_ppcs3';
            reccf.nin = ri.nin;
            reccf.nout = 1;
            reccf.l = floor(reccf.nin/2)+4;

            % cf is compactly supported
            reccf.cs = 1;
            
            % Initialize parameters
            reccf.lengthScale= [];
            reccf.magnSigma2 = [];
            
            % Set the function handles
            reccf.fh_pak = @gpcf_ppcs3_pak;
            reccf.fh_unpak = @gpcf_ppcs3_unpak;
            reccf.fh_e = @gpcf_ppcs3_e;
            reccf.fh_g = @gpcf_ppcs3_g;
            reccf.fh_cov = @gpcf_ppcs3_cov;
            reccf.fh_trcov  = @gpcf_ppcs3_trcov;
            reccf.fh_trvar  = @gpcf_ppcs3_trvar;
            %  gpcf.fh_sampling = @hmc2;
            %  reccf.sampling_opt = hmc2_opt;
            reccf.fh_recappend = @gpcf_ppcs3_recappend;  
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

        % record lengthScale
        if ~isempty(gpcf.lengthScale)
            reccf.lengthScale(ri,:)=gpcf.lengthScale;
            reccf.p.lengthScale = feval(gpp.lengthScale.fh_recappend, reccf.p.lengthScale, ri, gpcf.p.lengthScale);
        elseif ri==1
            reccf.lengthScale=[];
        end
        % record magnSigma2
        if ~isempty(gpcf.magnSigma2)
            reccf.magnSigma2(ri,:)=gpcf.magnSigma2;
        elseif ri==1
            reccf.magnSigma2=[];
        end
    end
end    