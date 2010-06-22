function gpcf = gpcf_ppcs1(do, varargin)
%GPCF_PPCS1   Create a piece wise polynomial (q=1) covariance function 
%
%	Description
%
%       GPCF = GPCF_PPCS1('INIT', 'nin', NIN, OPTIONS) Create and
%       initialize piece wise polynomial covariance function for
%       Gaussian process for input dimension NIN. OPTIONS is optional
%       parameter-value pair used as described below by
%       GPCF_PPCS1('set',...
%
%       GPCF = GPCF_PPCS1('SET', GPCF, OPTIONS) Set the fields of GPCF
%        as described by the parameter-value pairs ('FIELD', VALUE) in
%        the OPTIONS. The fields that can be modified are:
%
%             'magnSigma2'        : Magnitude (squared) for exponential 
%                                   part. (default 0.1)
%             'lengthScale'       : Length scale for each input. This 
%                                   can be either scalar corresponding 
%                                   to an isotropic function or vector 
%                                   defining own length-scale for each 
%                                   input direction. (default 10).
%             'l_nin'             : set gpcf.l = floor(l_nin/2) + 1. 
%                                   This parameter defines the order 
%                                   of the polynomial. Default is 
%                                   floor(nin/2) + 1  and this can 
%                                   only be increased
%             'magnSigma2_prior'  : prior structure for magnSigma2
%             'lengthScale_prior' : prior structure for lengthScale
%             'metric'            : metric structure into the 
%                                   covariance function
%
%       Note! If the prior structure is set to empty matrix
%       (e.g. 'magnSigma2_prior', []) then the parameter in question
%       is considered fixed and it is not handled in optimization,
%       grid integration, MCMC etc.
%
%       The piecewise polynomial function is the following:
%
%           k(x_i, x_j) = ma.*cs.^(l+1).*((l+1).*rn + 1);
%
%       where r = sum( (x_i,d - x_j,d).^2./l^2_d )
%             l = floor(l_nin/2) + 2  
%             cs = max(0,1-r);
%       and l_nin must be greater or equal to gpcf.nin
%       
%       NOTE2! Use of gpcf_ppcs1 requires that you have installed
%       GPstuff with SuiteSparse.
%
%	See also
%       gpcf_matern32, gp_init, gp_e, gp_g, gp_trcov, gp_cov, gp_unpak, gp_pak
    
% Copyright (c) 2009-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    
    ip=inputParser;
    ip.FunctionName = 'GPCF_PPCS1';
    ip.addRequired('do', @(x) ismember(x, {'init','set'}));
    ip.addOptional('gpcf', [], @isstruct);
    ip.addParamValue('nin',[], @(x) isscalar(x) && x>0 && mod(x,1)==0);
    ip.addParamValue('magnSigma2',[], @(x) isscalar(x) && x>0);
    ip.addParamValue('lengthScale',[], @(x) isvector(x) && all(x>0));
    ip.addParamValue('l_nin',[], @(x) isscalar(x) && x>0 && mod(x,1)==0);
    ip.addParamValue('metric',[], @isstruct);
    ip.addParamValue('magnSigma2_prior',[], @(x) isstruct(x) || isempty(x));
    ip.addParamValue('lengthScale_prior',[], @(x) isstruct(x) || isempty(x));
    ip.parse(do, varargin{:});
    do=ip.Results.do;
    gpcf=ip.Results.gpcf;
    nin=ip.Results.nin;
    magnSigma2=ip.Results.magnSigma2;
    lengthScale=ip.Results.lengthScale;
    l_nin=ip.Results.l_nin;
    metric=ip.Results.metric;
    magnSigma2_prior=ip.Results.magnSigma2_prior;
    lengthScale_prior=ip.Results.lengthScale_prior;
    
    switch do
      case 'init'
        % Initialize the covariance function
        if isempty(nin)
            error('nin has to be given in init: gpcf_ppcs1(''init'',''nin'',NIN,...)')
        end

        % Check that SuiteSparse is available
        if isempty(which('ldlchol'))
            error('SuiteSparse is not installed (or it is not in the path). gpcf_ppcs1 cannot be used!')
        end
        
        gpcf.type = 'gpcf_ppcs1';
        gpcf.nin = nin;
      
        if isempty(l_nin)
            gpcf.l = floor(nin/2) + 2;
        else
            if l_nin < gpcf.nin
                error('The l_nin has to be greater than or equal to the number of inputs!')
            end
            gpcf.l = floor(l_nin/2) + 2;
        end
      
        % cf is compactly supported
        gpcf.cs = 1;
        
        % Initialize parameters
        if isempty(lengthScale)
            gpcf.lengthScale = repmat(10, 1, nin); 
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
        if isempty(lengthScale_prior)
            gpcf.p.lengthScale=prior_unif('init');
        else
            gpcf.p.lengthScale=lengthScale_prior;
        end
        if isempty(magnSigma2_prior)
            gpcf.p.magnSigma2=prior_unif('init');
        else
            gpcf.p.magnSigma2=magnSigma2_prior;
        end
        
        % Initialize metric
        if ~isempty(metric)
            gpcf.metric = metric;
            gpcf = rmfield(gpcf, 'lengthScale');
        end
        
        % Set the function handles to the nested functions
        gpcf.fh_pak = @gpcf_ppcs1_pak;
        gpcf.fh_unpak = @gpcf_ppcs1_unpak;
        gpcf.fh_e = @gpcf_ppcs1_e;
        gpcf.fh_ghyper = @gpcf_ppcs1_ghyper;
        gpcf.fh_ginput = @gpcf_ppcs1_ginput;
        gpcf.fh_cov = @gpcf_ppcs1_cov;
        gpcf.fh_trcov  = @gpcf_ppcs1_trcov;
        gpcf.fh_trvar  = @gpcf_ppcs1_trvar;
        gpcf.fh_recappend = @gpcf_ppcs1_recappend;
        
      case 'set'
        % Set the parameter values of covariance function
        % go through all the parameter values that are changed
        if isempty(gpcf)
            error('with set option you have to provide the old covariance structure.')
        end        
        if ~isempty(magnSigma2);gpcf.magnSigma2=magnSigma2;end
        if ~isempty(lengthScale);gpcf.lengthScale=lengthScale;end
        if ~isempty(l_nin);
            if l_nin < gpcf.nin
                error('The l_nin has to be greater than egual to the number of inputs!')
            end
            gpcf.l=floor(l_nin/2) + 2;
        end
        if ~isempty(metric)
            gpcf.metric=metric;
            if isfield(gpcf, 'lengthScale')
                gpcf = rmfield(gpcf, 'lengthScale');
            end
        end
        if ~isempty(magnSigma2_prior);gpcf.p.magnSigma2=magnSigma2_prior;end
        if ~isempty(lengthScale_prior);gpcf.p.lengthScale=lengthScale_prior;end
    end
    
    function w = gpcf_ppcs1_pak(gpcf, w)
    %GPCF_PPCS1_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %   W = GPCF_PPCS1_PAK(GPCF) takes a covariance function data
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
    %	See also
    %	GPCF_PPCS1_UNPAK
        
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




    function [gpcf, w] = gpcf_ppcs1_unpak(gpcf, w)
    %GPCF_PPCS1_UNPAK  Sets the covariance function parameters pack into the structure
    %
    %	Description
    %   [GPCF, W] = GPCF_PPCS1_UNPAK(GPCF, W) takes a covariance
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
    %	GPCF_PPCS1_PAK
        
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
    
    function eprior =gpcf_ppcs1_e(gpcf, x, t)
    %GPCF_PPCS1_E     Evaluate the energy of prior of PPCS1 parameters
    %
    %	Description
    %   E = GPCF_PPCS1_E(GPCF, X, T) takes a covariance function data
    %   structure GPCF together with a matrix X of input vectors and a
    %   vector T of target vectors and evaluates log p(th) x J, where
    %   th is a vector of PPCS1 parameters and J is the Jacobian of
    %   transformation exp(w) = th. (Note that the parameters are log
    %   transformed, when packed.) 
    %
    %   Also the log prior of the hyperparameters of the covariance
    %   function parameters is added to E if hyper-hyperprior is
    %   defined.
    %
    %	See also
    %	GPCF_PPCS1_PAK, GPCF_PPCS1_UNPAK, GPCF_PPCS1_G, GP_E
        
        eprior = 0;
        gpp=gpcf.p;
        
        [n, m] =size(x);

        if isfield(gpcf,'metric')
            
            if ~isempty(gpcf.p.magnSigma2)
                eprior=eprior + feval(gpp.magnSigma2.fe, gpcf.magnSigma2, gpp.magnSigma2.a) -log(gpcf.magnSigma2);
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
    
    function [DKff, gprior]  = gpcf_ppcs1_ghyper(gpcf, x, x2, mask)
    %GPCF_PPCS1_GHYPER     Evaluate gradient of covariance function and hyper-prior with 
    %                     respect to the hyperparameters.
    %
    %	Description
    %	[DKff, GPRIOR] = GPCF_PPCS1_GHYPER(GPCF, X) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X) with respect to th (cell array with matrix
    %   elements), and GPRIOR = d log (p(th))/dth, where th is the
    %   vector of hyperparameters
    %
    %	[DKff, GPRIOR] = GPCF_PPCS1_GHYPER(GPCF, X, X2) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X2) with respect to th (cell array with matrix
    %   elements), and GPRIOR = d log (p(th))/dth, where th is the
    %   vector of hyperparameters
    %
    %	[DKff, GPRIOR] = GPCF_PPCS1_GHYPER(GPCF, X, [], MASK) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the diagonal of gradients of
    %   covariance matrix Kff = k(X,X2) with respect to th (cell array
    %   with matrix elements), and GPRIOR = d log (p(th))/dth, where
    %   th is the vector of hyperparameters. This is needed for
    %   example with FIC sparse approximation.
    %
    %	See also
    %   GPCF_PPCS1_PAK, GPCF_PPCS1_UNPAK, GPCF_PPCS1_E, GP_G
        
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
            Cdm = gpcf_ppcs1_trcov(gpcf, x);
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
                
                const1 = l+1;
                                        
                Dd = -(l+1).*cs.^l.*(const1.*d +1 );
                Dd = Dd + cs.^(l+1).*const1;
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
                        % In the case of isotropic ppcs1
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
                        const1 = l+1;
                        
                        D = -(l+1).*cs.^l.*(const1.*d +1 );
                        D = D + cs.^(l+1).*const1;
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
                        
                        %const1 = 2.*l^2+8.*l+6;
                        %const2 = (l+2)*0.5*const1;
                        %const3 = -ma2/3.*cs.^(l+1);
                        %Dd = const3.*(cs.*(const1.*d+3*l+6)-(const2.*d2+(l+2)*(3*l+6).*d+(l+2)*3));
                        %int = d ~= 0;
                        
                        const1 = l+1;
                        
                        Dd = -(l+1).*cs.^l.*(const1.*d +1 );
                        Dd = Dd + cs.^(l+1).*const1;
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
                    
                const1 = l+1;
                Dd = -(l+1).*cs.^l.*(const1.*d +1 );
                Dd = Dd + cs.^(l+1).*const1;
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
                        % In the case of isotropic ppcs1
                        s2 = 1./gpcf.lengthScale.^2;
                        ma2 = gpcf.magnSigma2;
                        
                        % Calculate the sparse distance (lower triangle) matrix
                        dist1 = 0;
                        for i=1:m
                            dist1 = dist1 + s2.*(bsxfun(@minus,x(:,i),x2(:,i)')).^2;
                        end
                        d1 = sqrt(dist1); 
                        cs1 = max(1-d1,0);
                        
                        const1 = l+1;
                        
                        DK_l = -(l+1).*cs1.^l.*(const1.*d1 +1 );
                        DK_l = DK_l + cs1.^(l+1).*const1;
                        DK_l = -d1.*ma2.*DK_l;
                        
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
                            dist1 = dist1 + s2(i).*bsxfun(@minus,x(:,i),x2(:,i)').^2;
                            d_l1{i} = s2(i).*(bsxfun(@minus,x(:,i),x2(:,i)')).^2;
                        end
                        d1 = sqrt(dist1); 
                        cs1 = max(1-d1,0);
                        
                        const1 = l+1;
                        
                        D = -(l+1).*cs1.^l.*(const1.*d1 +1 );
                        D = D + cs1.^(l+1).*const1;
                        D = ma2.*D;
                        
                        for i = 1:m
                            % Calculate the gradient matrix
                            DK_l = -D.*d_l1{i};
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
    
    function DKff  = gpcf_ppcs1_ginput(gpcf, x, x2)
    %GPCF_PPCS1_GINPUT     Evaluate gradient of covariance function with 
    %                     respect to x.
    %
    %	Description
    %	DKff = GPCF_PPCS1_GHYPER(GPCF, X) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X) with respect to X (cell array with matrix
    %   elements)
    %
    %	DKff = GPCF_PPCS1_GHYPER(GPCF, X, X2) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X2) with respect to X (cell array with matrix
    %   elements).
    %
    %	See also
    %   GPCF_PPCS1_PAK, GPCF_PPCS1_UNPAK, GPCF_PPCS1_E, GP_G
        
        [n, m] =size(x);

        i1=0;i2=1;
        DKff = {};
        gprior = [];
        
        % evaluate the gradient for training covariance
        if nargin == 2
            K = gpcf_ppcs1_trcov(gpcf, x);
            ii1=0;
            
            l = gpcf.l;
            [I,J] = find(K);
            
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
                    
                    [gd, gprior_dist] = feval(gpcf.metric.ginput, gpcf.metric, x(col_ind,:), x(ii,:));

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
                
                const1 = l+1;
                                        
                Dd = -(l+1).*cs.^l.*(const1.*d +1 );
                Dd = Dd + cs.^(l+1).*const1;
                Dd = ma2.*Dd;
                                               
                for i=1:length(gdist)
                    ii1 = ii1+1;
                    D = Dd.*gdist{i};
                    D = sparse(I,J,D,n,n);
                    DKff{ii1} = D + D';
                end
                
            else
                if length(gpcf.lengthScale) == 1
                    % In the case of an isotropic SEXP
                    s2 = repmat(1./gpcf.lengthScale.^2, 1, m);
                else
                    s2 = 1./gpcf.lengthScale.^2;
                end
                ma2 = gpcf.magnSigma2;
                        
                % Calculate the sparse distance (lower triangle) matrix
                % and the distance matrix for each component
                d2 = 0;
                for i = 1:m
                    d2 = d2 + s2(i).*(x(I,i) - x(J,i)).^2;
                end
                d = sqrt(d2);
                        
                % Create the 'compact support' matrix, that is, (1-R)_+,
                % where ()_+ truncates all non-positive inputs to zero.
                cs = 1-d;
                
                %const1 = 2.*l^2+8.*l+6;
                %const2 = (l+2)*0.5*const1;
                %const3 = -ma2/3.*cs.^(l+1);
                %Dd = const3.*(cs.*(const1.*d+3*l+6)-(const2.*d2+(l+2)*(3*l+6).*d+(l+2)*3));
                
                Dd = -(l+1).*cs.^l.*( (l+1).*d +1 );
                Dd = Dd + cs.^(l+1).*(l+1);
                        
                Dd = sparse(I,J,ma2.*Dd,n,n);
                d = sparse(I,J,d,n,n);
                        
                row = ones(n,1);
                cols = 1:n;
                for i = 1:m
                    for j = 1:n
                        % Calculate the gradient matrix
                        ind = find(d(:,j));
                        apu = full(Dd(:,j)).*s2(i).*(x(j,i)-x(:,i));
                        apu(ind) = apu(ind)./d(ind,j);
                        D = sparse(row*j, cols, apu, n, n);
                        D = D+D';
                                               
                        ii1 = ii1+1;
                        DKff{ii1} = D;
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
            n2 = size(x2,1);

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
                    [gd, gprior_dist] = feval(gpcf.metric.ginput, gpcf.metric, x, x2(ii,:));
                    
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
                    
                const1 = l+1;
                Dd = -(l+1).*cs.^l.*(const1.*d +1 );
                Dd = Dd + cs.^(l+1).*const1;
                Dd = ma2.*Dd;
                
                for i=1:length(gdist)
                    ii1 = ii1+1;
                    D = Dd.*gdist{i};
                    D = sparse(I,J,D,n1,n2);
                    DKff{ii1} = D;
                end

            else
                if length(gpcf.lengthScale) == 1
                    % In the case of an isotropic SEXP
                    s2 = repmat(1./gpcf.lengthScale.^2, 1, m);
                else
                    s2 = 1./gpcf.lengthScale.^2;
                end
                ma2 = gpcf.magnSigma2;
                
                % Calculate the sparse distance (lower triangle) matrix
                % and the distance matrix for each component
                dist1 = 0; 
                for i = 1:m
                    dist1 = dist1 + s2(i).*bsxfun(@minus,x(:,i),x2(:,i)').^2;
                end
                d = sqrt(dist1); 
                cs1 = max(1-d,0);
                
                const1 = l+1;
                        
                Dd = -(l+1).*cs1.^l.*(const1.*d +1 );
                Dd = Dd + cs1.^(l+1).*const1;
                Dd = ma2.*Dd;
                
                row = ones(n2,1);
                cols = 1:n2;
                for i = 1:m
                    for j = 1:n
                        % Calculate the gradient matrix
                        ind = find(d(j,:));
                        apu = Dd(j,:).*s2(i).*(x(j,i)-x2(:,i))';
                        apu(ind) = apu(ind)./d(j,ind);
                        D = sparse(row*j, cols, apu, n, n2);
                                               
                        ii1 = ii1+1;
                        DKff{ii1} = D;
                    end
                end

            end
        end
    end
    
    
    function C = gpcf_ppcs1_cov(gpcf, x1, x2, varargin)
    % GP_PPCS1_COV     Evaluate covariance matrix between two input vectors.
    %
    %         Description         
    %         C = GP_PPCS1_COV(GP, TX, X) takes in covariance function of a
    %         Gaussian process GP and two matrixes TX and X that
    %         contain input vectors to GP. Returns covariance matrix
    %         C. Every element ij of C contains covariance between
    %         inputs i in TX and j in X.
    %
    %
    %         See also
    %         GPCF_PPCS1_TRCOV, GPCF_PPCS1_TRVAR, GP_COV, GP_TRCOV

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
            const1 = l+1;
            C = ma.*cs.^(l+1).*(const1.*r + 1);
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
            
            const1 = l+1;
            C = ma.*cs.^(l+1).*(const1.*r + 1);
            C = sparse(I,J,C,n1,n2) + sparse(I0,J0,ma,n1,n2);
        end
    end
    
    function C = gpcf_ppcs1_trcov(gpcf, x)
    % GP_PPCS1_TRCOV     Evaluate training covariance matrix of inputs.
    %
    %         Description
    %         C = GP_PPCS1_TRCOV(GP, TX) takes in covariance function of a
    %         Gaussian process GP and matrix TX that contains training
    %         input vectors. Returns covariance matrix C. Every
    %         element ij of C contains covariance between inputs i and
    %         j in TX
    %
    %         See also
    %         GPCF_PPCS1_COV, GPCF_PPCS1_TRVAR, GP_COV, GP_TRCOV
        
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
            const1 = l+1;
            cs = max(0,1-rn);
            C = ma.*cs.^(l+1).*(const1.*rn + 1);
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
                const1 = l+1;
                cs = max(0,1-rn);
                C = ma.*cs.^(l+1).*(const1.*rn + 1);
                C = sparse(I,J,C,n,n);
                C = C + C' + sparse(1:n,1:n,ma,n,n);
            end
        end
    end    
    
    function C = gpcf_ppcs1_trvar(gpcf, x)
    % GP_PPCS1_TRVAR     Evaluate training variance vector
    %
    %         Description
    %         C = GP_PPCS1_TRVAR(GPCF, TX) takes in covariance function 
    %         of a Gaussian process GPCF and matrix TX that contains
    %         training inputs. Returns variance vector C. Every
    %         element i of C contains variance of input i in TX
    %
    %
    %         See also
    %         GPCF_PPCS1_COV, GP_COV, GP_TRCOV
        
        [n, m] =size(x);
        
        C = ones(n,1)*gpcf.magnSigma2;
        C(C<eps)=0;
    end

    function reccf = gpcf_ppcs1_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %
    %          Description
    %          RECCF = GPCF_PPCS1_RECAPPEND(RECCF, RI, GPCF)
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
            reccf.type = 'gpcf_ppcs1';
            reccf.nin = ri.nin;
            reccf.nout = 1;
            reccf.l = floor(reccf.nin/2)+4;

            % cf is compactly supported
            reccf.cs = 1;
            
            % Initialize parameters
            reccf.lengthScale= [];
            reccf.magnSigma2 = [];
            
            % Set the function handles
            reccf.fh_pak = @gpcf_ppcs1_pak;
            reccf.fh_unpak = @gpcf_ppcs1_unpak;
            reccf.fh_e = @gpcf_ppcs1_e;
            reccf.fh_g = @gpcf_ppcs1_g;
            reccf.fh_cov = @gpcf_ppcs1_cov;
            reccf.fh_trcov  = @gpcf_ppcs1_trcov;
            reccf.fh_trvar  = @gpcf_ppcs1_trvar;
            %  gpcf.fh_sampling = @hmc2;
            %  reccf.sampling_opt = hmc2_opt;
            reccf.fh_recappend = @gpcf_ppcs1_recappend;  
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