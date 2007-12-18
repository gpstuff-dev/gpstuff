function gpcf = gpcf_sexp(do, varargin)
%GPCF_SEXP	Create a squared exponential covariance function for Gaussian Process
%
%	Description
%
%	GPCF = GPCF_SEXP('INIT', NIN) Create and initialize squared exponential 
%       covariance function fo Gaussian process 
%
%	The fields and (default values) in GPCF_SEXP are:
%	  type           = 'gpcf_sexp'
%	  nin            = number of inputs (NIN)
%	  nout           = number of outputs: always 1
%	  magnSigma2     = general magnitude (squared) for exponential part  (sampled with HMC)
%                          (0.1)
%	  lengthScale    = length scale for each input. This can be either   (sampled with HMC)
%                          scalar (corresponding isotropic) or vector (corresponding ARD).
%                          (repmat(10, 1, nin))
%         p              = prior structure for covariance function   (p.lengthScale.a.s is sampled with HMC)
%                          parameters. 
%         fh_pak         = function handle to packing function
%                          (@gpcf_sexp_pak)
%         fh_unpak       = function handle to unpackin function
%                          (@gpcf_sexp_unpak)
%         fh_e           = function handle to error function
%                          (@gpcf_sexp_e)
%         fh_ghyper      = function handle to gradient function (with respect to hyperparameters)
%                          (@gpcf_sexp_ghyper)
%         fh_gind        = function handle to gradient function (with respect to inducing inputs)
%                          (@gpcf_sexp_gind)
%         fh_cov         = function handle to covariance function
%                          (@gpcf_sexp_cov)
%         fh_trcov       = function handle to training covariance function
%                          (@gpcf_sexp_trcov)
%         fh_covvec      = function handle to elementvice covariance function
%                          (@gpcf_sexp_covvec)
%         fh_trvar       = function handle to training variance function
%                          (@gpcf_sexp_trvar)
%         fh_sampling    = function handle to parameter sampling function
%                          (@hmc2)
%         sampling_opt   = options structure for fh_sampling
%                          (hmc2_opt)
%         fh_recappend   = function handle to record append function
%                          (gpcf_sexp_recappend)
%
%	GPCF = GPCF_SEXP('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in GPCF.
%
%	See also
%	
%
%

% Copyright (c) 1998,1999 Aki Vehtari
% Copyright (c) 2006-2007 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end
    
    % Initialize the covariance function
    if strcmp(do, 'init')
        nin = varargin{1};
        gpcf.type = 'gpcf_sexp';
        gpcf.nin = nin;
        gpcf.nout = 1;
        
        % Initialize parameters
        gpcf.lengthScale= repmat(10, 1, nin); 
        gpcf.magnSigma2 = 0.1;
        
        % Initialize prior structure
        gpcf.p=[];
        gpcf.p.lengthScale=[];
        gpcf.p.magnSigma2=[];
        
        % Set the function handles to the nested functions
        gpcf.fh_pak = @gpcf_sexp_pak;
        gpcf.fh_unpak = @gpcf_sexp_unpak;
        gpcf.fh_e = @gpcf_sexp_e;
        gpcf.fh_ghyper = @gpcf_sexp_ghyper;
        gpcf.fh_gind = @gpcf_sexp_gind;
        gpcf.fh_cov = @gpcf_sexp_cov;
        gpcf.fh_covvec = @gpcf_sexp_covvec;
        gpcf.fh_trcov  = @gpcf_sexp_trcov;
        gpcf.fh_trvar  = @gpcf_sexp_trvar;
        gpcf.fh_recappend = @gpcf_sexp_recappend;
        
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
              case 'magnSigma2'
                gpcf.magnSigma2 = varargin{i+1};
              case 'lengthScale'
                gpcf.lengthScale = varargin{i+1};
              case 'fh_sampling'
                gpcf.fh_sampling = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end    
        end
    end
    
    function w = gpcf_sexp_pak(gpcf, w)
    %GPcf_SEXP_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %	W = GP_SEXP_PAK(GPCF, W) takes a Gaussian Process data structure GP and
    %	combines the hyper-parameters into a single row vector W.
    %
    %	The ordering of the parameters in HP is defined by
    %	  hp = [hyper-params of gp.cf{1}, hyper-params of gp.cf{2}, ...];
    %
    %	See also
    %	GPCF_SEXP_UNPAK
    %
        
    % Copyright (c) 2000-2001 Aki Vehtari
    % Copyright (c) 2006      Jarno Vanhatalo
        
    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
        
        gpp=gpcf.p;
        
        i1=0;i2=1;
        if ~isempty(w)
            i1 = length(w);
        end
        i1 = i1+1;
        w(i1) = gpcf.magnSigma2;
        if isfield(gpp.lengthScale, 'p') && ~isempty(gpp.lengthScale.p)
            i1=i1+1;
            w(i1)=gpp.lengthScale.a.s;
            if any(strcmp(fieldnames(gpp.lengthScale.p),'nu'))
                i1=i1+1;
                w(i1)=gpp.lengthScale.a.nu;
            end
        end
        i2=i1+length(gpcf.lengthScale);
        i1=i1+1;
        w(i1:i2)=gpcf.lengthScale;
        i1=i2;
    end




    function [gpcf, w] = gpcf_sexp_unpak(gpcf, w)
    %GPCF_SEXP_UNPAK  Separate GP covariance function hyper-parameter vector into components. 
    %
    %	Description
    %	GP = GPCF_SEXP_UNPAK(GP, W) takes an Gaussian Process data structure GP
    %	and  a hyper-parameter vector W, and returns a Gaussian Process data
    %	structure  identical to the input model, except that the covariance
    %	hyper-parameters has been set to the of W.
    %
    %	See also
    %	GP_PAK
    %
        
    % Copyright (c) 2000-2001 Aki Vehtari
    % Copyright (c) 2006      Jarno Vanhatalo
        
    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
        
        
        gpp=gpcf.p;
        i1=0;i2=1;
        i1=i1+1;
        gpcf.magnSigma2=w(i1);
        if isfield(gpp.lengthScale, 'p') && ~isempty(gpp.lengthScale.p)
            i1=i1+1;
            gpcf.p.lengthScale.a.s=w(i1);
            if any(strcmp(fieldnames(gpp.lengthScale.p),'nu'))
                i1=i1+1;
                gpcf.p.lengthScale.a.nu=w(i1);
            end
        end
        i2=i1+length(gpcf.lengthScale);
        i1=i1+1;
        gpcf.lengthScale=w(i1:i2);
        i1=i2;
        w = w(i1+1:end);
    end
    
    function eprior =gpcf_sexp_e(gpcf, x, t)
    %GPCF_SEXP_E	Evaluate prior contribution of error of covariance function SE.
    %
    %	Description
    %	E = GPCF_SEXP_E(W, GP, X, T) takes a gp data structure GPCF together
    %	with a matrix X of input vectors and a matrix T of target vectors,
    %	and evaluates the error function E. Each row of X corresponds
    %	to one input vector and each row of T corresponds to one
    %	target vector.
    %
    %	See also
    %	GP2, GP2PAK, GP2UNPAK, GP2FWD, GP2R_G
    %
        
    % Copyright (c) 1998-2006 Aki Vehtari
        
    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
        [n, m] =size(x);
        
        % Evaluate the prior contribution to the error. The parameters that
        % are sampled are from space W = log(w) where w is all the "real" samples.  
        % On the other hand errors are evaluated in the W-space so we need take 
        % into account also the  Jakobian of transformation W -> w = exp(W).
        % See Gelman et.all., 2004, Bayesian data Analysis, second edition, p24.
        eprior = 0;
        gpp=gpcf.p;
        
        eprior=eprior...
               +feval(gpp.magnSigma2.fe, ...
                      gpcf.magnSigma2, gpp.magnSigma2.a)...
               -log(gpcf.magnSigma2);
        if isfield(gpp.lengthScale, 'p') && ~isempty(gpp.lengthScale.p)
            eprior=eprior...
                   +feval(gpp.lengthScale.p.s.fe, ...
                          gpp.lengthScale.a.s, gpp.lengthScale.p.s.a)...
                   -log(gpp.lengthScale.a.s);
            if any(strcmp(fieldnames(gpp.lengthScale.p),'nu'))
                eprior=eprior...
                       +feval(gpp.p.lengthScale.nu.fe, ...
                              gpp.lengthScale.a.nu, gpp.lengthScale.p.nu.a)...
                       -log(gpp.lengthScale.a.nu);
            end
        end
        eprior=eprior...
               +feval(gpp.lengthScale.fe, ...
                      gpcf.lengthScale, gpp.lengthScale.a)...
               -sum(log(gpcf.lengthScale));
        e_x=x;
        e_t=t;
        e_ls=gpcf.lengthScale;
        e_ms=gpcf.magnSigma2;
        e_e = eprior;
    end
    
    function [g, gdata, gprior]  = gpcf_sexp_ghyper(gpcf, x, t, g, gdata, gprior, varargin)
    %GPCF_SEXP_GHYPER     Evaluate gradient of error for SE covariance function
    %                     with respect to the hyperparameters.
    %
    %	Descriptioni
    %	G = GPCF_SEXP_GHYPER(W, GPCF, X, T, G, GDATA, GPRIOR, VARARGIN) takes a gp 
    %   hyper-parameter vector W, data structure GPCF a matrix X of input vectors a 
    %   matrix T of target vectors, inverse covariance function , 
    %	and evaluates the error gradient G. Each row of X corresponds to one 
    %   input vector and each row of T corresponds to one target vector.
    %
    %	[G, GDATA, GPRIOR] = GPCF_SEXP_GHYPER(GP, P, T) also returns separately  the
    %	data and prior contributions to the gradient.
    %
    %	See also
    %

    % Copyright (c) 1998-2001 Aki Vehtari
    % Copyright (c) 2006      Jarno Vanhatalo
        
    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
        
        gpp=gpcf.p;
        [n, m] =size(x);
        
        i1=0;i2=1;
        if ~isempty(g)
            i1 = length(g);
        end
        
        % First check if sparse model is used
        switch gpcf.type
          case 'FULL' 
            % Evaluate help arguments for gradient evaluation
            % instead of calculating trace(invC*Cdm) calculate sum(invCv.*Cdm(:)), when 
            % Cdm and invC are symmetric matricess of same size. This is 67 times faster 
            % with n=215 
            invC = varargin{1};
            Cdm = gpcf_sexp_trcov(gpcf, x);
            invCv=invC(:);
            b = varargin{2};
            if length(varargin) > 4
                b2 = varargin{5};
                b3 = varargin{6};
            end
            % loop over all the lengthScales
            if length(gpcf.lengthScale) == 1
                % In the case of isotropic SEXP
                s = 2./gpcf.lengthScale.^2;
                dist = 0;
                for i=1:m
                    D = gminus(x(:,i),x(:,i)');
                    dist = dist + D.^2;
                end
                D = Cdm.*s.*dist;
                Bdl = b'*(D*b);
                Cdl = sum(invCv.*D(:)); % help arguments for lengthScale 
                %Cdl = trace(C\D); % help arguments for lengthScale 

            else
                % In the case ARD is used
                for i=1:m  
                    s = 2./gpcf.lengthScale(i).^2;
                    dist = gminus(x(:,i),x(:,i)');
                    D = Cdm.*s.*dist.^2;
                    Bdl(i) = b'*(D*b);
                    Cdl(i) = sum(invCv.*D(:)); % help arguments for lengthScale 
                    %Cdl(i) = trace(C\D); % help arguments for lengthScale 
                end
            end
            Bdm = b'*(Cdm*b);
            Cdm = sum(invCv.*Cdm(:)); % help argument for magnSigma2
            %Cdm = trace(C\D); % help argument for magnSigma2
            
            
          case 'FIC' 
            % Evaluate the help matrices for the gradient evaluation (see
            % gpcf_sexp_trcov)
            
            L = varargin{1};             % u x u
            b = varargin{2};             % u x f
            iKuuKuf = varargin{3};             % mask(R, M) (block/band) diagonal
            La = varargin{4};                 % array of size
            if length(varargin) > 4
                b2 = varargin{5};
                b3 = varargin{6};
            end
            
            u = gpcf.X_u;
            
            % Derivatives of K_uu and K_uf with respect to magnitude sigma and lengthscale
            % NOTE! Here we have already taken into account that the parameters are transformed 
            % through log() and thus dK/dlog(p) = p * dK/dp
            K_uu = gpcf_sexp_trcov(gpcf, u);
            K_uf = gpcf_sexp_cov(gpcf, u, x);
            Cv_ff = gpcf_sexp_trvar(gpcf, x);

            % Evaluate help matrix for calculations of derivatives with respect to the lengthScale
            if length(gpcf.lengthScale) == 1
                % In the case of an isotropic SEXP
                s = 1./gpcf.lengthScale.^2;
                dist = 0;
                dist2 = 0;
                for i=1:m
                    D = gminus(u(:,i),x(:,i)');
                    D2= gminus(u(:,i),u(:,i)');
                    dist = dist + D.^2;
                    dist2 = dist2 + D2.^2;
                end
                DKuf_l = 2.*s.*K_uf.*dist;
                DKuu_l = 2.*s.*K_uu.*dist2;
            else
                % In the case ARD is used
                for i=1:m  
                    s = 1./gpcf.lengthScale(i).^2;        % set the length
                    dist = gminus(u(:,i),x(:,i)');
                    dist2 = gminus(u(:,i),u(:,i)');
                    DKuf_l{i} = 2.*s.*K_uf.*dist.^2;
                    DKuu_l{i} = 2.*s.*K_uu.*dist2.^2;
                end
            end
          case 'PIC_BLOCK'
            % Evaluate the help matrices for the gradient evaluation (see
            % gpcf_sexp_trcov)
            
            L = varargin{1};             % f x u
            b = varargin{2};             % 1 x f
            iKuuKuf = varargin{3};       % u x f
            Labl = varargin{4};          % array of size
            if length(varargin) > 4
                b2 = varargin{5};
                b3 = varargin{6};
            end

            u = gpcf.X_u;
            ind=gpcf.tr_index;
            
            % Derivatives of K_uu and K_uf with respect to magnitude sigma and lengthscale
            % NOTE! Here we have already taken into account that the parameters are transformed 
            % through log() and thus dK/dlog(p) = p * dK/dp
            K_uu = feval(gpcf.fh_trcov, gpcf, u); 
            K_uf = feval(gpcf.fh_cov, gpcf, u, x);
            for i=1:length(ind)
                K_ff{i} = feval(gpcf.fh_trcov, gpcf, x(ind{i},:));
            end

            % Evaluate help matrix for calculations of derivatives with respect to the lengthScale
            if length(gpcf.lengthScale) == 1
                % In the case of an isotropic SEXP
                s = 1./gpcf.lengthScale.^2;
                dist = 0;
                dist2 = 0;
                for j=1:length(ind)
                    dist3{j} = zeros(size(ind{j},1),size(ind{j},1));
                end
                for i=1:m
                    D = gminus(u(:,i),x(:,i)');
                    D2= gminus(u(:,i),u(:,i)');
                    dist = dist + D.^2;
                    dist2 = dist2 + D2.^2;
                    for j=1:length(ind)
                        dist3{j} = dist3{j} + (gminus(x(ind{j},i),x(ind{j},i)')).^2;
                    end
                end
                DKuf_l = 2.*s.*K_uf.*dist;
                DKuu_l = 2.*s.*K_uu.*dist2;
                for j=1:length(ind)
                    DKff_l{j} = 2.*s.*K_ff{j}.*dist3{j};
                end
            else
                % In the case ARD is used
                for i=1:m  
                    s = 1./gpcf.lengthScale(i).^2;        % set the length
                    dist = gminus(u(:,i),x(:,i)');
                    dist2 = gminus(u(:,i),u(:,i)');
                    DKuf_l{i} = 2.*s.*K_uf.*dist.^2;
                    DKuu_l{i} = 2.*s.*K_uu.*dist2.^2;
                    for j=1:length(ind)
                        dist3 = gminus(x(ind{j},i),x(ind{j},i)');
                        dist3 = 2.*s.*K_ff{j}.*dist3.^2;
                        DKff_l{j,i} = dist3;
                    end
                end
            end
          case 'PIC_BAND'
            % Evaluate the help matrices for the gradient evaluation (see
            % gpcf_sexp_trcov)
            
            L = varargin{1};             % f x u
            b = varargin{2};             % 1 x f
            iKuuKuf = varargin{3};       % u x f
            La = varargin{4};            % matrix of size
            if length(varargin) > 4
                b2 = varargin{5};
                b3 = varargin{6};
            end
          
            u = gpcf.X_u;
            ind=gpcf.tr_index;
            nzmax = size(ind,1);
            
            % Derivatives of K_uu and K_uf with respect to magnitude sigma and lengthscale
            % NOTE! Here we have already taken into account that the parameters are transformed 
            % through log() and thus dK/dlog(p) = p * dK/dp
            K_uu = feval(gpcf.fh_trcov, gpcf, u); 
            K_uf = feval(gpcf.fh_cov, gpcf, u, x);
            % Evaluate help matrix for calculations of derivatives with respect to the lengthScale
            if length(gpcf.lengthScale) == 1
                % In the case of an isotropic SEXP
                di2 = 0;
                s = 1./gpcf.lengthScale.^2;
                for i = 1:m
                    di2 = di2 + s.*(x(ind(:,1),i) - x(ind(:,2),i)).^2;
                end
                kv_ff = gpcf.magnSigma2.*exp(-di2);
                K_ff = sparse(ind(:,1),ind(:,2),kv_ff,n,n);

                dist = 0;
                dist2 = 0;
                dist3 = zeros(nzmax,1);
                for i=1:m
                    D = gminus(u(:,i),x(:,i)');
                    D2= gminus(u(:,i),u(:,i)');
                    dist = dist + D.^2;
                    dist2 = dist2 + D2.^2;
                    dist3 = dist3 + (x(ind(:,1),i)-x(ind(:,2),i)).^2;
                end
                DKuf_l = 2.*s.*K_uf.*dist;
                DKuu_l = 2.*s.*K_uu.*dist2;
                DKff_l = sparse(ind(:,1),ind(:,2), 2.*s.*kv_ff.*dist3 ,n,n);
            else
                % In the case ARD is used
                for i=1:m  
                    s = 1./gpcf.lengthScale(i).^2;        % set the length
                    dist = gminus(u(:,i),x(:,i)');
                    dist2 = gminus(u(:,i),u(:,i)');
                    dist = 2.*s.*K_uf.*dist.^2;
                    dist2 = 2.*s.*K_uu.*dist2.^2;
                    for j=1:length(ind)
                        dist3 = gminus(x(ind{j},i),x(ind{j},i)');
                        dist3 = 2.*s.*K_ff{j}.*dist3.^2;
                        DKff_l{j,i} = dist3;
                    end
                    
                    DKuf_l{i} = dist;         % 
                    DKuu_l{i} = dist2;        % 
                end
            end
          case 'CS+PIC'
            % Evaluate the help matrices for the gradient evaluation (see
            % gpcf_sexp_trcov)
            
            L = varargin{1};             % f x u
            b = varargin{2};             % 1 x f
            iKuuKuf = varargin{3};       % u x f
            La = varargin{4};            % matrix of size
            if length(varargin) > 4
                b2 = varargin{5};
                b3 = varargin{6};
            end

            u = gpcf.X_u;
            %ind=gpcf.tr_index;
            ind = gpcf.tr_indvec;
            nzmax = size(ind,1);
            
            % Derivatives of K_uu and K_uf with respect to magnitude sigma and lengthscale
            % NOTE! Here we have already taken into account that the parameters are transformed 
            % through log() and thus dK/dlog(p) = p * dK/dp
            K_uu = feval(gpcf.fh_trcov, gpcf, u);
            K_uf = feval(gpcf.fh_cov, gpcf, u, x);
            % Evaluate help matrix for calculations of derivatives with respect to the lengthScale
            if length(gpcf.lengthScale) == 1
                % In the case of an isotropic SEXP
                di2 = 0;
                s = 1./gpcf.lengthScale.^2;
                for i = 1:m
                    di2 = di2 + s.*(x(ind(:,1),i) - x(ind(:,2),i)).^2;
                end
                kv_ff = gpcf.magnSigma2.*exp(-di2);
                K_ff = sparse(ind(:,1),ind(:,2),kv_ff,n,n);

                dist = 0;
                dist2 = 0;
                dist3 = zeros(nzmax,1);
                for i=1:m
                    D = gminus(u(:,i),x(:,i)');
                    D2= gminus(u(:,i),u(:,i)');
                    dist = dist + D.^2;
                    dist2 = dist2 + D2.^2;
                    dist3 = dist3 + (x(ind(:,1),i)-x(ind(:,2),i)).^2;
                end
                DKuf_l = 2.*s.*K_uf.*dist;
                DKuu_l = 2.*s.*K_uu.*dist2;
                DKff_l = sparse(ind(:,1),ind(:,2), 2.*s.*kv_ff.*dist3 ,n,n);
            else
                % In the case ARD is used
                for i=1:m  
                    s = 1./gpcf.lengthScale(i).^2;        % set the length
                    dist = gminus(u(:,i),x(:,i)');
                    dist2 = gminus(u(:,i),u(:,i)');
                    dist = 2.*s.*K_uf.*dist.^2;
                    dist2 = 2.*s.*K_uu.*dist2.^2;
                    for j=1:length(ind)
                        dist3 = gminus(x(ind{j},i),x(ind{j},i)');
                        dist3 = 2.*s.*K_ff{j}.*dist3.^2;
                        DKff_l{j,i} = dist3;
                    end
                    
                    DKuf_l{i} = dist;         % 
                    DKuu_l{i} = dist2;        % 
                end
            end
        end

        % Evaluate the gdata and gprior with respect to magnSigma2
        i1 = i1+1;
        switch gpcf.type
          case 'FULL'
            gdata(i1) = 0.5.*(Cdm - Bdm);
          case 'FIC'
                KfuiKuuKuu = iKuuKuf'*K_uu;
                gdata(i1) = -0.5.*((2*b*K_uf'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*K_uf'*iKuuKuf))) - ...
                                   sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                
                gdata(i1) = gdata(i1) - 0.5.*(b.*Cv_ff')*b';
                gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(K_uf'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                gdata(i1) = gdata(i1) + 0.5.*(sum(Cv_ff./La) - sum(sum(L.*L)).*gpcf.magnSigma2);
                gdata(i1) = gdata(i1) + 0.5.*(2.*sum(sum(L.*L,2).*sum(K_uf'.*iKuuKuf',2)) - sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));
                if length(varargin) > 4
                    gdata(i1) = gdata(i1) - 0.5.*(2*b2*K_uf'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                    gdata(i1) = gdata(i1) - 0.5.*(b2.*Cv_ff')*b3;
                    gdata(i1) = gdata(i1) + 0.5.*(2.*b2.*sum(K_uf'.*iKuuKuf',2)'*b3- b2.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b3);
                end
          case 'PIC_BLOCK'
            KfuiKuuKuu = iKuuKuf'*K_uu;
            %            H = (2*K_uf'- KfuiKuuKuu)*iKuuKuf;
            % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
            gdata(i1) = -0.5.*((2*b*K_uf'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*K_uf'*iKuuKuf))) - ...
                               sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
            if length(varargin) > 4
                gdata(i1) = gdata(i1) -0.5.*(2*b2*K_uf'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
            end
            for i=1:length(K_ff)
                gdata(i1) = gdata(i1) ...                   %   + trace(Labl{i}\H(ind{i},ind{i})) ...
                    + 0.5.*(-b(ind{i})*K_ff{i}*b(ind{i})' ...
                    + 2.*b(ind{i})*K_uf(:,ind{i})'*iKuuKuf(:,ind{i})*b(ind{i})'- ...
                            b(ind{i})*KfuiKuuKuu(ind{i},:)*iKuuKuf(:,ind{i})*b(ind{i})' ...       %H(ind{i},ind{i})
                    + trace(Labl{i}\K_ff{i})...
                    - trace(L(ind{i},:)*(L(ind{i},:)'*K_ff{i})) ...               %- trace(Labl{i}\H(ind{i},ind{i})) 
                    + 2.*sum(sum(L(ind{i},:)'.*(L(ind{i},:)'*K_uf(:,ind{i})'*iKuuKuf(:,ind{i})))) - ...
                      sum(sum(L(ind{i},:)'.*((L(ind{i},:)'*KfuiKuuKuu(ind{i},:))*iKuuKuf(:,ind{i}))))); 
                %trace(L(ind{i},:)*(L(ind{i},:)'*H(ind{i},ind{i}))));
                if length(varargin) > 4
                    gdata(i1) = gdata(i1) ...                   %   + trace(Labl{i}\H(ind{i},ind{i})) ...
                    + 0.5.*(-b2(ind{i})*K_ff{i}*b3(ind{i}) ...
                    + 2.*b2(ind{i})*K_uf(:,ind{i})'*iKuuKuf(:,ind{i})*b3(ind{i})- ...
                            b2(ind{i})*KfuiKuuKuu(ind{i},:)*iKuuKuf(:,ind{i})*b3(ind{i})); 
                end
            end
          case {'PIC_BAND','CS+PIC'}
            KfuiKuuKuu = K_uf';
            KfuiKuuKuu = iKuuKuf'*K_uu;

            % Note! H = (2*K_uf'- KfuiKuuKuu)*iKuuKuf, but here we set actually H = mask(H) and the computations 
            % with full(H) are done with partition
            H = (2*K_uf - KfuiKuuKuu');
            
            H_diag = sum(H.*iKuuKuf,1);
            [I,J] = find(tril(K_ff,-1));
            H2 = zeros(1,size(I,1));
            for i = 1:size(H,1)
                H2 = H2 + H(i,I).*iKuuKuf(i,J);
            end
            H2 = sparse(I,J,H2,n,n);
            H = H2 + H2' + sparse(1:n,1:n,H_diag);
            
% $$$             H = sum(H(:,ind(:,1)).*iKuuKuf(:,ind(:,2)));
% $$$             H = sparse(ind(:,1),ind(:,2),H,n,n);
            
            % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
            gdata(i1) = -0.5.*((2*b*K_uf'-(b*KfuiKuuKuu))*(iKuuKuf*b'));
            gdata(i1) = gdata(i1) - sum(sum(L'.*(L'*K_uf'*iKuuKuf)));
            gdata(i1) = gdata(i1) + 0.5.*sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf)));
            gdata(i1) = gdata(i1) + sum(sum((La\(K_uf'-0.5.*KfuiKuuKuu))'.*iKuuKuf));
            gdata(i1) = gdata(i1) + 0.5.*-(b(ind(:,1)).*kv_ff')*b(ind(:,2))';
            gdata(i1) = gdata(i1) + 0.5.*b*H*b';
            gdata(i1) = gdata(i1) + 0.5.*trace(La\(K_ff-H));
            gdata(i1) = gdata(i1) + 0.5.*sum(sum(L'.*(L'*(H-K_ff))));               %- trace(Labl{i}\H(ind{i},ind{i})) 
            if length(varargin) > 4
                gdata(i1) = gdata(i1) -0.5.*((2*b2*K_uf'-(b2*KfuiKuuKuu))*(iKuuKuf*b3));
                gdata(i1) = gdata(i1) + 0.5.*-(b2(ind(:,1)).*kv_ff')*b3(ind(:,2));
                gdata(i1) = gdata(i1) + 0.5.*b2*H*b3;
            end
          case 'CS+PIC'
            KfuiKuuKuu = iKuuKuf'*K_uu;
            %            H = (2*K_uf'- KfuiKuuKuu)*iKuuKuf;
            % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
            gdata(i1) = -0.5.*((2*b*K_uf'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*K_uf'*iKuuKuf))) - ...
                               sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
            iKuuKufiLa = iKuuKuf/Labl;
            H2 = (2*K_uf' - KfuiKuuKuu);
            %H = (2*K_uf' - KfuiKuuKuu)*iKuuKuf;

% $$$             H = (2*K_uf - KfuiKuuKuu');
% $$$             [I,J] = find(iLa);
% $$$             H = sum(H(:,ind(:,1)).*iKuuKuf(:,ind(:,2)));
% $$$             H = sparse(I,J,H,n,n);
            
            %            gdata(i1) = gdata(i1) + 0.5*trace(iLa*H);
            gdata(i1) = gdata(i1) + 0.5*sum(sum(H2.*iKuuKufiLa'));
            for i=1:length(K_ff)
                gdata(i1) = gdata(i1) ...                   %   + trace(Labl{i}\H(ind{i},ind{i})) ...
                    + 0.5.*(-b(ind{i})*K_ff{i}*b(ind{i})' ...
                            + 2.*b(ind{i})*K_uf(:,ind{i})'*iKuuKuf(:,ind{i})*b(ind{i})'- ...
                            b(ind{i})*KfuiKuuKuu(ind{i},:)*iKuuKuf(:,ind{i})*b(ind{i}))' ...       %H(ind{i},ind{i})
                                                                                                  %    + trace(iLa(ind{i},ind{i})*K_ff{i})...
                                                                                                  %                    - sum(sum(iLa(ind{i},ind{i}).*H(ind{i},ind{i})'))...        
                    + sum(sum((H2(ind{i},:).*iKuuKufiLa(:,ind{i})')))...
                    - sum(sum(Kuu))...                            
                    - sum(sum(L(ind{i},:)'.*(L(ind{i},:)'*K_ff{i})) ...               %- trace(Labl{i}\H(ind{i},ind{i})) 
                            + 2.*sum(sum(L(ind{i},:)'.*(L(ind{i},:)'*K_uf(:,ind{i})'*iKuuKuf(:,ind{i})))) - ...
                            sum(sum(L(ind{i},:)'.*((L(ind{i},:)'*KfuiKuuKuu(ind{i},:))*iKuuKuf(:,ind{i}))))); 
                                                                %trace(L(ind{i},:)*(L(ind{i},:)'*H(ind{i},ind{i}))));
            end
        end
        gprior(i1)=feval(gpp.magnSigma2.fg, ...
                         gpcf.magnSigma2, ...
                         gpp.magnSigma2.a, 'x').*gpcf.magnSigma2 - 1;
        % Evaluate the prior contribution of gradient with respect to lengthScale.p.s (and lengthScale.p.nu)
        if isfield(gpp.lengthScale, 'p') && ~isempty(gpp.lengthScale.p)
            i1=i1+1;
            gprior(i1)=...
                feval(gpp.lengthScale.p.s.fg, ...
                      gpp.lengthScale.a.s,...
                      gpp.lengthScale.p.s.a, 'x').*gpp.lengthScale.a.s - 1 ...
                +feval(gpp.lengthScale.fg, ...
                       gpcf.lengthScale, ...
                       gpp.lengthScale.a, 's').*gpp.lengthScale.a.s;
            if any(strcmp(fieldnames(gpp.lengthScale.p),'nu'))
                i1=i1+1;
                gprior(i1)=...
                    feval(gpp.lengthScale.p.nu.fg, ...
                          gpp.lengthScale.a.nu,...
                          gpp.lengthScale.p.nu.a, 'x').*gpp.lengthScale.a.nu -1 ...
                    +feval(gpp.lengthScale.fg, ...
                           gpcf.lengthScale, ...
                           gpp.lengthScale.a, 'nu').*gpp.lengthScale.a.nu;
            end
        end
        % Evaluate the data contribution of gradient with respect to lengthScale
        if length(gpcf.lengthScale)>1
            for i2=1:gpcf.nin
                i1=i1+1;
                switch gpcf.type
                  case 'FULL'
                    gdata(i1)=0.5.*(Cdl(i2) - Bdl(i2));
                  case 'FIC'
                    KfuiKuuKuu = iKuuKuf'*DKuu_l{i2};
                    gdata(i1) = -0.5.*((2*b*DKuf_l{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf_l{i2}'*iKuuKuf))) - ...
                                       sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                    gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf_l{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                    gdata(i1) = gdata(i1) + 0.5.*(2.*sum(sum(L.*L,2).*sum(DKuf_l{i2}'.*iKuuKuf',2)) - sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));
                    if length(varargin) > 4
                        gdata(i1) = gdata(i1) -0.5.*(2*b2*DKuf_l{i2}'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                        gdata(i1) = gdata(i1) + 0.5.*(2.*b2.*sum(DKuf_l{i2}'.*iKuuKuf',2)'*b3 - b2.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b3);
                    end
                  case 'PIC_BLOCK'
                    KfuiKuuDKuu_l = iKuuKuf'*DKuu_l{i2};
                    % H = (2*DKuf_l'- KfuiKuuDKuu_l)*iKuuKuf;
                    % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
                    gdata(i1) = -0.5.*((2*b*DKuf_l{i2}'-(b*KfuiKuuDKuu_l))*(iKuuKuf*b') + 2.*sum(sum(L'.*((L'*DKuf_l{i2}')*iKuuKuf))) - ...
                                       sum(sum(L'.*((L'*KfuiKuuDKuu_l)*iKuuKuf))));
                    if length(varargin) > 4
                        gdata(i1) = gdata(i1) -0.5.*(2*b2*DKuf_l{i2}'-(b2*KfuiKuuDKuu_l))*(iKuuKuf*b3);
                    end
                    for i=1:length(K_ff)
                        gdata(i1) = gdata(i1) ...                   %   + trace(Labl{i}\H(ind{i},ind{i})) ...
                            + 0.5.*(-b(ind{i})*DKff_l{i,i2}*b(ind{i})' ...
                                    + 2.*b(ind{i})*DKuf_l{i2}(:,ind{i})'*iKuuKuf(:,ind{i})*b(ind{i})'- ...
                                    b(ind{i})*KfuiKuuDKuu_l(ind{i},:)*iKuuKuf(:,ind{i})*b(ind{i})' ...       %H(ind{i},ind{i})
                                    + trace(Labl{i}\DKff_l{i,i2})...
                                    - trace(L(ind{i},:)*(L(ind{i},:)'*DKff_l{i,i2})) ...               %- trace(Labl{i}\H(ind{i},ind{i})) 
                                    + 2.*sum(sum(L(ind{i},:)'.*(L(ind{i},:)'*DKuf_l{i2}(:,ind{i})'*iKuuKuf(:,ind{i})))) - ...
                                    sum(sum(L(ind{i},:)'.*((L(ind{i},:)'*KfuiKuuDKuu_l(ind{i},:))*iKuuKuf(:,ind{i}))))); 
                        %trace(L(ind{i},:)*(L(ind{i},:)'*H(ind{i},ind{i}))));
                        if length(varargin) > 4
                            gdata(i1) = gdata(i1) ...                   %   + trace(Labl{i}\H(ind{i},ind{i})) ...
                                + 0.5.*(-b2(ind{i})*DKff_l{i,i2}*b3(ind{i}) ...
                                    + 2.*b2(ind{i})*DKuf_l{i2}(:,ind{i})'*iKuuKuf(:,ind{i})*b3(ind{i})'- ...
                                    b2(ind{i})*KfuiKuuDKuu_l(ind{i},:)*iKuuKuf(:,ind{i})*b3(ind{i}));
                        end
                    end
                  case {'PIC_BAND','CS+PIC'}
                    KfuiKuuDKuu_l = iKuuKuf'*DKuu_l{i2};
                    % Note! H = (2*K_uf'- KfuiKuuKuu)*iKuuKuf, but here we set actually H = mask(H) and the computations 
                    % with full(H) are done with partition
                    H = (2*DKuf_l{i2} - KfuiKuuDKuu_l');
                    
% $$$                     H2 = zeros(1,size(ind,1));
% $$$                     for i = 1:size(H,1)
% $$$                         H2 = H2 + H(i,ind(:,1)).*iKuuKuf(i,ind(:,2));
% $$$                     end
% $$$                     H2 = sparse(ind(:,1),ind(:,2),H2,n,n);
% $$$                     H = H2;
                    
% $$$                     H_diag = sum(H.*iKuuKuf,1);
% $$$                     [I,J] = find(tril(K_ff,-1));
% $$$                     H2 = zeros(1,size(I,1));
% $$$                     for i = 1:size(H,1)
% $$$                         H2 = H2 + H(i,I).*iKuuKuf(i,J);
% $$$                     end
% $$$                     H2 = sparse(I,J,H2,n,n);
% $$$                     H = H2 + H2' + sparse(1:n,1:n,H_diag);

                    
                    H = sum(H(:,ind(:,1)).*iKuuKuf(:,ind(:,2)));
                    H = sparse(ind(:,1),ind(:,2),H,n,n);
                    % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
                    gdata(i1) = -0.5.*((2*b*DKuf_l{i2}'-(b*KfuiKuuDKuu_l))*(iKuuKuf*b'));
                    gdata(i1) = gdata(i1) - sum(sum(L'.*(L'*DKuf_l{i2}'*iKuuKuf)));
                    gdata(i1) = gdata(i1) + 0.5.*sum(sum(L'.*((L'*KfuiKuuDKuu_l)*iKuuKuf)));
                    gdata(i1) = gdata(i1) + sum(sum((La\(DKuf_l'-0.5.*KfuiKuuDKuu_l))'.*iKuuKuf));
                    gdata(i1) = gdata(i1) + 0.5.*(b*(H-DKff_l{:,i2}))*b';
                    gdata(i1) = gdata(i1) + 0.5.*trace(La\(DKff_l{:,i2}-H));
                    gdata(i1) = gdata(i1) + 0.5.*sum(sum(L'.*(L'*(H-DKff_l{:,i2}))));         
                    if length(varargin) > 4
                        gdata(i1) = -0.5.*((2*b2*DKuf_l{i2}'-(b2*KfuiKuuDKuu_l))*(iKuuKuf*b3));
                        gdata(i1) = gdata(i1) + 0.5.*(b2*(H-DKff_l{:,i2}))*b3;
                    end
                  case 'CS+PIC'
                    KfuiKuuDKuu_l = iKuuKuf'*DKuu_l{i2};
                    %            H = (2*DKuf_l'- KfuiKuuDKuu_l)*iKuuKuf;
                    % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
                    gdata(i1) = -0.5.*((2*b*DKuf_l{i2}'-(b*KfuiKuuDKuu_l))*(iKuuKuf*b') + 2.*sum(sum(L'.*((L'*DKuf_l{i2}')*iKuuKuf))) - ...
                                       sum(sum(L'.*((L'*KfuiKuuDKuu_l)*iKuuKuf))));
                    H = (2*DKuf_l{i2}' - KfuiKuuDKuu_l)*iKuuKuf;
                    gdata(i1) = gdata(i1) + 0.5*sum(sum(iLa.*H'));
                    for i=1:length(K_ff)
                        gdata(i1) = gdata(i1) ...                   %   + trace(Labl{i}\H(ind{i},ind{i})) ...
                            + 0.5.*(-b(ind{i})*DKff_l{i,i2}*b(ind{i})' ...
                                    + 2.*b(ind{i})*DKuf_l{i2}(:,ind{i})'*iKuuKuf(:,ind{i})*b(ind{i})'- ...
                                    b(ind{i})*KfuiKuuDKuu_l(ind{i},:)*iKuuKuf(:,ind{i})*b(ind{i})' ...       %H(ind{i},ind{i})
                                    + trace(iLa(ind{i},ind{i})*DKff_l{i,i2})...
                                    - trace(iLa(ind{i},ind{i})*H(ind{i},ind{i}))...
                                    - sum(sum(L(ind{i},:)'.*(L(ind{i},:)'*DKff_l{i,i2}))) ...    %- trace(Labl{i}\H(ind{i},ind{i})) 
                                    + 2.*sum(sum(L(ind{i},:)'.*(L(ind{i},:)'*DKuf_l{i2}(:,ind{i})'*iKuuKuf(:,ind{i})))) - ...
                                    sum(sum(L(ind{i},:)'.*((L(ind{i},:)'*KfuiKuuDKuu_l(ind{i},:))*iKuuKuf(:,ind{i}))))); 
                        %trace(L(ind{i},:)*(L(ind{i},:)'*H(ind{i},ind{i}))));
                    end
                end
                gprior(i1)=feval(gpp.lengthScale.fg, ...
                                 gpcf.lengthScale(i2), ...
                                 gpp.lengthScale.a, 'x').*gpcf.lengthScale(i2) - 1;
            end
        else
            i1=i1+1;
            switch gpcf.type
              case 'FULL'
                gdata(i1)=0.5.*(Cdl - Bdl);
              case 'FIC' 
                    KfuiKuuKuu = iKuuKuf'*DKuu_l;
                    gdata(i1) = -0.5.*((2*b*DKuf_l'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf_l'*iKuuKuf))) - ...
                                       sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                    gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf_l'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                    gdata(i1) = gdata(i1) + 0.5.*(2.*sum(sum(L.*L,2).*sum(DKuf_l'.*iKuuKuf',2)) - sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));
                    
                    if length(varargin) > 4
                        gdata(i1) = gdata(i1) -0.5.*(2*b2*DKuf_l'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                        gdata(i1) = gdata(i1) + 0.5.*(2.*b2.*sum(DKuf_l'.*iKuuKuf',2)'*b3- b2.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b3);
                    end
              case 'PIC_BLOCK'
                KfuiKuuDKuu_l = iKuuKuf'*DKuu_l;
                % H = (2*DKuf_l'- KfuiKuuDKuu_l)*iKuuKuf;
                % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
                gdata(i1) = -0.5.*((2*b*DKuf_l'-(b*KfuiKuuDKuu_l))*(iKuuKuf*b') + 2.*sum(sum(L'.*((L'*DKuf_l')*iKuuKuf))) - ...
                                   sum(sum(L'.*((L'*KfuiKuuDKuu_l)*iKuuKuf))));
                if length(varargin) > 4
                    gdata(i1) = gdata(i1) -0.5.*(2*b2*DKuf_l'-(b2*KfuiKuuDKuu_l))*(iKuuKuf*b3');
                end
                for i=1:length(K_ff)
                    gdata(i1) = gdata(i1) ...                   %   + trace(Labl{i}\H(ind{i},ind{i})) ...
                        + 0.5.*(-b(ind{i})*DKff_l{i}*b(ind{i})' ...
                                + 2.*b(ind{i})*DKuf_l(:,ind{i})'*iKuuKuf(:,ind{i})*b(ind{i})'- ...
                                b(ind{i})*KfuiKuuDKuu_l(ind{i},:)*iKuuKuf(:,ind{i})*b(ind{i})' ...       %H(ind{i},ind{i})
                                + trace(Labl{i}\DKff_l{i})...
                                - sum(sum(L(ind{i},:)'.*(L(ind{i},:)'*DKff_l{i}))) ...               %- trace(Labl{i}\H(ind{i},ind{i})) 
                                + 2.*sum(sum(L(ind{i},:)'.*(L(ind{i},:)'*DKuf_l(:,ind{i})'*iKuuKuf(:,ind{i})))) - ...
                                sum(sum(L(ind{i},:)'.*((L(ind{i},:)'*KfuiKuuDKuu_l(ind{i},:))*iKuuKuf(:,ind{i}))))); 
                    %trace(L(ind{i},:)*(L(ind{i},:)'*H(ind{i},ind{i}))));
                    if length(varargin) > 4
                        gdata(i1) = gdata(i1) ...                   %   + trace(Labl{i}\H(ind{i},ind{i})) ...
                            + 0.5.*(-b2(ind{i})*DKff_l{i}*b3(ind{i}) ...
                                    + 2.*b2(ind{i})*DKuf_l(:,ind{i})'*iKuuKuf(:,ind{i})*b3(ind{i})- ...
                                    b2(ind{i})*KfuiKuuDKuu_l(ind{i},:)*iKuuKuf(:,ind{i})*b3(ind{i})); 
                    end
                end
              case {'PIC_BAND','CS+PIC'}
                KfuiKuuDKuu_l = iKuuKuf'*DKuu_l;
                % Note! H = (2*K_uf'- KfuiKuuKuu)*iKuuKuf, but here we set actually H = mask(H) and the computations 
                % with full(H) are done with partition
                H = (2*DKuf_l - KfuiKuuDKuu_l');

% $$$                 H2 = zeros(1,size(ind,1));
% $$$                 for i = 1:size(H,1)
% $$$                     H2 = H2 + H(i,ind(:,1)).*iKuuKuf(i,ind(:,2));
% $$$                 end
% $$$                 H2 = sparse(ind(:,1),ind(:,2),H2,n,n);
% $$$                 H = H2;
                
% $$$                 H_diag = sum(H.*iKuuKuf);
% $$$                 H2 = zeros(1,size(I,1));
% $$$                 for i = 1:size(H,1)
% $$$                     H2 = H2 + H(i,I).*iKuuKuf(i,J);
% $$$                 end
% $$$                 H2 = sparse(I,J,H2,n,n);
% $$$                 H = H2 + H2' + sparse(1:n,1:n,H_diag,n,n);
                
                H = sum(H(:,ind(:,1)).*iKuuKuf(:,ind(:,2)));
                H = sparse(ind(:,1),ind(:,2),H,n,n);
                % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
                gdata(i1) = -0.5.*((2*b*DKuf_l'-(b*KfuiKuuDKuu_l))*(iKuuKuf*b'));
                gdata(i1) = gdata(i1) - sum(sum(L'.*(L'*DKuf_l'*iKuuKuf)));
                gdata(i1) = gdata(i1) + 0.5.*sum(sum(L'.*((L'*KfuiKuuDKuu_l)*iKuuKuf)));
                gdata(i1) = gdata(i1) + sum(sum((La\(DKuf_l'-0.5.*KfuiKuuDKuu_l))'.*iKuuKuf));
                gdata(i1) = gdata(i1) + 0.5.*(b*(H-DKff_l))*b';
                gdata(i1) = gdata(i1) + 0.5.*trace(La\(DKff_l-H));
                gdata(i1) = gdata(i1) + 0.5.*sum(sum(L'.*(L'*(H-DKff_l))));
                
                if length(varargin) > 4
                    gdata(i1) = -0.5.*((2*b2*DKuf_l'-(b2*KfuiKuuDKuu_l))*(iKuuKuf*b3));
                    gdata(i1) = gdata(i1) + 0.5.*(b2*(H-DKff_l))*b3;
                end

% $$$                 gdata(i1) = gdata(i1) + 0.5.*-(b(ind(:,1)).*kv_ff')*b(ind(:,2))';
% $$$                 gdata(i1) = gdata(i1) + 0.5.*b*H*b';
% $$$                 gdata(i1) = gdata(i1) + 0.5.*trace(La\(K_ff-H));
% $$$                 gdata(i1) = gdata(i1) + 0.5.*sum(sum(L'.*(L'*(H-K_ff))));               %- trace(Labl{i}\H(ind{i},ind{i})) 
% $$$ 
% $$$                 gdata(i1) = -0.5.*((2*b*DKuf_l'-(b*KfuiKuuDKuu_l))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf_l'*iKuuKuf))) - ...
% $$$                                    sum(sum(L'.*((L'*KfuiKuuDKuu_l)*iKuuKuf))) - 2.*sum(sum((La\DKuf_l')'.*iKuuKuf)) + ...
% $$$                                    sum(sum((La\KfuiKuuDKuu_l)'.*iKuuKuf)));
% $$$                 gdata(i1) = gdata(i1) + 0.5.*(b*(H-DKff_l))*b';
% $$$                 gdata(i1) = gdata(i1) + 0.5.*trace(La\(DKff_l-H));
% $$$                 gdata(i1) = gdata(i1) + 0.5.*sum(sum(L'.*(L'*(H-DKff_l))));
              case 'CS+PIC'
                KfuiKuuDKuu_l = iKuuKuf'*DKuu_l;
                %            H = (2*DKuf_l'- KfuiKuuDKuu_l)*iKuuKuf;
                % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
                gdata(i1) = -0.5.*((2*b*DKuf_l'-(b*KfuiKuuDKuu_l))*(iKuuKuf*b') + 2.*sum(sum(L'.*((L'*DKuf_l')*iKuuKuf))) - ...
                                   sum(sum(L'.*((L'*KfuiKuuDKuu_l)*iKuuKuf))));

                
                H = (2*DKuf_l' - KfuiKuuDKuu_l)*iKuuKuf;
                gdata(i1) = gdata(i1) + 0.5*sum(sum(iLa.*H'));

                for i=1:length(K_ff)
                    gdata(i1) = gdata(i1) ...                   %   + trace(Labl{i}\H(ind{i},ind{i})) ...
                        + 0.5.*(-b(ind{i})*DKff_l{i}*b(ind{i})' ...
                                + 2.*b(ind{i})*DKuf_l(:,ind{i})'*iKuuKuf(:,ind{i})*b(ind{i})'- ...
                                b(ind{i})*KfuiKuuDKuu_l(ind{i},:)*iKuuKuf(:,ind{i})*b(ind{i})' ...       %H(ind{i},ind{i})
                                + sum(sum(iLa(ind{i},ind{i}).*DKff_l{i}'))...
                                - sum(sum(L(ind{i},:)'.*(L(ind{i},:)'*DKff_l{i}))) ...               %- trace(Labl{i}\H(ind{i},ind{i})) 
                                - sum(sum(iLa(ind{i},ind{i}).*H(ind{i},ind{i})'))...
                                + 2.*sum(sum(L(ind{i},:)'.*(L(ind{i},:)'*DKuf_l(:,ind{i})'*iKuuKuf(:,ind{i})))) - ...
                                sum(sum(L(ind{i},:)'.*((L(ind{i},:)'*KfuiKuuDKuu_l(ind{i},:))*iKuuKuf(:,ind{i}))))); 
                    %trace(L(ind{i},:)*(L(ind{i},:)'*H(ind{i},ind{i}))));
                end
            end
            gprior(i1)=feval(gpp.lengthScale.fg, ...
                             gpcf.lengthScale, ...
                             gpp.lengthScale.a, 'x').*gpcf.lengthScale -1;
            
        end
        g = gdata + gprior;
    end
        
    
    function [g_ind, gdata_ind, gprior_ind]  = gpcf_sexp_gind(gpcf, x, t, g_ind, gdata_ind, gprior_ind, varargin)
    %GPCF_SEXP_GIND    Evaluate gradient of error for SE covariance function 
    %                  with respect to inducing inputs.
    %
    %	Descriptioni
    %	[DKuu_u, DKuf_u] = GPCF_SEXP_GIND(W, GPCF, X, T) 
    %
    %	See also
    %

    % Copyright (c) 2006      Jarno Vanhatalo
        
    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
        
        gpp=gpcf.p;
        [n, m] =size(x);
        
        L = varargin{1};             % u x u
        b = varargin{2};             % u x f
        iKuuKuf = varargin{3};       % mask(R, M) (block/band) diagonal
        La = varargin{4};            % array of size
        if length(varargin) > 4
            b2 = varargin{5};
            b3 = varargin{6};
        end
        
        u = gpcf.X_u;
        n_u = size(u,1);
                
        % First check if sparse model is used
        switch gpcf.type
          case 'FIC'            
            % Derivatives of K_uu and K_uf with respect to inducing inputs
            K_uu = feval(gpcf.fh_trcov, gpcf, u);
            K_uf = feval(gpcf.fh_cov, gpcf, u, x);
            
            if length(gpcf.lengthScale) == 1
                % In the case of an isotropic SEXP
                s = repmat(1./gpcf.lengthScale.^2, 1, m);
            else
                s = 1./gpcf.lengthScale.^2;
            end

            gradient = zeros(1,n_u*m);
            ind = 1; %j+(i-1)*n_u
            for i=1:m
                for j = 1:size(u,1)               
                    DKuf_u = zeros(size(K_uf));
                    DKuu_u = zeros(size(K_uu));
                    DKuf_u(j,:) = -2.*s(i).*gminus(u(j,i),x(:,i)');
                    DKuu_u(j,:) = -2.*s(i).* gminus(u(j,i),u(:,i)');
                    DKuu_u = DKuu_u + DKuu_u';
                                        
                    DKuf_u = DKuf_u.*K_uf;
                    DKuu_u = DKuu_u.*K_uu;      % dist2 = dist2 + dist2' - diag(diag(dist2));
                    
                    
                    KfuiKuuKuu = iKuuKuf'*DKuu_u;
                                        
                    gradient(ind) = -0.5.*((2*b*DKuf_u'-(b*KfuiKuuKuu))*(iKuuKuf*b') + ... 
                                                   2.*sum(sum(L'.*(L'*DKuf_u'*iKuuKuf))) - sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                    gradient(ind) = gradient(ind) + 0.5.*(2.*b.*sum(DKuf_u'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                    gradient(ind) = gradient(ind) + 0.5.*(2.*sum(sum(L.*L,2).*sum(DKuf_u'.*iKuuKuf',2)) - ...
                                                                      sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));
                    
                    if length(varargin) > 4
                        gradient(ind) = gradient(ind) -0.5.*(2*b2*DKuf_u'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                        gradient(ind) = gradient(ind) + 0.5.*(2.*b2.*sum(DKuf_u'.*iKuuKuf',2)'*b3- b2.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b3);
                    end
                    ind = ind +1;
                end
            end                             
          case 'PIC_BLOCK'
            trindex=gpcf.tr_index;
            
            % Derivatives of K_uu and K_uf with respect to inducing inputs
            K_uu = feval(gpcf.fh_trcov, gpcf, u);
            K_uf = feval(gpcf.fh_cov, gpcf, u, x);
            
            if length(gpcf.lengthScale) == 1       % In the case of an isotropic SEXP
                s = repmat(1./gpcf.lengthScale.^2, 1, m);
            else
                s = 1./gpcf.lengthScale.^2;
            end
            
            gradient = zeros(1,n_u*m);
            ind = 1; 
            for i=1:m
                for j = 1:size(u,1)               
                    DKuf_u = zeros(size(K_uf));
                    DKuu_u = zeros(size(K_uu));
                    DKuf_u(j,:) = -2.*s(i).*gminus(u(j,i),x(:,i)');
                    DKuu_u(j,:) = -2.*s(i).* gminus(u(j,i),u(:,i)');
                    DKuu_u = DKuu_u + DKuu_u';
                    
                    DKuf_u = DKuf_u.*K_uf;
                    DKuu_u = DKuu_u.*K_uu;      % dist2 = dist2 + dist2' - diag(diag(dist2));
                                        
                    KfuiKuuDKuu_u = iKuuKuf'*DKuu_u;
                   
                    gradient(ind) = -0.5.*((2*b*DKuf_u'-(b*KfuiKuuDKuu_u))*(iKuuKuf*b') + 2.*sum(sum(L'.*((L'*DKuf_u')*iKuuKuf))) - ...
                                       sum(sum(L'.*((L'*KfuiKuuDKuu_u)*iKuuKuf))));
                    if length(varargin) > 4
                        gradient(ind) = gradient(ind) -0.5.*(2*b2*DKuf_u'-(b2*KfuiKuuDKuu_u))*(iKuuKuf*b3);
                    end
                    for i1=1:length(trindex)
                        gradient(ind) = gradient(ind) + 0.5.*(2.*b(trindex{i1})*DKuf_u(:,trindex{i1})'*iKuuKuf(:,trindex{i1})*b(trindex{i1})'- ...
                                    b(trindex{i1})*KfuiKuuDKuu_u(trindex{i1},:)*iKuuKuf(:,trindex{i1})*b(trindex{i1})' ... 
                                    + 2.*sum(sum(L(trindex{i1},:)'.*(L(trindex{i1},:)'*DKuf_u(:,trindex{i1})'*iKuuKuf(:,trindex{i1})))) - ...
                                    sum(sum(L(trindex{i1},:)'.*((L(trindex{i1},:)'*KfuiKuuDKuu_u(trindex{i1},:))*iKuuKuf(:,trindex{i1}))))); 
                        if length(varargin) > 4
                            gradient(ind) = gradient(ind) + 0.5.*(2.*b2(trindex{i1})*DKuf_u(:,trindex{i1})'*iKuuKuf(:,trindex{i1})*b3(trindex{i1})- ...
                                    b2(trindex{i1})*KfuiKuuDKuu_u(trindex{i1},:)*iKuuKuf(:,trindex{i1})*b3(trindex{i1})'); 
                        end
                    end
                    ind = ind +1;
                end
            end
          case 'CS+PIC'
            
        end
        gdata_ind = gdata_ind + gradient;
        g_ind = gdata_ind;
    end
    
    
    function C = gpcf_sexp_cov(gpcf, x1, x2, varargin)
    % GP_SEXP_COV     Evaluate covariance matrix between two input vectors. 
    %
    %         Description
    %         C = GP_SEXP_COV(GP, TX, X) takes in covariance function of a Gaussian
    %         process GP and two matrixes TX and X that contain input vectors to 
    %         GP. Returns covariance matrix C. Every element ij of C contains  
    %         covariance between inputs i in TX and j in X.
    %
    %         For covariance function definition see manual or 
    %         Neal R. M. Regression and Classification Using Gaussian 
    %         Process Priors, Bayesian Statistics 6.
        
    % Copyright (c) 1998-2004 Aki Vehtari
    % Copyright (c) 2006      Aki Vehtari, Jarno Vanhatalo
        
    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
        
        if isempty(x2)
            x2=x1;
        end
        [n1,m1]=size(x1);
        [n2,m2]=size(x2);
        
        if m1~=m2
            error('the number of columns of X1 and X2 has to be same')
        end
                
        C=zeros(n1,n2);
        ma2 = gpcf.magnSigma2;
        
        % Evaluate the covariance
        if ~isempty(gpcf.lengthScale)  
            s = 1./gpcf.lengthScale.^2;
            if m1==1 && m2==1
                dd = gminus(x1,x2');
                dist=dd.^2*s;
            else
                % If ARD is not used make s a vector of 
                % equal elements 
                if size(s)==1
                    s = repmat(s,1,m1);
                end
                dist=zeros(n1,n2);
                for j=1:m1
                    dd = gminus(x1(:,j),x2(:,j)');
                    dist = dist + dd.^2.*s(:,j);
                end
            end
            C = ma2.*exp(-dist);
        end
        cov_x1=x1;
        cov_x2=x2;
        cov_ls=gpcf.lengthScale;
        cov_ms=gpcf.magnSigma2;
        cov_C=C;
    end
    
    function C = gpcf_sexp_trcov(gpcf, x)
    % GP_SEXP_TRCOV     Evaluate training covariance matrix of inputs. 
    %
    %         Description
    %         C = GP_SEXP_TRCOV(GP, TX) takes in covariance function of a Gaussian
    %         process GP and matrix TX that contains training input vectors to 
    %         GP. Returns covariance matrix C. Every element ij of C contains  
    %         covariance between inputs i and j in TX 
    %
    %         For covariance function definition see manual or 
    %         Neal R. M. Regression and Classification Using Gaussian 
    %         Process Priors, Bayesian Statistics 6.
        
    % Copyright (c) 1998-2004 Aki Vehtari
    % Copyright (c) 2006      Aki Vehtari, Jarno Vanhatalo
        
    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
        
        [n, m] =size(x);
        
        s = 1./(gpcf.lengthScale);
        s2 = s.^2;
        if size(s)==1
            s2 = repmat(s2,1,m);
        end
        ma = gpcf.magnSigma2;
        
        % Here we take advantage of the 
        % symmetry of covariance matrix
% $$$         C=zeros(n,n);
% $$$         for i1=2:n
% $$$             i1n=(i1-1)*n;
% $$$             for i2=1:i1-1
% $$$                 ii=i1+(i2-1)*n;
% $$$                 for i3=1:m
% $$$                     C(ii)=C(ii)+s2(i3).*(x(i1,i3)-x(i2,i3)).^2;       % the covariance function
% $$$                 end
% $$$                 C(i1n+i2)=C(ii); 
% $$$             end
% $$$         end
% $$$         C = ma.*exp(-C);
% $$$         C(C<eps)=0;
        
        C = zeros(n,n);
        for ii1=1:n-1
            d = zeros(n-ii1,1);
            col_ind = ii1+1:n;
            for ii2=1:m
                d = d+s2(ii2).*(x(col_ind,ii2)-x(ii1,ii2)).^2;
            end
            C(col_ind,ii1) = d;
        end
        C = C+C';
        C = ma.*exp(-C);

        trcov_x=x;
        trcov_ls=gpcf.lengthScale;
        trcov_ms=gpcf.magnSigma2;
        trcov_C=C;
    end
       
    function C = gpcf_sexp_covvec(gpcf, x1, x2, varargin)
    % GPCF_SEXP_COVVEC     Evaluate covariance vector between two input vectors. 
    %
    %         Description
    %         C = GPCF_SEXP_COVVEC(GP, TX, X) takes in Gaussian process GP and two
    %         matrixes TX and X that contain input vectors to GP. Returns 
    %         covariance vector C, where every element i of C contains covariance
    %         between input i in TX and i in X.
    %

        
        if isempty(x2)
            x2=x1;
        end
        [n1,m1]=size(x1);
        [n2,m2]=size(x2);
        
        if m1~=m2
            error('the number of columns of X1 and X2 has to be same')
        end
        
        ma2 = gpcf.magnSigma2;
        
        di2 = 0;
        s = 1./gpcf.lengthScale.^2;
        for i = 1:m1
            di2 = di2 + s.*(x1(:,i) - x2(:,i)).^2;
        end
        C = gpcf.magnSigma2.*exp(-di2);
    end
        
        
    function C = gpcf_sexp_trvar(gpcf, x)
    % GP_SEXP_TRVAR     Evaluate training variance vector of inputs. 
    %
    %         Description
    %         C = GP_SEXP_TRVAR(GP, TX) takes in covariance function of a Gaussian
    %         process GP and matrix TX that contains training input vectors to 
    %         GP. Returns variance vector C. Every element i of C contains  
    %         variance of input i in TX 
    %
    %         For covariance function definition see manual or 
    %         Neal R. M. Regression and Classification Using Gaussian 
    %         Process Priors, Bayesian Statistics 6.
        
    % Copyright (c) 1998-2004 Aki Vehtari
    % Copyright (c) 2006      Aki Vehtari, Jarno Vanhatalo
        
    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
        
        [n, m] =size(x);
        
        C = ones(n,1)*gpcf.magnSigma2;
        C(C<eps)=0;
    end

    function reccf = gpcf_sexp_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_SEXP_RECAPPEND(RECCF, RI, GPCF) takes old covariance 
    %          function record RECCF, record index RI, RECAPPEND returns a 
    %          structure RECCF containing following record fields:
    %          lengthHyper    = 
    %          lengthHyperNu  = 
    %          lengthScale    = 
    %          magnSigma2     = 
        
    % Initialize record
        if nargin == 2
            reccf.type = 'gpcf_sexp';
            reccf.nin = ri;
            reccf.nout = 1;
            
            % Initialize parameters
            reccf.lengthScale= [];
            reccf.magnSigma2 = [];
            
            % Set the function handles
            reccf.fh_pak = @gpcf_sexp_pak;
            reccf.fh_unpak = @gpcf_sexp_unpak;
            reccf.fh_e = @gpcf_sexp_e;
            reccf.fh_g = @gpcf_sexp_g;
            reccf.fh_cov = @gpcf_sexp_cov;
            reccf.fh_trcov  = @gpcf_sexp_trcov;
            reccf.fh_trvar  = @gpcf_sexp_trvar;
            %  gpcf.fh_sampling = @hmc2;
            %  reccf.sampling_opt = hmc2_opt;
            reccf.fh_recappend = @gpcf_sexp_recappend;  
            return
        end
        
        gpp = gpcf.p;
        % record lengthScale
        if ~isempty(gpcf.lengthScale)
            if ~isempty(gpp.lengthScale)
                reccf.lengthHyper(ri,:)=gpp.lengthScale.a.s;
                if isfield(gpp.lengthScale,'p')
                    if isfield(gpp.lengthScale.p,'nu')
                        reccf.lengthHyperNu(ri,:)=gpp.lengthScale.a.nu;
                    end
                end
            elseif ri==1
                reccf.lengthHyper=[];
            end
            reccf.lengthScale(ri,:)=gpcf.lengthScale;
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