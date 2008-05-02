function gpcf = gpcf_ppcs2(do, varargin)
%GPCF_SEXP	Create a piecewise polynomial covariance function with compact support
%               for Gaussian Process
%
%	Description
%
%	GPCF = GPCF_PPCS2('INIT', NIN) Create and initialize squared exponential 
%       covariance function fo Gaussian process 
%
%	The fields and (default values) in GPCF_PPCS2 are:
%	  type           = 'gpcf_ppcs2'
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
%                          (@gpcf_ppcs2_pak)
%         fh_unpak       = function handle to unpackin function
%                          (@gpcf_ppcs2_unpak)
%         fh_e           = function handle to error function
%                          (@gpcf_ppcs2_e)
%         fh_ghyper      = function handle to gradient function (with respect to hyperparameters)
%                          (@gpcf_ppcs2_ghyper)
%         fh_gind        = function handle to gradient function (with respect to inducing inputs)
%                          (@gpcf_ppcs2_gind)
%         fh_cov         = function handle to covariance function
%                          (@gpcf_ppcs2_cov)
%         fh_trcov       = function handle to training covariance function
%                          (@gpcf_ppcs2_trcov)
%         fh_covvec      = function handle to elementvice covariance function
%                          (@gpcf_ppcs2_covvec)
%         fh_trvar       = function handle to training variance function
%                          (@gpcf_ppcs2_trvar)
%         fh_sampling    = function handle to parameter sampling function
%                          (@hmc2)
%         sampling_opt   = options structure for fh_sampling
%                          (hmc2_opt)
%         fh_recappend   = function handle to record append function
%                          (gpcf_ppcs2_recappend)
%
%	GPCF = GPCF_PPCS2('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
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
        gpcf.type = sprintf('gpcf_ppcs2');
        gpcf.nin = nin;
        gpcf.nout = 1;
        gpcf.l = floor(nin/2) + 3;
        
        % cf is compactly supported
        gpcf.cs = 1;
        
        % Initialize parameters
        gpcf.lengthScale= repmat(10, 1, nin); 
        gpcf.magnSigma2 = 0.1;
        
        % Initialize prior structure
        gpcf.p=[];
        gpcf.p.lengthScale=[];
        gpcf.p.magnSigma2=[];

        
        % Set the function handles to the nested functions
        gpcf.fh_pak = @gpcf_ppcs2_pak;
        gpcf.fh_unpak = @gpcf_ppcs2_unpak;
        gpcf.fh_e = @gpcf_ppcs2_e;
        gpcf.fh_ghyper = @gpcf_ppcs2_ghyper;
        gpcf.fh_gind = @gpcf_ppcs2_gind;
        gpcf.fh_cov = @gpcf_ppcs2_cov;
        gpcf.fh_covvec = @gpcf_ppcs2_covvec;
        gpcf.fh_trcov  = @gpcf_ppcs2_trcov;
        gpcf.fh_trvar  = @gpcf_ppcs2_trvar;
        gpcf.fh_recappend = @gpcf_ppcs2_recappend;
        
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
    
    function w = gpcf_ppcs2_pak(gpcf, w)
    %GPcf_PPCS2_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %	W = GP_PPCS2_PAK(GPCF, W) takes a Gaussian Process data structure GP and
    %	combines the hyper-parameters into a single row vector W.
    %
    %	The ordering of the parameters in HP is defined by
    %	  hp = [hyper-params of gp.cf{1}, hyper-params of gp.cf{2}, ...];
    %
    %	See also
    %	GPCF_PPCS2_UNPAK
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




    function [gpcf, w] = gpcf_ppcs2_unpak(gpcf, w)
    %GPCF_PPCS2_UNPAK  Separate GP covariance function hyper-parameter vector into components. 
    %
    %	Description
    %	GP = GPCF_PPCS2_UNPAK(GP, W) takes an Gaussian Process data structure GP
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
    
    function eprior =gpcf_ppcs2_e(gpcf, x, t)
    %GPCF_PPCS2_E	Evaluate prior contribution of error of covariance function SE.
    %
    %	Description
    %	E = GPCF_PPCS2_E(W, GP, X, T) takes a gp data structure GPCF together
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
    
    function [gprior, DKff, DKuu, DKuf]  = gpcf_ppcs2_ghyper(gpcf, x, t, g, gdata, gprior, varargin)
    %GPCF_PPCS2_GHYPER     Evaluate gradient of error for SE covariance function
    %                     with respect to the hyperparameters.
    %
    %	Descriptioni
    %	G = GPCF_PPCS2_GHYPER(W, GPCF, X, T, G, GDATA, GPRIOR, VARARGIN) takes a gp 
    %   hyper-parameter vector W, data structure GPCF a matrix X of input vectors a 
    %   matrix T of target vectors, inverse covariance function , 
    %	and evaluates the error gradient G. Each row of X corresponds to one 
    %   input vector and each row of T corresponds to one target vector.
    %
    %	[G, GDATA, GPRIOR] = GPCF_PPCS2_GHYPER(GP, P, T) also returns separately  the
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
        if ~isempty(gprior)
            i1 = length(gprior);
        end
        
        % First check if sparse model is used
        switch gpcf.type
          case 'FULL'
            % Evaluate help arguments for gradient evaluation
            % instead of calculating trace(invC*Cdm) calculate sum(invCv.*Cdm(:)), when
            % Cdm and invC are symmetric matricess of same size. This is 67 times faster
            % with n=215
            Cdm = gpcf_ppcs2_trcov(gpcf, x);

            ii1=1;
            DKff{ii1} = Cdm;
                
            l = gpcf.l;
            [I,J] = find(Cdm);
            % loop over all the lengthScales
            if length(gpcf.lengthScale) == 1
                % In the case of isotropic PPCS2
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
                const1 = 2.*l^2+8.*l+6;
                const2 = l^2+4.*l+3;
                D = -ma2.*cs.^(l+1).*d.*(cs.*(const1.*d+3.*l+6)-(l+2).*(const2.*d2+(3.*l+6).*d+3))/3;
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
                    d2 = d2 + s2(i).*(x(I,i) - x(J,i)).^2;
                    d_l2(:,i) = s2(i).*(x(I,i) - x(J,i)).^2;
                end
                d = sqrt(d2);
                d_l = d_l2;

                % Create the 'compact support' matrix, that is, (1-R)_+,
                % where ()_+ truncates all non-positive inputs to zero.
                cs = 1-d;

                const1 = 2.*l^2+8.*l+6;
                const2 = l^2+4.*l+3;

                for i = 1:m
                    % Calculate the gradient matrix
                    D = -ma2.*cs.^(l+1).*d_l(:,i).*(cs.*(const1.*d+3*l+6)-(l+2)*(const2.*d2+(3*l+6)*d+3))/3;
                    % Divide by r in cases where r is non-zero
                    D(d ~= 0) = D(d ~= 0)./d(d ~= 0);
                    %D(r ~= 0) = D(r ~= 0)./r(r ~= 0);
                    D = sparse(I,J,D,n,n);
                    
                    ii1 = ii1+1;
                    DKff{ii1} = D;
                end
            end
          case 'FIC'
            % Evaluate help arguments for gradient evaluation
            % instead of calculating trace(invC*Cdm) calculate sum(invCv.*Cdm(:)), when
            % Cdm and invC are symmetric matricess of same size. This is 67 times faster
            % with n=215
            u = gpcf.X_u;

            % Derivatives of K_uu and K_uf with respect to magnitude sigma and lengthscale
            % NOTE! Here we have already taken into account that the parameters are transformed
            % through log() and thus dK/dlog(p) = p * dK/dp
            K_uu = feval(gpcf.fh_trcov, gpcf, u);
            K_uf = feval(gpcf.fh_cov, gpcf, u, x);
            DKff = feval(gpcf.fh_trvar, gpcf, x);

            l = gpcf.l;

            ii1=1;
            DKuu{ii1} = K_uu;
            DKuf{ii1} = K_uf;
            
            % loop over all the lengthScales
            if length(gpcf.lengthScale) == 1
                % In the case of isotropic PPCS2
                s2 = 1./gpcf.lengthScale.^2;
                ma2 = gpcf.magnSigma2;

                % Calculate the sparse distance (lower triangle) matrix
                dist1 = 0; dist2 = 0;
                for i=1:m
                    dist1 = dist1 + s2.*(gminus(u(:,i),x(:,i)')).^2;
                    dist2 = dist2 + s2.*(gminus(u(:,i),u(:,i)')).^2;
                end
                d1 = sqrt(dist1); d2 = sqrt(dist2);
                cs1 = max(1-d1,0);  cs2 = max(1-d2,0);
                const1 = 2.*l^2+8.*l+6;
                const2 = l^2+4.*l+3;

                DKuf_l = -ma2.*cs1.^(l+1).*d1.*(cs1.*(const1.*d1+3.*l+6)-(l+2).*(const2.*dist1+(3.*l+6).*d1+3))/3;
                DKuu_l = -ma2.*cs2.^(l+1).*d2.*(cs2.*(const1.*d2+3.*l+6)-(l+2).*(const2.*dist2+(3.*l+6).*d2+3))/3;
                
                ii1=ii1+1;
                DKuu{ii1} = DKuu_l;
                DKuf{ii1} = DKuf_l;
            else

                % In the case ARD is used
                s2 = 1./gpcf.lengthScale.^2;
                ma2 = gpcf.magnSigma2;

                % Calculate the sparse distance (lower triangle) matrix
                % and the distance matrix for each component
                dist1 = 0; dist2 = 0;
                d_l1 = [];  d_l2=[];
                for i = 1:m
                    dist1 = dist1 + s2(i).*gminus(u(:,i),x(:,i)').^2;
                    dist2 = dist2 + s2(i).*gminus(u(:,i),u(:,i)').^2;
                    d_l1{i} = s2(i).*(gminus(u(:,i),x(:,i)')).^2;
                    d_l2{i} = s2(i).*(gminus(u(:,i),u(:,i)')).^2;
                end
                d1 = sqrt(dist1); d2 = sqrt(dist2);
                cs1 = max(1-d1,0); cs2 = max(1-d2,0);

                const1 = l^2+4.*l+3;
                const2 = 3.*l+6;

                for i = 1:m
                    % Calculate the gradient matrix
                    DKuf_l = ma2.*(l+2).*d_l1{i}.*cs1.^(l+1).*(const1.*dist1 + const2.*d1 + 3)./3;
                    DKuf_l = DKuf_l - ma2.*cs1.^(l+2).*d_l1{i}.*(2.*const1.*d1 + const2)./3;
                    % Divide by r in cases where r is non-zero
                    DKuf_l(d1 ~= 0) = DKuf_l(d1 ~= 0)./d1(d1 ~= 0);
                    
                    DKuu_l = ma2.*(l+2).*d_l2{i}.*cs2.^(l+1).*(const1.*dist2 + const2.*d2 + 3)./3;
                    DKuu_l = DKuu_l - ma2.*cs2.^(l+2).*d_l2{i}.*(2.*const1.*d2 + const2)./3;
                    % Divide by r in cases where r is non-zero
                    DKuu_l(d2 ~= 0) = DKuu_l(d2 ~= 0)./d2(d2 ~= 0);
                    
                    ii1=ii1+1;
                    DKuu{ii1} = DKuu_l;
                    DKuf{ii1} = DKuf_l;
                end
                
            end
          case 'PIC_BLOCK'
            % Evaluate help arguments for gradient evaluation
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
            
            ii1=1;
            DKuu{ii1} = K_uu;
            DKuf{ii1} = K_uf;
            DKff{ii1} = K_ff;

            l = gpcf.l;

            % loop over all the lengthScales
            if length(gpcf.lengthScale) == 1
                % In the case of isotropic PPCS2
                s2 = 1./gpcf.lengthScale.^2;
                ma2 = gpcf.magnSigma2;

                % Calculate the sparse distance (lower triangle) matrix
                dist1 = 0; dist2 = 0;
                for j=1:length(ind)
                    dist3{j} = zeros(numel(ind{j}),numel(ind{j}));
                end
                for i=1:m
                    dist1 = dist1 + s2.*(gminus(u(:,i),x(:,i)')).^2;
                    dist2 = dist2 + s2.*(gminus(u(:,i),u(:,i)')).^2;
                    for j=1:length(ind)
                        dist3{j} = dist3{j} + s2.*(gminus(x(ind{j},i),x(ind{j},i)')).^2;
                    end
                end
                d1 = sqrt(dist1); d2 = sqrt(dist2);
                cs1 = max(1-d1,0);  cs2 = max(1-d2,0);
                const1 = 2.*l^2+8.*l+6;
                const2 = l^2+4.*l+3;

                DKuf_l = -ma2.*cs1.^(l+1).*d1.*(cs1.*(const1.*d1+3.*l+6)-(l+2).*(const2.*dist1+(3.*l+6).*d1+3))/3;
                DKuu_l = -ma2.*cs2.^(l+1).*d2.*(cs2.*(const1.*d2+3.*l+6)-(l+2).*(const2.*dist2+(3.*l+6).*d2+3))/3;
                
                for j=1:length(ind)
                    d3 = sqrt(dist3{j});
                    cs3 = max(1-d3,0);
                    DKff_l{j} = -ma2.*cs3.^(l+1).*d3.*(cs3.*(const1.*d3+3.*l+6)-(l+2).*(const2.*dist3{j}+(3.*l+6).*d3+3))/3;
                end
                
                ii1=ii1+1;
                DKuu{ii1} = DKuu_l;
                DKuf{ii1} = DKuf_l;
                DKff{ii1} = DKff_l;
            else

                % In the case ARD is used
                s2 = 1./gpcf.lengthScale.^2;
                ma2 = gpcf.magnSigma2;

                % Calculate the sparse distance (lower triangle) matrix
                % and the distance matrix for each component
                dist1 = 0; dist2 = 0;
                d_l1 = [];  d_l2=[];
                for j=1:length(ind)
                    dist3{j} = zeros(size(ind{j},1),size(ind{j},1));
                end
                for i = 1:m
                    dist1 = dist1 + s2(i).*gminus(u(:,i),x(:,i)').^2;
                    dist2 = dist2 + s2(i).*gminus(u(:,i),u(:,i)').^2;
                    d_l1{i} = s2(i).*(gminus(u(:,i),x(:,i)')).^2;
                    d_l2{i} = s2(i).*(gminus(u(:,i),u(:,i)')).^2;
                    for j=1:length(ind)
                        dist3{j} = dist3{j} + s2(i).*(gminus(x(ind{j},i),x(ind{j},i)')).^2;
                        d_l3{j,i} = s2(i).*(gminus(x(ind{j},i),x(ind{j},i)')).^2;
                    end
                end
                d1 = sqrt(dist1); d2 = sqrt(dist2);
                cs1 = max(1-d1,0); cs2 = max(1-d2,0);

                const1 = l^2+4.*l+3;
                const2 = 3.*l+6;

                for i = 1:m
                    % Calculate the gradient matrix
                    DKuf_l = ma2.*(l+2).*d_l1{i}.*cs1.^(l+1).*(const1.*dist1 + const2.*d1 + 3)./3;
                    DKuf_l = DKuf_l - ma2.*cs1.^(l+2).*d_l1{i}.*(2.*const1.*d1 + const2)./3;
                    % Divide by r in cases where r is non-zero
                    DKuf_l(d1 ~= 0) = DKuf_l(d1 ~= 0)./d1(d1 ~= 0);
                    
                    DKuu_l = ma2.*(l+2).*d_l2{i}.*cs2.^(l+1).*(const1.*dist2 + const2.*d2 + 3)./3;
                    DKuu_l = DKuu_l - ma2.*cs2.^(l+2).*d_l2{i}.*(2.*const1.*d2 + const2)./3;
                    % Divide by r in cases where r is non-zero
                    DKuu_l(d2 ~= 0) = DKuu_l(d2 ~= 0)./d2(d2 ~= 0);
                    
                    for j=1:length(ind)
                        d3 = sqrt(dist3{j});
                        cs3 = max(1-d3,0);
                        DKff_l{j} = ma2.*(l+2).*d_l3{j,i}.*cs3.^(l+1).*(const1.*dist3{j} + const2.*d3 + 3)./3;
                        DKff_l{j} = DKff_l{j} - ma2.*cs3.^(l+2).*d_l3{j,i}.*(2.*const1.*d3 + const2)./3;
                        % Divide by r in cases where r is non-zero
                        DKff_l{j}(d3 ~= 0) = DKff_l{j}(d3 ~= 0)./d3(d3 ~= 0);
                    end
                    ii1=ii1+1;
                    DKuu{ii1} = DKuu_l;
                    DKuf{ii1} = DKuf_l;
                    DKff{ii1} = DKff_l;
                end
                
            end
          case {'CS+FIC'}
            % Evaluate help arguments for gradient evaluation
            % instead of calculating trace(invC*Cdm) calculate sum(invCv.*Cdm(:)), when
            % Cdm and invC are symmetric matricess of same size. This is 67 times faster
            % with n=215

            Cdm = gpcf_ppcs2_trcov(gpcf, x);
            l = gpcf.l;
            [I,J] = find(Cdm);

            ii1=1;
            DKff{ii1} = Cdm;
            
            % loop over all the lengthScales
            if length(gpcf.lengthScale) == 1
                % In the case of isotropic PPCS2
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
                const1 = 2.*l^2+8.*l+6;
                const2 = l^2+4.*l+3;
                D = -ma2.*cs.^(l+1).*d.*(cs.*(const1.*d+3.*l+6)-(l+2).*(const2.*d2+(3.*l+6).*d+3))/3;
                D = sparse(I,J,D,n,n);

                ii1=ii1+1;
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

                const1 = 2.*l^2+8.*l+6;
                const2 = l^2+4.*l+3;

                D = {};
                for i = 1:m
                    % Calculate the gradient matrix
                    indt = d ~= 0;
                    D = -ma2.*cs.^(l+1).*d_l(:,i).*(cs.*(const1.*d+3*l+6)-(l+2)*(const2.*d2+(3*l+6)*d+3))/3;
                    % Divide by r in cases where r is non-zero
                    D(indt) = D(indt)./d(indt);
                    %D(r ~= 0) = D(r ~= 0)./r(r ~= 0);
                    D = sparse(I,J,D,n,n);

                    ii1=ii1+1;
                    DKff{ii1} = D;
                end
            end
            %D_ma = Cdm;
            %Z = sinv(La,Cdm);
            %Bdm = b'*(Cdm*b);
            %Cdm = sum(invCv.*Cdm(:)); % help argument for magnSigma2
        end

        % Evaluate the gdata and gprior with respect to magnSigma2
        i1 = i1+1;
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
                gprior(i1)=feval(gpp.lengthScale.fg, ...
                                 gpcf.lengthScale(i2), ...
                                 gpp.lengthScale.a, 'x').*gpcf.lengthScale(i2) - 1;
            end
        else
            i1=i1+1;
            gprior(i1)=feval(gpp.lengthScale.fg, ...
                             gpcf.lengthScale, ...
                             gpp.lengthScale.a, 'x').*gpcf.lengthScale -1;
            
        end
    end
    
    
    function [gprior_ind, DKuu, DKuf]  = gpcf_ppcs2_gind(gpcf, x, t, g_ind, gdata_ind, gprior_ind, varargin)
    %GPCF_PPCS2_GIND    Evaluate gradient of error for SE covariance function 
    %                  with respect to inducing inputs.
    %
    %	Descriptioni
    %	[DKuu_u, DKuf_u] = GPCF_PPCS2_GIND(W, GPCF, X, T) 
    %
    %	See also
    %

    % Copyright (c) 1998-2001 Aki Vehtari
    % Copyright (c) 2006      Jarno Vanhatalo
        
    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.

        DKuu={};
        DKuf={};
    end
    
    
    function C = gpcf_ppcs2_cov(gpcf, x1, x2, varargin)
    % GP_PPCS2_COV     Evaluate covariance matrix between two input vectors. 
    %
    %         Description
    %         C = GP_PPCS2_COV(GP, TX, X) takes in covariance function of a Gaussian
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
    %         if isempty(x2)
    %             x2=x1;
    %         end
    %         [n1,m1]=size(x1);
    %         [n2,m2]=size(x2);
    %         
    %         if m1~=m2
    %             error('the number of columns of X1 and X2 has to be same')
    %         end
    %                 
    %         C=zeros(n1,n2);
    %         ma2 = gpcf.magnSigma2;
    %         l = gpcf.l;
    %         % Evaluate the covariance
    %         if ~isempty(gpcf.lengthScale)  
    %             s = 1./gpcf.lengthScale.^2;
    %             if m1==1 && m2==1
    %                 dd = gminus(x1,x2');
    %                 dist=dd.^2*s;
    %             else
    %                 % If ARD is not used make s a vector of 
    %                 % equal elements 
    %                 if size(s)==1
    %                     s = repmat(s,1,m1);
    %                 end
    %                 dist=zeros(n1,n2);
    %                 for j=1:m1
    %                     dd = gminus(x1(:,j),x2(:,j)');
    %                     dist = dist + dd.^2.*s(:,j);
    %                 end
    %             end
    %             r = sqrt(dist);
    %             cs = sparse(max(0, 1-r));
    %             C = ma2.*cs.^(l+2).*((l^2+4*l+3).*r.^2+(3*l+6).*r+3)/3;
    %             %C = ma2.*cs.^(l+2).*((l^2+4*l+3).*r.^2+(3*l+6).*r+3);
    %         end


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
        RR=zeros(n1,n2);
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
            %                 if (ntriplets > len)
            %                     I(2*len) = 0;
            %                     J(2*len) = 0;
            %                     R(2*len) = 0;
            %                 end
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
        C = ma.*cs.^(l+2).*((l^2+4*l+3).*r.^2+(3*l+6).*r+3)/3;
        C = sparse(I,J,C,n1,n2) + sparse(I0,J0,ma,n1,n2);
    end
    
    function C = gpcf_ppcs2_trcov(gpcf, x)
    % GP_PPCS2_TRCOV     Evaluate training covariance matrix of inputs. 
    %
    %         Description
    %         C = GP_PPCS2_TRCOV(GP, TX) takes in covariance function of a Gaussian
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
        const1 = l^2+4*l+3;
        const2 = 3*l+6;
        cs = max(0,1-rn);
        C = ma.*cs.^(l+2).*(const1.*rn.^2+const2.*rn+3)/3;
        C = sparse(I,J,C,n,n);
        C = C + C' + sparse(1:n,1:n,ma,n,n);
    end
    
    function C = gpcf_ppcs2_covvec(gpcf, x1, x2, varargin)
    % GPCF_PPCS2_COVVEC     Evaluate covariance vector between two input vectors. 
    %
    %         Description
    %         C = GPCF_PPCS2_COVVEC(GP, TX, X) takes in Gaussian process GP and two
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
        
        l = gpcf.l;
        ma2 = gpcf.magnSigma2;
        
        di2 = 0;
        s = 1./gpcf.lengthScale.^2;
        for i = 1:m1
            di2 = di2 + s.*(x1(:,i) - x2(:,i)).^2;
        end
        di2 = sqrt(di2);
        cs = max(0, 1-di2);
        C = gpcf.magnSigma2.*cs.^(l+2).*((l^2+4*l+3).*di2.^2+(3*l+6).*di2+3)/3;
        %C = gpcf.magnSigma2.*cs.^(l+2).*((l^2+4*l+3).*di2.^2+(3*l+6).*di2+3);            
    end
    
    
    function C = gpcf_ppcs2_trvar(gpcf, x)
    % GP_PPCS2_TRVAR     Evaluate training variance vector of inputs. 
    %
    %         Description
    %         C = GP_PPCS2_TRVAR(GP, TX) takes in covariance function of a Gaussian
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

    function reccf = gpcf_ppcs2_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %          DescriptionK_cs = gp_trcov(gp,x);
    %          RECCF = GPCF_PPCS2_RECAPPEND(RECCF, RI, GPCF) takes old covariance 
    %          function record RECCF, record index RI, RECAPPEND returns a 
    %          structure RECCF containing following record fields:
    %          lengthHyper    = 
    %          lengthHyperNu  = 
    %          lengthScale    = 
    %          magnSigma2     = 
        
    % Initialize record
        if nargin == 2
            reccf.type = 'gpcf_ppcs2';
            reccf.nin = ri;
            reccf.nout = 1;
            reccf.l = floor(reccf.nin/2)+3;

            % cf is compactly supported
            reccf.cs = 1;
            
            % Initialize parameters
            reccf.lengthScale= [];
            reccf.magnSigma2 = [];
            
            % Set the function handles
            reccf.fh_pak = @gpcf_ppcs2_pak;
            reccf.fh_unpak = @gpcf_ppcs2_unpak;
            reccf.fh_e = @gpcf_ppcs2_e;
            reccf.fh_g = @gpcf_ppcs2_g;
            reccf.fh_cov = @gpcf_ppcs2_cov;
            reccf.fh_covvec = @gpcf_ppcs2_covvec;
            reccf.fh_trcov  = @gpcf_ppcs2_trcov;
            reccf.fh_trvar  = @gpcf_ppcs2_trvar;
            %  gpcf.fh_sampling = @hmc2;
            %  reccf.sampling_opt = hmc2_opt;
            reccf.fh_recappend = @gpcf_ppcs2_recappend;  
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