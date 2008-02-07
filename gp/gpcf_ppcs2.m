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
        trcov_I0 = [];
        trcov_J0 = [];
        trcov_c0 = [];
        %        trcov_x0 = x;
        trcov_ls0 = zeros(size(gpcf.lengthScale));
        trcov_ms0 = zeros(size(gpcf.magnSigma2));

        
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
    
    function [g, gdata, gprior]  = gpcf_ppcs2_ghyper(gpcf, x, t, g, gdata, gprior, varargin)
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
            invCv=invC(:);
            Cdm = gpcf_ppcs2_trcov(gpcf, x);

            b = varargin{2};
            if length(varargin) > 4
                b2 = varargin{5};
                b3 = varargin{6};
            end
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
                
                %size(D)
                %size(b)
                Bdl = b'*(D*b);
                Cdl = sum(invCv.*D(:)); % help arguments for lengthScale 
                
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
                
                    Bdl(i) = b'*(D*b);
                    Cdl(i) = sum(invCv.*D(:)); % help arguments for lengthScale 
                end
            end
            Bdm = b'*(Cdm*b);
            Cdm = sum(invCv.*Cdm(:)); % help argument for magnSigma2

            
          case {'CS+PIC' 'CS+FIC'}
            % Evaluate help arguments for gradient evaluation
            % instead of calculating trace(invC*Cdm) calculate sum(invCv.*Cdm(:)), when 
            % Cdm and invC are symmetric matricess of same size. This is 67 times faster 
            % with n=215 
            L = varargin{1};
            b = varargin{2};
            iKuuKuf = varargin{3};       % u x f
            La = varargin{4};          % array of siz
                                         %iLamLL = inv(Labl)-L*L';
            if length(varargin) > 4
                b2 = varargin{5};
                b3 = varargin{6};
            end
            
            Cdm = gpcf_ppcs2_trcov(gpcf, x);
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
                
                %Bdl = b*(D*b');
                %Cdl = sum(invCv.*D(:)); % help arguments for lengthScale 
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
                
                D = {};
                for i = 1:m 
                    % Calculate the gradient matrix                    
                    D{i} = -ma2.*cs.^(l+1).*d_l(:,i).*(cs.*(const1.*d+3*l+6)-(l+2)*(const2.*d2+(3*l+6)*d+3))/3;
                    % Divide by r in cases where r is non-zero
                    D{i}(d ~= 0) = D{i}(d ~= 0)./d(d ~= 0);
                    %D(r ~= 0) = D(r ~= 0)./r(r ~= 0);
                    D{i} = sparse(I,J,D{i},n,n);
                
                    %Bdl(i) = b*(D*b');
                    %Cdl(i) = sum(invCv.*D(:)); % help arguments for lengthScale 
                end
            end
            D_ma = Cdm;
            %Bdm = b'*(Cdm*b);
            %Cdm = sum(invCv.*Cdm(:)); % help argument for magnSigma2
        end

        % Evaluate the gdata and gprior with respect to magnSigma2
        i1 = i1+1;
        switch gpcf.type
          case 'FULL'
            gdata(i1) = 0.5.*(Cdm - Bdm);
          case 'FIC'
            gdata(i1) = DE_Kuu(:)'*K_uu(:) + DE_Kuf(:)'*K_uf(:) + gpcf.magnSigma2.*sum(DE_Kff);
          case 'PIC_BLOCK'
            KfuiKuuKuu = iKuuKuf'*K_uu;
            %            H = (2*K_uf'- KfuiKuuKuu)*iKuuKuf;
            % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
            gdata(i1) = -0.5.*((2*b*K_uf'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*K_uf'*iKuuKuf))) - ...
                               sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
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
            end
          case 'PIC_BAND'
            KfuiKuuKuu = iKuuKuf'*K_uu;
            % Note! H = (2*K_uf'- KfuiKuuKuu)*iKuuKuf, but here we set actually H = mask(H) and the computations 
            % with full(H) are done with partition
            H = (2*K_uf - KfuiKuuKuu');
            H = sum(H(:,ind(:,1)).*iKuuKuf(:,ind(:,2)));
            H = sparse(ind(:,1),ind(:,2),H,n,n);
            % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
            gdata(i1) = -0.5.*((2*b*K_uf'-(b*KfuiKuuKuu))*(iKuuKuf*b'));
            gdata(i1) = gdata(i1) - sum(sum(L'.*(L'*K_uf'*iKuuKuf)));
            gdata(i1) = gdata(i1) + 0.5.*sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf)));
            gdata(i1) = gdata(i1) + sum(sum((La\(K_uf'-0.5.*KfuiKuuKuu))'.*iKuuKuf));
            gdata(i1) = gdata(i1) + 0.5.*-(b(ind(:,1)).*kv_ff')*b(ind(:,2))';
            gdata(i1) = gdata(i1) + 0.5.*b*H*b';
            gdata(i1) = gdata(i1) + 0.5.*trace(La\(K_ff-H));
            gdata(i1) = gdata(i1) + 0.5.*sum(sum(L'.*(L'*(H-K_ff))));               %- trace(Labl{i}\H(ind{i},ind{i})) 
          case {'CS+FIC' 'CS+PIC'}
            %gdata(i1) = 0.5*(trace(iLamLL*D_ma) - b*D_ma*b');
            %gdata(i1) = 0.5*(trace(La\D_ma) - trace(L*L'*D_ma) - b*D_ma*b');
            gdata(i1) = 0.5*(trace(La\D_ma) - sum(sum(L.*(L'*D_ma')')) - b*D_ma*b');
            if length(varargin) > 4
                        gdata(i1) = gdata(i1) + 0.5.*b2*D_ma*b3;
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
                    gdata(i1)= DE_Kuu(:)'*DKuu_l(:,i2) + DE_Kuf(:)'*DKuf_l(:,i2);
                  case 'PIC_BLOCK'
                    KfuiKuuDKuu_l = iKuuKuf'*DKuu_l{i2};
                    %            H = (2*DKuf_l'- KfuiKuuDKuu_l)*iKuuKuf;
                    % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
                    gdata(i1) = -0.5.*((2*b*DKuf_l{i2}'-(b*KfuiKuuDKuu_l))*(iKuuKuf*b') + 2.*sum(sum(L'.*((L'*DKuf_l{i2}')*iKuuKuf))) - ...
                                       sum(sum(L'.*((L'*KfuiKuuDKuu_l)*iKuuKuf))));
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
                    end
                  case {'CS+PIC' 'CS+FIC'}
                    %gdata(i1) = 0.5*(trace(iLamLL*D{i2}) - b*D{i2}*b');
                    %gdata(i1) = 0.5*(trace(La\D{i2}) - trace(L*L'*D{i2}) - b*D{i2}*b');
                    gdata(i1) = 0.5*(trace(La\D{i2}) - sum(sum(L.*(L'*D{i2})')) - b*D{i2}*b');
                   if length(varargin) > 4
                        gdata(i1) = gdata(i1) + 0.5.*b2*D{i2}*b3;
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
                gdata(i1)= DE_Kuu(:)'*DKuu_l(:) + DE_Kuf(:)'*DKuf_l(:);
              case 'PIC_BLOCK'
                KfuiKuuDKuu_l = iKuuKuf'*DKuu_l;
                %            H = (2*DKuf_l'- KfuiKuuDKuu_l)*iKuuKuf;
                % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
                gdata(i1) = -0.5.*((2*b*DKuf_l'-(b*KfuiKuuDKuu_l))*(iKuuKuf*b') + 2.*sum(sum(L'.*((L'*DKuf_l')*iKuuKuf))) - ...
                                   sum(sum(L'.*((L'*KfuiKuuDKuu_l)*iKuuKuf))));
                for i=1:length(K_ff)
                    gdata(i1) = gdata(i1) ...                   %   + trace(Labl{i}\H(ind{i},ind{i})) ...
                        + 0.5.*(-b(ind{i})*DKff_l{i}*b(ind{i})' ...
                                + 2.*b(ind{i})*DKuf_l(:,ind{i})'*iKuuKuf(:,ind{i})*b(ind{i})'- ...
                                b(ind{i})*KfuiKuuDKuu_l(ind{i},:)*iKuuKuf(:,ind{i})*b(ind{i})' ...       %H(ind{i},ind{i})
                                + trace(Labl{i}\DKff_l{i})...
                                - trace(L(ind{i},:)*(L(ind{i},:)'*DKff_l{i})) ...               %- trace(Labl{i}\H(ind{i},ind{i})) 
                                + 2.*sum(sum(L(ind{i},:)'.*(L(ind{i},:)'*DKuf_l(:,ind{i})'*iKuuKuf(:,ind{i})))) - ...
                                sum(sum(L(ind{i},:)'.*((L(ind{i},:)'*KfuiKuuDKuu_l(ind{i},:))*iKuuKuf(:,ind{i}))))); 
                    %trace(L(ind{i},:)*(L(ind{i},:)'*H(ind{i},ind{i}))));
                end
              case 'PIC_BAND'
                KfuiKuuDKuu_l = iKuuKuf'*DKuu_l;
                % Note! H = (2*K_uf'- KfuiKuuKuu)*iKuuKuf, but here we set actually H = mask(H) and the computations 
                % with full(H) are done with partition
                H = (2*DKuf_l - KfuiKuuDKuu_l');
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
              case {'CS+PIC' 'CS+FIC'}
                %gdata(i1) = 0.5*(trace(iLamLL*D) - b*D*b');
                %gdata(i1) = 0.5*(trace(La\D) - sum(sum((L*L').*D')) - b*D*b');
                gdata(i1) = 0.5*(trace(La\D) - sum(sum(L.*(L'*D)')) - b*D*b');
                if length(varargin) > 4
                     gdata(i1) = gdata(i1) + 0.5.*b2*D*b3;
                end
            end
            gprior(i1)=feval(gpp.lengthScale.fg, ...
                             gpcf.lengthScale, ...
                             gpp.lengthScale.a, 'x').*gpcf.lengthScale -1;
            
        end
        g = gdata + gprior;
    end
        
    
    function [DKuu_u, DKuf_u]  = gpcf_ppcs2_gind(gpcf, x, t, varargin)
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
        
        gpp=gpcf.p;
        [n, m] =size(x);
                
        % First check if sparse model is used
        switch gpcf.type
           case {'FIC', 'PIC_BLOCK', 'PIC_BAND'}
            % Evaluate the help matrices for the gradient evaluation (see
            % gpcf_ppcs2_trcov)

            u = gpcf.X_u;
            n_u = size(u,1);
            
            % Derivatives of K_uu and K_uf with respect to inducing inputs
            K_uu = feval(gpcf.fh_trcov, gpcf, u);
            K_uf = feval(gpcf.fh_cov, gpcf, u, x);
            
            if length(gpcf.lengthScale) == 1
                % In the case of an isotropic PPCS2
                s = repmat(1./gpcf.lengthScale.^2, 1, m);
            else
                s = 1./gpcf.lengthScale.^2;
            end
            for i=1:m
                for j = 1:size(u,1)
                    dist = zeros(size(u,1),n);
                    dist2 = zeros(size(K_uu));
                    
                    dist(j,:) = -2.*s(i).*gminus(u(j,i),x(:,i)');
                    dist2(j,:) = -2.*s(i).* gminus(u(j,i),u(:,i)');
                    dist2 = dist2 + dist2';
                                        
                    dist = dist.*K_uf;
                    dist2 = dist2.*K_uu;
                    %                    dist2 = dist2 + dist2' - diag(diag(dist2));                    
                    
                    DKuf_u(:,j+(i-1)*n_u) = dist(:);         % Matrix of size uf x mu
                    DKuu_u(:,j+(i-1)*n_u) = dist2(:);        % Matrix of size uu x mu
                end
            end
        end
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
        l = gpcf.l;
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
            r = sqrt(dist);
            cs = sparse(max(0, 1-r));
            C = ma2.*cs.^(l+2).*((l^2+4*l+3).*r.^2+(3*l+6).*r+3)/3;
            %C = ma2.*cs.^(l+2).*((l^2+4*l+3).*r.^2+(3*l+6).*r+3);
        end
        cov_x1=x1;
        cov_x2=x2;
        cov_ls=gpcf.lengthScale;
        cov_ms=gpcf.magnSigma2;
        cov_C=C;
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

% $$$         s2 = 1./(gpcf.lengthScale.^2);
% $$$          
% $$$         ma = gpcf.magnSigma2;
% $$$         l = gpcf.l;
% $$$ 
% $$$         const1 = l^2+4*l+3;
% $$$         const2 = 3*l+6;
% $$$         
% $$$         C = zeros(n,n);
% $$$         for i = 1:n
% $$$             for j = 1:n
% $$$                 d2 = 0;
% $$$                 for k = 1:m
% $$$                     d2 = d2 + s2(k).*(x(i,k)-x(j,k)).^2;
% $$$                 end
% $$$                 d = sqrt(d2);
% $$$                 if d > 1
% $$$                     C(i,j) = 0;
% $$$                 else
% $$$                     C(i,j) = ma.*(1-d).^(l+2).*(const1.*d.^2+const2.*d+3)/3;
% $$$                 end
% $$$                 
% $$$                 
% $$$                  
% $$$             end
% $$$         end
% $$$         
% $$$         return
       
        
        if abs(trcov_ls0-gpcf.lengthScale)<1e-10 & abs(trcov_ms0-gpcf.magnSigma2)<1e-10
            C = sparse(trcov_I0, trcov_J0, trcov_c0, n, n);
        else
            s = 1./(gpcf.lengthScale);
            s2 = s.^2;
            if size(s)==1
                s2 = repmat(s2,1,m);
            end
            ma = gpcf.magnSigma2;
            l = gpcf.l;
            
            % Compute the sparse distance matrix.
            ntriplets = floor(0.03*n*n);
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
                d = sqrt(d);
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
                I(ntrip_prev+1:ntriplets) = ii1+I2;
                J(ntrip_prev+1:ntriplets) = ii1;
                R(ntrip_prev+1:ntriplets) = R2;
            end
            R = sparse(I(1:ntriplets),J(1:ntriplets),R(1:ntriplets),n,n);

% $$$             % Compute the sparse distance matrix.
% $$$             R = sparse([],[],[],n,n,0);
% $$$             for ii1=1:n-1
% $$$                 d = zeros(n-ii1,1);
% $$$                 col_ind = ii1+1:n;
% $$$                 for ii2=1:m
% $$$                     d = d+s2(ii2).*(x(col_ind,ii2)-x(ii1,ii2)).^2;
% $$$                 end
% $$$                 d = sqrt(d);
% $$$                 d(d >= 1) = 0;
% $$$                 R(col_ind,ii1) = d;
% $$$             end
            
            % Find the non-zero elements of R.
            [I,J,rn] = find(R);
% $$$             
% $$$             % Create the 'compact support' matrix.
% $$$             cs = sparse(I,J,1-rn,n,n) + speye(n,n);
% $$$ 
% $$$             % Calculate the covariance matrix
            const1 = l^2+4*l+3;
            const2 = 3*l+6;
% $$$             C = ma.*cs.^(l+2).*(const1.*R.^2+const2.*R+3);
% $$$             
% $$$             % Add the upper triangle matrix.
% $$$             C = C + tril(C,-1)';
% $$$ 
            % An other way to construct C
            cs = 1-rn;
            C2 = ma.*cs.^(l+2).*(const1.*rn.^2+const2.*rn+3)/3;
            %C2 = ma.*cs.^(l+2).*(const1.*rn.^2+const2.*rn+3);
            C2 = sparse(I,J,C2,n,n);
            C = C2 + C2' + sparse(1:n,1:n,ma,n,n);
            %C = C2 + C2' + sparse(1:n,1:n,ma.*3,n,n);
            
            [I,J,c] = find(C);
            trcov_I0 = I;
            trcov_J0 = J;
            trcov_c0 = c;
            %        trcov_x0 = x;
            trcov_ls0 = gpcf.lengthScale;
            trcov_ms0 = gpcf.magnSigma2;
        end
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
    %          Description
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