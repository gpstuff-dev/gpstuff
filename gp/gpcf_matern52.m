function gpcf = gpcf_matern52(do, varargin)
%GPCF_SEXP	Create a squared exponential covariance function for Gaussian Process
%
%	Description
%
%	GPCF = GPCF_MATERN52('INIT', NIN) Create and initialize squared exponential 
%       covariance function fo Gaussian process 
%
%	The fields and (default values) in GPCF_MATERN52 are:
%	  type           = 'gpcf_matern52'
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
%                          (@gpcf_matern52_pak)
%         fh_unpak       = function handle to unpackin function
%                          (@gpcf_matern52_unpak)
%         fh_e           = function handle to error function
%                          (@gpcf_matern52_e)
%         fh_ghyper      = function handle to gradient function (with respect to hyperparameters)
%                          (@gpcf_sexp_ghyper)
%         fh_gind        = function handle to gradient function (with respect to inducing inputs)
%                          (@gpcf_sexp_gind)
%         fh_cov         = function handle to covariance function
%                          (@gpcf_matern52_cov)
%         fh_trcov       = function handle to training covariance function
%                          (@gpcf_matern52_trcov)
%         fh_trvar       = function handle to training variance function
%                          (@gpcf_matern52_trvar)
%         fh_sampling    = function handle to parameter sampling function
%                          (@hmc2)
%         sampling_opt   = options structure for fh_sampling
%                          (hmc2_opt)
%         fh_recappend   = function handle to record append function
%                          (gpcf_matern52_recappend)
%
%	GPCF = GPCF_MATERN52('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in GPCF.
%
%	See also
%	
%
%

% Copyright (c) 1996,1997 Christopher M Bishop, Ian T Nabney
% Copyright (c) 1998,1999 Aki Vehtari
% Copyright (c) 2006      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end

    % Initialize the covariance function
    if strcmp(do, 'init')
        nin = varargin{1};
        gpcf.type = 'gpcf_matern52';
        gpcf.nin = nin;
        gpcf.nout = 1;
        
        % Initialize parameters
        gpcf.lengthScale= repmat(10, 1, nin); 
        gpcf.magnSigma2 = 0.1;
        
        % Initialize prior structure
        gpcf.p=[];
        gpcf.p.lengthScale=[];
        gpcf.p.magnSigma2=[];
        
        % initialise the return values of nested functions
        % and the variables that are used to check the similarity
        % Similarity chechkers
        % function [g, gdata, gprior]  = gpcf_matern52_g(gpcf, x, t, g, gdata, gprior, invC, varargin)
        % function reccf = gpcf_matern52_recappend(reccf, ri, gpcf)
        
        cov_x1=[]; cov_x2=[]; cov_ls=[]; cov_ms=[]; cov_C=[];
        trcov_x=[]; trcov_ls=[]; trcov_ms=[]; trcov_C=[];
        trvar_x=[]; trvar_ls=[]; trvar_ms=[]; trvar_C=[];
        e_x=[]; e_t=[]; e_ls=[]; e_ms=[]; e_e=[];
        
        % Set the function handles to the nested functions
        gpcf.fh_pak = @gpcf_matern52_pak;
        gpcf.fh_unpak = @gpcf_matern52_unpak;
        gpcf.fh_e = @gpcf_matern52_e;
        gpcf.fh_ghyper = @gpcf_matern52_ghyper;
        gpcf.fh_gind = @gpcf_matern52_gind;
        gpcf.fh_cov = @gpcf_matern52_cov;
        gpcf.fh_trcov  = @gpcf_matern52_trcov;
        gpcf.fh_trvar  = @gpcf_matern52_trvar;
        %  gpcf.fh_sampling = @hmc2;
        %  gpcf.sampling_opt = hmc2_opt;
        gpcf.fh_recappend = @gpcf_matern52_recappend;
        
        if length(varargin) > 1
            if mod(nargin,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=2:2:length(varargin)-1
                if strcmp(varargin{i},'magnSigma2')
                    gpcf.magnSigma2 = varargin{i+1};
                elseif strcmp(varargin{i},'lengthScale')
                    gpcf.lengthScale = varargin{i+1};
                elseif strcmp(varargin{i},'fh_sampling')
                    gpcf.fh_sampling = varargin{i+1};
                else
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
            if strcmp(varargin{i},'magnSigma2')
                gpcf.magnSigma2 = varargin{i+1};
            elseif strcmp(varargin{i},'lengthScale')
                gpcf.lengthScale = varargin{i+1};
            elseif strcmp(varargin{i},'fh_sampling')
                gpcf.fh_sampling = varargin{i+1};
            else
                error('Wrong parameter name!')
            end    
        end
    end

    function w = gpcf_matern52_pak(gpcf, w)
    %GPcf_MATERN52_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %	W = GP_MATERN52_PAK(GPCF, W) takes a Gaussian Process data structure GP and
    %	combines the hyper-parameters into a single row vector W.
    %
    %	The ordering of the parameters in HP is defined by
    %	  hp = [hyper-params of gp.cf{1}, hyper-params of gp.cf{2}, ...];
    %
    %	See also
    %	GPCF_MATERN52_UNPAK
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

    
    function [gpcf, w] = gpcf_matern52_unpak(gpcf, w)
    %GPCF_MATERN52_UNPAK  Separate GP covariance function hyper-parameter vector into components. 
    %
    %	Description
    %	GP = GPCF_MATERN52_UNPAK(GP, W) takes an Gaussian Process data structure GP
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

    function eprior =gpcf_matern52_e(gpcf, x, t)
    %GPCF_MATERN52_E	Evaluate prior contribution of error of covariance function SE.
    %
    %	Description
    %	E = GPCF_MATERN52_E(W, GP, X, T) takes a gp data structure GPCF together
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
        
    % $$$     if issame(e_x,x) && issame(e_t,t) && issame(e_ls,gpcf.lengthScale) && issame(e_ms,gpcf.magnSigma2)
    % $$$         eprior = e_e;
    % $$$     else
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
    % $$$     end
    end
    
    function [g, gdata, gprior]  = gpcf_matern52_ghyper(gpcf, x, t, g, gdata, gprior, varargin)
    %GPCF_MATERN52_G Evaluate gradient of error for MATERN52 covariance function.
    %                   with respect to the hyperparameters
    %
    %	Descriptioni
    %	G = GPCF_MATERN52_G(W, GPCF, X, T, G, GDATA, GPRIOR, VARARGIN) takes a gp 
    %   hyper-parameter vector W, data structure GPCF a matrix X of input vectors a 
    %   matrix T of target vectors, inverse covariance function , 
    %	and evaluates the error gradient G. Each row of X corresponds to one 
    %   input vector and each row of T corresponds to one target vector.
    %
    %	[G, GDATA, GPRIOR] = GPCF_MATERN52_G(GP, P, T) also returns separately  the
    %	data and prior contributions to the gradient.
    %
    %	See also
    %

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
            Cdm = gpcf_matern52_trcov(gpcf, x);
            invCv=invC(:);
            b = varargin{2};
            ma2 = gpcf.magnSigma2;
            % loop over all the lengthScales
            if length(gpcf.lengthScale) == 1
                % In the case of isotropic MATERN52
                s = 1./gpcf.lengthScale;
                dist = 0;
                for i=1:m
                    D = gminus(x(:,i),x(:,i)');
                    dist = dist + D.^2;
                end
                D = ma2./3.*(5.*dist.*s^2 + 5.*sqrt(5.*dist).*dist.*s.^3).*exp(-sqrt(5.*dist).*s);
                Bdl = b'*(D*b);
                Cdl = sum(invCv.*D(:)); % help arguments for lengthScale 
            else
                % In the case ARD is used
                s = 1./gpcf.lengthScale.^2;
                dist = 0;
                for i=1:m
                    dist = dist + s(i).*(gminus(x(:,i),x(:,i)')).^2;
                end
                dist=sqrt(dist);
                for i=1:m
                    D = ma2.*s(i).*((5+5.*sqrt(5).*dist)/3).*(gminus(x(:,i),x(:,i)')).^2.*exp(-sqrt(5).*dist);
                    Bdl(i) = b'*(D*b);
                    Cdl(i) = sum(invCv.*D(:)); % help arguments for lengthScale 
                end
            end
            Bdm = b'*(Cdm*b);
            Cdm = sum(invCv.*Cdm(:)); % help argument for magnSigma2        
          case 'FIC'
            % Evaluate the help matrices for the gradient evaluation (see
            % gpcf_sexp_trcov)
            
            DE_Kuu = varargin{1};             % u x u
            DE_Kuf = varargin{2};             % u x f
            DE_Kff = varargin{3};             % mask(R, M) (block/band) diagonal
            
            u = gpcf.X_u;
            
            % Derivatives of K_uu and K_uf with respect to magnitude sigma and lengthscale
            % NOTE! Here we have already taken into account that the parameters are transformed 
            % through log() and thus dK/dlog(p) = p * dK/dp
            K_uu = feval(gpcf.fh_trcov, gpcf, u);
            K_uf = feval(gpcf.fh_cov, gpcf, u, x);
            Cv_ff = feval(gpcf.fh_trvar, gpcf, x);
            
            % Evaluate help matrix for calculations of derivatives with respect to the lengthScale
            if length(gpcf.lengthScale) == 1
                s = 1./gpcf.lengthScale.^2;
                ma2 = gpcf.magnSigma2;
                dist = 0; dist2 = 0;
                for i=1:m
                    dist = dist + s.*(gminus(u(:,i),x(:,i)')).^2;
                    dist2 = dist2 + s.*(gminus(u(:,i),u(:,i)')).^2;
                end

                D1 = s.^2.*ma2.*exp(-sqrt(5.*dist));
                D2 = s.^2.*ma2.*exp(-sqrt(5.*dist2));
                D1 = (5./(3.*sqrt(s)) + 5.*sqrt(5)/3.*sqrt(dist)).*D1.*dist;
                D2 = (5./(3.*sqrt(s)) + 5.*sqrt(5)/3.*sqrt(dist2)).*D2.*dist2;
                
                DKuf_l = D1(:);      % Matrix of size uf x m
                DKuu_l = D2(:);      % Matrix of size uu x m
            else
                % In the case ARD is used
                s = 1./gpcf.lengthScale.^2;
                ma2 = gpcf.magnSigma2;
                dist = 0; dist2 = 0;
                for i=1:m
                    dist = dist + s(i).*(gminus(u(:,i),x(:,i)')).^2;
                    dist2 = dist2 + s(i).*(gminus(u(:,i),u(:,i)')).^2;
                end
                for i=1:m
                    D1 = ma2.*exp(-sqrt(5.*dist)).*s(i).^(3/2).*(gminus(u(:,i),x(:,i)')).^2;;
                    D2 = ma2.*exp(-sqrt(5.*dist2)).*s(i).^(3/2).*(gminus(u(:,i),u(:,i)')).^2;;
                    D1 = (5./3 + 5.*sqrt(5.*dist)/3).*D1;
                    D2 = (5./3 + 5.*sqrt(5.*dist2)/3).*D2;
                    
                    DKuf_l(:,i) = D1(:);      % Matrix of size uf x m
                    DKuu_l(:,i) = D2(:);      % Matrix of size uu x m
                end     
            end
          case 'PIC_BLOCK'
            % Evaluate the help matrices for the gradient evaluation (see
            % gpcf_sexp_trcov)
            
            L = varargin{1};             % f x u
            b = varargin{2};             % 1 x f
            iKuuKuf = varargin{3};       % u x f
            Labl = varargin{4};          % array of size
            
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
                s = 1./gpcf.lengthScale.^2;
                ma2 = gpcf.magnSigma2;
                dist = 0; dist2 = 0;
                for j=1:length(ind)
                    dist3{j} = zeros(size(ind{j},1),size(ind{j},1));
                end
                for i=1:m
                    dist = dist + s.*(gminus(u(:,i),x(:,i)')).^2;
                    dist2 = dist2 + s.*(gminus(u(:,i),u(:,i)')).^2;
                    for j=1:length(ind)
                        dist3{j} = dist3{j} + s.*(gminus(x(ind{j},i),x(ind{j},i)')).^2;
                    end
                end

                D1 = ma2.*exp(-sqrt(5.*dist));
                D2 = ma2.*exp(-sqrt(5.*dist2));
                DKuf_l = (5./3 + 5.*sqrt(5.*dist)/3) .*D1.*dist;
                DKuu_l = (5./3 + 5.*sqrt(5.*dist2)/3) .*D2.*dist2;
                
                for j=1:length(ind)
                    D3 = ma2.*exp(-sqrt(5.*dist3{j}));
                    DKff_l{j} = (5./3 + 5.*sqrt(5.*dist3{j})/3).*D3.*dist3{j};
                end
            else
                % In the case ARD is used
                s = 1./gpcf.lengthScale.^2;
                ma2 = gpcf.magnSigma2;
                dist = 0; dist2 = 0;
                for i=1:m
                    dist = dist + s(i).*(gminus(u(:,i),x(:,i)')).^2;
                    dist2 = dist2 + s(i).*(gminus(u(:,i),u(:,i)')).^2;
                end
                for i=1:m
                    D1 = ma2.*exp(-sqrt(5.*dist)).*s(i).^(3/2).*(gminus(u(:,i),x(:,i)')).^2;;
                    D2 = ma2.*exp(-sqrt(5.*dist2)).*s(i).^(3/2).*(gminus(u(:,i),u(:,i)')).^2;;
                    D1 = (5./3 + 5.*sqrt(5.*dist)/3).*D1;
                    D2 = (5./3 + 5.*sqrt(5.*dist2)/3).*D2;
                    
                    DKuf_l(:,i) = D1(:);      % Matrix of size uf x m
                    DKuu_l(:,i) = D2(:);      % Matrix of size uu x m
                end     
            end
          case 'PIC_BAND'
            % Evaluate the help matrices for the gradient evaluation (see
            % gpcf_sexp_trcov)
            
            L = varargin{1};             % f x u
            b = varargin{2};             % 1 x f
            iKuuKuf = varargin{3};       % u x f
            La = varargin{4};            % matrix of size
            
            u = gpcf.X_u;
            ind=gpcf.tr_index;
            nzmax = size(ind,1);
            
            % Derivatives of K_uu and K_uf with respect to magnitude sigma and lengthscale
            % NOTE! Here we have already taken into account that the parameters are transformed 
            % through log() and thus dK/dlog(p) = p * dK/dp
            K_uu = feval(gpcf.fh_trcov, gpcf, u);
            K_uf = feval(gpcf.fh_cov, gpcf, u, x);
            di2 = 0;
            s = 1./gpcf.lengthScale.^2;
            for i = 1:m
                di2 = di2 + s.*(x(ind(:,1),i) - x(ind(:,2),i)).^2;
            end
            di = sqrt(5.*di2);
            kv_ff = gpcf.magnSigma2.*(1 + di + 5.*di2./3).*exp(-di);
% $$$             kv_ff = zeros(nzmax,1);
% $$$             for i = 1:size(ind,1)
% $$$                 kv_ff(i) = feval(gpcf.fh_cov, gpcf, x(ind(i,1),:), x(ind(i,2),:));
% $$$             end
            K_ff = sparse(ind(:,1),ind(:,2),kv_ff,n,n);
            
            % Evaluate help matrix for calculations of derivatives with respect to the lengthScale
            if length(gpcf.lengthScale) == 1
                s = 1./gpcf.lengthScale.^2;
                ma2 = gpcf.magnSigma2;
                dist = 0; dist2 = 0;
                dist3 = zeros(nzmax,1);
                for i=1:m
                    dist = dist + s.*(gminus(u(:,i),x(:,i)')).^2;
                    dist2 = dist2 + s.*(gminus(u(:,i),u(:,i)')).^2;
                    dist3 = dist3 + (x(ind(:,1),i)-x(ind(:,2),i)).^2;
                end

                D1 = ma2.*exp(-sqrt(5.*dist));
                D2 = ma2.*exp(-sqrt(5.*dist2));
                DKuf_l = (5./3 + 5.*sqrt(5.*dist)/3) .*D1.*dist;
                DKuu_l = (5./3 + 5.*sqrt(5.*dist2)/3) .*D2.*dist2;
                D3 = ma2.*exp(-sqrt(5.*dist3));
                DKff_l = sparse(ind(:,1),ind(:,2), (5./3 + 5.*sqrt(5.*dist3)/3).*D3.*dist3 ,n,n);
            else
                % In the case ARD is used
                s = 1./gpcf.lengthScale.^2;
                ma2 = gpcf.magnSigma2;
                dist = 0; dist2 = 0;
                for i=1:m
                    dist = dist + s(i).*(gminus(u(:,i),x(:,i)')).^2;
                    dist2 = dist2 + s(i).*(gminus(u(:,i),u(:,i)')).^2;
                end
                for i=1:m
                    D1 = ma2.*exp(-sqrt(5.*dist)).*s(i).^(3/2).*(gminus(u(:,i),x(:,i)')).^2;;
                    D2 = ma2.*exp(-sqrt(5.*dist2)).*s(i).^(3/2).*(gminus(u(:,i),u(:,i)')).^2;;
                    D1 = (5./3 + 5.*sqrt(5.*dist)/3).*D1;
                    D2 = (5./3 + 5.*sqrt(5.*dist2)/3).*D2;
                    
                    DKuf_l(:,i) = D1(:);      % Matrix of size uf x m
                    DKuu_l(:,i) = D2(:);      % Matrix of size uu x m
                end     
            end
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
            for i = 1:size(ind,1)
                H(i) = (2*K_uf(:,ind(i,1))'- KfuiKuuKuu(ind(i,1),:))*iKuuKuf(:,ind(i,2));
            end
            H = sparse(ind(:,1), ind(:,2), H, n,n);
            % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
            gdata(i1) = -0.5.*((2*b*K_uf'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*K_uf'*iKuuKuf))) - ...
                               sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))) - 2.*trace((La\K_uf')*iKuuKuf) + ...
                                   trace((La\KfuiKuuKuu)*iKuuKuf));
            gdata(i1) = gdata(i1) ...                             %   + trace(Labl{i}\H(ind{i},ind{i})) ...
                + 0.5.*(-(b(ind(:,1)).*kv_ff')*b(ind(:,2))' ...
                        + b*H*b' ...
                        + trace(La\(K_ff-H))...
                        - sum(sum(L'.*(L'*K_ff))) ...               %- trace(Labl{i}\H(ind{i},ind{i})) 
                        + sum(sum(L'.*(L'*H))));
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
                  case {'PIC_BLOCK', 'PIC_BAND'}
                    gdata(i1)= DE_Kuu(:)'*DKuu_l(:,i2) + DE_Kuf(:)'*DKuf_l(:,i2);
                    for i=1:length(ind)
                        gdata(i1) =  gdata(i1) + DE_Kff{i}(:)'*DKff_l{i}(:,i2);
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
                H=zeros(1,size(ind,1));
                for i = 1:size(ind,1)
                    H(i) = (2*DKuf_l(:,ind(i,1))'- KfuiKuuDKuu_l(ind(i,1),:))*iKuuKuf(:,ind(i,2));
                end
                H = sparse(ind(:,1), ind(:,2), H, n,n);
                % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
                gdata(i1) = -0.5.*((2*b*DKuf_l'-(b*KfuiKuuDKuu_l))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf_l'*iKuuKuf))) - ...
                                   sum(sum(L'.*((L'*KfuiKuuDKuu_l)*iKuuKuf))) - 2.*trace((La\DKuf_l')*iKuuKuf) + ...
                                   trace((La\KfuiKuuDKuu_l)*iKuuKuf));
                gdata(i1) = gdata(i1) ...                             %   + trace(Labl{i}\H(ind{i},ind{i})) ...
                    + 0.5.*(-(b*DKff_l')*b' ...
                            + b*H*b' ...
                            + trace(La\(DKff_l-H))...
                            - sum(sum(L'.*(L'*DKff_l))) ...               %- trace(Labl{i}\H(ind{i},ind{i})) 
                            + sum(sum(L'.*(L'*H))));                
            end            
            gprior(i1)=feval(gpp.lengthScale.fg, ...
                             gpcf.lengthScale, ...
                             gpp.lengthScale.a, 'x').*gpcf.lengthScale - 1;
        end
        
        g = gdata + gprior;
    end
    
    function [DKuu_u, DKuf_u]  = gpcf_matern52_gind(gpcf, x, t, varargin)
    %GPCF_MATERN52_GIND    Evaluate gradient of error for SE covariance function 
    %                      with respect to inducing inputs.
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
                
        % First check if sparse model is used
        switch gpcf.type
           case {'FIC', 'PIC_BLOCK', 'PIC_BAND'}
            % Evaluate the help matrices for the gradient evaluation (see
            % gpcf_sexp_trcov)

            u = gpcf.X_u;
            n_u = size(u,1);
                        
            ma2 = gpcf.magnSigma2;
            if length(gpcf.lengthScale) == 1
                % In the case of an isotropic EXP
                s = repmat(1./gpcf.lengthScale.^2, 1, m);
            else
                s = 1./gpcf.lengthScale.^2;
            end
            dist=0; dist2=0;
            for i2=1:nin
                dist = dist + s(i2).*(gminus(u(:,i2),x(:,i2)')).^2;
                dist2 = dist2 + s(i2).*(gminus(u(:,i2),u(:,i2)')).^2;
            end
            dist=sqrt(dist);  dist2=sqrt(dist2);
            for i=1:m
                for j = 1:size(u,1)
                    D1 = zeros(size(u,1),n);
                    D2 = zeros(size(u,1),size(u,1));
                    D1(j,:) = sqrt(s(i)).*gminus(u(j,i),x(:,i)');
                    D2(j,:) = sqrt(s(i)).*gminus(u(j,i),u(:,i)');
                    D2 = D2 + D2';
                    
                    D1 = ma2.*(10/3 - 5 - 5.*sqrt(5).*dist./3).*exp(-sqrt(5).*dist).*D1;
                    D2 = ma2.*(10/3 - 5 - 5.*sqrt(5).*dist2./3).*exp(-sqrt(5).*dist2).*D2;
                    
                    DKuf_u(:,j+(i-1)*n_u) = D1(:);         % Matrix of size uf x mu
                    DKuu_u(:,j+(i-1)*n_u) = D2(:);         % Matrix of size uu x mu
                end
            end
        end
    end


    function C = gpcf_matern52_cov(gpcf, x1, x2)
    % GP_MATERN52_COV     Evaluate covariance matrix between two input vectors. 
    %
    %         Description
    %         C = GP_MATERN52_COV(GP, TX, X) takes in covariance function of a Gaussian
    %         process GP and two matrixes TX and X that contain input vectors to 
    %         GP. Returns covariance matrix C. Every element ij of C contains  
    %         covariance between inputs i in TX and j in X.
    %
    %         For covariance function definition see manual or 
    %         Neal R. M. Regression and Classification Using Gaussian 
    %         Process Priors, Bayesian Statistics 6.
        
    % Copyright (c) 1998-2004 Aki Vehtari
    % Copyright (c) 2006      Jarno Vanhatalo
        
    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
        
    % $$$         if issame(cov_x1,x1) && issame(cov_x2,x2) && issame(cov_ls,gpcf.lengthScale) && issame(cov_ms,gpcf.magnSigma2)
    % $$$             C = cov_C;
    % $$$         else
        
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
            s2 = 1./gpcf.lengthScale.^2;
            % If ARD is not used make s a vector of 
            % equal elements 
            if size(s2)==1
                s2 = repmat(s2,1,m1);
            end
            dist2=zeros(n1,n2);
            for j=1:m1
                dist2 = dist2 + s2(:,j).*(gminus(x1(:,j),x2(:,j)')).^2;
            end
            dist = sqrt(5.*dist2);
            C = ma2.*(1 + dist + 5.*dist2./3).*exp(-dist);
        end
        C(C<eps)=0;
        cov_x1=x1;
        cov_x2=x2;
        cov_ls=gpcf.lengthScale;
        cov_ms=gpcf.magnSigma2;
        cov_C=C;
    % $$$         end
    end

    function C = gpcf_matern52_trcov(gpcf, x)
    % GP_MATERN52_TRCOV     Evaluate training covariance matrix of inputs. 
    %
    %         Description
    %         C = GP_MATERN52_TRCOV(GP, TX) takes in covariance function of a Gaussian
    %         process GP and matrix TX that contains training input vectors to 
    %         GP. Returns covariance matrix C. Every element ij of C contains  
    %         covariance between inputs i and j in TX 
    %
    %         For covariance function definition see manual or 
    %         Neal R. M. Regression and Classification Using Gaussian 
    %         Process Priors, Bayesian Statistics 6.
        
    % Copyright (c) 1998-2004 Aki Vehtari
    % Copyright (c) 2006      Jarno Vanhatalo
        
    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
        
    % $$$         if issame(trcov_x,x) && issame(trcov_ls,gpcf.lengthScale) && issame(trcov_ms,gpcf.magnSigma2)
    % $$$             C = trcov_C;
    % $$$         else
        [n, m] =size(x);
        
        s2 = 1./(gpcf.lengthScale).^2;
        if size(s2)==1
            s2 = repmat(s2,1,m);
        end
        ma2 = gpcf.magnSigma2;
        
        % Here we take advantage of the 
        % symmetry of covariance matrix
        C=zeros(n,n);
        for i1=2:n
            i1n=(i1-1)*n;
            for i2=1:i1-1
                ii=i1+(i2-1)*n;
                for i3=1:m
                    C(ii)=C(ii)+s2(i3).*(x(i1,i3)-x(i2,i3)).^2;       % the covariance function
                end
                C(i1n+i2)=C(ii); 
            end
        end
        dist = sqrt(5.*C);
        C = ma2.*(1 + dist + 5.*C./3).*exp(-dist);
        C(C<eps)=0;
        
        trcov_x=x;
        trcov_ls=gpcf.lengthScale;
        trcov_ms=gpcf.magnSigma2;
        trcov_C=C;
    % $$$         end
    end

    function C = gpcf_matern52_trvar(gpcf, x)
    % GP_MATERN52_TRVAR     Evaluate training variance vector of inputs. 
    %
    %         Description
    %         C = GP_MATERN52_TRVAR(GP, TX) takes in covariance function of a Gaussian
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

    function reccf = gpcf_matern52_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_MATERN52_RECAPPEND(RECCF, RI, GPCF) takes old covariance 
    %          function record RECCF, record index RI, RECAPPEND returns a 
    %          structure RECCF containing following record fields:
    %          lengthHyper    = 
    %          lengthHyperNu  = 
    %          lengthScale    = 
    %          magnSigma2     = 
        
    % Initialize record
        if nargin == 2
            reccf.type = 'gpcf_matern52';
            reccf.nin = ri;
            gpcf.nout = 1;
            
            % Initialize parameters
            reccf.lengthScale= [];
            reccf.magnSigma2 = [];
            
            % Set the function handles
            reccf.fh_pak = @gpcf_matern52_pak;
            reccf.fh_unpak = @gpcf_matern52_unpak;
            reccf.fh_e = @gpcf_matern52_e;
            reccf.fh_g = @gpcf_matern52_g;
            reccf.fh_cov = @gpcf_matern52_cov;
            reccf.fh_trcov  = @gpcf_matern52_trcov;
            reccf.fh_trvar  = @gpcf_matern52_trvar;
            %  gpcf.fh_sampling = @hmc2;
            %  reccf.sampling_opt = hmc2_opt;
            reccf.fh_recappend = @gpcf_matern52_recappend;  
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

