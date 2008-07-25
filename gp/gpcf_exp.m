function gpcf = gpcf_exp(do, varargin)
%GPCF_EXP	Create an exponential covariance function for Gaussian Process
%
%	Description
%
%	GPCF = GPCF_EXP('INIT', NIN) Create and initialize exponential
%       covariance function for Gaussian process
%
%	The fields and (default values) in GPCF_EXP are:
%	  type           = 'gpcf_exp'
%	  nin            = Number of inputs. (NIN)
%	  nout           = Number of outputs. (always 1)
%	  magnSigma2     = Magnitude (squared) for exponential part. 
%                          (0.1)
%	  lengthScale    = Length scale for each input. This can be either scalar corresponding 
%                          isotropic or vector corresponding ARD. 
%                          (repmat(10, 1, nin))
%         p              = Prior structure for covariance function parameters. 
%                          (e.g. p.lengthScale.)
%         fh_pak         = function handle to pack function
%                          (@gpcf_sexp_pak)
%         fh_unpak       = function handle to unpack function
%                          (@gpcf_sexp_unpak)
%         fh_e           = function handle to energy function
%                          (@gpcf_sexp_e)
%         fh_ghyper      = function handle to gradient of energy with respect to hyperparameters
%                          (@gpcf_sexp_ghyper)
%         fh_gind        = function handle to gradient of function with respect to inducing inputs
%                          (@gpcf_sexp_gind)
%         fh_cov         = function handle to covariance function
%                          (@gpcf_sexp_cov)
%         fh_trcov       = function handle to training covariance function
%                          (@gpcf_sexp_trcov)
%         fh_trvar       = function handle to training variance function
%                          (@gpcf_sexp_trvar)
%         fh_recappend   = function handle to append the record function 
%                          (gpcf_sexp_recappend)
%
%	GPCF = GPCF_EXP('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in GPCF.
%
%	See also
%       gpcf_sexp, gpcf_matern32, gpcf_matern52, gpcf_ppcs2, gp_init, gp_e, gp_g, gp_trcov
%       gp_cov, gp_unpak, gp_pak
    
% Copyright (c) 2000-2001 Aki Vehtari
% Copyright (c) 2006      Helsinkin University of Technology (author Jarno Vanhatalo)
% Copyright (c) 2007-2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end

    % Initialize the covariance function
    if strcmp(do, 'init')
        nin = varargin{1};
        gpcf.type = 'gpcf_exp';
        gpcf.nin = nin;
        gpcf.nout = 1;
        
        % Initialize parameters
        gpcf.lengthScale= repmat(10, 1, nin); 
        gpcf.magnSigma2 = 0.1;
        
        % Initialize prior structure
        gpcf.p=[];
        gpcf.p.lengthScale=[];
        gpcf.p.magnSigma2=[];
        
        % Set the function handles
        gpcf.fh_pak = @gpcf_exp_pak;
        gpcf.fh_unpak = @gpcf_exp_unpak;
        gpcf.fh_e = @gpcf_exp_e;
        gpcf.fh_ghyper = @gpcf_exp_ghyper;
        gpcf.fh_gind = @gpcf_exp_gind;
        gpcf.fh_cov = @gpcf_exp_cov;
        gpcf.fh_covvec = @gpcf_exp_covvec;
        gpcf.fh_trcov  = @gpcf_exp_trcov;
        gpcf.fh_trvar  = @gpcf_exp_trvar;
        gpcf.fh_recappend = @gpcf_exp_recappend;

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

    function w = gpcf_exp_pak(gpcf, w)
    %GPCF_EXP_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %	W = GPCF_EXP_PAK(GPCF, W) takes a covariance function data structure GPCF and
    %	combines the hyper-parameters into a single row vector W.
    %
    %	The ordering of the parameters in W is:
    %       w = [gpcf.magnSigma2 (hyperparameters of gpcf.lengthScale) gpcf.lengthScale]
    %	  
    %
    %	See also
    %	GPCF_EXP_UNPAK
        
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


    function [gpcf, w] = gpcf_exp_unpak(gpcf, w)
    %GPCF_EXP_UNPAK  Separate covariance function hyper-parameter vector into components.
    %
    %	Description
    %	[GPCF, W] = GPCF_EXP_UNPAK(GPCF, W) takes a covariance function data structure GPCF
    %	and  a hyper-parameter vector W, and returns a covariance function data
    %	structure  identical to the input, except that the covariance hyper-parameters 
    %   has been set to the values in W. Deletes the values set to GPCF from W and returns 
    %   the modeified W. 
    %
    %	See also
    %	GPCF_EXP_PAK
    
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


    function eprior =gpcf_exp_e(gpcf, x, t)
    %GPCF_EXP_E     Evaluate the energy of prior of EXP parameters
    %
    %	Description
    %	E = GPCF_EXP_E(GPCF, X, T) takes a covariance function data structure 
    %   GPCF together with a matrix X of input vectors and a matrix T of target 
    %   vectors and evaluates log p(th) x J, where th is a vector of SEXP parameters 
    %   and J is the Jakobian of transformation exp(w) = th. (Note that the parameters 
    %   are log transformed, when packed.)
    %
    %	See also
    %	GPCF_EXP_PAK, GPCF_EXP_UNPAK, GPCF_EXP_G, GP_E
        
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
    end


    function [gprior, DKff, DKuu, DKuf]  = gpcf_exp_ghyper(gpcf, x, t, g, gdata, gprior, varargin)
    %GPCF_EXP_GHYPER     Evaluate gradient of covariance function and hyper-prior with 
    %                    respect to the hyperparameters.
    %
    %	Descriptioni
    %	[GPRIOR, DKff, DKuu, DKuf] = GPCF_EXP_GHYPER(GPCF, X, T, G, GDATA, GPRIOR, VARARGIN) 
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
    %   GPCF_EXP_PAK, GPCF_EXP_UNPAK, GPCF_EXP_E, GP_G
        
        gpp=gpcf.p;
        [n, m] =size(x);

        i1=0;i2=1;
        if ~isempty(gprior)
            i1 = length(gprior);
        end

        % First check if sparse model is used
        switch gpcf.GPtype
          case 'FULL'
            % Evaluate: DKff{1} = d Kff / d magnSigma2
            %           DKff{2} = d Kff / d lengthScale
            
            Cdm = gpcf_exp_trcov(gpcf, x);
            ii1=1;
            DKff{ii1} = Cdm;
            
            % loop over all the lengthScales
            if length(gpcf.lengthScale) == 1
                % In the case of isotropic EXP (no ARD)
                s = 1./gpcf.lengthScale;
                dist = 0;
                for i=1:nin
                    dist = dist + (gminus(x(:,i),x(:,i)')).^2;
                end
                D = Cdm.*s.*sqrt(dist);
                ii1 = ii1+1;
                DKff{ii1} = D;
            else
                % In the case ARD is used
                s = 1./gpcf.lengthScale.^2;
                dist = 0;
                dist2 = 0;
                for i=1:nin
                    dist = dist + s(i).*(gminus(x(:,i),x(:,i)')).^2;
                end
                dist = sqrt(dist);
                for i=1:nin                      
                    D = s(i).*Cdm.*(gminus(x(:,i),x(:,i)')).^2 ;
                    D(dist~=0) = D(dist~=0)./dist(dist~=0);
                    ii1 = ii1+1;
                    DKff{ii1} = D;
                end
            end
          case {'FIC' 'CS+FIC'}
            % Evaluate: DKff{1} = d mask(Kff,I) / d magnSigma2
            %           DKff{2} = d mask(Kff,I) / d lengthScale
            %           
            %           DKuu{1} = d Kuu / d magnSigma2
            %           DKuu{2} = d Kuu / d lengthScale
            %
            %           DKuf{1} = d Kuf / d magnSigma2
            %           DKuf{2} = d Kuf / d lengthScale
            %
            % NOTE! Here we have already taken into account that the parameters are transformed
            % through log() and thus dK/dlog(p) = p * dK/dp

            u = gpcf.X_u;                        
            K_uu = feval(gpcf.fh_trcov, gpcf, u);
            K_uf = feval(gpcf.fh_cov, gpcf, u, x);
            DKff = feval(gpcf.fh_trvar, gpcf, x);    % d mask(Kff,I) / d magnSigma2
            
            % Set d Kuu / d magnSigma2 and d Kuf / d magnSigma2
            ii1=1;
            DKuu{ii1} = K_uu;
            DKuf{ii1} = K_uf;
            
            % Evaluate help matrix for calculations of derivatives with respect to the lengthScale
            if length(gpcf.lengthScale) == 1
                % In the case of an isotropic EXP
                s = 1./gpcf.lengthScale;
                dist = 0;
                dist2 = 0;
                for i=1:m
                    dist = dist + (gminus(u(:,i),x(:,i)')).^2;
                    dist2 = dist2 + (gminus(u(:,i),u(:,i)')).^2;
                end
                DKuf_l = s.*K_uf.*sqrt(dist);
                DKuu_l = s.*K_uu.*sqrt(dist2);
                ii1=ii1+1;
                DKuu{ii1} = DKuu_l;
                DKuf{ii1} = DKuf_l;
            else
                % In the case ARD is used
                s = 1./gpcf.lengthScale.^2;        % set the length
                dist = 0; dist2 = 0;
                for i=1:nin
                    dist = dist + s(i).*(gminus(u(:,i),x(:,i)')).^2;
                    dist2 = dist2 + s(i).*(gminus(u(:,i),u(:,i)')).^2;
                end
                dist = sqrt(dist); dist2 = sqrt(dist2);
                for i=1:nin
                    D1 = s(i).*K_uf.* gminus(u(:,i),x(:,i)').^2;
                    D2 = s(i).*K_uu.* gminus(u(:,i),u(:,i)').^2;
                    D1(dist~=0) = D1(dist~=0)./dist(dist~=0);
                    D2(dist2~=0) = D2(dist2~=0)./dist2(dist2~=0);
                    DKuf_l = D1;      % Matrix of size uf x m
                    DKuu_l = D2;      % Matrix of size uu x m
                    ii1=ii1+1;
                    DKuu{ii1} = DKuu_l;
                    DKuf{ii1} = DKuf_l;
                end
            end
          case 'PIC_BLOCK'
            % Evaluate: DKff{1} = d mask(Kff,I) / d magnSigma2
            %           DKff{2} = d mask(Kff,I) / d lengthScale
            %           
            %           DKuu{1} = d Kuu / d magnSigma2
            %           DKuu{2} = d Kuu / d lengthScale
            %
            %           DKuf{1} = d Kuf / d magnSigma2
            %           DKuf{2} = d Kuf / d lengthScale
            %
            % NOTE! Here we have already taken into account that the parameters are transformed
            % through log() and thus dK/dlog(p) = p * dK/dp
            
            u = gpcf.X_u;
            ind=gpcf.tr_index;
            K_uu = feval(gpcf.fh_trcov, gpcf, u); 
            K_uf = feval(gpcf.fh_cov, gpcf, u, x);
            for i=1:length(ind)
                K_ff{i} = feval(gpcf.fh_trcov, gpcf, x(ind{i},:));
            end
            
            % Set d mask(Kff,I) / d magnSigma2, d Kuu / d magnSigma2 and d Kuf / d magnSigma2
            ii1=1;
            DKuu{ii1} = K_uu;
            DKuf{ii1} = K_uf;
            DKff{ii1} = K_ff;

            % Evaluate help matrix for calculations of derivatives with respect to the lengthScale
            if length(gpcf.lengthScale) == 1
                % In the case of an isotropic EXP
                s = 1./gpcf.lengthScale;
                dist = 0;
                dist2 = 0;
                for j=1:length(ind)
                    dist3{j} = zeros(size(ind{j},1),size(ind{j},1));
                end
                for i=1:m
                    dist = dist + (gminus(u(:,i),x(:,i)')).^2;
                    dist2 = dist2 + (gminus(u(:,i),u(:,i)')).^2;
                    for j=1:length(ind)
                        dist3{j} = dist3{j} + (gminus(x(ind{j},i),x(ind{j},i)')).^2;
                    end
                end
                DKuf_l = s.*K_uf.*sqrt(dist);
                DKuu_l = s.*K_uu.*sqrt(dist2);
                for j=1:length(ind)
                    DKff_l{j} = s.*K_ff{j}.*sqrt(dist3{j});
                end
                ii1=ii1+1;
                DKuu{ii1} = DKuu_l;
                DKuf{ii1} = DKuf_l;
                DKff{ii1} = DKff_l;
            else
                % In the case ARD is used
                s = 1./gpcf.lengthScale.^2;        % set the length
                dist = 0; dist2 = 0;

                for j=1:length(ind)
                    dist3{j} = zeros(size(ind{j},1),size(ind{j},1));
                end
                
                for i=1:m
                    dist = dist + s(i).*(gminus(u(:,i),x(:,i)')).^2;
                    dist2 = dist2 + s(i).*(gminus(u(:,i),u(:,i)')).^2;
                    for j=1:length(ind)
                        dist3{j} = dist3{j} + s(i).*(gminus(x(ind{j},i),x(ind{j},i)')).^2;
                    end
                end
                dist = sqrt(dist); dist2 = sqrt(dist2);
                for j=1:length(ind)
                    dist3{j} = sqrt(dist3{j});
                end
                for i=1:m
                    D1 = s(i).*K_uf.*(gminus(u(:,i),x(:,i)')).^2;
                    D2 = s(i).*K_uu.*(gminus(u(:,i),u(:,i)')).^2;
                    D1(dist~=0) = D1(dist~=0)./dist(dist~=0);
                    D2(dist2~=0) = D2(dist2~=0)./dist2(dist2~=0);
                    DKuf_l = D1;       
                    DKuu_l = D2;       
                    for j=1:length(ind)
                        D3 = s(i).*K_ff{j}.*(gminus(x(ind{j},i),x(ind{j},i)')).^2;
                        D3(dist3{j}~=0) = D3(dist3{j}~=0)./dist3{j}(dist3{j}~=0);
                        DKff_l{j} = D3;
                    end
                    ii1=ii1+1;
                    DKuu{ii1} = DKuu_l;
                    DKuf{ii1} = DKuf_l;
                    DKff{ii1} = DKff_l;
                end
            end
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

    function [gprior_ind, DKuu, DKuf]  = gpcf_exp_gind(gpcf, x, t, g_ind, gdata_ind, gprior_ind, varargin)
    %GPCF_EXP_GIND     Evaluate gradient of covariance function with 
    %                  respect to the inducing inputs.
    %
    %	Descriptioni
    %	[GPRIOR_IND, DKuu, DKuf] = GPCF_EXP_GIND(GPCF, X, T, G, GDATA_IND, GPRIOR_IND, VARARGIN) 
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
    %   GPCF_EXP_PAK, GPCF_EXP_UNPAK, GPCF_EXP_E, GP_G
        
        gpp=gpcf.p;
        [n, m] =size(x);
        u = gpcf.X_u;
        n_u = size(u,1);
        
        % First check if sparse model is used
        switch gpcf.GPtype
          case 'FIC'
            % Derivatives of K_uu and K_uf with respect to inducing inputs
            K_uu = feval(gpcf.fh_trcov, gpcf, u);
            K_uf = feval(gpcf.fh_cov, gpcf, u, x);
            
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
            dist = sqrt(dist); dist2 = sqrt(dist2);
            ii1 = 0;
            for i=1:m
                for j = 1:size(u,1)
                    D1 = zeros(size(u,1),n);
                    D2 = zeros(size(K_uu));
                    D1(j,:) = -s(i).*gminus(u(j,i),x(:,i)');
                    D2(j,:) = -s(i).* gminus(u(j,i),u(:,i)');
                    D2 = D2 + D2';
                    
                    D1(dist~=0) = D1(dist~=0)./dist(dist~=0);
                    D2(dist2~=0) = D2(dist2~=0)./dist2(dist2~=0);
                    
                    DKuf_u = D1.*K_uf;
                    DKuu_u = D2.*K_uu;

                    ii1 = ii1 + 1;
                    DKuf{ii1} = DKuf_u;
                    DKuu{ii1} = DKuu_u;                
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
            
            dist=0; dist2=0;
            for i2=1:nin
                dist = dist + s(i2).*(gminus(u(:,i2),x(:,i2)')).^2;
                dist2 = dist2 + s(i2).*(gminus(u(:,i2),u(:,i2)')).^2;
            end
            dist = sqrt(dist); dist2 = sqrt(dist2);
            ii1 = 0;
            for i=1:m
                for j = 1:size(u,1)               
                    D1 = zeros(size(u,1),n);
                    D2 = zeros(size(K_uu));
                    D1(j,:) = -s(i).*gminus(u(j,i),x(:,i)');
                    D2(j,:) = -s(i).* gminus(u(j,i),u(:,i)');
                    D2 = D2 + D2';
                    
                    D1(dist~=0) = D1(dist~=0)./dist(dist~=0);
                    D2(dist2~=0) = D2(dist2~=0)./dist2(dist2~=0);
                    
                    DKuf_u = D1.*K_uf;
                    DKuu_u = D2.*K_uu;
                    
                    ii1 = ii1 + 1;
                    DKuf{ii1} = DKuf_u;
                    DKuu{ii1} = DKuu_u;
                end
            end
          case 'CS+FIC'
            % Derivatives of K_uu and K_uf with respect to inducing inputs
            K_uu = feval(gpcf.fh_trcov, gpcf, u);
            K_uf = feval(gpcf.fh_cov, gpcf, u, x);
            
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
            dist = sqrt(dist); dist2 = sqrt(dist2);
            ii1 = 0;
            for i=1:m
                for j = 1:size(u,1)
                    D1 = zeros(size(u,1),n);
                    D2 = zeros(size(K_uu));
                    D1(j,:) = -s(i).*gminus(u(j,i),x(:,i)');
                    D2(j,:) = -s(i).* gminus(u(j,i),u(:,i)');
                    D2 = D2 + D2';
                    
                    D1(dist~=0) = D1(dist~=0)./dist(dist~=0);
                    D2(dist2~=0) = D2(dist2~=0)./dist2(dist2~=0);
                    
                    DKuf_u = D1.*K_uf;
                    DKuu_u = D2.*K_uu;
                    
                    ii1 = ii1 + 1;
                    DKuf{ii1} = DKuf_u;
                    DKuu{ii1} = DKuu_u;
                end
            end
        end
    end
    
    
    function C = gpcf_exp_cov(gpcf, x1, x2)
    % GP_EXP_COV     Evaluate covariance matrix between two input vectors.
    %
    %         Description
    %         C = GP_EXP_COV(GP, TX, X) takes in covariance function of a Gaussian
    %         process GP and two matrixes TX and X that contain input vectors to
    %         GP. Returns covariance matrix C. Every element ij of C contains
    %         covariance between inputs i in TX and j in X.
    %
    %
    %         See also
    %         GPCF_EXP_TRCOV, GPCF_EXP_TRVAR, GP_COV, GP_TRCOV
        
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
            dist=zeros(n1,n2);
            for j=1:m1
                dist = dist + s2(j).*(gminus(x1(:,j),x2(:,j)')).^2;
            end
            C = ma2.*exp(-sqrt(dist));
        end
    end


    function C = gpcf_exp_trcov(gpcf, x)
    % GP_EXP_TRCOV     Evaluate training covariance matrix of inputs.
    %
    %         Description
    %         C = GP_EXP_TRCOV(GP, TX) takes in covariance function of a Gaussian
    %         process GP and matrix TX that contains training input vectors. 
    %         Returns covariance matrix C. Every element ij of C contains covariance 
    %         between inputs i and j in TX
    %
    %
    %         See also
    %         GPCF_EXP_COV, GPCF_EXP_TRVAR, GP_COV, GP_TRCOV

    C = trcov(gpcf, x);
% $$$         [n, m] =size(x);
% $$$ 
% $$$         s = 1./(gpcf.lengthScale);
% $$$         s2 = s.^2;
% $$$         if size(s)==1
% $$$             s2 = repmat(s2,1,m);
% $$$         end
% $$$         ma = gpcf.magnSigma2;
% $$$ 
% $$$         % Here we take advantage of the 
% $$$         % symmetry of covariance matrix
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
% $$$         C = ma.*exp(-sqrt(C));
% $$$         C(C<eps)=0;
    end

    function C = gpcf_exp_covvec(gpcf, x1, x2, varargin)
    % GPCF_EXP_COVVEC     Evaluate covariance vector between two input vectors.
    %
    %         Description
    %         C = GPCF_EXP_COVVEC(GP, TX, X) takes in Gaussian process GP and two
    %         matrixes TX and X that contain input vectors to GP. Returns
    %         covariance vector C, where every element i of C contains covariance
    %         between input i in TX and i in X.
    %
    %
    %         See also
    %         GPCF_EXP_COV, GPCF_EXP_TRVAR, GP_COV, GP_TRCOV
        
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
        C = gpcf.magnSigma2.*exp(-sqrt(di2));
    end
    
    
    function C = gpcf_exp_trvar(gpcf, x)
    % GP_EXP_TRVAR     Evaluate training variance vector
    %
    %         Description
    %         C = GP_EXP_TRVAR(GPCF, TX) takes in covariance function of a Gaussian
    %         process GPCF and matrix TX that contains training inputs. Returns variance 
    %         vector C. Every element i of C contains variance of input i in TX
    %
    %
    %         See also
    %         GPCF_EXP_COV, GPCF_EXP_COVVEC, GP_COV, GP_TRCOV

        [n, m] =size(x);

        C = ones(n,1).*gpcf.magnSigma2;
        C(C<eps)=0;
    end

    function reccf = gpcf_exp_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_EXP_RECAPPEND(RECCF, RI, GPCF) takes old covariance
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
            reccf.type = 'gpcf_exp';
            reccf.nin = ri;
            gpcf.nout = 1;
            
            % Initialize parameters
            reccf.lengthScale= [];
            reccf.magnSigma2 = [];
            
            % Set the function handles
            reccf.fh_pak = @gpcf_exp_pak;
            reccf.fh_unpak = @gpcf_exp_unpak;
            reccf.fh_e = @gpcf_exp_e;
            reccf.fh_g = @gpcf_exp_g;
            reccf.fh_cov = @gpcf_exp_cov;
            reccf.fh_trcov  = @gpcf_exp_trcov;
            reccf.fh_trvar  = @gpcf_exp_trvar;
            %  gpcf.fh_sampling = @hmc2;
            %  reccf.sampling_opt = hmc2_opt;
            reccf.fh_recappend = @gpcf_exp_recappend;  
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