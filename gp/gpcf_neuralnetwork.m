function gpcf = gpcf_neuralnetwork(do, varargin)
%GPCF_NEURALNETWORK	Create a neural network covariance function for Gaussian Process
%
%	Description
%
%	GPCF = GPCF_NEURALNETWORK('INIT', NIN) Create and initialize neural
%       network covariance function for Gaussian process
%
%	The fields and (default values) in GPCF_NEURALNETWORK are:
%	  type           = 'gpcf_neuralnetwork'
%	  nin            = Number of inputs. (NIN)
%	  nout           = Number of outputs. (always 1)
%	  biasSigma2     = Prior variance on the network bias term. 
%                          (0.1)
%	  weightSigma2   = Prior variances on the network weights. This can be either scalar corresponding 
%                          isotropic or vector corresponding ARD. 
%                          (repmat(10, 1, nin))
%         p              = Prior structure for covariance function parameters. 
%                          (e.g. p.weightSigma2.)
%         fh_pak         = function handle to pack function
%                          (@gpcf_neuralnetwork_pak)
%         fh_unpak       = function handle to unpack function
%                          (@gpcf_neuralnetwork_unpak)
%         fh_e           = function handle to energy function
%                          (@gpcf_neuralnetwork_e)
%         fh_ghyper      = function handle to gradient of energy with respect to hyperparameters
%                          (@gpcf_neuralnetwork_ghyper)
%         fh_ginput      = function handle to gradient of function with respect to inducing inputs
%                          (@gpcf_neuralnetwork_ginput)
%         fh_cov         = function handle to covariance function
%                          (@gpcf_neuralnetwork_cov)
%         fh_trcov       = function handle to training covariance function
%                          (@gpcf_neuralnetwork_trcov)
%         fh_trvar       = function handle to training variance function
%                          (@gpcf_neuralnetwork_trvar)
%         fh_recappend   = function handle to append the record function 
%                          (gpcf_neuralnetwork_recappend)
%
%	GPCF = GPCF_NEURALNETWORK('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in GPCF.
%
%	See also
%       gpcf_exp, gpcf_matern32, gpcf_matern52, gpcf_ppcs2, gp_init, gp_e, gp_g, gp_trcov
%       gp_cov, gp_unpak, gp_pak
    
% Copyright (c) 2000-2001 Aki Vehtari
% Copyright (c) 2007-2008 Jarno Vanhatalo
% Copyright (c) 2009 Jaakko Riihimaki

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end

    % Initialize the covariance function
    if strcmp(do, 'init')
        nin = varargin{1};
        gpcf.type = 'gpcf_neuralnetwork';
        gpcf.nin = nin;
        gpcf.nout = 1;

        % Initialize parameters
        gpcf.weightSigma2= repmat(10, 1, nin);
        gpcf.biasSigma2 = 0.1;

        % Initialize prior structure
        gpcf.p=[];
        gpcf.p.weightSigma2=[];
        gpcf.p.biasSigma2=[];

        % Set the function handles to the nested functions
        gpcf.fh_pak = @gpcf_neuralnetwork_pak;
        gpcf.fh_unpak = @gpcf_neuralnetwork_unpak;
        gpcf.fh_e = @gpcf_neuralnetwork_e;
        gpcf.fh_ghyper = @gpcf_neuralnetwork_ghyper;
        gpcf.fh_ginput = @gpcf_neuralnetwork_ginput;
        gpcf.fh_cov = @gpcf_neuralnetwork_cov;
        gpcf.fh_trcov  = @gpcf_neuralnetwork_trcov;
        gpcf.fh_trvar  = @gpcf_neuralnetwork_trvar;
        gpcf.fh_recappend = @gpcf_neuralnetwork_recappend;

        if length(varargin) > 1
            if mod(nargin,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=2:2:length(varargin)-1
                switch varargin{i}
                  case 'biasSigma2'
                    gpcf.biasSigma2 = varargin{i+1};
                  case 'weightSigma2'
                    gpcf.weightSigma2 = varargin{i+1};
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
              case 'biasSigma2'
                gpcf.biasSigma2 = varargin{i+1};
              case 'weightSigma2'
                gpcf.weightSigma2 = varargin{i+1};
              case 'fh_sampling'
                gpcf.fh_sampling = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end

    function w = gpcf_neuralnetwork_pak(gpcf, w)
    %GPCF_NEURALNETWORK_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %	W = GPCF_NEURALNETWORK_PAK(GPCF, W) takes a covariance function data structure GPCF and
    %	combines the hyper-parameters into a single row vector W.
    %
    %	The ordering of the parameters in W is:
    %       w = [gpcf.biasSigma2 (hyperparameters of gpcf.weightSigma2) gpcf.weightSigma2]
    %	  
    %
    %	See also
    %	GPCF_NEURALNETWORK_UNPAK
        gpp=gpcf.p;

        i1=0;i2=1;
        if ~isempty(w)
            i1 = length(w);
        end
        i1 = i1+1;
        w(i1) = gpcf.biasSigma2;
        i2=i1+length(gpcf.weightSigma2);
        i1=i1+1;
        w(i1:i2)=gpcf.weightSigma2;
        i1=i2;
        
        % Hyperparameters of weightSigma2
        if isfield(gpp.weightSigma2, 'p') && ~isempty(gpp.weightSigma2.p)
            i1=i1+1;
            w(i1)=gpp.weightSigma2.a.s;
            if any(strcmp(fieldnames(gpp.weightSigma2.p),'nu'))
                i1=i1+1;
                w(i1)=gpp.weightSigma2.a.nu;
            end
        end
    end




    function [gpcf, w] = gpcf_neuralnetwork_unpak(gpcf, w)
    %GPCF_NEURALNETWORK_UNPAK  Separate covariance function hyper-parameter vector into components.
    %
    %	Description
    %	[GPCF, W] = GPCF_NEURALNETWORK_UNPAK(GPCF, W) takes a covariance function data structure GPCF
    %	and  a hyper-parameter vector W, and returns a covariance function data
    %	structure  identical to the input, except that the covariance hyper-parameters 
    %   has been set to the values in W. Deletes the values set to GPCF from W and returns 
    %   the modeified W. 
    %
    %	See also
    %	GPCF_NEURALNETWORK_PAK
    %
        gpp=gpcf.p;
        i1=0;i2=1;
        i1=i1+1;
        gpcf.biasSigma2=w(i1);
        i2=i1+length(gpcf.weightSigma2);
        i1=i1+1;
        gpcf.weightSigma2=w(i1:i2);
        i1=i2;
        % Hyperparameters of weightSigma2
        if isfield(gpp.weightSigma2, 'p') && ~isempty(gpp.weightSigma2.p)
            i1=i1+1;
            gpcf.p.weightSigma2.a.s=w(i1);
            if any(strcmp(fieldnames(gpp.weightSigma2.p),'nu'))
                i1=i1+1;
                gpcf.p.weightSigma2.a.nu=w(i1);
            end
        end        
        w = w(i1+1:end);
    end

    function eprior =gpcf_neuralnetwork_e(gpcf, x, t)
    %GPCF_NEURALNETWORK_E     Evaluate the energy of prior of NEURALNETWORK parameters
    %
    %	Description
    %	E = GPCF_NEURALNETWORK_E(GPCF, X, T) takes a covariance function data structure 
    %   GPCF together with a matrix X of input vectors and a matrix T of target 
    %   vectors and evaluates log p(th) x J, where th is a vector of NEURALNETWORK parameters 
    %   and J is the Jakobian of transformation exp(w) = th. (Note that the parameters 
    %   are log transformed, when packed.)
    %
    %	See also
    %	GPCF_NEURALNETWORK_PAK, GPCF_NEURALNETWORK_UNPAK, GPCF_NEURALNETWORK_G, GP_E
    %
        [n, m] =size(x);

        % Evaluate the prior contribution to the error. The parameters that
        % are sampled are from space W = log(w) where w is all the "real" samples.
        % On the other hand errors are evaluated in the W-space so we need take
        % into account also the  Jakobian of transformation W -> w = exp(W).
        % See Gelman et.all., 2004, Bayesian data Analysis, second edition, p24.
        eprior = 0;
        gpp=gpcf.p;

        eprior=eprior...
               +feval(gpp.biasSigma2.fe, ...
                      gpcf.biasSigma2, gpp.biasSigma2.a)...
               -log(gpcf.biasSigma2);
        if isfield(gpp.weightSigma2, 'p') && ~isempty(gpp.weightSigma2.p)
            eprior=eprior...
                   +feval(gpp.weightSigma2.p.s.fe, ...
                          gpp.weightSigma2.a.s, gpp.weightSigma2.p.s.a)...
                   -log(gpp.weightSigma2.a.s);
            if any(strcmp(fieldnames(gpp.weightSigma2.p),'nu'))
                eprior=eprior...
                       +feval(gpp.p.weightSigma2.nu.fe, ...
                              gpp.weightSigma2.a.nu, gpp.weightSigma2.p.nu.a)...
                       -log(gpp.weightSigma2.a.nu);
            end
        end
        eprior=eprior...
               +feval(gpp.weightSigma2.fe, ...
                      gpcf.weightSigma2, gpp.weightSigma2.a)...
               -sum(log(gpcf.weightSigma2));

    end

    function [DKff, gprior]  = gpcf_neuralnetwork_ghyper(gpcf, x, x2, mask)  % , t, g, gdata, gprior, varargin
    %GPCF_NEURALNETWORK_GHYPER     Evaluate gradient of covariance function and hyper-prior with 
    %                     respect to the hyperparameters.
    %
    %	Descriptioni
    %	[GPRIOR, DKff, DKuu, DKuf] = GPCF_NEURALNETWORK_GHYPER(GPCF, X, T, G, GDATA, GPRIOR, VARARGIN) 
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
    %   GPCF_NEURALNETWORK_PAK, GPCF_NEURALNETWORK_UNPAK, GPCF_NEURALNETWORK_E, GP_G
        
        gpp=gpcf.p;
        [n, m] =size(x);
        
        i1=0;
        
        % Evaluate: DKff{1} = d Kff / d biasSigma2
        %           DKff{2} = d Kff / d weightSigma2
        % NOTE! Here we have already taken into account that the parameters are transformed
        % through log() and thus dK/dlog(p) = p * dK/dp
        
        % evaluate the gradient for training covariance
        if nargin == 2
            
            x_aug=[ones(size(x,1),1) x];
            
            if length(gpcf.weightSigma2) == 1
                % In the case of an isotropic NEURALNETWORK
                s = gpcf.weightSigma2*ones(1,m);
            else
                s = gpcf.weightSigma2;
            end
            
            S_nom=2*x_aug*diag([gpcf.biasSigma2 s])*x_aug';
            
            S_den_tmp=(2*sum(repmat([gpcf.biasSigma2 s], n, 1).*x_aug.^2,2)+1);
            S_den2=S_den_tmp*S_den_tmp';
            S_den=sqrt(S_den2);
            
            C_tmp=2/pi./sqrt(1-(S_nom./S_den).^2);
            % C(abs(C)<=eps) = 0;
            C_tmp = (C_tmp+C_tmp')./2;
            
            bnom_g=2*ones(n);
            bden_g=(0.5./S_den).*(bnom_g.*repmat(S_den_tmp',n,1)+repmat(S_den_tmp,1,n).*bnom_g);
            bg=gpcf.biasSigma2*C_tmp.*(bnom_g.*S_den-bden_g.*S_nom)./S_den2;

            DKff{1}=(bg+bg')/2;
            
            if length(gpcf.weightSigma2) == 1
                    wnom_g=2*x*x';
                    tmp_g=sum(2*x.^2,2);
                    wden_g=0.5./S_den.*(tmp_g*S_den_tmp'+S_den_tmp*tmp_g');
                    wg=s(1)*C_tmp.*(wnom_g.*S_den-wden_g.*S_nom)./S_den2;
                    
                    DKff{2}=(wg+wg')/2;
            else
                for d1=1:m
                    wnom_g=2*x(:,d1)*x(:,d1)';
                    tmp_g=2*x(:,d1).^2;
                    wden_g=0.5./S_den.*(tmp_g*S_den_tmp'+S_den_tmp*tmp_g');
                    wg=s(d1)*C_tmp.*(wnom_g.*S_den-wden_g.*S_nom)./S_den2;
                    
                    DKff{d1+1}=(wg+wg')/2;
                end
            end
            
        % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
        elseif nargin == 3
            
            if size(x,2) ~= size(x2,2)
                error('gpcf_neuralnetwork -> _ghyper: The number of columns in x and x2 has to be the same. ')
            end
            
            n2 =size(x2,1);
            
            x_aug=[ones(size(x,1),1) x];
            x_aug2=[ones(size(x2,1),1) x2];
            
            if length(gpcf.weightSigma2) == 1
                % In the case of an isotropic NEURALNETWORK
                s = gpcf.weightSigma2*ones(1,m);
            else
                s = gpcf.weightSigma2;
            end
            
            S_nom=2*x_aug*diag([gpcf.biasSigma2 s])*x_aug2';
            
            S_den_tmp1=(2*sum(repmat([gpcf.biasSigma2 s], n, 1).*x_aug.^2,2)+1);
            S_den_tmp2=(2*sum(repmat([gpcf.biasSigma2 s], n2, 1).*x_aug2.^2,2)+1);
            
            S_den2=S_den_tmp1*S_den_tmp2';
            S_den=sqrt(S_den2);
            
            C_tmp=2/pi./sqrt(1-(S_nom./S_den).^2);
            %C(abs(C)<=eps) = 0;
            
            bnom_g=2*ones(n, n2);            
            bden_g=(0.5./S_den).*(bnom_g.*repmat(S_den_tmp2',n,1)+repmat(S_den_tmp1,1,n2).*bnom_g);
            
            DKff{1}=gpcf.biasSigma2*C_tmp.*(bnom_g.*S_den-bden_g.*S_nom)./S_den2;
            
            if length(gpcf.weightSigma2) == 1
                    wnom_g=2*x*x2';
                    tmp_g1=sum(2*x.^2,2);
                    tmp_g2=sum(2*x2.^2,2);
                    wden_g=0.5./S_den.*(tmp_g1*S_den_tmp2'+S_den_tmp1*tmp_g2');

                    DKff{2}=s(1)*C_tmp.*(wnom_g.*S_den-wden_g.*S_nom)./S_den2;
            else
                for d1=1:m
                    wnom_g=2*x(:,d1)*x2(:,d1)';
                    tmp_g1=2*x(:,d1).^2;
                    tmp_g2=2*x2(:,d1).^2;
                    wden_g=0.5./S_den.*(tmp_g1*S_den_tmp2'+S_den_tmp1*tmp_g2');

                    DKff{d1+1}=s(d1)*C_tmp.*(wnom_g.*S_den-wden_g.*S_nom)./S_den2;
                end
            end

            % Evaluate: DKff{1}    = d mask(Kff,I) / d biasSigma2
            %           DKff{2...} = d mask(Kff,I) / d weightSigma2
        elseif nargin == 4
            
            x_aug=[ones(size(x,1),1) x];
            
            if length(gpcf.weightSigma2) == 1
                % In the case of an isotropic NEURALNETWORK
                s = gpcf.weightSigma2*ones(1,m);
            else
                s = gpcf.weightSigma2;
            end
            
            S_nom=2*sum(repmat([gpcf.biasSigma2 s],n,1).*x_aug.^2,2);
            
            S_den=(S_nom+1);
            S_den2=S_den.^2;
            
            C_tmp=2/pi./sqrt(1-(S_nom./S_den).^2);
            %C(abs(C)<=eps) = 0;
            
            bnom_g=2*ones(n,1);
            bden_g=(0.5./S_den).*(2*bnom_g.*S_den);
            
            DKff{1}=gpcf.biasSigma2*C_tmp.*(bnom_g.*S_den-bden_g.*S_nom)./S_den2;
            
            if length(gpcf.weightSigma2) == 1
                    wnom_g=sum(2*x.^2,2);
                    wden_g=0.5./S_den.*(2*wnom_g.*S_den);

                    DKff{2}=s(1)*C_tmp.*(wnom_g.*S_den-wden_g.*S_nom)./S_den2;
            else
                for d1=1:m
                    wnom_g=2*x(:,d1).^2;
                    wden_g=0.5./S_den.*(2*wnom_g.*S_den);

                    DKff{d1+1}=s(d1)*C_tmp.*(wnom_g.*S_den-wden_g.*S_nom)./S_den2;
                end
            end
        end
        if nargout > 1
            % Evaluate the gprior with respect to biasSigma2
            i1 = i1+1;
            gprior(i1)=feval(gpp.biasSigma2.fg, ...
                             gpcf.biasSigma2, ...
                             gpp.biasSigma2.a, 'x').*gpcf.biasSigma2 - 1;
            % Evaluate the data contribution of gradient with respect to weightSigma2
            if length(gpcf.weightSigma2)>1
                for i2=1:gpcf.nin
                    i1=i1+1;
                    gprior(i1)=feval(gpp.weightSigma2.fg, ...
                                     gpcf.weightSigma2(i2), ...
                                     gpp.weightSigma2.a, 'x').*gpcf.weightSigma2(i2) - 1;
                end
            else
                i1=i1+1;
                gprior(i1)=feval(gpp.weightSigma2.fg, ...
                                 gpcf.weightSigma2, ...
                                 gpp.weightSigma2.a, 'x').*gpcf.weightSigma2 -1;
            end
            % Evaluate the prior contribution of gradient with respect to weightSigma2.p.s (and weightSigma2.p.nu)
            if isfield(gpp.weightSigma2, 'p') && ~isempty(gpp.weightSigma2.p)
                i1=i1+1;
                gprior(i1)=...
                    feval(gpp.weightSigma2.p.s.fg, ...
                          gpp.weightSigma2.a.s,...
                          gpp.weightSigma2.p.s.a, 'x').*gpp.weightSigma2.a.s - 1 ...
                    +feval(gpp.weightSigma2.fg, ...
                           gpcf.weightSigma2, ...
                           gpp.weightSigma2.a, 's').*gpp.weightSigma2.a.s;
                if any(strcmp(fieldnames(gpp.weightSigma2.p),'nu'))
                    i1=i1+1;
                    gprior(i1)=...
                        feval(gpp.weightSigma2.p.nu.fg, ...
                              gpp.weightSigma2.a.nu,...
                              gpp.weightSigma2.p.nu.a, 'x').*gpp.weightSigma2.a.nu -1 ...
                        +feval(gpp.weightSigma2.fg, ...
                               gpcf.weightSigma2, ...
                               gpp.weightSigma2.a, 'nu').*gpp.weightSigma2.a.nu;
                end
            end
        end
    end


    function [DKff, gprior]  = gpcf_neuralnetwork_ginput(gpcf, x, x2)
    %GPCF_NEURALNETWORK_GIND     Evaluate gradient of covariance function with 
    %                   respect to x.
    %
    %	Descriptioni
    %	[GPRIOR_IND, DKuu, DKuf] = GPCF_NEURALNETWORK_GIND(GPCF, X, T, G, GDATA_IND, GPRIOR_IND, VARARGIN) 
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
    %   GPCF_NEURALNETWORK_PAK, GPCF_NEURALNETWORK_UNPAK, GPCF_NEURALNETWORK_E, GP_G
        
        [n, m] =size(x);
        
        if nargin == 2
            
            if length(gpcf.weightSigma2) == 1
                % In the case of an isotropic NEURALNETWORK
                s = gpcf.weightSigma2*ones(1,m);
            else
                s = gpcf.weightSigma2;
            end
            
            x_aug=[ones(size(x,1),1) x];
            
            S_nom=2*x_aug*diag([gpcf.biasSigma2 s])*x_aug';
            S_den_tmp=(2*sum(repmat([gpcf.biasSigma2 s], n, 1).*x_aug.^2,2)+1);
            
            S_den2=S_den_tmp*S_den_tmp';
            S_den=sqrt(S_den2);
            
            C_tmp=2/pi./sqrt(1-(S_nom./S_den).^2);
            %C(abs(C)<=eps) = 0;
            C_tmp = (C_tmp+C_tmp')./2;
            
            ii1=0;
            for d1=1:m
                for j=1:n
                    
                    DK = zeros(n);
                    DK(j,:)=s(d1)*x(:,d1)';
                    DK = DK + DK';
                    inom_g=2*DK;
                    
                    tmp_g=zeros(n);
                    tmp_g(j,:)=2*s(d1)*2*x(j,d1)*S_den_tmp';
                    tmp_g=tmp_g+tmp_g';
                    
                    iden_g=0.5./S_den.*(tmp_g);
                    
                    ii1=ii1+1;
                    DKff{ii1}=C_tmp.*(inom_g.*S_den-iden_g.*S_nom)./S_den2;
                    
                    gprior(ii1) = 0;
                end
            end
            
        elseif nargin == 3
            
            if length(gpcf.weightSigma2) == 1
                % In the case of an isotropic NEURALNETWORK
                s = gpcf.weightSigma2*ones(1,m);
            else
                s = gpcf.weightSigma2;
            end
            
            n2 =size(x2,1);
            
            x_aug=[ones(size(x,1),1) x];
            x_aug2=[ones(size(x2,1),1) x2];
            
            S_nom=2*x_aug*diag([gpcf.biasSigma2 s])*x_aug2';
            
            S_den_tmp1=(2*sum(repmat([gpcf.biasSigma2 s], n, 1).*x_aug.^2,2)+1);
            S_den_tmp2=(2*sum(repmat([gpcf.biasSigma2 s], n2, 1).*x_aug2.^2,2)+1);
            
            S_den2=S_den_tmp1*S_den_tmp2';
            S_den=sqrt(S_den2);
            
            C_tmp=2/pi./sqrt(1-(S_nom./S_den).^2);
            % C(abs(C)<=eps) = 0;
            
            ii1 = 0;
            for d1=1:m
                for j = 1:n
                    
                    DK = zeros(n, n2);
                    DK(j,:)=s(d1)*x2(:,d1)';
                    inom_g=2*DK;
                                        
                    tmp_g=zeros(n, n2);
                    tmp_g(j,:)=2*s(d1)*2*x(j,d1)*S_den_tmp2';
                    
                    iden_g=0.5./S_den.*(tmp_g);
                    
                    ii1=ii1+1;
                    DKff{ii1}=C_tmp.*(inom_g.*S_den-iden_g.*S_nom)./S_den2;

                    gprior(ii1) = 0;
                end
            end
        end
    end


    function C = gpcf_neuralnetwork_cov(gpcf, x1, x2, varargin)
    % GP_NEURALNETWORK_COV     Evaluate covariance matrix between two input vectors.
    %
    %         Description
    %         C = GP_NEURALNETWORK_COV(GP, TX, X) takes in covariance function of a Gaussian
    %         process GP and two matrixes TX and X that contain input vectors to
    %         GP. Returns covariance matrix C. Every element ij of C contains
    %         covariance between inputs i in TX and j in X.
    %
    %
    %         See also
    %         GPCF_NEURALNETWORK_TRCOV, GPCF_NEURALNETWORK_TRVAR, GP_COV, GP_TRCOV
        
        if isempty(x2)
            x2=x1;
        end
        
        [n1,m1]=size(x1);
        [n2,m2]=size(x2);
        
        if m1~=m2
            error('the number of columns of X1 and X2 has to be same')
        end
        
        x_aug1=[ones(n1,1) x1];
        x_aug2=[ones(n2,1) x2];
        
        if length(gpcf.weightSigma2) == 1
             % In the case of an isotropic NEURALNETWORK
             s = gpcf.weightSigma2*ones(1,m1);
        else
             s = gpcf.weightSigma2;
        end
        
        S_nom=2*x_aug1*diag([gpcf.biasSigma2 s])*x_aug2';
        
        S_den_tmp1=(2*sum(repmat([gpcf.biasSigma2 s], n1, 1).*x_aug1.^2,2)+1);
        S_den_tmp2=(2*sum(repmat([gpcf.biasSigma2 s], n2, 1).*x_aug2.^2,2)+1);
        S_den2=S_den_tmp1*S_den_tmp2';
        
        C=2/pi*asin(S_nom./sqrt(S_den2));
        
        C(abs(C)<=eps) = 0;
    end
    
    
    function C = gpcf_neuralnetwork_trcov(gpcf, x)
    % GP_NEURALNETWORK_TRCOV     Evaluate training covariance matrix of inputs.
    %
    %         Description
    %         C = GP_NEURALNETWORK_TRCOV(GP, TX) takes in covariance function of a Gaussian
    %         process GP and matrix TX that contains training input vectors. 
    %         Returns covariance matrix C. Every element ij of C contains covariance 
    %         between inputs i and j in TX
    %
    %
    %         See also
    %         GPCF_NEURALNETWORK_COV, GPCF_NEURALNETWORK_TRVAR, GP_COV, GP_TRCOV
        
        [n,m]=size(x);
        x_aug=[ones(n,1) x];
        
        if length(gpcf.weightSigma2) == 1
         	% In the case of an isotropic NEURALNETWORK
             s = gpcf.weightSigma2*ones(1,m);
        else
             s = gpcf.weightSigma2;
        end
        
        S_nom=2*x_aug*diag([gpcf.biasSigma2 s])*x_aug';
        
        S_den_tmp=(2*sum(repmat([gpcf.biasSigma2 s], n, 1).*x_aug.^2,2)+1);        
        S_den2=S_den_tmp*S_den_tmp';
        
        C=2/pi*asin(S_nom./sqrt(S_den2));
        
        C(abs(C)<=eps) = 0;
        C = (C+C')./2;
        
    end

    function C = gpcf_neuralnetwork_trvar(gpcf, x)
    % GP_NEURALNETWORK_TRVAR     Evaluate training variance vector
    %
    %         Description
    %         C = GP_NEURALNETWORK_TRVAR(GPCF, TX) takes in covariance function of a Gaussian
    %         process GPCF and matrix TX that contains training inputs. Returns variance 
    %         vector C. Every element i of C contains variance of input i in TX
    %
    %
    %         See also
    %         GPCF_NEURALNETWORK_COV, GPCF_NEURALNETWORK_COVVEC, GP_COV, GP_TRCOV
    
        [n,m]=size(x);
        x_aug=[ones(n,1) x];
        
        if length(gpcf.weightSigma2) == 1
        	% In the case of an isotropic NEURALNETWORK
            s = gpcf.weightSigma2*ones(1,m);
        else
            s = gpcf.weightSigma2;
        end

        s_tmp=sum(repmat([gpcf.biasSigma2 s], n, 1).*x_aug.^2,2);
        
        C=2/pi*asin(2*s_tmp./(1+2*s_tmp));
        C(C<eps)=0;
        
    end

    function reccf = gpcf_neuralnetwork_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_NEURALNETWORK_RECAPPEND(RECCF, RI, GPCF) takes old covariance
    %          function record RECCF, record index RI and covariance function structure. 
    %          Appends the parameters of GPCF to the RECCF in the ri'th place.
    %
    %          RECAPPEND returns a structure RECCF containing following record fields:
    %          lengthHyper    
    %          lengthHyperNu  
    %          weightSigma2    
    %          biasSigma2     
    %
    %          See also
    %          GP_MC and GP_MC -> RECAPPEND

    % Initialize record
        if nargin == 2
            reccf.type = 'gpcf_neuralnetwork';
            reccf.nin = ri;
            reccf.nout = 1;

            % Initialize parameters
            reccf.weightSigma2= [];
            reccf.biasSigma2 = [];

            % Set the function handles
            reccf.fh_pak = @gpcf_neuralnetwork_pak;
            reccf.fh_unpak = @gpcf_neuralnetwork_unpak;
            reccf.fh_e = @gpcf_neuralnetwork_e;
            reccf.fh_g = @gpcf_neuralnetwork_g;
            reccf.fh_cov = @gpcf_neuralnetwork_cov;
            reccf.fh_trcov  = @gpcf_neuralnetwork_trcov;
            reccf.fh_trvar  = @gpcf_neuralnetwork_trvar;
            reccf.fh_recappend = @gpcf_neuralnetwork_recappend;
            reccf.p=[];
            reccf.p.weightSigma2=[];
            reccf.p.biasSigma2=[];
            if ~isempty(ri.p.weightSigma2)
                reccf.p.weightSigma2 = ri.p.weightSigma2;
            end
            if ~isempty(ri.p.biasSigma2)
                reccf.p.biasSigma2 = ri.p.biasSigma2;
            end
            
            return
        end

        gpp = gpcf.p;
        % record weightSigma2
        if ~isempty(gpcf.weightSigma2)
            if ~isempty(gpp.weightSigma2)
                reccf.lengthHyper(ri,:)=gpp.weightSigma2.a.s;
                if isfield(gpp.weightSigma2,'p')
                    if isfield(gpp.weightSigma2.p,'nu')
                        reccf.lengthHyperNu(ri,:)=gpp.weightSigma2.a.nu;
                    end
                end
            end
            reccf.weightSigma2(ri,:)=gpcf.weightSigma2;
        elseif ri==1
            reccf.weightSigma2=[];
        end
        % record biasSigma2
        if ~isempty(gpcf.biasSigma2)
            reccf.biasSigma2(ri,:)=gpcf.biasSigma2;
        elseif ri==1
            reccf.biasSigma2=[];
        end
    end
end