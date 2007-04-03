function gpcf = gpcf_exp(do, varargin)
%GPCF_EXP	Create a squared exponential covariance function for Gaussian Process.
%
%	Description
%
%	GPCF = GPCF_EXP('INIT', NIN) Create and initialize squared exponential 
%       covariance function fo Gaussian process 
%
%	The fields and (default values) in GPCF_EXP are:
%	  type           = 'gpcf_exp'
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
%                          (@gpcf_exp_pak)
%         fh_unpak       = function handle to unpackin function
%                          (@gpcf_exp_unpak)
%         fh_e           = function handle to error function
%                          (@gpcf_exp_e)
%         fh_g           = function handle to gradient function
%                          (@gpcf_exp_g)
%         fh_cov         = function handle to covariance function
%                          (@gpcf_exp_cov)
%         fh_trcov       = function handle to training covariance function
%                          (@gpcf_exp_trcov)
%         fh_trvar       = function handle to training variance function
%                          (@gpcf_exp_trvar)
%         fh_sampling    = function handle to parameter sampling function
%                          (@hmc2)
%         sampling_opt   = options structure for fh_sampling
%                          (hmc2_opt)
%         fh_recappend   = function handle to record append function
%                          (gpcf_exp_recappend)
%
%	GPCF = GPCF_EXP('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
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
    gpcf.fh_g = @gpcf_exp_g;
    gpcf.fh_cov = @gpcf_exp_cov;
    gpcf.fh_trcov  = @gpcf_exp_trcov;
    gpcf.fh_trvar  = @gpcf_exp_trvar;
    %  gpcf.fh_sampling = @hmc2;
    %  gpcf.sampling_opt = hmc2_opt;
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
    %GPcf_EXP_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %	W = GP_EXP_PAK(GPCF, W) takes a Gaussian Process data structure GP and
    %	combines the hyper-parameters into a single row vector W.
    %
    %	The ordering of the parameters in HP is defined by
    %	  hp = [hyper-params of gp.cf{1}, hyper-params of gp.cf{2}, ...];
    %
    %	See also
    %	GPCF_EXP_UNPAK
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


    function [gpcf, w] = gpcf_exp_unpak(gpcf, w)
    %GPCF_EXP_UNPAK  Separate GP covariance function hyper-parameter vector into components. 
    %
    %	Description
    %	GP = GPCF_EXP_UNPAK(GP, W) takes an Gaussian Process data structure GP
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


    function eprior =gpcf_exp_e(gpcf, x, t)
    %GPCF_EXP_E	Evaluate prior contribution of error of covariance function SE.
    %
    %	Description
    %	E = GPCF_EXP_E(W, GP, X, T) takes a gp data structure GPCF together
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
    end


    function [g, gdata, gprior]  = gpcf_exp_g(gpcf, x, t, g, gdata, gprior, varargin)
    %GPCF_SEXP_G Evaluate gradient of error for EXP covariance function.
    %
    %	Descriptioni
    %	G = GPCF_EXP_G(W, GPCF, X, T, invC, B) takes a gp hyper-parameter  
    %       vector W, data structure GPCF a matrix X of input vectors a matrix T
    %       of target vectors, inverse covariance function invC and B(=invC*t), 
    %	and evaluates the error gradient G. Each row of X corresponds to one 
    %       input vector and each row of T corresponds to one target vector.
    %
    %	[G, GDATA, GPRIOR] = GPCF_EXP_G(GP, P, T) also returns separately  the
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
        Cdm = gpcf_exp_trcov(gpcf, x);
        invCv=invC(:);
        b = varargin{2};
        % loop over all the lengthScales
        if length(gpcf.lengthScale) == 1
            % In the case of isotropic SEXP (no ARD)
            s = 1./gpcf.lengthScale;
            dist = 0;
            for i=1:nin
                dist = dist + (gminus(x(:,i),x(:,i)')).^2;
            end
            D = Cdm.*s.*sqrt(dist);
            Bdl = b'*(D*b);
            Cdl = sum(invCv.*D(:)); % help arguments for lengthScale 
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
                Bdl(i) = b'*(D*b);
                Cdl(i) = sum(invCv.*D(:)); % help arguments for lengthScale 
            end
        end
        Bdm = b'*(Cdm*b);
        Cdm = sum(invCv.*Cdm(:)); % help argument for magnSigma2
      case {'FIC', 'PIC_BLOCK', 'PIC_BAND'}
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
            % In the case of an isotropic SEXP
            s = 1./gpcf.lengthScale.^2;
            dist = 0;
            dist2 = 0;
            for i=1:m
                dist = dist + (gminus(u(:,i),x(:,i)')).^2;
                dist2 = dist2 + (gminus(u(:,i),u(:,i)')).^2;
            end
            dist = s.*K_uf.*sqrt(dist);
            dist2 = s.*K_uu.*sqrt(dist2);
            DKuf_l = dist(:);
            DKuu_l = dist2(:);
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
      case {'PIC_BLOCK', 'PIC_BAND'}
        
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
              case {'FIC', 'PIC_BLOCK', 'PIC_BAND'}
                gdata(i1)= DE_Kuu(:)'*DKuu_l(:,i2) + DE_Kuf(:)'*DKuf_l(:,i2);
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
          case {'FIC', 'PIC_BLOCK', 'PIC_BAND'}
            gdata(i1)= DE_Kuu(:)'*DKuu_l(:) + DE_Kuf(:)'*DKuf_l(:);
        end
        gprior(i1)=feval(gpp.lengthScale.fg, ...
                         gpcf.lengthScale, ...
                         gpp.lengthScale.a, 'x').*gpcf.lengthScale -1;
    end

    g = gdata + gprior;
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
        s = 1./gpcf.lengthScale;
        s2 = s.^2;
        if m1==1 && m2==1
            dist = s.*abs(gminus(x1,x2'));
        else
            % If ARD is not used make s a vector of 
            % equal elements 
            if size(s)==1
                s2 = repmat(s2,1,m1);
            end
            dist=zeros(n1,n2);
            for j=1:m1
                dist = dist + s2(j).*(gminus(x1(:,j),x2(:,j)')).^2;
            end
        end
        C = ma2.*exp(-sqrt(dist));
    end
    end


    function C = gpcf_exp_trcov(gpcf, x)
    % GP_EXP_TRCOV     Evaluate training covariance matrix of inputs. 
    %
    %         Description
    %         C = GP_EXP_TRCOV(GP, TX) takes in covariance function of a Gaussian
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
    C = ma.*exp(-sqrt(C));
    C(C<eps)=0;
    end

    function C = gpcf_exp_trvar(gpcf, x)
    % GP_EXP_TRVAR     Evaluate training variance vector of inputs. 
    %
    %         Description
    %         C = GP_EXP_TRVAR(GP, TX) takes in covariance function of a Gaussian
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

    C = ones(n,1).*gpcf.magnSigma2;
    C(C<eps)=0;
    end

    function reccf = gpcf_exp_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_EXP_RECAPPEND(RECCF, RI, GPCF) takes old covariance 
    %          function record RECCF, record index RI, RECAPPEND returns a 
    %          structure RECCF containing following record fields:
    %          lengthHyper    = 
    %          lengthHyperNu  = 
    %          lengthScale    = 
    %          magnSigma2     = 

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