function gpcf = gpcf_SSsexp(varargin)
%GPCF_SEXP Create a squared exponential covariance function for Gaussian Process
%
%  Description
%
%	GPCF = GPCF_SEXP('INIT','nin', NIN) Create and initialize 
%       squared exponential covariance function fo Gaussian process
%
%	The fields and (default values) in GPCF_SEXP are:
%	  type           = 'gpcf_sexp'
%	  nin            = number of inputs (NIN)
%	  magnSigma2     = general magnitude (squared) for exponential part  (sampled with HMC)
%                          (0.1)
%	  lengthScale    = length scale for each input. This can be either   (sampled with HMC)
%                          scalar (corresponding isotropic) or vector (corresponding ARD).
%                          (repmat(10, 1, NIN))
%     freq           =
%     nfreq          =
%
%	GPCF = GPCF_SEXP(GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in GPCF.
%
%  See also
%
%
%

% Copyright (c) 2008 Jarno Vanhatalo
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
    ip.FunctionName = 'GPCF_SSSEXP';
    ip.addOptional('gpcf', [], @isstruct);
    ip.addParamValue('nin',[], @(x) isscalar(x) && x>0 && mod(x,1)==0);
    ip.addParamValue('magnSigma2',[], @(x) isscalar(x) && x>0);
    ip.addParamValue('lengthScale',[], @(x) isvector(x) && all(x>0));
    ip.addParamValue('nfreq',[], @isscalar)
    ip.addParamValue('frequency',[], @isvector)
    ip.parse(varargin{:});
    gpcf=ip.Results.gpcf;
    magnSigma2=ip.Results.magnSigma2;
    lengthScale=ip.Results.lengthScale;
    nfreq=ip.Results.nfreq;
    nin=ip.Results.nin;
    frequency=ip.Results.frequency;

    switch do
        case 'init'
            gpcf.type = 'gpcf_SSsexp';
            gpcf.nin = nin;
            
            % Initialize parameters
            if isempty(lengthScale)
                gpcf.lengthScale = 10;
            else
                gpcf.lengthScale = lengthScale;
            end
            if isempty(magnSigma2)
                gpcf.magnSigma2 = 0.1;
            else
                gpcf.magnSigma2=magnSigma2;
            end
            if isempty(frequency)
                gpcf.frequency = sqrt(2).*erfinv(2.*hammersley(nin,100) - 1);
            else
                if size(frequency) ~= gpcf.nin
                    error('The size of the frequency matrix has to be m x nin!')
                else
                    gpcf.frequency = frequency;
                    gpcf.nfreq = size(gpcf.frequency,1);
                end
            end
            if isempty(nfreq)
                gpcf.nfreq = size(gpcf.frequency,2);
            else
                gpcf.nfreq = nfreq;
                gpcf.frequency = sqrt(2).*erfinv(2.*hammersley(gpcf.nin,gpcf.nfreq) - 1);
            end

            % Initialize prior structure
            gpcf.p=[];
            gpcf.p.lengthScale=[];
            gpcf.p.magnSigma2=[];

            % Set the function handles to the nested functions
            gpcf.fh.pak = @gpcf_SSsexp_pak;
            gpcf.fh.unpak = @gpcf_SSsexp_unpak;
            gpcf.fh.e = @gpcf_SSsexp_e;
            gpcf.fh.ghyper = @gpcf_SSsexp_ghyper;
            gpcf.fh.gind = @gpcf_SSsexp_gind;
            gpcf.fh.cov = @gpcf_SSsexp_cov;
            gpcf.fh.covvec = @gpcf_SSsexp_covvec;
            gpcf.fh.trcov  = @gpcf_SSsexp_trcov;
            gpcf.fh.trvar  = @gpcf_SSsexp_trvar;
            gpcf.fh.recappend = @gpcf_SSsexp_recappend;

        case 'set'
            % Set the parameter values of covariance function
            % go through all the parameter values that are changed
            if ~isempty(lengthScale)
                gpcf.lengthScale = lengthScale;
            end
            if ~isempty(magnSigma2)
                gpcf.magnSigma2=magnSigma2;
            end
            if ~isempty(frequency)
                if size(frequency) ~= gpcf.nin
                    error('The size of the frequency matrix has to be m x nin!')
                else
                    gpcf.frequency = frequency;
                    gpcf.nfreq = size(gpcf.frequency,1);
                end
            end
            if ~isempty(nfreq)
                gpcf.nfreq =nfreq;
                gpcf.frequency = sqrt(2).*erfinv(2.*hammersley(gpcf.nin,gpcf.nfreq) - 1);
            end
    end
    

    function w = gpcf_SSsexp_pak(gpcf, w, param)
        %GPcf_SEXP_PAK	 Combine GP covariance function hyper-parameters into one vector.
        %
        %  Description
        %	W = GP_SEXP_PAK(GPCF, W) takes a Gaussian Process data structure GP and
        %	combines the hyper-parameters into a single row vector W.
        %
        %	The ordering of the parameters in HP is defined by
        %	  hp = [hyper-params of gp.cf{1}, hyper-params of gp.cf{2}, ...];
        %
        %  See also
        %	GPCF_SEXP_UNPAK
        %

        % Copyright (c) 2008      Jarno Vanhatalo

        % This software is distributed under the GNU General Public
        % License (version 2 or later); please refer to the file
        % License.txt, included with the software, for details.

        if nargin == 2
            param = 'hyper';
        end
        
        switch param
            case 'hyper'
                gpp=gpcf.p;

                i1=0;i2=1;
                if ~isempty(w)
                    i1 = length(w);
                end
                i1 = i1+1;
                w(i1) = gpcf.magnSigma2;
                i2=i1+length(gpcf.lengthScale);
                i1=i1+1;
                w(i1:i2)=gpcf.lengthScale;
                i1=i2+1;
                if isfield(gpp.lengthScale, 'p') && ~isempty(gpp.lengthScale.p)
                    i1=i1+1;
                    w(i1)=gpp.lengthScale.a.s;
                    if any(strcmp(fieldnames(gpp.lengthScale.p),'nu'))
                        i1=i1+1;
                        w(i1)=gpp.lengthScale.a.nu;
                    end
                end
            case 'spectral'
                i1=0;
                if ~isempty(w)
                    i1 = length(w);
                end
                i2=i1+length(gpcf.frequency(:));
                w(i1:i2) = gp.frequency(:)';
                i1=i2; i2=i1+length(gpcf.phase(:));
                w(i1:i2) = gpcf.phase(:)
                i1=i2+1;
        end
    end




    function [gpcf, w] = gpcf_SSsexp_unpak(gpcf, w, param)
        %GPCF_SEXP_UNPAK  Separate GP covariance function hyper-parameter vector into components.
        %
        %  Description
        %	GP = GPCF_SEXP_UNPAK(GP, W) takes an Gaussian Process data structure GP
        %	and  a hyper-parameter vector W, and returns a Gaussian Process data
        %	structure  identical to the input model, except that the covariance
        %	hyper-parameters has been set to the of W.
        %
        %  See also
        %	GP_PAK
        %

        % Copyright (c) 2008      Jarno Vanhatalo

        % This software is distributed under the GNU General Public
        % License (version 2 or later); please refer to the file
        % License.txt, included with the software, for details.

        if nargin == 2
            param = 'hyper';
        end

        switch param
            case 'hyper'
                gpp=gpcf.p;
                i1=0;i2=1;
                i1=i1+1;
                gpcf.magnSigma2=w(i1);
                i2=i1+length(gpcf.lengthScale);
                i1=i1+1;
                gpcf.lengthScale=w(i1:i2);
                i1=i2;
                if isfield(gpp.lengthScale, 'p') && ~isempty(gpp.lengthScale.p)
                    i1=i1+1;
                    gpcf.p.lengthScale.a.s=w(i1);
                    if any(strcmp(fieldnames(gpp.lengthScale.p),'nu'))
                        i1=i1+1;
                        gpcf.p.lengthScale.a.nu=w(i1);
                    end
                end
                w = w(i1+1:end);
            case 'spectral'
                i1=1;i2=length(gpcf.frequency(:));
                gpcf.frequency = reshape(w(i1:i2), size(gpcf.frequency));
                i1 = i2; i2 = i1 + length(gpcf.frequency(:));
                i1 = i1+1; 
                gpcf.phase = reshape(w(i1:i2), size(gpcf.phase));
                w = w(i1+1:end);
        end
    end

    function eprior =gpcf_SSsexp_e(gpcf, x, t)
        %GPCF_SEXP_E	Evaluate prior contribution of error of covariance function SE.
        %
        %  Description
        %	E = GPCF_SEXP_E(W, GP, X, T) takes a gp data structure GPCF together
        %	with a matrix X of input vectors and a matrix T of target vectors,
        %	and evaluates the error function E. Each row of X corresponds
        %	to one input vector and each row of T corresponds to one
        %	target vector.
        %
        %  See also
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
        % into account also the  Jacobian of transformation W -> w = exp(W).
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

    function [DKff, gprior]  = gpcf_SSsexp_ghyper(gpcf, x, x2)
    %GPCF_SEXP_GHYPER     Evaluate gradient of error for SE covariance function
    %                     with respect to the hyperparameters.
    %
    %  Descriptioni
    %	G = GPCF_SEXP_GHYPER(W, GPCF, X, T, G, GDATA, GPRIOR, VARARGIN) takes a gp
    %   hyper-parameter vector W, data structure GPCF a matrix X of input vectors a
    %   matrix T of target vectors, inverse covariance function ,
    %	and evaluates the error gradient G. Each row of X corresponds to one
    %   input vector and each row of T corresponds to one target vector.
    %
    %	[G, GDATA, GPRIOR] = GPCF_SEXP_GHYPER(GP, P, T) also returns separately  the
    %	data and prior contributions to the gradient.
    %
    %  See also
    %

    % Copyright (c) 2008      Jarno Vanhatalo

    % This software is distributed under the GNU General Public
    % License (version 2 or later); please refer to the file
    % License.txt, included with the software, for details.
        
        gpp=gpcf.p;
        [n, m] =size(x);
        
        i1=0;i2=1;

        Cdm = gpcf_SSsexp_trcov(gpcf, x)./2;
        
        ii1=1;
        DKff{ii1} = Cdm;

        % loop over all the lengthScales
        if length(gpcf.lengthScale) == 1
            % In the case of isotropic SSSEXP                               
            l = gpcf.lengthScale;
            ma2 = gpcf.magnSigma2;
            l = repmat(l,1,m);
            
            s = gpcf.frequency./(repmat(l(:),1,gpcf.nfreq).*sqrt(2).*pi);
            C1 = x*s.*sin(2*pi*x*s);
            C2 = -x*s.*cos(2*pi*x*s);
            D = [C1 ; C2];
            D = 2.*pi.*reshape(D,n,2*gpcf.nfreq).*sqrt(ma2/gpcf.nfreq);
            
            ii1 = ii1+1;
            DKff{ii1} = D;
        else
            % In the case ARD is used
            l = gpcf.lengthScale;
            ma2 = gpcf.magnSigma2;
            if size(l)==1
                l = repmat(l,1,m);
            end
            
            s = gpcf.frequency./(repmat(l(:),1,gpcf.nfreq).*sqrt(2).*pi);
            C1 = 2.*pi.*sin(2*pi*x*s);
            C2 = -2.*pi.*cos(2*pi*x*s);
            C = [C1 ; C2];
            C = reshape(C,n,2*gpcf.nfreq).*sqrt(ma2/gpcf.nfreq);
            
            for i=1:m
                D1 = x(:,i)*s(i,:).*C1;
                D2 = x(:,i)*s(i,:).*C2;
                D = [D1 ; D2];
                D = reshape(D,n,2*gpcf.nfreq).*sqrt(ma2/gpcf.nfreq);
                
                ii1 = ii1+1;
                DKff{ii1} = D;
            end
        end
        
        % Evaluate the gprior with respect to magnSigma2
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


    function [g_ind, gdata_ind, gprior_ind]  = gpcf_SSsexp_gind(gpcf, x, t, g_ind, gdata_ind, gprior_ind, varargin)
    %GPCF_SEXP_GIND    Evaluate gradient of error for SE covariance function
    %                  with respect to inducing inputs.
    %
    %  Description
    %	[DKuu_u, DKuf_u] = GPCF_SEXP_GIND(W, GPCF, X, T)
    %
    %  See also
        %

        % Copyright (c) 2006      Jarno Vanhatalo

        % This software is distributed under the GNU General Public
        % License (version 2 or later); please refer to the file
        % License.txt, included with the software, for details.

        gdata_ind = gdata_ind + gradient;
        g_ind = gdata_ind;
    end


      function C = gpcf_SSsexp_trcov(gpcf, x)
        % GP_SSEXP_TRCOV     Evaluate training covariance matrix of inputs.
        %
        %         Description
        %         C = GP_SSEXP_TRCOV(GP, TX) takes in covariance function of a Gaussian
        %         process GP and matrix TX that contains training input vectors to
        %         GP. Returns covariance matrix C. Every element ij of C contains
        %         covariance between inputs i and j in TX
        %
        %         For covariance function definition see ...

        % Copyright (c) 2008      Jarno Vanhatalo

        % This software is distributed under the GNU General Public
        % License (version 2 or later); please refer to the file
        % License.txt, included with the software, for details.

        [n, m] =size(x);
        
        l = gpcf.lengthScale;
        ma2 = gpcf.magnSigma2;
        if size(l)==1
            l = repmat(l,1,m);
        end
        
        s = gpcf.frequency./(repmat(l(:),1,gpcf.nfreq).*sqrt(2).*pi);
        C1 = cos(2*pi*x*s);
        C2 = sin(2*pi*x*s);
        C = [C1 ; C2];
        C = reshape(C,n,2*gpcf.nfreq).*sqrt(ma2/gpcf.nfreq);
                    
% $$$         [n, m] =size(x);
% $$$         
% $$$         l = gpcf.lengthScale;
% $$$         ma = gpcf.magnSigma2;
% $$$         if size(l)==1
% $$$             l = repmat(l,1,m);
% $$$         end
% $$$         s = gpcf.frequency./repmat(l,gpcf.nfreq,1);
% $$$         phi = gpcf.phase;
% $$$ 
% $$$         C = cos(2*pi*x*s + phi);
    end

    function C = gpcf_SSsexp_trvar(gpcf, x)
        % GP_SSEXP_TRVAR     Evaluate training variance vector of inputs.
        %
        %         Description
        %         C = GP_SSEXP_TRVAR(GP, TX) takes in covariance function of a Gaussian
        %         process GP and matrix TX that contains training input vectors to
        %         GP. Returns variance vector C. Every element i of C contains
        %         variance of input i in TX
        %
        %         For covariance function definition see manual or
        %         Neal R. M. Regression and Classification Using Gaussian
        %         Process Priors, Bayesian Statistics 6.

        % Copyright (c) 2008      Jarno Vanhatalo

        % This software is distributed under the GNU General Public
        % License (version 2 or later); please refer to the file
        % License.txt, included with the software, for details.

% $$$         [n, m] =size(x);
% $$$ 
% $$$         C = ones(n,1)*gpcf.magnSigma2*2/gpcf.nfreq;
% $$$         C(C<eps)=0;
    end

    function reccf = gpcf_SSsexp_recappend(reccf, ri, gpcf)
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
            reccf.type = 'gpcf_SSsexp';
            reccf.nin = ri;
            reccf.nout = 1;

            % Initialize parameters
            reccf.lengthScale= [];
            reccf.magnSigma2 = [];

            % Set the function handles
            reccf.fh.pak = @gpcf_SSsexp_pak;
            reccf.fh.unpak = @gpcf_SSsexp_unpak;
            reccf.fh.e = @gpcf_SSsexp_e;
            reccf.fh.g = @gpcf_SSsexp_g;
            reccf.fh.cov = @gpcf_SSsexp_cov;
            reccf.fh.trcov  = @gpcf_SSsexp_trcov;
            reccf.fh.trvar  = @gpcf_SSsexp_trvar;
            reccf.fh.recappend = @gpcf_SSsexp_recappend;
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
