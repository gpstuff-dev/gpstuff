function gpcf = gpcf_noiset(do, varargin)
%GPCF_NOISET	Create a scale mixture noise covariance function (~Student-t) 
%               for Gaussian Process.
%
%	Description
%
%	GPCF = GPCF_NOISET('INIT', NDATA) Create and initialize noise
%       covariance function fo Gaussian process. NOTE! In contrast to most of 
%       the gpcf functions gpcf_noiset needs the number of data points NDATA.
%
%	The fields and (default values) in GPCF_NOISET are:
%	  type           = 'gpcf_se'
%         ndata          = number of data points
%	  nout           = number of outputs: always 1
%	  noiseSigmas2   = scale of residual distribution
%                          Variation for normal distribution (0.01)
%         U              = (1)
%         tau2           = (0.1)
%         alpha          = (0.5)
%         nu             = degrees of freedom (4)
%         r              = the residuals
%         fix_nu         = 0 for sampling also nu, 1 for not sampling nu (1)
%         p              = prior structure for covariance function
%                          parameters. 
%         fh_pak         = function handle to packing function
%                          (@gpcf_se_pak)
%         fh_unpak       = function handle to unpackin function
%                          (@gpcf_se_unpak)
%         fh_e           = function handle to error function
%                          (@gpcf_se_e)
%         fh_ghyper      = function handle to gradient function
%                          (@gpcf_se_g)
%         fh_cov         = function handle to covariance function
%                          (@gpcf_se_cov)
%         fh_trcov       = function handle to training covariance function
%                          (@gpcf_se_trcov)
%         fh_trvar       = function handle to training variance function
%                          (@gpcf_se_trvar)
%         fh_gibbs       = function handle to parameter sampling function (Gibbs sampling)
%         sampling_opt   = options structure for fh_sampling
%                          (hmc2_opt)
%         fh_recappend   = function handle to record append function
%                          (gpcf_noise_recappend)
%
%	GPCF = GPCF_NOISET('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in GPCF. The fields that 
%       can be modified are:
%
%             'noiseSigmas2'     : set the noiseSigmas2
%             'U'                : set the vector U
%             'tau2'             : set tau^2
%             'alpha'            : set alpha
%             'nu'               : set nu
%             'fix_nu'           : set fix_nu to 0 or 1
%             'nu_prior'         : set the prior structure for nu. 
%                  (nu is the only parameter whose prior we can adjust)
%
%       NOTE!
%       The Student-t residual model is greated as in Gelman et. al. (2004) page 304-305:
%    
%          y-E[y] ~ N(0, alpha^2 * U), where U = diag(u_1, u_2, ..., u_n)
%             u_i ~ Inv-Chi^2(nu, tau^2)
%       
%          The degrees of freedom nu are given a 1/nu prior and they are sampled via 
%          slice sampling.
%
%	See also
%	gpcf_sexp, gpcf_noise
%
%

% Copyright (c) 1998,1999 Aki Vehtari
% Copyright (c) 2006-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin < 2
    error('Not enough arguments')
end

% Initialize the covariance function
if strcmp(do, 'init')
   if isempty(varargin{2})
        error('Not enough arguments. NDATA is missing')
    end
    
    gpcf.type = 'gpcf_noiset';
    gpcf.ndata = varargin{1};
    gpcf.nout = 1;
    
    % Initialize parameters
    gpcf.noiseSigmas2 = 0.1^2.*ones(varargin{1},1);
    gpcf.U = ones(varargin{1},1);
    gpcf.tau2 = 0.1;
    gpcf.alpha = 0.5;
    gpcf.nu = 4;
    gpcf.r = zeros(varargin{1},1);
    gpcf.fix_nu = 1;
    
    % Initialize prior structure
    gpcf.p=[];
    gpcf.p.noiseSigmas2=[];
    gpcf.p.nu = prior_logunif('init');
    
    % Set the function handles
    gpcf.fh_pak = @gpcf_noiset_pak;
    gpcf.fh_unpak = @gpcf_noiset_unpak;
    gpcf.fh_e = @gpcf_noiset_e;
    gpcf.fh_ghyper = @gpcf_noiset_ghyper;
    gpcf.fh_cov = @gpcf_noiset_cov;
    gpcf.fh_trcov  = @gpcf_noiset_trcov;
    gpcf.fh_trvar  = @gpcf_noiset_trvar;
    gpcf.fh_gibbs = @gpcf_noiset_gibbs;
    %    gpcf.sampling_opt = 'noiset_opt';
    gpcf.fh_recappend = @gpcf_noiset_recappend;
    
    if length(varargin) > 1
        if mod(nargin,2) ~= 0
            error('Wrong number of arguments')
        end
        % Loop through all the parameter values that are changed
        for i=2:2:length(varargin)-1
            switch varargin{i}
              case 'noiseSigmas2'
                if size(varargin{i+1},1) == gpcf.ndata & size(varargin{i+1},2) == 1
                    gpcf.noiseSigmas2 = varargin{i+1};
                else
                    error('the size of has to be NDATAx1')
                end
              case 'fh_sampling'
                gpcf.fh_sampling = varargin{i+1};
              case 'U'
                if size(varargin{i+1},1) == gpcf.ndata
                    gpcf.U = varargin{i+1};
                else
                    error('the size of U is wrong, it has to be NDATAx1')
                end
              case 'tau2'
                gpcf.tau2 = varargin{i+1};
              case 'alpha'
                gpcf.alpha = varargin{i+1};
              case 'nu'
                gpcf.nu = varargin{i+1};
              case 'fix_nu'
                gpcf.fix_nu = varargin{i+1};
              case 'nu_prior'
                gpcf.p.nu = varargin{i+1};
              case 'censored'
                gpcf.censored = varargin{i+1}{1};
                yy = varargin{i+1}{2};
                if gpcf.censored(1) >= gpcf.censored(2)
                    error('gpcf_noiset -> if censored model is used the limits have to be given in increasing order.')
                end
                
                imis1 = [];
                imis2 = [];
                if gpcf.censored(1) > -inf
                    imis1 = find(yy<=gpcf.censored(1));
                end            
                if gpcf.censored(1) < inf
                    imis2 = find(yy>=gpcf.censored(2));
                end                                
                gpcf.cy = yy([imis1 ; imis2])';
                gpcf.imis = [imis1 ; imis2];
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
          case 'noiseSigmas2'
            if size(varargin{i+1},1) == gpcf.ndata & size(varargin{i+1},2) == 1
                gpcf.noiseSigmas2 = varargin{i+1};
            else
                error('the size of has to be NDATAx1')
            end
          case 'fh_sampling'
            gpcf.fh_sampling = varargin{i+1};
          case 'U'
            if size(varargin{i+1},1) == gpcf.ndata
                gpcf.U = varargin{i+1};
            else
                error('the size of U is wrong, it has to be NDATAx1')
            end
          case 'tau2'
            gpcf.tau2 = varargin{i+1};
          case 'alpha'
            gpcf.alpha = varargin{i+1};
          case 'nu'
            gpcf.nu = varargin{i+1};
          case 'fix_nu'
            gpcf.fix_nu = varargin{i+1};
          case 'nu_prior'
            gpcf.p.nu = varargin{i+1};
          case 'censored'
            gpcf.censored = varargin{i+1}{1};
            yy = varargin{i+1}{2};
            if gpcf.censored(1) >= gpcf.censored(2)
                error('gpcf_noiset -> if censored model is used the limits have to be given in increasing order.')
            end
            
            imis1 = [];
            imis2 = [];
            if gpcf.censored(1) > -inf
                imis1 = find(yy<=gpcf.censored(1));
            end            
            if gpcf.censored(1) < inf
                imis2 = find(yy>=gpcf.censored(2));
            end            
            gpcf.cy = yy([imis1 ; imis2])';
            gpcf.imis = [imis1 ; imis2];
          otherwise
            error('Wrong parameter name!')
        end    
    end
end



    function w = gpcf_noiset_pak(gpcf)
    %GPcf_NOISET_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %	W = GP_NOISET_PAK(GPCF, W) takes a Gaussian Process covariance function
    %	GPCF and combines the hyper-parameters into a single row vector W.
    %
    %	The ordering of the parameters in HP is defined by
    %	  hp = [hyper-params of gp.cf{1}, hyper-params of gp.cf{2}, ...];
    %
    %	See also
    %	GPCF_NOISET_UNPAK
    %

        w = [];
    end


    function [gpcf, w] = gpcf_noiset_unpak(gpcf, w)
    %GPCF_SE_UNPAK  Separate GP covariance function hyper-parameter vector into components. 
    %
    %	Description
    %	GP = GPCF_NOISET_UNPAK(GP, W) takes a Gaussian Process covariance function
    %	GPCF and  a hyper-parameter vector W, and returns a covariance function data 
    %	structure  identical to the input model, except that the covariance
    %	hyper-parameters has been set to the of W.
    %
    %	See also
    %	GP_NOISET_PAK, GP_PAK

    end

    function eprior =gpcf_noiset_e(gpcf, p, t)
    %GPCF_NOISET_E	Evaluate prior contribution of error of covariance function noiset.
    %
    %	Description
    %	E = GPCF_NOISET_E(W, GP, P, T) takes a gp data structure GPCF together
    %	with a matrix P of input vectors and a matrix T of target vectors,
    %	and evaluates the error function E. Each row of P corresponds
    %	to one input vector and each row of T corresponds to one
    %	target vector.
    %
    %	See also
    %	GP2, GP2PAK, GP2UNPAK, GP2FWD, GP2R_G
    %
        eprior = 0;
    end

    function [DCff, gprior]  = gpcf_noiset_ghyper(gpcf, p, t, g, gdata, gprior, invC, varargin)
    %GPCF_NOISE_G Evaluate gradient of error for SE covariance function.
    %
    %	Description
    %	G = GPCF_NOISET_G(W, GPCF, X, T) takes a gp hyper-parameter vector W, 
    %       data structure GPCF a matrix X of input vectors and a matrix T 
    %       of target vectors, and evaluates the error gradient G. Each row of X
    %	corresponds to one input vector and each row of T corresponds
    %       to one target vector.
    %
    %	[G, GDATA, GPRIOR] = GPCF_NOISET_G(GP, P, T) also returns separately  the
    %	data and prior contributions to the gradient.
    %
    %	See also
    %
        
        DCff = [];
        gprior = [];
    end

    function C = gpcf_noiset_cov(gpcf, x1, x2)
    % GP_NOISET_COV     Evaluate covariance matrix between two input vectors. 
    %
    %         Description
    %         C = GP_NOISET_COV(GP, TX, X) takes in covariance function of a Gaussian
    %         process GP and two matrixes TX and X that contain input vectors to 
    %         GP. Returns covariance matrix C. Every element ij of C contains  
    %         covariance between inputs i in TX and j in X.
    %
    %         For covariance function definition see manual or 
    %         Neal R. M. Regression and Classification Using Gaussian 
    %         Process Priors, Bayesian Statistics 6.

        C = 0;
    end


    function C = gpcf_noiset_trcov(gpcf, x)
    % GP_SE_COV     Evaluate training covariance matrix of inputs. 
    %
    %         Description
    %         C = GP_SE_COV(GP, TX) takes in covariance function of a Gaussian
    %         process GP and matrix TX that contains training input vectors to 
    %         GP. Returns covariance matrix C. Every element ij of C contains  
    %         covariance between inputs i and j in TX 
    %
    %         For covariance function definition see manual or 
    %         Neal R. M. Regression and Classification Using Gaussian 
    %         Process Priors, Bayesian Statistics 6.


        [n, m] =size(x);
        n1=n+1;
        
        if n ~= gpcf.ndata
            error(['gpcf_noiset -> _trcov: The training covariance can be evaluated'... 
                  '      only for training data.                                   '])
        end
        
        C = sparse(1:n, 1:n, gpcf.noiseSigmas2, n, n);
    end

    function C = gpcf_noiset_trvar(gpcf, x)
    % GP_NOISE_TRVAR     Evaluate training variance vector of inputs. 
    %
    %         Description
    %         C = GP_NOISE_TRVAR(GP, TX) takes in covariance function of a Gaussian
    %         process GP and matrix TX that contains training input vectors to 
    %         GP. Returns variance vector C. Every element i of C contains  
    %         variance of input i in TX 

        
        [n, m] =size(x);
        if n ~= gpcf.ndata
            error(['gpcf_noiset -> _trvar: The training variance can be evaluated'... 
                  '      only for training data.                                 '])
        end
        C = gpcf.noiseSigmas2;
        
    end
    
    function [gpcf, y] = gpcf_noiset_gibbs(gp, gpcf, opt, x, y)
    % GPCF_NOISET_GIBBS     Function for sampling the noiseSigmas2:s
    %
    %         Description
    %         Perform Gibbs sampling for the covariance function parameters.

            
        [n,m] = size(x);
        
        % Draw a sample of the mean of y. Its distribution is
        % f ~ N(K*inv(C)*y, K - K*inv(C)*K')
        switch gp.type
          case {'FULL', 'FIC'}
            sampy = gp_rnd(gp, x, y, x, [], 1:n);
          case {'PIC' 'PIC_BLOCK'}
            sampy = gp_rnd(gp, x, y, x, [], gp.tr_index);
        end
        % Calculate the residual
        r = y-sampy;
        
        U = gpcf.U;
        t2 = gpcf.tau2;
        alpha = gpcf.alpha;
        nu = gpcf.nu;
        rss2=alpha.^2.*U;
        
        % Perform the gibbs sampling (Gelman et.al. (2004) page 304-305)
        % Notice that 'sinvchi2rand' is parameterized as in Gelman et. al.
% $$$         U=invgamrand((nu.*t2+(r./alpha).^2)./(nu+1),nu+1);
% $$$         t2=gamrand(hmean(U),n*nu);
% $$$         alpha2=invgamrand(mean(r.^2./U),n);
% $$$         rss2=alpha2.*U;
% $$$         %nu=sls1mm(@invgam_nu_e,nu,soptnu,[],t2,U);
% $$$         nu=sls1mm(@(nu) -sum(sinvchi2_lpdf(U,nu,t2))+log(nu),nu,opt);
        
        U=sinvchi2rand(nu+1, (nu.*t2+(r./alpha).^2)./(nu+1));
        
% $$$         U2=invgamrand((nu.*t2+(r./alpha).^2)./(nu+1),nu+1);
        shape = n*nu./2;                               % These are parameters...
        invscale = nu.*sum(1./U)./2;                   % used in Gelman        
        t2=gamrnd(shape, 1./invscale);                 % Notice! The matlab parameterization is different from Gelmans
% $$$         t2=gamrand(hmean(U),n*nu);
        alpha2=sinvchi2rand(n,mean(r.^2./U));
% $$$         alpha2=invgamrand(mean(r.^2./U),n);
        rss2=alpha2.*U;
        % Sample nu
        if gpcf.fix_nu == 0 && ~isempty(gpcf.p.nu)
            pp = gpcf.p.nu;            
            nu=sls1mm( @(nu) (-sum(sinvchi2_lpdf(U,nu,t2))+feval(pp.fh_e, nu, pp)) ,nu,opt ) ;
        end
        gpcf.noiseSigmas2 = rss2;
        gpcf.U = U;
        gpcf.tau2 = t2;
        gpcf.alpha = sqrt(alpha2);
        gpcf.nu = nu;
        gpcf.r = r;
        if isfield(gpcf, 'censored')   
            imis1 = [];
            imis2 = [];
            if gpcf.censored(1) > -inf
                imis1 = find(y<=gpcf.censored(1));
                y(imis1)=normrtrand(sampy(imis1),alpha2*U(imis1),gpcf.censored(1));
            end
            
            if gpcf.censored(1) < inf
                imis2 = find(y>=gpcf.censored(2));
                y(imis2)=normltrand(sampy(imis2),alpha2*U(imis2),gpcf.censored(2));
            end
            gpcf.cy = y([imis1 ; imis2]);
        end
    end

    function reccf = gpcf_noiset_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_NOISET_RECAPPEND(RECCF, RI, GPCF) takes old covariance 
    %          function record RECCF, record index RI, RECAPPEND returns a 
    %          structure REC containing following record fields:
        
    % Initialize record
    if nargin == 2
        reccf.type = 'gpcf_noiset';
        gpcf.ndata = [];
        
        % Initialize parameters
        reccf.noiseSigmas2 = []; 
        
        % Set the function handles
        reccf.fh_pak = @gpcf_noiset_pak;
        reccf.fh_unpak = @gpcf_noiset_unpak;
        reccf.fh_e = @gpcf_noiset_e;
        reccf.fh_g = @gpcf_noiset_g;
        reccf.fh_cov = @gpcf_noiset_cov;
        reccf.fh_trcov  = @gpcf_noiset_trcov;
        reccf.fh_trvar  = @gpcf_noiset_trvar;
        reccf.fh_gibbs = @gpcf_noiset_gibbs;
        reccf.fh_recappend = @gpcf_noiset_recappend;
        return
    end

    reccf.ndata = gpcf.ndata;
    gpp = gpcf.p;

    % record noiseSigma
    if ~isempty(gpcf.noiseSigmas2)
        reccf.noiseSigmas2(ri,:)=gpcf.noiseSigmas2;
    elseif ri==1
        reccf.noiseSigmas2=[];
    end
    if ~isempty(gpcf.nu)
        reccf.nu(ri,:)=gpcf.nu;
        reccf.U(ri,:) = gpcf.U;
        reccf.tau2(ri,:) = gpcf.tau2;
        reccf.alpha(ri,:) = gpcf.alpha;
        reccf.r(ri,:) = gpcf.r;
    elseif ri==1
        reccf.noiseSigmas2=[];
    end
    if isfield(gpcf, 'censored')
        reccf.cy(ri,:) = gpcf.cy';
    end

end
end

