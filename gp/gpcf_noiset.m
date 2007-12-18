function gpcf = gpcf_noiset(do, varargin)
%GPCF_NOISET	Create a Student-t noise covariance function for Gaussian Process.
%
%	Description
%
%	GPCF = GPCF_NOISET('INIT', NIN, NDATA) Create and initialize noise
%       covariance function fo Gaussian process. NOTE! In contrast to most of 
%       the gpcf functions gpcf_noiset needs the number of data points NDATA.
%
%	The fields and (default values) in GPCF_NOISET are:
%	  type           = 'gpcf_se'
%	  nin            = number of inputs (NIN)
%         ndata          = number of data points
%	  nout           = number of outputs: always 1
%	  noiseSigmas2   = scale of residual distribution
%                          Variation for normal distribution
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
%         fh_sample      = function handle to parameter sampling function
%         sampling_opt   = options structure for fh_sampling
%                          (hmc2_opt)
%         fh_recappend   = function handle to record append function
%                          (gpcf_noise_recappend)
%
%	GPCF = GPCF_NOISET('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in GPCF.
%
%       NOTE!
%       The Student-t noise covariance is greated as in Gelman et. al. (2004) page 304-305:
%    
%          C = alfa^2 * U, where U = diag(u_1, u_2, ..., u_n)
%          u_i ~ Inv-Chi^2(nu, tau^2)
%       
%          The degrees of freedom nu are given a 1/nu prior and they are sampled via 
%          slice sampling.
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
    if isempty(varargin{2})
        error('Not enough arguments. NDATA is missing')
    end
    nin = varargin{1};
    
    gpcf.type = 'gpcf_noiset';
    gpcf.nin = nin;
    gpcf.ndata = varargin{2};
    gpcf.nout = 1;
    
    % Initialize parameters
    gpcf.noiseSigmas2 = 0.1^2.*ones(varargin{2},1);
    gpcf.U = ones(varargin{2},1);
    gpcf.tau2 = 0.1;
    gpcf.alpha = 0.5;
    gpcf.nu = 4;
    
    % Initialize prior structure
    gpcf.p=[];
    gpcf.p.noiseSigmas2=[];
    
    % Set the function handles
    gpcf.fh_pak = @gpcf_noiset_pak;
    gpcf.fh_unpak = @gpcf_noiset_unpak;
    gpcf.fh_e = @gpcf_noiset_e;
    gpcf.fh_ghyper = @gpcf_noiset_ghyper;
    gpcf.fh_cov = @gpcf_noiset_cov;
    gpcf.fh_trcov  = @gpcf_noiset_trcov;
    gpcf.fh_trvar  = @gpcf_noiset_trvar;
    gpcf.fh_sample = @gpcf_fh_sample;
    %    gpcf.sampling_opt = 'noiset_opt';
    gpcf.fh_recappend = @gpcf_noiset_recappend;
    
    if length(varargin) > 2
        if mod(nargin,2) ==0
            error('Wrong number of arguments')
        end
        % Loop through all the parameter values that are changed
        for i=3:2:length(varargin)-1
            if strcmp(varargin{i},'noiseSigmas2')
                if size(varargin{i+1},1) == gpcf.ndata
                    gpcf.noiseSigmas2 = varargin{i+1};
                else
                    error('the size of noiseSigmas2 is wrong, it has to be NDATAx1')
                end
            elseif strcmp(varargin{i},'fh_sampling')
                gpcf.fh_sampling = varargin{i+1};
            elseif strcmp(varargin{i},'U')
                if size(varargin{i+1},1) == gpcf.ndata
                    gpcf.U = varargin{i+1};;
                else
                    error('the size of U is wrong, it has to be NDATAx1')
                end
            elseif strcmp(varargin{i},'tau2')
                gpcf.tau2 = varargin{i+1};;
            elseif strcmp(varargin{i},'alpha')
                gpcf.alpha = varargin{i+1};;
            elseif strcmp(varargin{i},'nu')
                gpcf.nu = varargin{i+1};;
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
        if strcmp(varargin{i},'noiseSigmas2')
            if size(varargin{i+1},2) == gpcf.ndata
                gpcf.noiseSigmas2 = varargin{i+1};
            else
                error('the size of noiseSigmas2 is wrong, has to be 1xNDATA')
            end
        elseif strcmp(varargin{i},'fh_sampling')
            gpcf.fh_sampling = varargin{i+1};
        else
            error('Wrong parameter name!')
        end    
    end
end



    function w = gpcf_noiset_pak(gpcf, w)
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

    % Copyright (c) 2000-2001 Aki Vehtari
    % Copyright (c) 2006      Jarno Vanhatalo

    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
        
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

    % Copyright (c) 2000-2001 Aki Vehtari
    % Copyright (c) 2006      Jarno Vanhatalo

    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
        
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

    % Copyright (c) 1998-2006 Aki Vehtari

    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
        eprior = 0;
    end

    function [g, gdata, gprior]  = gpcf_noiset_ghyper(gpcf, p, t, g, gdata, gprior, invC, varargin)
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

    % Copyright (c) 1998-2001 Aki Vehtari
    % Copyright (c) 2006      Jarno Vanhatalo

    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
        
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

    % Copyright (c) 2006  Jarno Vanhatalo

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

    C = sparse([],[],[],n1,n2,0);
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

    % Copyright (c) 1998-2004 Aki Vehtari

    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.

    [n, m] =size(x);
    n1=n+1;

    C = spdiags(gpcf.noiseSigmas2, 0, n, n);
    end

    function C = gpcf_noiset_trvar(gpcf, x)
    % GP_NOISE_TRVAR     Evaluate training variance vector of inputs. 
    %
    %         Description
    %         C = GP_NOISE_TRVAR(GP, TX) takes in covariance function of a Gaussian
    %         process GP and matrix TX that contains training input vectors to 
    %         GP. Returns variance vector C. Every element i of C contains  
    %         variance of input i in TX 

        
    % Copyright (c) 1998-2004 Aki Vehtari
    % Copyright (c) 2006      Aki Vehtari, Jarno Vanhatalo
        
    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
        
        [n, m] =size(x);
        C = gpcf.noiseSigmas2;
        
    end
    
    function gpcf = gpcf_fh_sample(gp, gpcf, opt, x, y)
    % GPCF_FH_SAMPLE     Function for sampling the noiseSigmas2:s
    %
    %         Description
    %         

    % Copyright (c) 1998-2004 Aki Vehtari
    % Copyright (c) 2007 Jarno Vanhatalo

    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
            
        [n,m] = size(x);
        
        % Draw a sample of the mean of y. Its distribution is
        % y ~ N(K*inv(C)*y, K - K*inv(C)*K')
        switch gp.type
          case 'FULL'
            [meanY, varY, sampy] = gp_fwd(gp, x, y, x);
          case 'FIC'
            [meanY, varY, sampy] = gp_fwd(gp, x, y, x);
          case 'PIC_BLOCK'
            [meanY, varY, sampy] = gp_fwd(gp, x, y, x, gp.tr_index);
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
        % but 'gamrand' is parameterized as in Neal (1996).
        U=sinvchi2rand(nu+1, (nu.*t2+(r./alpha).^2)./(nu+1));
        shape = n*nu./2;                         % These are parameters...
        invscale = nu.*sum(1./U)./2;             % used in Gelman
        t2=gamrand(shape/invscale, 2.*shape);    % This is written as in Neal (1996)
        alpha2=sinvchi2rand(n,mean(r.^2./U));
        rss2=alpha2.*U;
        % Sample nu
        nu_energy = @(nu) (log(nu) - sinvchi2_lpdf(U,nu,t2));
        nu=sls1mm(nu_energy, nu, opt);
    
        gpcf.noiseSigmas2 = rss2;
        gpcf.U = U;
        gpcf.tau2 = t2;
        gpcf.alpha = sqrt(alpha2);
        gpcf.nu = nu;
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
        reccf.nin = ri;
        gpcf.nout = 1;
        
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
        reccf.fh_sampling = @gpcf_fh_sample;
        reccf.fh_sampling = @gpcf_fh_sampling;
        %  reccf.sampling_opt = noiset_opt;
        reccf.fh_recappend = @gpcf_noiset_recappend;  
        return
    end


    gpp = gpcf.p;

    % record noiseSigma
    if ~isempty(gpcf.noiseSigmas2)
        reccf.noiseSigmas2(ri,:)=gpcf.noiseSigmas2;
    elseif ri==1
        reccf.noiseSigmas2=[];
    end
    if ~isempty(gpcf.nu)
        reccf.nu(ri,:)=gpcf.nu;
        reccf.U = gpcf.U;
        reccf.tau2 = gpcf.tau2;
        reccf.alpha = gpcf.alpha;
    elseif ri==1
        reccf.noiseSigmas2=[];
    end
end
end

