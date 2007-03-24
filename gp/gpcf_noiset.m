function gpcf = gpcf_noiset(do, varargin)
%GPCF_NOISET	Create a Student-t noiset covariance function for Gaussian Process.
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
%         fh_g           = function handle to gradient function
%                          (@gpcf_se_g)
%         fh_cov         = function handle to covariance function
%                          (@gpcf_se_cov)
%         fh_trcov       = function handle to training covariance function
%                          (@gpcf_se_trcov)
%         fh_sampling    = function handle to parameter sampling function
%                          (@hmc2)
%         sampling_opt   = options structure for fh_sampling
%                          (hmc2_opt)
%         fh_recappend   = function handle to record append function
%                          (gpcf_noise_recappend)
%
%	GPCF = GPCF_NOISET('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in GPCF.
%
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
    gpcf.noiseSigmas2 = 0.1^2.*ones(1,varargin{2});
    
    % Initialize prior structure
    gpcf.p=[];
    gpcf.p.noiseSigmas2=[];
    
    % Set the function handles
    gpcf.fh_pak = @gpcf_noiset_pak;
    gpcf.fh_unpak = @gpcf_noiset_unpak;
    gpcf.fh_e = @gpcf_noiset_e;
    gpcf.fh_g = @gpcf_noiset_g;
    gpcf.fh_cov = @gpcf_noiset_cov;
    gpcf.fh_trcov  = @gpcf_noiset_trcov;
    gpcf.fh_sampling = @gpcf_fh_sampling;
    %    gpcf.sampling_opt = 'noiset_opt';
    gpcf.fh_recappend = @gpcf_noiset_recappend;
    
    if length(varargin) > 2
        if mod(nargin,2) ==0
            error('Wrong number of arguments')
        end
        % Loop through all the parameter values that are changed
        for i=3:2:length(varargin)-1
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

w=w;

% $$$ gpp=gpcf.p;
% $$$ 
% $$$ i1=0;i2=1;
% $$$ if ~isempty(w)
% $$$   i1 = length(w);
% $$$ end
% $$$ 
% $$$ if ~isempty(gpcf.noiseSigmas2) & ~isempty(gpp.noiseSigmas2)
% $$$   if ~isempty(gpp.noiseSigmas2.p)
% $$$     if ~isempty(gpp.noiseSigmas2.p.s.p)
% $$$       i1=i1+1;
% $$$       w(i1)=gpp.noiseSigmas2.p.s.a.s;
% $$$     end
% $$$     i2=i1+length(gpp.noiseSigmas2.a.s);
% $$$     i1=i1+1;
% $$$     w(i1:i2)=gpp.noiseSigmas2.a.s;
% $$$     i1=i2;
% $$$     if any(strcmp(fieldnames(gpp.noiseSigmas2.p),'nu'))
% $$$       if ~isempty(gpp.noiseSigmas2.p.nu.p)
% $$$ 	i1=i1+1;
% $$$ 	w(i1)=gpp.noiseSigmas2.p.nu.a.s;
% $$$       end
% $$$       i2=i1+length(gpp.noiseSigmas2.a.nu);
% $$$       i1=i1+1;
% $$$       w(i1:i2)=gpp.noiseSigmas2.a.nu;
% $$$       i1=i2;
% $$$     end
% $$$   end
% $$$ end


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


% $$$ gpp=gpcf.p;
% $$$ i1=0;i2=1;
% $$$ 
% $$$ if ~isempty(gpcf.noiseSigmas2) & ~isempty(gpp.noiseSigmas2)
% $$$   if ~isempty(gpp.noiseSigmas2.p)
% $$$     if ~isempty(gpp.noiseSigmas2.p.s.p)
% $$$       i1=i1+1;
% $$$       gpcf.p.noiseSigmas2.p.s.a.s=w(i1);
% $$$     end
% $$$     i2=i1+length(gpp.noiseSigmas2.a.s);
% $$$     i1=i1+1;
% $$$     gpcf.p.noiseSigmas2.a.s=w(i1:i2);
% $$$     i1=i2;
% $$$     if any(strcmp(fieldnames(gpp.noiseSigmas2.p),'nu'))
% $$$       if ~isempty(gpp.noiseSigmas2.p.nu.p)
% $$$ 	i1=i1+1;
% $$$ 	gpcf.p.noiseSigmas2.p.nu.a.s=w(i1);
% $$$       end
% $$$       i2=i1+length(gpp.noiseSigmas2.a.nu);
% $$$       i1=i1+1;
% $$$       gpcf.p.noiseSigmas2.a.nu=w(i1:i2);
% $$$       i1=i2;
% $$$     end
% $$$   end
% $$$ end
% $$$ w = w(i1+1:end);

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

[n, m] =size(p);

% Evaluate the prior contribution to the error.
eprior = 0;
gpp=gpcf.p;

if ~isempty(gpcf.noiseSigmas2) & ~isempty(gpp.noiseSigmas2)
  if ~isempty(gpp.noiseSigmas2.p)
    if ~isempty(gpp.noiseSigmas2.p.s.p)
      eprior=eprior...
	     +feval(gpp.noiseSigmas2.p.s.p.s.fe, ...
		    gpp.noiseSigmas2.p.s.a.s, gpp.noiseSigmas2.p.s.p.s.a)...
	     -log(gpp.noiseSigmas2.p.s.a.s);
    end
    eprior=eprior...
	   +feval(gpp.noiseSigmas2.p.s.fe, ...
		  gpp.noiseSigmas2.a.s, gpp.noiseSigmas2.p.s.a)...
	   -sum(log(gpp.noiseSigmas2.a.s));
    if any(strcmp(fieldnames(gpp.noiseSigmas2.p),'nu'))
      if ~isempty(gpp.noiseSigmas2.p.nu.p)
	eprior=eprior...
	       +feval(gpp.noiseSigmas2.p.nu.p.s.fe, ...
		      gpp.noiseSigmas2.p.nu.a.s, gpp.noiseSigmas2.p.nu.p.s.a)...
	       -log(gpp.noiseSigmas2.p.nu.a.s);
      end
      eprior=eprior...
	     +feval(gpp.noiseSigmas2.p.nu.fe, ...
		    gpp.noiseSigmas2.a.nu, gpp.noiseSigmas2.p.nu.a)...
	     -sum(log(gpp.noiseSigmas2.a.nu));
    end
  end
  eprior=eprior...
	 +feval(gpp.noiseSigmas2.fe, ...
		gpcf.noiseSigmas2, gpp.noiseSigmas2.a)...
	 -sum(log(gpcf.noiseSigmas2));
end


function [g, gdata, gprior]  = gpcf_noiset_g(gpcf, p, t, g, gdata, gprior, invC, varargin)
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

C=zeros(n1,n2);

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

C=zeros(n,n);

% If noise variances are given add them to diagonal elements
if ~isempty(gpcf.noiseSigmas2)
  C(1:n1:end)=C(1:n1:end)+gpcf.noiseSigmas2;
end

function gpcf = gpcf_fh_sampling(gp, gpcf, opt, x, y)

% Function for sampling the noiseSigmas2:s

p = x;
t = y;

% First sample the latent values
% First covariance matrices
[K,C] = gp_trcov(gp,p);
L=inv(chol(C));
invC=L*L'; 
% Evaluate the expected value and variance of latent values
r=K'*invC;
y=r*t;

lcov=K-r*K;
% Then the latent values
lcov=(lcov+lcov')/2;  % ensure the matrix is symmetric
ly=y+chol(lcov)'*randn(size(t));

% The residual is now
r = ly-t;

% from this below sample 
gpcf.noiseSigmas2 = ones(size(gpcf.noiseSigmas2)); %Change this line



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
  gpcf.fh_sampling = @gpcf_fh_sampling;
  %  reccf.sampling_opt = noiset_opt;
  reccf.fh_recappend = @gpcf_noiset_recappend;  
  return
end


gpp = gpcf.p;

% record noiseSigma
if ~isempty(gpcf.noiseSigmas2)
  if ~isempty(gpp.noiseSigmas2)
    reccf.noiseHyper(ri,:)=gpp.noiseSigmas2.a.s;
    if ~isempty(gpp.noiseSigmas2.p) & ~isempty(gpp.noiseSigmas2.p.s.p)
      reccf.noiseHyperHyper(ri,:)=gpp.noiseSigmas2.p.s.a.s;
    end
    if ~isempty(gpp.noiseSigmas2.p) & any(strcmp(fieldnames(gpp.noiseSigmas2.p),'nu'))
      reccf.noiseHyperNus(ri,:)=gpp.noiseSigmas2.a.nu;
      if ~isempty(gpp.noiseSigmas2.p.nu.p)
	reccf.noiseHyperNusHyper(ri,:)=gpp.noiseSigmas2.p.nu.a.s;
      end
    end
  elseif ri==1
    reccf.noiseHyper=[];
  end
  reccf.noiseSigmas2(ri,:)=gpcf.noiseSigmas2;
elseif ri==1
  reccf.noiseSigmas2=[];
end
