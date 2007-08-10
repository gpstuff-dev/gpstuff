function gpcf = gpcf_noise(do, varargin)
%GPCF_NOISE	Create a noise covariance function for Gaussian Process.
%
%	Description
%
%	GPCF = GPCF_NOISE('INIT', NIN) Create and initialize noise
%       covariance function fo Gaussian process 
%
%	The fields and (default values) in GPCF_NOISE are:
%	  type           = 'gpcf_se'
%	  nin            = number of inputs (NIN)
%	  nout           = number of outputs: always 1
%	  noiseSigmas2   = scale of residual distribution
%                          Variation for normal distribution 
%                          Degrees of freedom squared for t-distribution 
%                          (0.1^2)
%         p              = prior structure for covariance function
%                          parameters. 
%         fh_pak         = function handle to packing function
%                          (@gpcf_se_pak)
%         fh_unpak       = function handle to unpackin function
%                          (@gpcf_se_unpak)
%         fh_e           = function handle to error function
%                          (@gpcf_se_e)
%         fh_ghyper      = function handle to gradient function (with respect to hyperparameters)
%                          (@gpcf_se_ghyper)
%         fh_gind        = function handle to gradient function (with respect to inducing inputs)
%                          (@gpcf_se_gind)
%         fh_cov         = function handle to covariance function
%                          (@gpcf_se_cov)
%         fh_trcov       = function handle to training covariance function
%                          (@gpcf_se_trcov)
%         fh_trvar       = function handle to training variance function
%                          (@gpcf_se_trvar)
%         fh_sampling    = function handle to parameter sampling function
%                          (@hmc2)
%         sampling_opt   = options structure for fh_sampling
%                          (hmc2_opt)
%         fh_recappend   = function handle to record append function
%                          (gpcf_noise_recappend)
%
%	GPCF = GPCF_NOISE('SET', GPCF, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
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
  nin = varargin{1};
  
  gpcf.type = 'gpcf_noise';
  gpcf.nin = nin;
  gpcf.nout = 1;
  
  % Initialize parameters
  gpcf.noiseSigmas2 = 0.1^2; 
  
  % Initialize prior structure
  gpcf.p=[];
  gpcf.p.noiseSigmas2=[];
  
  % Set the function handles
  gpcf.fh_pak = @gpcf_noise_pak;
  gpcf.fh_unpak = @gpcf_noise_unpak;
  gpcf.fh_e = @gpcf_noise_e;
  gpcf.fh_ghyper = @gpcf_noise_ghyper;
  gpcf.fh_gind = @gpcf_noise_gind;
  gpcf.fh_cov = @gpcf_noise_cov;
  gpcf.fh_trcov  = @gpcf_noise_trcov;
  gpcf.fh_trvar  = @gpcf_noise_trvar;
  gpcf.fh_sampling = @hmc2;
  gpcf.sampling_opt = hmc2_opt;
  gpcf.fh_recappend = @gpcf_noise_recappend;
  
  if length(varargin) > 1
    if mod(nargin,2) ~=0
      error('Wrong number of arguments')
    end
    % Loop through all the parameter values that are changed
    for i=2:2:length(varargin)-1
      if strcmp(varargin{i},'noiseSigmas2')
	gpcf.noiseSigmas2 = varargin{i+1};
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
      gpcf.noiseSigmas2 = varargin{i+1};
    elseif strcmp(varargin{i},'fh_sampling')
      gpcf.fh_sampling = varargin{i+1};
    else
      error('Wrong parameter name!')
    end    
  end
end



function w = gpcf_noise_pak(gpcf, w)
%GPcf_NOISE_PAK	 Combine GP covariance function hyper-parameters into one vector.
%
%	Description
%	W = GP_NOISE_PAK(GPCF, W) takes a Gaussian Process covariance function
%	GPCF and combines the hyper-parameters into a single row vector W.
%
%	The ordering of the parameters in HP is defined by
%	  hp = [hyper-params of gp.cf{1}, hyper-params of gp.cf{2}, ...];
%
%	See also
%	GPCF_NOISE_UNPAK
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

i2=i1+length(gpcf.noiseSigmas2);
i1=i1+1;
w(i1:i2)=gpcf.noiseSigmas2;

function [gpcf, w] = gpcf_noise_unpak(gpcf, w)
%GPCF_SE_UNPAK  Separate GP covariance function hyper-parameter vector into components. 
%
%	Description
%	GP = GPCF_NOISE_UNPAK(GP, W) takes a Gaussian Process covariance function
%	GPCF and  a hyper-parameter vector W, and returns a covariance function data 
%	structure  identical to the input model, except that the covariance
%	hyper-parameters has been set to the of W.
%
%	See also
%	GP_NOISE_PAK, GP_PAK

% Copyright (c) 2000-2001 Aki Vehtari
% Copyright (c) 2006      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

gpp=gpcf.p;
i1=0;i2=1;

i2=i1+length(gpcf.noiseSigmas2);
i1=i1+1;
gpcf.noiseSigmas2=w(i1:i2);
w = w(i1+1:end);

function eprior =gpcf_noise_e(gpcf, p, t)
%GPCF_NOISE_E	Evaluate prior contribution of error of covariance function noise.
%
%	Description
%	E = GPCF_NOISE_E(W, GP, P, T) takes a gp data structure GPCF together
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

eprior=eprior...
       +feval(gpp.noiseSigmas2.fe, ...
              gpcf.noiseSigmas2, gpp.noiseSigmas2.a)...
       -sum(log(gpcf.noiseSigmas2));


function [g, gdata, gprior]  = gpcf_noise_ghyper(gpcf, x, t, g, gdata, gprior, varargin)
%GPCF_NOISE_GHYPER Evaluate gradient of error for NOISE covariance function.
%
%	Description
%	G = GPCF_NOISE_G(W, GPCF, X, T, C_gp, B) takes a gp hyper-parameter  
%       vector W, data structure GPCF a matrix X of input vectors a matrix T
%       of target vectors, covariance function C_gp and b(=invC*t), 
%	and evaluates the error gradient G. Each row of X corresponds to one 
%       input vector and each row of T corresponds to one target vector.
%
%	[G, GDATA, GPRIOR] = GPCF_NOISE_G(GP, P, T) also returns separately  the
%	data and prior contributions to the gradient.
%
%	See also
%

% Copyright (c) 1998-2001 Aki Vehtari
% Copyright (c) 2006-2007 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

gpp=gpcf.p;

i1=0;i2=1;
if ~isempty(g)
  i1 = length(g);
end

i1=i1+1;
D=gpcf.noiseSigmas2;
switch gpcf.type
  case 'FULL'
    %C_gp = varargin{1};
    invC = varargin{1};
    b = varargin{2};
    %B=trace(C_gp\eye(size(C_gp)));
    B = trace(invC);
    C=b'*b;    
    
    gdata(i1)=0.5.*D.*(B - C); 
  case 'FIC'       % NOTE! Here noiseSigmas2 is different from the beta used by Neill! 
                   % This affects that the derivative is also slightly different.
    R =varargin{3};
    gdata(i1) = D.*sum(R);
  case 'PIC_BLOCK'
    L = varargin{1};
    b =varargin{2};
    Labl = varargin{4};
    gdata(i1)= -0.5*D.*b*b';
    ind = gpcf.tr_index;
    for i=1:length(ind)
        gdata(i1)= gdata(i1) + 0.5*trace((inv(Labl{i})-L(ind{i},:)*L(ind{i},:)')).*D;
    end
  case {'PIC_BAND','CS+PIC'}
    L = varargin{1};
    b =varargin{2};
    La = varargin{4};
    gdata(i1)= - 0.5*D.*b*b';
    ind = gpcf.tr_index;
    %gdata(i1)= gdata(i1) + 0.5*trace((inv(La)-L*L')).*D;
    gdata(i1)= gdata(i1) + 0.5*trace((inv(La)-L*L')).*D;
end

gprior(i1)=feval(gpp.noiseSigmas2.fg, ...
                 gpcf.noiseSigmas2, ...
                 gpp.noiseSigmas2.a, 'x').*gpcf.noiseSigmas2-1;

g = gdata + gprior;

function [DKuu_u, DKuf_u]  = gpcf_noise_gind(gpcf, x, t, varargin)
    %GPCF_SEXP_GIND    Evaluate gradient of error for SE covariance function 
    %                  with respect to inducing inputs.
    %
    %	Descriptioni
    %	[DKuu_u, DKuf_u] = GPCF_SEXP_GIND(W, GPCF, X, T) 
    %
    %	See also
    %

    % Copyright (c) 1998-2001 Aki Vehtari
    % Copyright (c) 2006      Jarno Vanhatalo
        
    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
    DKuu_u = 0;
    DKuf_u = 0;


function C = gpcf_noise_cov(gpcf, x1, x2)
% GP_NOISE_COV     Evaluate covariance matrix between two input vectors. 
%
%         Description
%         C = GP_NOISE_COV(GP, TX, X) takes in covariance function of a Gaussian
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

%C=zeros(n1,n2);
C = sparse([],[],[],n1,n2,0);


function C = gpcf_noise_trcov(gpcf, x)
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

%C=zeros(n,n);
C = sparse([],[],[],n,n,0);
C(1:n1:end)=C(1:n1:end)+gpcf.noiseSigmas2;

function C = gpcf_noise_trvar(gpcf, x)
% GP_NOISE_TRVAR     Evaluate training variance vector of inputs. 
%
%         Description
%         C = GP_NOISE_TRVAR(GP, TX) takes in covariance function of a Gaussian
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
C=ones(n,1)*gpcf.noiseSigmas2;

function reccf = gpcf_noise_recappend(reccf, ri, gpcf)
% RECAPPEND - Record append
%          Description
%          RECCF = GPCF_NOISE_RECAPPEND(RECCF, RI, GPCF) takes old covariance 
%          function record RECCF, record index RI, RECAPPEND returns a 
%          structure REC containing following record fields:

% Initialize record
if nargin == 2
  reccf.type = 'gpcf_noise';
  reccf.nin = ri;
  gpcf.nout = 1;
  
  % Initialize parameters
  reccf.noiseSigmas2 = []; 
    
  % Set the function handles
  reccf.fh_pak = @gpcf_noise_pak;
  reccf.fh_unpak = @gpcf_noise_unpak;
  reccf.fh_e = @gpcf_noise_e;
  reccf.fh_g = @gpcf_noise_g;
  reccf.fh_cov = @gpcf_noise_cov;
  reccf.fh_trcov  = @gpcf_noise_trcov;
  reccf.fh_trvar  = @gpcf_noise_trvar;
%  gpcf.fh_sampling = @hmc2;
  reccf.sampling_opt = hmc2_opt;
  reccf.fh_recappend = @gpcf_noise_recappend;  
  return
end

gpp = gpcf.p;

% record noiseSigma
if ~isempty(gpcf.noiseSigmas2)
  if ~isempty(gpp.noiseSigmas2)
    reccf.noiseHyper(ri,:)=gpp.noiseSigmas2.a.s;
  elseif ri==1
    reccf.noiseHyper=[];
  end
  reccf.noiseSigmas2(ri,:)=gpcf.noiseSigmas2;
elseif ri==1
  reccf.noiseSigmas2=[];
end
