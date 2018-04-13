function gpcf = gpcf_constant(varargin)
%GPCF_CONSTANT  Create a constant covariance function
%
%  Description
%    GPCF = GPCF_CONSTANT('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates a constant covariance function structure in which the
%    named parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%    GPCF = GPCF_CONSTANT(GPCF,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a covariance function structure with the named
%    parameters altered with the specified values.
%  
%    Parameters for constant covariance function [default]
%      constSigma2       - magnitude (squared) [.1]
%      constSigma2_prior - prior for constSigma2 [prior_sqrtt]
%
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%  See also
%    GP_SET, GPCF_*, PRIOR_*, MEAN_*
%
% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010 Jaakko Riihimaki, Aki Vehtari
% Copyright (c) 2014 Arno Solin

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPCF_CONSTANT';
  ip.addOptional('gpcf', [], @isstruct);
  ip.addParamValue('constSigma2',.1, @(x) isscalar(x) && x>0);
  ip.addParamValue('constSigma2_prior',prior_sqrtt(), @(x) isstruct(x) || isempty(x));
  ip.parse(varargin{:});
  gpcf=ip.Results.gpcf;

  if isempty(gpcf)
    init=true;
    gpcf.type = 'gpcf_constant';
  else
    if ~isfield(gpcf,'type') && ~isequal(gpcf.type,'gpcf_constant')
      error('First argument does not seem to be a valid covariance function structure')
    end
    init=false;
  end
  
  % Initialize parameter
  if init || ~ismember('constSigma2',ip.UsingDefaults)
    gpcf.constSigma2=ip.Results.constSigma2;
  end

  % Initialize prior structure
  if init
    gpcf.p=[];
  end
  if init || ~ismember('constSigma2_prior',ip.UsingDefaults)
    gpcf.p.constSigma2=ip.Results.constSigma2_prior;
  end
  
  if init
    % Set the function handles to the subfunctions
    gpcf.fh.pak = @gpcf_constant_pak;
    gpcf.fh.unpak = @gpcf_constant_unpak;
    gpcf.fh.lp = @gpcf_constant_lp;
    gpcf.fh.lpg = @gpcf_constant_lpg;
    gpcf.fh.cfg = @gpcf_constant_cfg;   
    gpcf.fh.cfdg = @gpcf_constant_cfdg;
    gpcf.fh.cfdg2 = @gpcf_constant_cfdg2;
    gpcf.fh.ginput = @gpcf_constant_ginput;
    gpcf.fh.ginput2 = @gpcf_constant_ginput2;
    gpcf.fh.ginput3 = @gpcf_constant_ginput3;
    gpcf.fh.ginput4 = @gpcf_constant_ginput4;
    gpcf.fh.cov = @gpcf_constant_cov;
    gpcf.fh.trcov  = @gpcf_constant_trcov;
    gpcf.fh.trvar  = @gpcf_constant_trvar;
    gpcf.fh.recappend = @gpcf_constant_recappend;
    gpcf.fh.cf2ss = @gpcf_constant_cf2ss;
  end        

end

function [w, s, h] = gpcf_constant_pak(gpcf, w)
%GPCF_CONSTANT_PAK  Combine GP covariance function parameters into
%                   one vector.
%
%  Description
%    W = GPCF_CONSTANT_PAK(GPCF) takes a covariance function
%    structure GPCF and combines the covariance function
%    parameters and their hyperparameters into a single row
%    vector W. This is a mandatory subfunction used for example 
%    in energy and gradient computations.
%
%       w = [ log(gpcf.constSigma2)
%             (hyperparameters of gpcf.constSigma2)]'
%
%  See also
%    GPCF_CONSTANT_UNPAK
  
  w = []; s = {}; h=[];
  
  if ~isempty(gpcf.p.constSigma2)
    w = log(gpcf.constSigma2);
    s = [s 'log(constant.constSigma2)'];
    h = 1;
    % Hyperparameters of constSigma2
    [wh, sh, hh] = gpcf.p.constSigma2.fh.pak(gpcf.p.constSigma2);
    sh=strcat(repmat('prior-', size(sh,1),1),sh);
    w = [w wh];
    s = [s sh];
    h = [h 1+hh];
  end        
end

function [gpcf, w] = gpcf_constant_unpak(gpcf, w)
%GPCF_CONSTANT_UNPAK  Sets the covariance function parameters
%                     into the structure
%
%  Description
%    [GPCF, W] = GPCF_CONSTANT_UNPAK(GPCF, W) takes a covariance
%    function structure GPCF and a parameter vector W, and
%    returns a covariance function structure identical to the
%    input, except that the covariance parameters have been set
%    to the values in W. Deletes the values set to GPCF from W
%    and returns the modified W. This is a mandatory subfunction 
%    used for example in energy and gradient computations.
%
%    Assignment is inverse of  
%       w = [ log(gpcf.constSigma2)
%             (hyperparameters of gpcf.constSigma2)]'
%
%  See also
%   GPCF_CONSTANT_PAK

  gpp=gpcf.p;
  if ~isempty(gpp.constSigma2)
    gpcf.constSigma2 = exp(w(1));
    w = w(2:end);
    % Hyperparameters of magnSigma2
    [p, w] = gpcf.p.constSigma2.fh.unpak(gpcf.p.constSigma2, w);
    gpcf.p.constSigma2 = p;
  end
end

function lp = gpcf_constant_lp(gpcf)
%GPCF_CONSTANT_LP  Evaluate the log prior of covariance function parameters
%
%  Description
%    LP = GPCF_CONSTANT_LP(GPCF) takes a covariance function
%    structure GPCF and returns log(p(th)), where th collects the
%    parameters. This is a mandatory subfunction used for example 
%    in energy computations.
%
%  See also
%    GPCF_CONSTANT_PAK, GPCF_CONSTANT_UNPAK, GPCF_CONSTANT_LPG, GP_E

% Evaluate the prior contribution to the error. The parameters that
% are sampled are from space W = log(w) where w is all the
% "real" samples. On the other hand errors are evaluated in the
% W-space so we need take into account also the Jacobian of
% transformation W -> w = exp(W). See Gelman et al. (2013),
% Bayesian Data Analysis, third edition, p. 21.
  
  lp = 0;
  gpp=gpcf.p;
  if ~isempty(gpp.constSigma2)
    lp = gpp.constSigma2.fh.lp(gpcf.constSigma2, gpp.constSigma2) +log(gpcf.constSigma2);
  end
end

function lpg = gpcf_constant_lpg(gpcf)
%GPCF_CONSTANT_LPG  Evaluate gradient of the log prior with respect
%               to the parameters.
%
%  Description
%    LPG = GPCF_CONSTANT_LPG(GPCF) takes a covariance function
%    structure GPCF and returns LPG = d log (p(th))/dth, where th
%    is the vector of parameters. This is a mandatory subfunction 
%    used for example in gradient computations.
%
%  See also
%    GPCF_CONSTANT_PAK, GPCF_CONSTANT_UNPAK, GPCF_CONSTANT_LP, GP_G

  lpg = [];
  gpp=gpcf.p;
  
  if ~isempty(gpcf.p.constSigma2)            
    lpgs = gpp.constSigma2.fh.lpg(gpcf.constSigma2, gpp.constSigma2);
    lpg = [lpg lpgs(1).*gpcf.constSigma2+1 lpgs(2:end)];
  end
end

function DKff = gpcf_constant_cfg(gpcf, x, x2, mask, i1)  
%GPCF_CONSTANT_CFG  Evaluate gradient of covariance function
%                   with respect to the parameters
%
%  Description
%    DKff = GPCF_CONSTANT_CFG(GPCF, X) takes a
%    covariance function structure GPCF, a matrix X of input
%    vectors and returns DKff, the gradients of covariance matrix
%    Kff = k(X,X) with respect to th (cell array with matrix
%    elements). This is a mandatory subfunction used in gradient 
%    computations.
%
%    DKff = GPCF_CONSTANT_CFG(GPCF, X, X2) takes a
%    covariance function structure GPCF, a matrix X of input
%    vectors and returns DKff, the gradients of covariance matrix
%    Kff = k(X,X2) with respect to th (cell array with matrix
%    elements). This subfunction is needed when using sparse 
%    approximations (e.g. FIC).
%
%    DKff = GPCF_CONSTANT_CFG(GPCF, X, [], MASK)
%    takes a covariance function structure GPCF, a matrix X of
%    input vectors and returns DKff, the diagonal of gradients of
%    covariance matrix Kff = k(X,X2) with respect to th (cell
%    array with matrix elements). This subfunction is needed when 
%    using sparse approximations (e.g. FIC).
%
%  See also
%    GPCF_CONSTANT_PAK, GPCF_CONSTANT_UNPAK, GPCF_CONSTANT_LP, GP_G

  [n, m] =size(x);

  DKff = {};
  
  if nargin==5
    % Use memory save option
    if i1==0
      % Return number of hyperparameters
      if ~isempty(gpcf.p.constSigma2)
        DKff=1;
      else
        DKff=0;
      end
      return
    end
  end
  
  % Evaluate: DKff{1} = d Kff / d constSigma2
  %           DKff{2} = d Kff / d coeffSigma2
  % NOTE! Here we have already taken into account that the parameters are transformed
  % through log() and thus dK/dlog(p) = p * dK/dp
  
  % evaluate the gradient for training covariance
  if nargin == 2 || (isempty(x2) && isempty(mask))
    
    if ~isempty(gpcf.p.constSigma2)
      DKff{1}=ones(n)*gpcf.constSigma2;
    end
    
    % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
  elseif nargin == 3 || isempty(mask)
    if size(x,2) ~= size(x2,2)
      error('gpcf_constant -> _ghyper: The number of columns in x and x2 has to be the same. ')
    end

    if ~isempty(gpcf.p.constSigma2)
      DKff{1}=ones([n size(x2,1)])*gpcf.constSigma2;
    end
    
    % Evaluate: DKff{1}    = d mask(Kff,I) / d constSigma2
    %           DKff{2...} = d mask(Kff,I) / d coeffSigma2
  elseif nargin == 4 || nargin == 5

    if ~isempty(gpcf.p.constSigma2)
      DKff{1}=ones(n,1)*gpcf.constSigma2; % d mask(Kff,I) / d constSigma2
    end
  end
  if nargin==5
    DKff=DKff{1};
  end

end


function DKff = gpcf_constant_cfdg(gpcf, x, x2, dims)
%GPCF_CONSTANT_CFDG  Evaluate gradient of covariance function, of
%                which has been taken partial derivative with
%                respect to x, with respect to parameters.
%
%  Description
%    DKff = GPCF_CONSTANT_CFDG(GPCF, X) takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns
%    DKff, the gradients of derivatived covariance matrix
%    dK(df,f)/dhyp = d(d k(X,X)/dx)/dhyp, with respect to the
%    parameters
%
%    Evaluate: DKff{1:m} = d Kff / d coeffSigma2
%    m is the dimension of inputs. If ARD is used, then multiple
%    coefficients. This subfunction is needed when using derivative 
%    observations.
%
%  See also
%    GPCF_CONSTANT_GINPUT

[n,m]=size(x);
if nargin<3
    x2=x;
end
if nargin < 4 || isempty(dims)
    dims = 1:m;
end
ii1=0;
DKff={};
if ~isempty(gpcf.p.constSigma2)
    dd=zeros(size(x,1),size(x2,1));
    for i=dims
        ii1=ii1+1;
        DKff{ii1}=dd;
    end
end
end

function DKff = gpcf_constant_cfdg2(gpcf, x, x2, dims1, dims2)
%GPCF_CONSTANT_CFDG2  Evaluate gradient of covariance function, of
%                 which has been taken partial derivatives with
%                 respect to both input variables x, with respect
%                 to parameters.
%
%  Description
%    DKff = GPCF_CONSTANT_CFDG2(GPCF, X) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of derivative covariance matrix
%    dK(df,df)/dhyp = d(d^2 k(X1,X2)/dX1dX2)/dhyp with respect to
%    the parameters
%
%    Evaluate: DKff{1:m} = d Kff / d coeffSigma 
%    m is the dimension of inputs. If ARD is used, then multiple
%    lengthScales. This subfunction is needed when using derivative 
%    observations.
%
%  See also
%   GPCF_CONSTANT_GINPUT, GPCF_CONSTANT_GINPUT2

if nargin < 4 || isempty(dims1)
    %dims1 = 1:m;
    error('dims1 needs to be given')
end
if nargin < 5 || isempty(dims2)
    %dims2 = 1:m;
    error('dims2 needs to be given')
end

[n,m]=size(x);
ii1=0;
%dd=zeros(size(x,1),size(x2,1));
dd=0;
DKff={};
if ~isempty(gpcf.p.constSigma2)
  ii1=ii1+1;
  DKff{ii1}=dd;
end
end

function DKff = gpcf_constant_ginput(gpcf, x, x2, i1)
%GPCF_CONSTANT_GINPUT  Evaluate gradient of covariance function with 
%                      respect to x.
%
%  Description
%    DKff = GPCF_CONSTANT_GINPUT(GPCF, X) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X) with respect to X (cell array with matrix elements).
%    This subfunction is needed when computing gradients with 
%    respect to inducing inputs in sparse approximations.
%
%    DKff = GPCF_CONSTANT_GINPUT(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2) with respect to X (cell array with matrix elements).
%    This subfunction is needed when computing gradients with 
%    respect to inducing inputs in sparse approximations.
%
%    DKff = GPCF_CONSTANT_GINPUT(GPCF, X, X2, i) takes a covariance
%    function structure GPCF, a matrix X of input vectors
%    and returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2), or k(X,X) if X2 is empty, with respect to ith 
%    covariate in X. This subfunction is needed when using 
%    memory save option in gp_set.
%
%  See also
%    GPCF_CONSTANT_PAK, GPCF_CONSTANT_UNPAK, GPCF_CONSTANT_LP, GP_G
  
  [n, m] =size(x);
  if nargin==4
    % Use memory save option
    if i1==0
      % Return number of covariates
      if isfield(gpcf,'selectedVariables')
        DKff=length(gpcf.selectedVariables);
      else
        DKff=m;
      end
      return
    end
  end
  
  if nargin == 2 || isempty(x2)
    ii1 = 0;
    for j = 1:n
      for i=1:m
        ii1 = ii1 + 1;
        DKff{ii1} = zeros(n);
      end
    end
    
  elseif nargin == 3 || nargin == 4
    %K = feval(gpcf.fh.cov, gpcf, x, x2);
    
    ii1 = 0;
    for j = 1:n
      for i=1:m
        ii1 = ii1 + 1;
        DKff{ii1} = zeros(n, size(x2,1));
        gprior(ii1) = 0; 
      end
    end
  end
  if nargin==5
    DKff=DKff{1};
  end
end


function DKff = gpcf_constant_ginput2(gpcf, x, x2, dims, takeOnlyDiag)
%GPCF_CONSTANT_GINPUT2  Evaluate gradient of covariance function with
%                   respect to both input variables x and x2 (in
%                   same dimension).
%
%  Description
%    DKff = GPCF_CONSTANT_GINPUT2(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of twice derivatived covariance
%    matrix K(df,df) = dk(X1,X2)/dX1dX2 (cell array with matrix
%    elements). Input variable's dimensions are expected to be
%    same. The function returns also DKff1 and DKff2 which are
%    parts of DKff and needed with CFDG2. DKff = DKff1 -
%    DKff2. This subfunction is needed when using derivative 
%    observations.
%   
%  See also
%    GPCF_CONSTANT_GINPUT, GPCF_CONSTANT_GINPUT2, GPCF_CONSTANT_CFDG2       

[n,m]=size(x);
if nargin<4 || isempty(dims)
    dims=1:m;
end
ii1=0;
if nargin==5 && isequal(takeOnlyDiag,'takeOnlyDiag')
    for i=dims
        ii1=ii1+1;
        DKff{ii1} = kron(0,zeros(n,1));
    end
    %DKff = kron(zeros(m,1),zeros(n,1));
else
    DK=zeros(size(x,1),size(x2,1));
    for i=dims
        ii1=ii1+1;
        DKff{ii1}=DK;
    end
end
end

function DKff = gpcf_constant_ginput3(gpcf, x, x2, dims1, dims2)
%GPCF_CONSTANT_GINPUT3  Evaluate gradient of covariance function with
%                   respect to both input variables x and x2 (in
%                   different dimensions).
%
%  Description
%    DKff = GPCF_CONSTANT_GINPUT3(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of twice derivatived covariance
%    matrix K(df,df) = dk(X1,X2)/dX1dX2 (cell array with matrix
%    elements). The derivative is calculated in multidimensional
%    problem between input's observation dimensions which are not
%    same. This subfunction is needed when using derivative 
%    observations.
%   
%  See also
%    GPCF_CONSTANT_GINPUT, GPCF_CONSTANT_GINPUT2, GPCF_CONSTANT_CFDG2        

if nargin<4 || isempty(dims1)
    dims1=1:m;
end
if nargin<5 || isempty(dims2)
    dims2=1:m;
end

[n,m]=size(x);
ii1=0;
DK=zeros(size(x,1),size(x2,1));
for i=dims1
  for j=dims2
    ii1=ii1+1;
    DKff{ii1}=DK;
  end
end
end

function DKff = gpcf_constant_ginput4(gpcf, x, x2, dims)
%GPCF_CONSTANT_GINPUT  Evaluate gradient of covariance function with 
%                  respect to x. Simplified and faster version of
%                  constant_ginput, returns full matrices.
%
%  Description
%    DKff = GPCF_CONSTANT_GINPUT4(GPCF, X) takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns
%    DKff, the gradients of covariance matrix Kff = k(X,X) with
%    respect to X (whole matrix). This subfunction is needed when 
%    using derivative observations.
%
%    DKff = GPCF_CONSTANT_GINPUT4(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2) with respect to X (whole matrix). This subfunction 
%    is needed when using derivative observations.
%
%  See also
%    GPCF_CONSTANT_PAK, GPCF_CONSTANT_UNPAK, GPCF_CONSTANT_LP, GP_G

[n,m]=size(x);
if nargin<4
    dims=1:m;
end
ii1=0;
if nargin==2
  x2=x;
end
DK=zeros(size(x,1),size(x2,1));
for i=dims
  ii1=ii1+1;
  DKff{ii1}=DK;
end
end


function C = gpcf_constant_cov(gpcf, x1, x2, varargin)
%GP_CONSTANT_COV  Evaluate covariance matrix between two input vectors
%
%  Description         
%    C = GP_CONSTANT_COV(GP, TX, X) takes in covariance function
%    of a Gaussian process GP and two matrixes TX and X that
%    contain input vectors to GP. Returns covariance matrix C. 
%    Every element ij of C contains covariance between inputs i
%    in TX and j in X. This is a mandatory subfunction used for 
%    example in prediction and energy computations.
%
%  See also
%    GPCF_CONSTANT_TRCOV, GPCF_CONSTANT_TRVAR, GP_COV, GP_TRCOV
  
  if isempty(x2)
    x2=x1;
  end
  [n1,m1]=size(x1);
  [n2,m2]=size(x2);

  if m1~=m2
    error('the number of columns of X1 and X2 has to be same')
  end

  C = ones(n1,n2)*gpcf.constSigma2;
end

function C = gpcf_constant_trcov(gpcf, x)
%GP_CONSTANT_TRCOV  Evaluate training covariance matrix of inputs
%
%  Description
%    C = GP_CONSTANT_TRCOV(GP, TX) takes in covariance function
%    of a Gaussian process GP and matrix TX that contains
%    training input vectors. Returns covariance matrix C. Every
%    element ij of C contains covariance between inputs i and j
%    in TX. This is a mandatory subfunction used for example in
%    prediction and energy computations.
%
%  See also
%    GPCF_CONSTANT_COV, GPCF_CONSTANT_TRVAR, GP_COV, GP_TRCOV

  n =size(x,1);
  C = ones(n,n)*gpcf.constSigma2;

end


function C = gpcf_constant_trvar(gpcf, x)
%GP_CONSTANT_TRVAR  Evaluate training variance vector
%
%  Description
%    C = GP_CONSTANT_TRVAR(GPCF, TX) takes in covariance function 
%    of a Gaussian process GPCF and matrix TX that contains
%    training inputs. Returns variance vector C. Every
%    element i of C contains variance of input i in TX. This is 
%    a mandatory subfunction used for example in prediction and 
%    energy computations.
%
%  See also
%    GPCF_CONSTANT_COV, GP_COV, GP_TRCOV

  n =size(x,1);
  C = ones(n,1)*gpcf.constSigma2;
  
end

function reccf = gpcf_constant_recappend(reccf, ri, gpcf)
%RECAPPEND Record append
%
%  Description
%    RECCF = GPCF_CONSTANT_RECAPPEND(RECCF, RI, GPCF) takes a
%    covariance function record structure RECCF, record index RI
%    and covariance function structure GPCF with the current MCMC
%    samples of the parameters. Returns RECCF which contains all
%    the old samples and the current samples from GPCF. This 
%    subfunction is needed when using MCMC sampling (gp_mc).
%
%  See also
%    GP_MC and GP_MC -> RECAPPEND

  if nargin == 2
    % Initialize the record
    reccf.type = 'gpcf_constant';

    % Initialize parameters
    reccf.constSigma2 = [];

    % Set the function handles
    reccf.fh.pak = @gpcf_constant_pak;
    reccf.fh.unpak = @gpcf_constant_unpak;
    reccf.fh.lp = @gpcf_constant_lp;
    reccf.fh.lpg = @gpcf_constant_lpg;
    reccf.fh.cfg = @gpcf_constant_cfg;
    reccf.fh.cfdg = @gpcf_constant_cfdg;
    reccf.fh.cfdg2 = @gpcf_constant_cfdg2;
    reccf.fh.ginput = @gpcf_constant_ginput;
    reccf.fh.ginput2 = @gpcf_constant_ginput2;
    reccf.fh.ginput3 = @gpcf_constant_ginput3;
    reccf.fh.ginput4 = @gpcf_constant_ginput4;
    reccf.fh.cov = @gpcf_constant_cov;
    reccf.fh.trcov  = @gpcf_constant_trcov;
    reccf.fh.trvar  = @gpcf_constant_trvar;
    reccf.fh.recappend = @gpcf_constant_recappend;
    reccf.p=[];
    reccf.p.constSigma2=[];
    if ~isempty(ri.p.constSigma2)
      reccf.p.constSigma2 = ri.p.constSigma2;
    end

  else
    % Append to the record
    gpp = gpcf.p;

    % record constSigma2
    reccf.constSigma2(ri,:)=gpcf.constSigma2;
    if isfield(gpp,'constSigma2') && ~isempty(gpp.constSigma2)
      reccf.p.constSigma2 = gpp.constSigma2.fh.recappend(reccf.p.constSigma2, ri, gpcf.p.constSigma2);
    end
  end
end

function [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = gpcf_constant_cf2ss(gpcf,x)
%GPCF_CONSTANT_CF2SS Convert the covariance function to state space form
%
%  Description
%    Convert the covariance function to state space form such that
%    the process can be described by the stochastic differential equation
%    of the form:
%      df(t)/dt = F f(t) + L w(t),
%    where w(t) is a white noise process. The observation model now 
%    corresponds to y_k = H f(t_k) + r_k, where r_k ~ N(0,sigma2).
%
%

  % Check inputs
  if nargin<2, x=[]; end

  % Define the model
  F      = 0; 
  L      = 1; 
  Qc     = 0; 
  H      = 1;
  Pinf   = gpcf.constSigma2;
  dF     = 0;
  dQc    = 0;
  dPinf  = 1;

  % Set params
  params.stationary = true;
  
  % Check which parameters are optimized
  if isempty(gpcf.p.constSigma2), ind(1) = false; else ind(1) = true; end
  
  % Return only those derivatives that are needed
  dF    = dF(:,:,ind);
  dQc   = dQc(:,:,ind);
  dPinf = dPinf(:,:,ind);
  
end

