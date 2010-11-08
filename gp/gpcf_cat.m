function gpcf = gpcf_cat(varargin)
%GPCF_CAT  Create a categorigal covariance function
%
%  Description
%    GPCF = GPCF_CAT('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates a categorigal covariance function structure in
%    which the named parameters have the specified values. Any
%    unspecified parameters are set to default values.
%
%    GPCF = GPCF_CAT(GPCF,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a covariance function structure with the named
%    parameters altered with the specified values.
%  
%    Categorigal covariance function returns correlation 1 if input
%    values X_i and X_j are equal and 0 otherwise.
%
%    Parameters for categorical covariance function
%      selectedVariables = vector defining which inputs are used
%
%  See also
%    GP_SET, GPCF_*, PRIOR_*, MEAN_*

% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2008-2010 Jaakko Riihimäki
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPCF_CAT';
  ip.addOptional('gpcf', [], @isstruct);
  ip.addParamValue('selectedVariables',[], ...
                   @(x) isempty(x) || (isvector(x) && all(x>0)));
  ip.parse(varargin{:});
  gpcf=ip.Results.gpcf;

  if isempty(gpcf)
    % Initialize a covariance function structure
    init=true;
    gpcf.type = 'gpcf_cat';
  else
    % Modify a covariance function structure
    if ~isfield(gpcf,'type') && ~isequal(gpcf.type,'gpcf_cat')
      error('First argument does not seem to be a valid covariance function structure')
    end
    init=false;
  end
  
  if ~ismember('selectedVariables',ip.UsingDefaults)
    if ~isempty(ip.Results.selectedVariables)
      gpcf.selectedVariables = ip.Results.selectedVariables;
    elseif isfield(gpcf,'selectedVariables')
      gpcf=rmfield(gpcf,'selectedVariables');
    end
  end
  if init
    gpcf.fh.pak = @gpcf_cat_pak;
    gpcf.fh.unpak = @gpcf_cat_unpak;
    gpcf.fh.e = @gpcf_cat_e;
    gpcf.fh.ghyper = @gpcf_cat_ghyper;
    gpcf.fh.ginput = @gpcf_cat_ginput;
    gpcf.fh.cov = @gpcf_cat_cov;
    gpcf.fh.trcov  = @gpcf_cat_trcov;
    gpcf.fh.trvar  = @gpcf_cat_trvar;
    gpcf.fh.recappend = @gpcf_cat_recappend;
  end

  function [w,s] = gpcf_cat_pak(gpcf, w)
  %GPCF_CAT_PAK  Combine GP covariance function hyper-parameters into
  %              one vector.
  %
  %  Description
  %    W = GPCF_CAT_PAK(GPCF) takes a covariance function data
  %    structure GPCF and combines the covariance function
  %    parameters and their hyperparameters into a single row
  %    vector W and takes a logarithm of the covariance function
  %    parameters.
  %
  %       w = []
  %
  %  See also
  %    GPCF_CAT_UNPAK
    
    w = []; s = {};
  end

  function [gpcf, w] = gpcf_cat_unpak(gpcf, w)
  %GPCF_CAT_UNPAK  Sets the covariance function parameters pack into
  %                the structure
  %
  %  Description
  %    [GPCF, W] = GPCF_CAT_UNPAK(GPCF, W) takes a covariance
  %    function data structure GPCF and a hyper-parameter vector W,
  %    and returns a covariance function data structure identical
  %    to the input, except that the covariance hyper-parameters
  %    have been set to the values in W. Deletes the values set to
  %    GPCF from W and returns the modified W.
  %
  %  See also
  %   GPCF_CAT_PAK
    
  end

  function eprior = gpcf_cat_e(gpcf, x, t)
  %GPCF_CAT_E  Evaluate the energy of prior of covariance function parameters
  %
  %  Description
  %    E = GPCF_CAT_E(GPCF, X, T) takes a covariance function data
  %    structure GPCF together with a matrix X of input vectors and
  %    a vector T of target vectors and evaluates log p(th) x J,
  %    where th is a vector of covariance function parameters and J
  %    is the Jacobian of transformation g(w) = th.
  %
  %    Also the log prior of the hyperparameters of the covariance
  %    function parameters is added to E if hyper-hyperprior is
  %    defined.
  %
  %  See also
  %    GPCF_CAT_PAK, GPCF_CAT_UNPAK, GPCF_CAT_G, GP_E

    eprior = 0;
    
  end

  function [DKff, gprior]  = gpcf_cat_ghyper(gpcf, x, x2, mask)
  %GPCF_CAT_GHYPER  Evaluate gradient of covariance function and
  %                 hyper-prior with respect to the hyperparameters.
  %
  %  Description
  %    [DKff, GPRIOR] = GPCF_CAT_GHYPER(GPCF, X) takes a covariance
  %    function data structure GPCF, a matrix X of input vectors
  %    and returns DKff, the gradients of covariance matrix Kff =
  %    k(X,X) with respect to th (cell array with matrix elements),
  %    and GPRIOR = d log (p(th))/dth, where th is the vector of
  %    hyperparameters
  %
  %    [DKff, GPRIOR] = GPCF_CAT_GHYPER(GPCF, X, X2) takes a
  %    covariance function data structure GPCF, a matrix X of input
  %    vectors and returns DKff, the gradients of covariance matrix
  %    Kff = k(X,X2) with respect to th (cell array with matrix
  %    elements), and GPRIOR = d log (p(th))/dth, where th is the
  %    vector of hyperparameters
  %
  %    [DKff, GPRIOR] = GPCF_CAT_GHYPER(GPCF, X, [], MASK) takes a
  %    covariance function data structure GPCF, a matrix X of input
  %    vectors and returns DKff, the diagonal of gradients of
  %    covariance matrix Kff = k(X,X2) with respect to th (cell
  %    array with matrix elements), and GPRIOR = d log (p(th))/dth,
  %    where th is the vector of hyperparameters. This is needed
  %    for example with FIC sparse approximation.
  %
  %  See also
  %    GPCF_CAT_PAK, GPCF_CAT_UNPAK, GPCF_CAT_E, GP_G

    DKff = {};
    gprior = [];
    
  end

  function [DKff, gprior]  = gpcf_cat_ginput(gpcf, x, x2)
  %GPCF_CAT_GINPUT  Evaluate gradient of covariance function with 
  %                 respect to x.
  %
  %  Description
  %    DKff = GPCF_CAT_GHYPER(GPCF, X) takes a covariance function
  %    data structure GPCF, a matrix X of input vectors and returns
  %    DKff, the gradients of covariance matrix Kff = k(X,X) with
  %    respect to X (cell array with matrix elements)
  %
  %    DKff = GPCF_CAT_GHYPER(GPCF, X, X2) takes a covariance
  %    function data structure GPCF, a matrix X of input vectors
  %    and returns DKff, the gradients of covariance matrix Kff =
  %    k(X,X2) with respect to X (cell array with matrix elements).
  %
  %  See also
  %   GPCF_CAT_PAK, GPCF_CAT_UNPAK, GPCF_CAT_E, GP_G
    
    [n, m] =size(x);
    
    if nargin == 2
      ii1 = 0;
      for i=1:m
        for j = 1:n
          ii1 = ii1 + 1;
          DKff{ii1} = zeros(n);
          gprior(ii1) = 0;
        end
      end
    elseif nargin == 3
      ii1 = 0;
      for i=1:m
        for j = 1:n
          ii1 = ii1 + 1;
          DKff{ii1} = zeros(n, size(x2,1));
          gprior(ii1) = 0; 
        end
      end
    end
  end

  function C = gpcf_cat_cov(gpcf, x1, x2, varargin)
  %GP_CAT_COV  Evaluate covariance matrix between two input vectors.
  %
  %  Description         
  %    C = GP_CAT_COV(GP, TX, X) takes in covariance function of a
  %    Gaussian process GP and two matrixes TX and X that contain
  %    input vectors to GP. Returns covariance matrix C. Every
  %    element ij of C contains covariance between inputs i in TX
  %    and j in X.
  %
  %  See also
  %    GPCF_CAT_TRCOV, GPCF_CAT_TRVAR, GP_COV, GP_TRCOV
    
    if isempty(x2)
      x2=x1;
    end
    [n1,m1]=size(x1);
    [n2,m2]=size(x2);
    
    if m1~=m2
      error('the number of columns of X1 and X2 has to be same')
    end

    C=repmat(true,n1,n2);
    if isfield(gpcf, 'selectedVariables')
      for j = 1:length(gpcf.selectedVariables)
        jj=gpcf.selectedVariables(j);
        C = C & bsxfun(@eq,x1(:,jj),x2(:,jj)');
      end
    else
      for j = 1:m1
        C = C & bsxfun(@eq,x1(:,j),x2(:,j)');
      end
    end
    C=double(C);
    
  end

  function C = gpcf_cat_trcov(gpcf, x)
  %GP_CAT_TRCOV  Evaluate training covariance matrix of inputs.
  %
  %  Description
  %    C = GP_CAT_TRCOV(GP, TX) takes in covariance function of a
  %    Gaussian process GP and matrix TX that contains training
  %    input vectors. Returns covariance matrix C. Every element ij
  %    of C contains covariance between inputs i and j in TX
  %
  %  See also
  %    GPCF_CAT_COV, GPCF_CAT_TRVAR, GP_COV, GP_TRCOV

    [n,m]=size(x);

    C=repmat(true,n,n);
    if isfield(gpcf, 'selectedVariables')
      for j = 1:length(gpcf.selectedVariables)
        jj=gpcf.selectedVariables(j);
        C = C & bsxfun(@eq,x(:,jj),x(:,jj)');
      end
    else
      for j = 1:m
        C = C & bsxfun(@eq,x(:,j),x(:,j)');
      end
    end
    C=double(C);
    
  end

  function C = gpcf_cat_trvar(gpcf, x)
  %GP_CAT_TRVAR  Evaluate training variance vector
  %
  %  Description
  %    C = GP_CAT_TRVAR(GPCF, TX) takes in covariance function of a
  %    Gaussian process GPCF and matrix TX that contains training
  %    inputs. Returns variance vector C. Every element i of C
  %    contains variance of input i in TX
  %
  %  See also
  %    GPCF_CAT_COV, GP_COV, GP_TRCOV

    [n,m]=size(x);
    C=ones(n,1);
    
  end

  function reccf = gpcf_cat_recappend(reccf, ri, gpcf)
  %RECAPPEND  Record append
  %
  %  Description
  %    RECCF = GPCF_CAT_RECAPPEND(RECCF, RI, GPCF) takes a
  %    covariance function record structure RECCF, record index RI
  %    and covariance function structure GPCF with the current MCMC
  %    samples of the hyperparameters. Returns RECCF which contains
  %    all the old samples and the current samples from GPCF .
  %
  %  See also
  %    GP_MC and GP_MC -> RECAPPEND

  % Initialize record
    if nargin == 2
      reccf.type = 'gpcf_cat';

      % Initialize parameters
      reccf.coeffSigma2= [];

      % Set the function handles
      reccf.fh.pak = @gpcf_cat_pak;
      reccf.fh.unpak = @gpcf_cat_unpak;
      reccf.fh.e = @gpcf_cat_e;
      reccf.fh.g = @gpcf_cat_g;
      reccf.fh.cov = @gpcf_cat_cov;
      reccf.fh.trcov  = @gpcf_cat_trcov;
      reccf.fh.trvar  = @gpcf_cat_trvar;
      reccf.fh.recappend = @gpcf_cat_recappend;

      return
    end

    if isfield(gpcf, 'selectedVariables')
      reccf.selectedVariables = gpcf.selectedVariables;
    end
    
  end
end