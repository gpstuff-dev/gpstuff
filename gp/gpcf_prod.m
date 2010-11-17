function gpcf = gpcf_prod(varargin)
%GPCF_PROD  Create a product form covariance function for Gaussian Process
%
%  Description
%    GPCF = GPCF_PROD('cf', {GPCF_1, GPCF_2, ...}) 
%    creates a product form covariance function
%          GPCF = GPCF_1 .* GPCF_2 .* ... .* GPCF_N
%
%  See also
%    GP_SET, GPCF_*
  
% Copyright (c) 2009-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPCF_PROD';
  ip.addOptional('gpcf', [], @isstruct);
  ip.addParamValue('cf',[], @iscell);
  ip.parse(varargin{:});
  gpcf=ip.Results.gpcf;

  if isempty(gpcf)
    init=true;
    gpcf.type = 'gpcf_prod';
  else
    if ~isfield(gpcf,'type') && ~isequal(gpcf.type,'gpcf_prod')
      error('First argument does not seem to be a valid covariance function structure')
    end
    init=false;
  end
  
  if init || ~ismember('cf',ip.UsingDefaults)
    % Initialize parameters
    gpcf.cf = {};
    cfs=ip.Results.cf;
    if ~isempty(cfs)
      for i = 1:length(cfs)
        gpcf.cf{i} = cfs{i};
      end
    else
      error('At least one covariance function has to be given in cf');
    end
  end
  
  if init
    % Set the function handles to the nested functions
    gpcf.fh.pak = @gpcf_prod_pak;
    gpcf.fh.unpak = @gpcf_prod_unpak;
    gpcf.fh.e = @gpcf_prod_e;
    gpcf.fh.ghyper = @gpcf_prod_ghyper;
    gpcf.fh.ginput = @gpcf_prod_ginput;
    gpcf.fh.cov = @gpcf_prod_cov;
    gpcf.fh.trcov  = @gpcf_prod_trcov;
    gpcf.fh.trvar  = @gpcf_prod_trvar;
    gpcf.fh.recappend = @gpcf_prod_recappend;
  end
  
  function [w, s] = gpcf_prod_pak(gpcf)
  %GPCF_PROD_PAK  Combine GP covariance function parameters into one vector.
  %
  %  Description
  %    W = GPCF_PROD_PAK(GPCF, W) loops through all the covariance
  %    functions and packs their parameters into one vector as
  %    described in the respective functions.
  %
  %  See also
  %    GPCF_PROD_UNPAK
    
    ncf = length(gpcf.cf);
    w = []; s = {};
    
    for i=1:ncf
      cf = gpcf.cf{i};
      [wi si] = feval(cf.fh.pak, cf);
      w = [w wi];
      s = [s; si];
    end
  end

  function [gpcf, w] = gpcf_prod_unpak(gpcf, w)
  %GPCF_PROD_UNPAK  Separate covariance function parameter
  %                 vector into components.
  %
  %  Description
  %    [GPCF, W] = GPCF_PROD_UNPAK(GPCF, W) loops through all the
  %    covariance functions and unpacks their parameters from W to
  %    each covariance function structure.
  %
  %  See also
  %    GPCF_PROD_PAK
  %
    ncf = length(gpcf.cf);
    
    for i=1:ncf
      cf = gpcf.cf{i};
      [cf, w] = feval(cf.fh.unpak, cf, w);
      gpcf.cf{i} = cf;
    end

  end
  
  function eprior =gpcf_prod_e(gpcf, x, t)
  %GPCF_PROD_E     Evaluate the energy of prior of PROD parameters
  %
  %  Description
  %    E = GPCF_PROD_E(GPCF, X, T) takes a covariance function data
  %    structure GPCF together with a matrix X of input vectors and
  %    a vector T of target vectors and evaluates log p(th) x J,
  %    where th is a vector of PROD parameters and J is the
  %    Jacobian of transformation exp(w) = th. (Note that the
  %    parameters are log transformed, when packed.)
  %
  %    Also the log prior of the hyperparameters of the covariance
  %    function parameters is added to E if prior is defined.
  %
  %  See also
  %    GPCF_PROD_PAK, GPCF_PROD_UNPAK, GPCF_PROD_G, GP_E
    
    eprior = 0;
    ncf = length(gpcf.cf);
    for i=1:ncf
      cf = gpcf.cf{i};
      eprior = eprior + feval(cf.fh.e, cf, x, t);
    end
    
  end

  function [DKff, gprior]  = gpcf_prod_ghyper(gpcf, x, x2, mask)
  %GPCF_PROD_GHYPER  Evaluate gradient of covariance function and
  %                  prior with respect to the parameters.
  %
  %  Description
  %    [DKff, GPRIOR] = GPCF_PROD_GHYPER(GPCF, X) takes a
  %    covariance function structure GPCF, a matrix X of input
  %    vectors and returns DKff, the gradients of covariance matrix
  %    Kff = k(X,X) with respect to th (cell array with matrix
  %    elements), and GPRIOR = d log (p(th))/dth, where th is the
  %    vector of parameters
  %
  %    [DKff, GPRIOR] = GPCF_PROD_GHYPER(GPCF, X, X2) takes a
  %    covariance function structure GPCF, a matrix X of input
  %    vectors and returns DKff, the gradients of covariance matrix
  %    Kff = k(X,X2) with respect to th (cell array with matrix
  %    elements), and GPRIOR = d log (p(th))/dth, where th is the
  %    vector of parameters
  %
  %    [DKff, GPRIOR] = GPCF_PROD_GHYPER(GPCF, X, [], MASK) takes a
  %    covariance function structure GPCF, a matrix X of input
  %    vectors and returns DKff, the diagonal of gradients of
  %    covariance matrix Kff = k(X,X2) with respect to th (cell
  %    array with matrix elements), and GPRIOR = d log (p(th))/dth,
  %    where th is the vector of parameters. This is needed for
  %    example with FIC sparse approximation.
  %
  %  See also
  %    GPCF_PROD_PAK, GPCF_PROD_UNPAK, GPCF_PROD_E, GP_G

    [n, m] =size(x);

    gprior = [];
    DKff = {};
    % Evaluate: DKff{1} = d Kff / d magnSigma2
    %           DKff{2} = d Kff / d lengthScale
    % NOTE! Here we have already taken into account that the parameters are transformed
    % through log() and thus dK/dlog(p) = p * dK/dp

    % evaluate the gradient for training covariance
    if nargin == 2
      
      ncf = length(gpcf.cf);
      
      % evaluate the individual covariance functions
      for i=1:ncf
        cf = gpcf.cf{i};
        C{i} = feval(cf.fh.trcov, cf, x);
      end
      
      % Evaluate the gradients
      ind = 1:ncf;
      DKff = {};
      for i=1:ncf
        cf = gpcf.cf{i};
        [DK, gpr] = feval(cf.fh.ghyper, cf, x);
        gprior = [gprior gpr];
        
        CC = 1;
        for kk = ind(ind~=i)
          CC = CC.*C{kk};
        end
        
        for j = 1:length(DK)
          DKff{end+1} = DK{j}.*CC;
        end
      end
      
      % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
    elseif nargin == 3
      if size(x,2) ~= size(x2,2)
        error('gpcf_prod -> _ghyper: The number of columns in x and x2 has to be the same. ')
      end
      
      ncf = length(gpcf.cf);
      
      % evaluate the individual covariance functions
      for i=1:ncf
        cf = gpcf.cf{i};
        C{i} = feval(cf.fh.cov, cf, x, x2);
      end
      
      % Evaluate the gradients
      ind = 1:ncf;
      DKff = {};
      for i=1:ncf
        cf = gpcf.cf{i};
        [DK, gpr] = feval(cf.fh.ghyper, cf, x, x2);
        gprior = [gprior gpr];
        
        CC = 1;
        for kk = ind(ind~=i)
          CC = CC.*C{kk};
        end
        
        for j = 1:length(DK)
          DKff{end+1} = DK{j}.*CC;
        end
      end

      
      
      % Evaluate: DKff{1}    = d mask(Kff,I) / d magnSigma2
      %           DKff{2...} = d mask(Kff,I) / d lengthScale
    elseif nargin == 4
      ncf = length(gpcf.cf);
      
      % evaluate the individual covariance functions
      for i=1:ncf
        cf = gpcf.cf{i};
        C{i} = feval(cf.fh.trvar, cf, x);
      end
      
      % Evaluate the gradients
      ind = 1:ncf;
      DKff = {};
      for i=1:ncf
        cf = gpcf.cf{i};
        [DK, gpr] = feval(cf.fh.ghyper, cf, [], 1);
        gprior = [gprior gpr;]
        
        CC = 1;
        for kk = ind(ind~=i)
          CC = CC.*C{kk};
        end
        
        for j = 1:length(DK)
          DKff{end+1} = DK{j}.*CC;
        end
      end
    end
  end


  function DKff  = gpcf_prod_ginput(gpcf, x, x2)
  %GPCF_PROD_GINPUT  Evaluate gradient of covariance function with 
  %                  respect to x.
  %
  %  Description
  %    DKff = GPCF_PROD_GHYPER(GPCF, X) takes a covariance function
  %    structure GPCF, a matrix X of input vectors and returns
  %    DKff, the gradients of covariance matrix Kff = k(X,X) with
  %    respect to X (cell array with matrix elements)
  %
  %    DKff = GPCF_PROD_GHYPER(GPCF, X, X2) takes a covariance
  %    function structure GPCF, a matrix X of input vectors and
  %    returns DKff, the gradients of covariance matrix Kff =
  %    k(X,X2) with respect to X (cell array with matrix elements).
  %
  %  See also
  %    GPCF_PROD_PAK, GPCF_PROD_UNPAK, GPCF_PROD_E, GP_G
    
    [n, m] =size(x);

    % evaluate the gradient for training covariance
    if nargin == 2
      
      ncf = length(gpcf.cf);
      
      % evaluate the individual covariance functions
      for i=1:ncf
        cf = gpcf.cf{i};
        C{i} = feval(cf.fh.trcov, cf, x);
      end
      
      % Evaluate the gradients
      ind = 1:ncf;
      for i=1:ncf
        cf = gpcf.cf{i};
        [DK, gpr] = feval(cf.fh.g, cf, x);
        
        CC = 1;
        for kk = ind(ind~=i)
          CC = CC.*C{kk};
        end
        
        for j = 1:length(DK)
          DKff{i+j-1} = DK{j}.*CC;
        end
      end

      % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
    elseif nargin == 3
      if size(x,2) ~= size(x2,2)
        error('gpcf_prod -> _ghyper: The number of columns in x and x2 has to be the same. ')
      end
      
      ncf = length(gpcf.cf);
      
      % evaluate the individual covariance functions
      for i=1:ncf
        cf = gpcf.cf{i};
        C{i} = feval(cf.fh.cov, cf, x, x2);
      end
      
      % Evaluate the gradients
      ind = 1:ncf;
      for i=1:ncf
        cf = gpcf.cf{i};
        [DK, gpr] = feval(cf.fh.g, cf, x, x2);
        
        CC = 1;
        for kk = ind(ind~=i)
          CC = CC.*C{kk};
        end
        
        for j = 1:length(DK)
          DKff{i+j-1} = DK{j}.*CC;
        end
      end
    end
    
  end


  function C = gpcf_prod_cov(gpcf, x1, x2)
  %GP_PROD_COV  Evaluate covariance matrix between two input vectors.
  %
  %  Description         
  %    C = GP_PROD_COV(GP, TX, X) takes in covariance function of a
  %    Gaussian process GP and two matrixes TX and X that contain
  %    input vectors to GP. Returns covariance matrix C. Every
  %    element ij of C contains covariance between inputs i in TX
  %    and j in X.
  %
  %
  %  See also
  %    GPCF_PROD_TRCOV, GPCF_PROD_TRVAR, GP_COV, GP_TRCOV
    
    if isempty(x2)
      x2=x1;
    end
    [n1,m1]=size(x1);
    [n2,m2]=size(x2);

    if m1~=m2
      error('the number of columns of X1 and X2 has to be same')
    end

    ncf = length(gpcf.cf);
    
    % evaluate the individual covariance functions
    C = 1;
    for i=1:ncf
      cf = gpcf.cf{i};
      C = C.*feval(cf.fh.cov, cf, x1, x2);
    end        
  end

  function C = gpcf_prod_trcov(gpcf, x)
  %GP_PROD_TRCOV     Evaluate training covariance matrix of inputs.
  %
  %  Description
  %    C = GP_PROD_TRCOV(GP, TX) takes in covariance function of a
  %    Gaussian process GP and matrix TX that contains training
  %    input vectors. Returns covariance matrix C. Every element ij
  %    of C contains covariance between inputs i and j in TX
  %
  %  See also
  %    GPCF_PROD_COV, GPCF_PROD_TRVAR, GP_COV, GP_TRCOV
    ncf = length(gpcf.cf);
    
    % evaluate the individual covariance functions
    C = 1;
    for i=1:ncf
      cf = gpcf.cf{i};
      C = C.*feval(cf.fh.trcov, cf, x);
    end
  end

  function C = gpcf_prod_trvar(gpcf, x)
  % GP_PROD_TRVAR     Evaluate training variance vector
  %
  %  Description
  %    C = GP_PROD_TRVAR(GPCF, TX) takes in covariance function of
  %    a Gaussian process GPCF and matrix TX that contains training
  %    inputs. Returns variance vector C. Every element i of C
  %    contains variance of input i in TX
  %
  %  See also
  %    GPCF_PROD_COV, GP_COV, GP_TRCOV


    ncf = length(gpcf.cf);
    
    % evaluate the individual covariance functions
    C = 1;
    for i=1:ncf
      cf = gpcf.cf{i};
      C = C.*feval(cf.fh.trvar, cf, x);
    end
  end

  function reccf = gpcf_prod_recappend(reccf, ri, gpcf)
  %RECAPPEND  Record append
  %
  %  Description
  %    RECCF = GPCF_PROD_RECAPPEND(RECCF, RI, GPCF) takes a
  %    covariance function record structure RECCF, record index RI
  %    and covariance function structure GPCF with the current MCMC
  %    samples of the parameters. Returns RECCF which contains all
  %    the old samples and the current samples from GPCF .
  %
  %  See also
  %    GP_MC, GP_MC->RECAPPEND
    
  % Initialize record
    if nargin == 2
      reccf.type = 'gpcf_prod';

      % Initialize parameters
      ncf = length(ri.cf);
      for i=1:ncf
        cf = ri.cf{i};
        reccf.cf{i} = feval(cf.fh.recappend, [], ri.cf{i});
      end
      
      % Set the function handles
      reccf.fh.pak = @gpcf_prod_pak;
      reccf.fh.unpak = @gpcf_prod_unpak;
      reccf.fh.e = @gpcf_prod_e;
      reccf.fh.g = @gpcf_prod_g;
      reccf.fh.cov = @gpcf_prod_cov;
      reccf.fh.trcov  = @gpcf_prod_trcov;
      reccf.fh.trvar  = @gpcf_prod_trvar;
      reccf.fh.recappend = @gpcf_prod_recappend;
      return
    end
    
    %loop over all of the covariance functions
    ncf = length(gpcf.cf);
    for i=1:ncf
      cf = gpcf.cf{i};
      reccf.cf{i} = feval(cf.fh.recappend, reccf.cf{i}, ri, cf);
    end
  end
end
