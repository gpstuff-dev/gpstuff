function lik = lik_gaussian(varargin)
%LIK_GAUSSIAN  Create a Gaussian likelihood structure
%
%  Description
%    LIK = LIK_GAUSSIAN('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates a Gaussian likelihood structure in which the named
%    parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%    LIK = LIK_GAUSSIAN(LIK,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a likelihhod function structure with the named
%    parameters altered with the specified values.
%
%    Parameters for Gaussian likelihood function [default]
%      sigma2       - variance [0.1]
%      sigma2_prior - prior for sigma2 [prior_logunif]
%
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%  See also
%    GP_SET, PRIOR_*, LIK_*

% Internal note: Because Gaussian noise can be combined
% analytically to the covariance matrix, lik_gaussian is internally
% little between lik_* and gpcf_* functions.
  
% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_GAUSSIAN';
  ip.addOptional('lik', [], @isstruct);
  ip.addParamValue('sigma2',0.1, @(x) isscalar(x) && x>0);
  ip.addParamValue('sigma2_prior',prior_logunif(), @(x) isstruct(x) || isempty(x));
  ip.parse(varargin{:});
  lik=ip.Results.lik;

  if isempty(lik)
    init=true;
    lik.type = 'Gaussian';
  else
    if ~isfield(lik,'type') && ~isequal(lik.type,'Gaussian')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end
  
  % Initialize parameters
  if init || ~ismember('sigma2',ip.UsingDefaults)
    lik.sigma2 = ip.Results.sigma2;
  end
  % Initialize prior structure
  if init
    lik.p=[];
  end
  if init || ~ismember('sigma2_prior',ip.UsingDefaults)
    lik.p.sigma2=ip.Results.sigma2_prior;
  end
  if init
    % Set the function handles to the nested functions
    lik.fh.pak = @lik_gaussian_pak;
    lik.fh.unpak = @lik_gaussian_unpak;
    lik.fh.lp = @lik_gaussian_lp;
    lik.fh.lpg = @lik_gaussian_lpg;
    lik.fh.llg = @lik_gaussian_llg;
    lik.fh.trcov  = @lik_gaussian_trcov;
    lik.fh.trvar  = @lik_gaussian_trvar;
    lik.fh.recappend = @lik_gaussian_recappend;
  end
  
  function [w s] = lik_gaussian_pak(lik)
  %LIK_GAUSSIAN_PAK  Combine likelihood parameters into one vector.
  %
  %  Description
  %    W = LIK_GAUSSIAN_PAK(LIK) takes a likelihood structure LIK
  %    and combines the parameters into a single row vector W.
  %
  %       w = [ log(lik.sigma2)
  %             (hyperparameters of lik.magnSigma2)]'
  %     
  %  See also
  %    LIK_GAUSSIAN_UNPAK

    w = []; s = {};
    if ~isempty(lik.p.sigma2)
      w = [w log(lik.sigma2)];
      s = [s 'log(gaussian.sigma2)'];
      % Hyperparameters of sigma2
      [wh sh] = feval(lik.p.sigma2.fh.pak, lik.p.sigma2);
      w = [w wh];
      s = [s sh];
    end    

  end

  function [lik, w] = lik_gaussian_unpak(lik, w)
  %LIK_GAUSSIAN_UNPAK  Extract likelihood parameters from the vector.
  %
  %  Description
  %    W = LIK_GAUSSIAN_UNPAK(W, LIK) takes a likelihood structure
  %    LIK and extracts the parameters from the vector W to the LIK
  %    structure.
  %
  %    Assignment is inverse of  
  %       w = [ log(lik.sigma2)
  %             (hyperparameters of lik.magnSigma2)]'
  %
  %  See also
  %    LIK_GAUSSIAN_PAK
    
    if ~isempty(lik.p.sigma2)
      lik.sigma2 = exp(w(1));
      w = w(2:end);
      
      % Hyperparameters of sigma2
      [p, w] = feval(lik.p.sigma2.fh.unpak, lik.p.sigma2, w);
      lik.p.sigma2 = p;
    end
  end

  function lp = lik_gaussian_lp(lik)
  %LIK_GAUSSIAN_LP  Evaluate the log prior of likelihood parameters
  %
  %  Description
  %    LP = LIK_T_LP(LIK) takes a likelihood structure LIK and
  %    returns log(p(th)), where th collects the parameters.
  %
  %  See also
  %    LIK_GAUSSIAN_PAK, LIK_GAUSSIAN_UNPAK, LIK_GAUSSIAN_G, GP_E

    lp = 0;

    if ~isempty(lik.p.sigma2)
      likp=lik.p;
      lp = feval(likp.sigma2.fh.lp, lik.sigma2, likp.sigma2) + log(lik.sigma2);
    end
  end

  function lpg = lik_gaussian_lpg(lik)
  %LIK_GAUSSIAN_LPG  Evaluate gradient of the log prior with respect
  %                  to the parameters.
  %
  %  Description
  %    LPG = LIK_GAUSSIAN_LPG(LIK) takes a Gaussian likelihood
  %    function structure LIK and returns LPG = d log (p(th))/dth,
  %    where th is the vector of parameters.
  %
  %  See also
  %    LIK_GAUSSIAN_PAK, LIK_GAUSSIAN_UNPAK, LIK_GAUSSIAN_E, GP_G

    lpg = [];

    if ~isempty(lik.p.sigma2)
      likp=lik.p;
      
      D{1}=lik.sigma2;
      
      ggs = feval(likp.sigma2.fh.lpg, lik.sigma2, likp.sigma2);
      lpg = ggs(1).*lik.sigma2 + 1;
      if length(ggs) > 1
        lpg = [lpg ggs(2:end)];
      end            
    end
  end

  function DKff = lik_gaussian_llg(lik, x, x2)
  %LIK_GAUSSIAN_LLG  Evaluate gradient of the log likelihood
  %
  %  Description
  %    Gaussian likelihood is a special case since it can be
  %    analytically combined with covariance functions and thus in
  %    gradient computation we need to take this into account.
  %
  %    DKff = LIK_GAUSSIAN_LLG(LIK, X) takes a Gaussian likelihood
  %    function structure LIK, a matrix X of input vectors and
  %    returns DKff, the gradients of Gaussian likelihood induced
  %    covariance matrix Kff = k(X,X) with respect to th (cell
  %    array with matrix elements).
  %
  %    DKff = LIK_GAUSSIAN_LLG(LIK, X, X2) takes a Gaussian
  %    likelihood function structure LIK, a matrix X of input
  %    vectors and returns DKff, the gradients of Gaussian
  %    likelihood induced covariance matrix Kff = k(X,X2) with
  %    respect to th (cell array with matrix elements).
  %
  %  See also
  %    LIK_GAUSSIAN_PAK, LIK_GAUSSIAN_UNPAK, LIK_GAUSSIAN_E, GP_G

    DKff = {};
    if ~isempty(lik.p.sigma2)
      DKff{1}=lik.sigma2;
    end
  end
  
  function DKff  = lik_gaussian_ginput(lik, x, t, g_ind, gdata_ind, gprior_ind, varargin)
  %LIK_GAUSSIAN_GINPUT  Evaluate gradient of likelihood function with 
  %                     respect to x.
  %
  %  Description
  %    DKff = LIK_GAUSSIAN_GINPUT(LIK, X) takes a likelihood
  %    function structure LIK, a matrix X of input vectors and
  %    returns DKff, the gradients of likelihood matrix Kff =
  %    k(X,X) with respect to X (cell array with matrix elements)
  %
  %    DKff = LIK_GAUSSIAN_GINPUT(LIK, X, X2) takes a likelihood
  %    function structure LIK, a matrix X of input vectors and
  %    returns DKff, the gradients of likelihood matrix Kff =
  %    k(X,X2) with respect to X (cell array with matrix elements).
  %
  %  See also
  %    LIK_GAUSSIAN_PAK, LIK_GAUSSIAN_UNPAK, LIK_GAUSSIAN_E, GP_G

  end

  function C = lik_gaussian_trcov(lik, x)
  %LIK_GAUSSIAN_TRCOV  Evaluate training covariance matrix
  %                    corresponding to Gaussian noise
  %  Description
  %    C = LIK_GAUSSIAN_TRCOV(GP, TX) takes in covariance function
  %    of a Gaussian process GP and matrix TX that contains
  %    training input vectors. Returns covariance matrix C. Every
  %    element ij of C contains covariance between inputs i and j
  %    in TX
  %
  %  See also
  %    LIK_GAUSSIAN_COV, LIK_GAUSSIAN_TRVAR, GP_COV, GP_TRCOV

    [n, m] =size(x);
    n1=n+1;

    C = sparse([],[],[],n,n,0);
    C(1:n1:end)=C(1:n1:end)+lik.sigma2;

  end

  function C = lik_gaussian_trvar(lik, x)
  %LIK_GAUSSIAN_TRVAR  Evaluate training variance vector
  %                    corresponding to Gaussian noise
  %
  %  Description
  %    C = LIK_GAUSSIAN_TRVAR(LIK, TX) takes in covariance function
  %    of a Gaussian process LIK and matrix TX that contains
  %    training inputs. Returns variance vector C. Every element i
  %    of C contains variance of input i in TX
  %
  %
  %  See also
  %    LIK_GAUSSIAN_COV, GP_COV, GP_TRCOV

    [n, m] =size(x);
    C=repmat(lik.sigma2,n,1);

  end

  function reccf = lik_gaussian_recappend(reccf, ri, lik)
  %RECAPPEND  Record append
  %
  %  Description
  %    RECCF = LIK_GAUSSIAN_RECAPPEND(RECCF, RI, LIK) takes a
  %    likelihood function record structure RECCF, record index RI
  %    and likelihood function structure LIK with the current MCMC
  %    samples of the parameters. Returns RECCF which contains all
  %    the old samples and the current samples from LIK .
  %
  %  See also
  %    GP_MC and GP_MC -> RECAPPEND

  % Initialize record
    if nargin == 2
      reccf.type = 'lik_gaussian';
      
      % Initialize parameters
      reccf.sigma2 = []; 
      
      % Set the function handles
      reccf.fh.pak = @lik_gaussian_pak;
      reccf.fh.unpak = @lik_gaussian_unpak;
      reccf.fh.lp = @lik_gaussian_lp;
      reccf.fh.lpg = @lik_gaussian_lpg;
      reccf.fh.llg = @lik_gaussian_llg;
      reccf.fh.trcov  = @lik_gaussian_trcov;
      reccf.fh.trvar  = @lik_gaussian_trvar;
      reccf.fh.recappend = @lik_gaussian_recappend;  
      reccf.p=[];
      reccf.p.sigma2=[];
      if ~isempty(ri.p.sigma2)
        reccf.p.sigma2 = ri.p.sigma2;
      end
      return
    end

    likp = lik.p;

    % record sigma
    if ~isempty(lik.sigma2)
      reccf.sigma2(ri,:)=lik.sigma2;
      reccf.p.sigma2 = feval(likp.sigma2.fh.recappend, reccf.p.sigma2, ri, likp.sigma2);
    elseif ri==1
      reccf.sigma2=[];
    end
  end

end
