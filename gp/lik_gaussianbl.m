function lik = lik_gaussianbl(varargin)
%LIK_GAUSSIAN  Create a Gaussian likelihood structure
%
%  Description
%    LIK = LIK_GAUSSIANBL('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates a Gaussian likelihood structure in which the named
%    parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%    LIK = LIK_GAUSSIANBL(LIK,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a likelihood function structure with the named
%    parameters altered with the specified values.
%
%    Parameters for Gaussian likelihood function [default]
%      sigma2       - variance of the independent noise [0.1] for each
%                     block of inputs. If noiseSigma2 is a vector each
%                     entry of the vector specifies noise variance for
%                     a block of inputs defined by the last column of
%                     the input matrix X. The variances are set for blocks
%                     according to 'bl_indic' field so that sigma2(i) is a
%                     noise variance of the inputs whose last column equals
%                     bl_indic(i). 
%      bl_indic     - block indicator vector [empty matrix]. If
%                     length(sigma2)>1 bl_indic has to be the same length
%                     as sigma2.
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
%
% Copyright (c) 2007-2011 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_GAUSSIANBL';
  ip.addOptional('lik', [], @isstruct);
  ip.addParamValue('sigma2',0.1, @(x) isvector(x) && all(x>0));
  ip.addParamValue('sigma2_prior',prior_logunif(), @(x) isstruct(x) || isempty(x));
  ip.addParamValue('bl_indic',0.1, @(x) isvector(x));
  ip.parse(varargin{:});
  lik=ip.Results.lik;
  
  if isempty(lik)
    init=true;
    lik.type = 'GaussianBL';
  else
    if ~isfield(lik,'type') || ~isequal(lik.type,'GaussianBL')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end
  
  % Initialize parameters
  if init || ~ismember('sigma2',ip.UsingDefaults)
    lik.sigma2 = ip.Results.sigma2;
    lik.bl_indic = ip.Results.bl_indic;
  end
  
  if length(lik.sigma2)> 1 || length(lik.bl_indic) > 1
      if length(lik.sigma2) ~= length(lik.bl_indic)
         error('sigma2 and bl_indic has to be same length') 
      end
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
    lik.fh.pak = @lik_gaussianbl_pak;
    lik.fh.unpak = @lik_gaussianbl_unpak;
    lik.fh.lp = @lik_gaussianbl_lp;
    lik.fh.lpg = @lik_gaussianbl_lpg;
    lik.fh.cfg = @lik_gaussianbl_cfg;
    lik.fh.trcov  = @lik_gaussianbl_trcov;
    lik.fh.trvar  = @lik_gaussianbl_trvar;
    lik.fh.recappend = @lik_gaussianbl_recappend;
  end
  
  function [w s,h] = lik_gaussianbl_pak(lik)
  %LIK_GAUSSIANBL_PAK  Combine likelihood parameters into one vector.
  %
  %  Description
  %    W = LIK_GAUSSIANBL_PAK(LIK) takes a likelihood structure LIK
  %    and combines the parameters into a single row vector W.
  %    This is a mandatory subfunction used for example in energy 
  %    and gradient computations.
  %
  %       w = [ log(lik.sigma2)
  %             (hyperparameters of lik.magnSigma2)]'
  %     
  %  See also
  %    LIK_GAUSSIANBL_UNPAK

    w = []; s = {}; h=[];
    if ~isempty(lik.p.sigma2)
      w = log(lik.sigma2);
      if numel(lik.sigma2)>1
          s = [s; sprintf('log(gaussian.sigma2 x %d)',numel(lik.sigma2))];
        else
          s = [s; 'log(gaussian.sigma2)'];
      end
      h = [h zeros(1,numel(lik.sigma))];
      % Hyperparameters of noiseSigma2
      [wh, sh,hh] = lik.p.sigma2.fh.pak(lik.p.sigma2);
      w = [w wh];
      s = [s sh];
      h = [h hh];
    end    
  end

  function [lik, w] = lik_gaussianbl_unpak(lik, w)
  %LIK_GAUSSIANBL_UNPAK  Extract likelihood parameters from the vector.
  %
  %  Description
  %    W = LIK_GAUSSIANBL_UNPAK(W, LIK) takes a likelihood structure
  %    LIK and extracts the parameters from the vector W to the LIK
  %    structure. This is a mandatory subfunction used for example 
  %    in energy and gradient computations.
  %
  %    Assignment is inverse of  
  %       w = [ log(lik.sigma2)
  %             (hyperparameters of lik.magnSigma2)]'
  %
  %  See also
  %    LIK_GAUSSIANBL_PAK
    
    if ~isempty(lik.p.sigma2)
      i2=length(lik.sigma2);
      lik.sigma2 = exp(w(1:i2));
      w = w(i2+1:end);
      
      % Hyperparameters of sigma2
      [p, w] = lik.p.sigma2.fh.unpak(lik.p.sigma2, w);
      lik.p.sigma2 = p;
    end
  end

  function lp = lik_gaussianbl_lp(lik)
  %LIK_GAUSSIANBL_LP  Evaluate the log prior of likelihood parameters
  %
  %  Description
  %    LP = LIK_T_LP(LIK) takes a likelihood structure LIK and
  %    returns log(p(th)), where th collects the parameters. This
  %    subfunction is needed when there are likelihood parameters.
  %
  %  See also
  %    LIK_GAUSSIANBL_PAK, LIK_GAUSSIANBL_UNPAK, LIK_GAUSSIANBL_G, GP_E

    lp = 0;

    if ~isempty(lik.p.sigma2)
      likp=lik.p;
      lp = likp.sigma2.fh.lp(lik.sigma2, likp.sigma2) + sum(log(lik.sigma2));
    end
  end

  function lpg = lik_gaussianbl_lpg(lik)
  %LIK_GAUSSIANBL_LPG  Evaluate gradient of the log prior with respect
  %                  to the parameters.
  %
  %  Description
  %    LPG = LIK_GAUSSIANBL_LPG(LIK) takes a Gaussian likelihood
  %    function structure LIK and returns LPG = d log (p(th))/dth,
  %    where th is the vector of parameters. This subfunction is
  %    needed when there are likelihood parameters.
  %
  %  See also
  %    LIK_GAUSSIANBL_PAK, LIK_GAUSSIANBL_UNPAK, LIK_GAUSSIANBL_E, GP_G

    lpg = [];

    if ~isempty(lik.p.sigma2)
      likp=lik.p;
      i2=length(lik.sigma2);
      
      lpgs = likp.sigma2.fh.lpg(lik.sigma2, likp.sigma2);
      lpg = lpgs(1:i2).*lik.sigma2 + 1;
      if length(lpgs) > 1
        lpg = [lpg lpgs(i2+1:end)];
      end
    end
  end

  function DKff = lik_gaussianbl_cfg(lik, x, x2)
  %LIK_GAUSSIANBL_CFG  Evaluate gradient of covariance with respect to
  %                 Gaussian noise
  %
  %  Description
  %    Gaussian likelihood is a special case since it can be
  %    analytically combined with covariance functions and thus we
  %    compute gradient of covariance instead of gradient of likelihood.
  %
  %    DKff = LIK_GAUSSIANBL_CFG(LIK, X) takes a Gaussian likelihood
  %    function structure LIK, a matrix X of input vectors and
  %    returns DKff, the gradients of Gaussian noise covariance
  %    matrix Kff = k(X,X) with respect to th (cell array with
  %    matrix elements). This subfunction is needed only in 
  %    Gaussian likelihoods.
  %
  %    DKff = LIK_GAUSSIANBL_CFG(LIK, X, X2) takes a Gaussian
  %    likelihood function structure LIK, a matrix X of input
  %    vectors and returns DKff, the gradients of Gaussian noise
  %    covariance matrix Kff = k(X,X) with respect to th (cell
  %    array with matrix elements). This subfunction is needed only in 
  %    Gaussian likelihoods.
  %
  %  See also
  %    LIK_GAUSSIANBL_PAK, LIK_GAUSSIANBL_UNPAK, LIK_GAUSSIANBL_E, GP_G

    [n, m] =size(x);

    if length(lik.sigma2)==1
        DKff{1} = lik.sigma2;
    else
        for i1 = 1:length(lik.bl_indic)
            ind = find(x(:,end)==lik.bl_indic(i1));
            DKff{i1} = sparse(ind,ind,lik.sigma2(i1),n,n);
        end
    end
  
  end
  
  function DKff  = lik_gaussianbl_ginput(lik, x, t, g_ind, gdata_ind, gprior_ind, varargin)
  %LIK_GAUSSIANBL_GINPUT  Evaluate gradient of likelihood function with 
  %                     respect to x.
  %
  %  Description
  %    DKff = LIK_GAUSSIANBL_GINPUT(LIK, X) takes a likelihood
  %    function structure LIK, a matrix X of input vectors and
  %    returns DKff, the gradients of likelihood matrix Kff =
  %    k(X,X) with respect to X (cell array with matrix elements).
  %    This subfunction is needed when computing gradients with 
  %    respect to inducing inputs in sparse approximations.
  %
  %    DKff = LIK_GAUSSIANBL_GINPUT(LIK, X, X2) takes a likelihood
  %    function structure LIK, a matrix X of input vectors and
  %    returns DKff, the gradients of likelihood matrix Kff =
  %    k(X,X2) with respect to X (cell array with matrix elements).
  %    This subfunction is needed when computing gradients with 
  %    respect to inducing inputs in sparse approximations.
  %
  %  See also
  %    LIK_GAUSSIANBL_PAK, LIK_GAUSSIANBL_UNPAK, LIK_GAUSSIANBL_E, GP_G

  end

  function C = lik_gaussianbl_trcov(lik, x)
  %LIK_GAUSSIANBL_TRCOV  Evaluate training covariance matrix
  %                    corresponding to Gaussian noise
  %  Description
  %    C = LIK_GAUSSIANBL_TRCOV(GP, TX) takes in covariance function
  %    of a Gaussian process GP and matrix TX that contains
  %    training input vectors. Returns covariance matrix C. Every
  %    element ij of C contains covariance between inputs i and j
  %    in TX. This subfunction is needed only in Gaussian likelihoods.
  %
  %  See also
  %    LIK_GAUSSIANBL_COV, LIK_GAUSSIANBL_TRVAR, GP_COV, GP_TRCOV

    [n, m] =size(x);

    s2 = zeros(n,1);
    if length(lik.sigma2)==1
        s2 = ones(n,1).*lik.sigma2;
    else
        for i1 = 1:length(lik.bl_indic)
            s2(x(:,end)==lik.bl_indic(i1)) = lik.sigma2(i1);
        end
    end
    
    C = sparse(1:n,1:n,s2,n,n);

  end

  function C = lik_gaussianbl_trvar(lik, x)
  %LIK_GAUSSIANBL_TRVAR  Evaluate training variance vector
  %                    corresponding to Gaussian noise
  %
  %  Description
  %    C = LIK_GAUSSIANBL_TRVAR(LIK, TX) takes in covariance function
  %    of a Gaussian process LIK and matrix TX that contains
  %    training inputs. Returns variance vector C. Every element i
  %    of C contains variance of input i in TX. This subfunction is 
  %    needed only in Gaussian likelihoods.
  %
  %  See also
  %    LIK_GAUSSIANBL_COV, GP_COV, GP_TRCOV

    [n, m] =size(x);
    
    C = zeros(n,1);
    if length(lik.sigma2)==1
        C = ones(n,1).*lik.sigma2;
    else
        for i1 = 1:length(lik.bl_indic)
            C(x(:,end)==lik.bl_indic(i1)) = lik.sigma2(i1);
        end
    end
 
  end

  function reccf = lik_gaussianbl_recappend(reccf, ri, lik)
  %RECAPPEND  Record append
  %
  %  Description
  %    RECCF = LIK_GAUSSIANBL_RECAPPEND(RECCF, RI, LIK) takes a
  %    likelihood function record structure RECCF, record index RI
  %    and likelihood function structure LIK with the current MCMC
  %    samples of the parameters. Returns RECCF which contains all
  %    the old samples and the current samples from LIK. This 
  %    subfunction is needed when using MCMC sampling (gp_mc).
  %
  %  See also
  %    GP_MC and GP_MC -> RECAPPEND

  % Initialize record
    if nargin == 2
      reccf.type = 'lik_gaussianbl';
      
      % Initialize parameters
      reccf.sigma2 = []; 
      reccf.bl_indic = [];
      
      % Set the function handles
      reccf.fh.pak = @lik_gaussianbl_pak;
      reccf.fh.unpak = @lik_gaussianbl_unpak;
      reccf.fh.lp = @lik_gaussianbl_lp;
      reccf.fh.lpg = @lik_gaussianbl_lpg;
      reccf.fh.cfg = @lik_gaussianbl_cfg;
      reccf.fh.trcov  = @lik_gaussianbl_trcov;
      reccf.fh.trvar  = @lik_gaussianbl_trvar;
      reccf.fh.recappend = @lik_gaussianbl_recappend;  
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
      reccf.bl_indic(ri,:)=lik.bl_indic;
      if ~isempty(lik.p.sigma2)
          reccf.p.sigma2 = likp.sigma2.fh.recappend(reccf.p.sigma2, ri, likp.sigma2);
      end
    elseif ri==1
      reccf.sigma2=[];
    end
  end

end
