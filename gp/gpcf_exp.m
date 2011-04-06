function gpcf = gpcf_exp(varargin)
%GPCF_EXP  Create an exponential covariance function
%
%  Description
%    GPCF = GPCF_EXP('PARAM1',VALUE1,'PARAM2,VALUE2,...) creates a
%    exponential covariance function structure in which the named
%    parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%    GPCF = GPCF_EXP(GPCF,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a covariance function structure with the named
%    parameters altered with the specified values.
%  
%    Parameters for exponential covariance function [default]
%      magnSigma2        - magnitude (squared) [0.1]
%      lengthScale       - length scale for each input. [1]
%                          This can be either scalar corresponding
%                          to an isotropic function or vector
%                          defining own length-scale for each input
%                          direction.
%      magnSigma2_prior  - prior for magnSigma2  [prior_sqrtunif]
%      lengthScale_prior - prior for lengthScale [prior_unif]
%      metric            - metric structure used by the covariance function []
%      selectedVariables - vector defining which inputs are used [all]
%                          selectedVariables is shorthand for using
%                          metric_euclidean with corresponding components
%
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%  See also
%    GP_SET, GPCF_*, PRIOR_*, METRIC_*

% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

  if nargin>0 && ischar(varargin{1}) && ismember(varargin{1},{'init' 'set'})
    % remove init and set
    varargin(1)=[];
  end
  
  ip=inputParser;
  ip.FunctionName = 'GPCF_EXP';
  ip.addOptional('gpcf', [], @isstruct);
  ip.addParamValue('magnSigma2',0.1, @(x) isscalar(x) && x>0);
  ip.addParamValue('lengthScale',1, @(x) isvector(x) && all(x>0));
  ip.addParamValue('metric',[], @isstruct);
  ip.addParamValue('magnSigma2_prior', prior_sqrtunif(), ...
                   @(x) isstruct(x) || isempty(x));
  ip.addParamValue('lengthScale_prior',prior_unif(), ...
                   @(x) isstruct(x) || isempty(x));
  ip.addParamValue('selectedVariables',[], @(x) isempty(x) || ...
                   (isvector(x) && all(x>0)));
  ip.parse(varargin{:});
  gpcf=ip.Results.gpcf;
  
  if isempty(gpcf)
    init=true;
    gpcf.type = 'gpcf_exp';
  else
    if ~isfield(gpcf,'type') && ~isequal(gpcf.type,'gpcf_exp')
      error('First argument does not seem to be a valid covariance function structure')
    end
    init=false;
  end
  if init
    % Set the function handles to the subfunctions
    gpcf.fh.pak = @gpcf_exp_pak;
    gpcf.fh.unpak = @gpcf_exp_unpak;
    gpcf.fh.lp = @gpcf_exp_lp;
    gpcf.fh.lpg = @gpcf_exp_lpg;
    gpcf.fh.cfg = @gpcf_exp_cfg;
    gpcf.fh.ginput = @gpcf_exp_ginput;
    gpcf.fh.cov = @gpcf_exp_cov;
    gpcf.fh.trcov  = @gpcf_exp_trcov;
    gpcf.fh.trvar  = @gpcf_exp_trvar;
    gpcf.fh.recappend = @gpcf_exp_recappend;
  end
  
  % Initialize parameters
  if init || ~ismember('lengthScale',ip.UsingDefaults)
    gpcf.lengthScale = ip.Results.lengthScale;
  end
  if init || ~ismember('magnSigma2',ip.UsingDefaults)
    gpcf.magnSigma2 = ip.Results.magnSigma2;
  end

  % Initialize prior structure
  if init
    gpcf.p=[];
  end
  if init || ~ismember('lengthScale_prior',ip.UsingDefaults)
    gpcf.p.lengthScale=ip.Results.lengthScale_prior;
  end
  if init || ~ismember('magnSigma2_prior',ip.UsingDefaults)
    gpcf.p.magnSigma2=ip.Results.magnSigma2_prior;
  end

  %Initialize metric
  if ~ismember('metric',ip.UsingDefaults)
    if ~isempty(ip.Results.metric)
      gpcf.metric = ip.Results.metric;
      gpcf = rmfield(gpcf, 'lengthScale');
      gpcf.p = rmfield(gpcf.p, 'lengthScale');
    elseif isfield(gpcf,'metric')
      if ~isfield(gpcf,'lengthScale')
        gpcf.lengthScale = gpcf.metric.lengthScale;
      end
      if ~isfield(gpcf.p,'lengthScale')
        gpcf.p.lengthScale = gpcf.metric.p.lengthScale;
      end
      gpcf = rmfield(gpcf, 'metric');
    end
  end
  
  % selectedVariables options implemented using metric_euclidean
  if ~ismember('selectedVariables',ip.UsingDefaults)
    if ~isfield(gpcf,'metric')
      if ~isempty(ip.Results.selectedVariables)
        gpcf.metric=metric_euclidean('components',...
                                     num2cell(ip.Results.selectedVariables),...
                                     'lengthScale',gpcf.lengthScale,...
                                     'lengthScale_prior',gpcf.p.lengthScale);
        gpcf = rmfield(gpcf, 'lengthScale');
        gpcf.p = rmfield(gpcf.p, 'lengthScale');
      end
    elseif isfield(gpcf,'metric') 
      if ~isempty(ip.Results.selectedVariables)
        gpcf.metric=metric_euclidean(gpcf.metric,...
                                     'components',...
                                     num2cell(ip.Results.selectedVariables));
        if ~ismember('lengthScale',ip.UsingDefaults)
          gpcf.metric.lengthScale=ip.Results.lengthScale;
          gpcf = rmfield(gpcf, 'lengthScale');
        end
        if ~ismember('lengthScale_prior',ip.UsingDefaults)
          gpcf.metric.p.lengthScale=ip.Results.lengthScale_prior;
          gpcf.p = rmfield(gpcf.p, 'lengthScale');
        end
      else
        if ~isfield(gpcf,'lengthScale')
          gpcf.lengthScale = gpcf.metric.lengthScale;
        end
        if ~isfield(gpcf.p,'lengthScale')
          gpcf.p.lengthScale = gpcf.metric.p.lengthScale;
        end
        gpcf = rmfield(gpcf, 'metric');
      end
    end
  end
  
end

function [w,s] = gpcf_exp_pak(gpcf)
%GPCF_EXP_PAK  Combine GP covariance function parameters into
%              one vector
%
%  Description
%    W = GPCF_EXP_PAK(GPCF) takes a covariance function
%    structure GPCF and combines the covariance function
%    parameters and their hyperparameters into a single row
%    vector W.
%
%       w = [ log(gpcf.magnSigma2)
%             (hyperparameters of gpcf.magnSigma2) 
%             log(gpcf.lengthScale(:))
%             (hyperparameters of gpcf.lengthScale)]'
%
%  See also
%    GPCF_EXP_UNPAK

  w = []; s = {};
  
  if ~isempty(gpcf.p.magnSigma2)
    w = [w log(gpcf.magnSigma2)];
    s = [s; 'log(exp.magnSigma2)'];
    % Hyperparameters of magnSigma2
    [wh sh] = feval(gpcf.p.magnSigma2.fh.pak, gpcf.p.magnSigma2);
    w = [w wh];
    s = [s; sh];
  end        
  
  if isfield(gpcf,'metric')
    [wm sm] = feval(gpcf.metric.fh.pak, gpcf.metric);
    w = [w wm];
    s = [s; sm];
  else
    if ~isempty(gpcf.p.lengthScale)
      w = [w log(gpcf.lengthScale)];
      if numel(gpcf.lengthScale)>1
        s = [s; sprintf('log(exp.lengthScale x %d)',numel(gpcf.lengthScale))];
      else
        s = [s; 'log(exp.lengthScale)'];
      end
      % Hyperparameters of lengthScale
      w = [w feval(gpcf.p.lengthScale.fh.pak, gpcf.p.lengthScale)];
      w = [w wh];
      s = [s; sh];
    end
  end
  
end

function [gpcf, w] = gpcf_exp_unpak(gpcf, w)
%GPCF_EXP_UNPAK  Sets the covariance function parameters into
%                the structure
%
%  Description
%    [GPCF, W] = GPCF_EXP_UNPAK(GPCF, W) takes a covariance
%    function structure GPCF and a hyper-parameter vector W,
%    and returns a covariance function structure identical
%    to the input, except that the covariance hyper-parameters
%    have been set to the values in W. Deletes the values set to
%    GPCF from W and returns the modified W.
%
%    Assignment is inverse of  
%       w = [ log(gpcf.magnSigma2)
%             (hyperparameters of gpcf.magnSigma2)
%             log(gpcf.lengthScale(:))
%             (hyperparameters of gpcf.lengthScale)]'
%
%  See also
%    GPCF_EXP_PAK
  
  gpp=gpcf.p;
  if ~isempty(gpp.magnSigma2)
    gpcf.magnSigma2 = exp(w(1));
    w = w(2:end);
    % Hyperparameters of magnSigma2
    [p, w] = feval(gpcf.p.magnSigma2.fh.unpak, gpcf.p.magnSigma2, w);
    gpcf.p.magnSigma2 = p;
  end
  
  if isfield(gpcf,'metric')
    [metric, w] = feval(gpcf.metric.fh.unpak, gpcf.metric, w);
    gpcf.metric = metric;
  else            
    if ~isempty(gpp.lengthScale)
      i1=1;
      i2=length(gpcf.lengthScale);
      gpcf.lengthScale = exp(w(i1:i2));
      w = w(i2+1:end);
      % Hyperparameters of lengthScale
      [p, w] = feval(gpcf.p.lengthScale.fh.unpak, gpcf.p.lengthScale, w);
      gpcf.p.lengthScale = p;
    end
  end
  
end

function lp = gpcf_exp_lp(gpcf)
%GPCF_EXP_LP  Evaluate the log prior of covariance function parameters
%
%  Description
%    LP = GPCF_EXP_LP(GPCF, X, T) takes a covariance function
%    structure GPCF and returns log(p(th)), where th collects the
%    parameters.
%
%  See also
%    GPCF_EXP_PAK, GPCF_EXP_UNPAK, GPCF_EXP_LPG, GP_E

% Evaluate the prior contribution to the error. The parameters that
% are sampled are transformed, e.g., W = log(w) where w is all
% the "real" samples. On the other hand errors are evaluated in
% the W-space so we need take into account also the Jacobian of
% transformation, e.g., W -> w = exp(W). See Gelman et.al., 2004,
% Bayesian data Analysis, second edition, p24.
  lp = 0;
  gpp=gpcf.p;
  
  if ~isempty(gpcf.p.magnSigma2)
    lp = lp +feval(gpp.magnSigma2.fh.lp, gpcf.magnSigma2, ...
                   gpp.magnSigma2) +log(gpcf.magnSigma2);
  end
  
  if isfield(gpcf,'metric')
    lp = lp +feval(gpcf.metric.fh.lp, gpcf.metric);
  elseif ~isempty(gpp.lengthScale)
    lp = lp +feval(gpp.lengthScale.fh.lp, gpcf.lengthScale, ...
                   gpp.lengthScale) +sum(log(gpcf.lengthScale));
  end
end

function lpg = gpcf_exp_lpg(gpcf)
%GPCF_EXP_LPG  Evaluate gradient of the log prior with respect
%              to the parameters.
%
%  Description
%    LPG = GPCF_EXP_LPG(GPCF) takes a covariance function
%    structure GPCF and returns LPG = d log (p(th))/dth, where th
%    is the vector of parameters.
%
%  See also
%    GPCF_EXP_PAK, GPCF_EXP_UNPAK, GPCF_EXP_LP, GP_G

  lpg = [];
  gpp=gpcf.p;
  
  if ~isempty(gpcf.p.magnSigma2)            
    lpgs = feval(gpp.magnSigma2.fh.lpg, gpcf.magnSigma2, gpp.magnSigma2);
    lpg = [lpg lpgs(1).*gpcf.magnSigma2+1 lpgs(2:end)];
  end
  
  if isfield(gpcf,'metric')
    lpg_dist = feval(gpcf.metric.fh.lpg, gpcf.metric);
    lpg=[lpg lpg_dist];
  else
    if ~isempty(gpcf.p.lengthScale)
      lll = length(gpcf.lengthScale);
      lpgs = feval(gpp.lengthScale.fh.lpg, gpcf.lengthScale, gpp.lengthScale);
      lpg = [lpg lpgs(1:lll).*gpcf.lengthScale+1 lpgs(lll+1:end)];
    end
  end
  
end

function DKff = gpcf_exp_cfg(gpcf, x, x2, mask)
%GPCF_EXP_CFG  Evaluate gradient of covariance function
%                 with respect to the parameters
%
%  Description
%    DKff = GPCF_EXP_CFG(GPCF, X) takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns
%    DKff, the gradients of covariance matrix Kff = k(X,X) with
%    respect to th (cell array with matrix elements).
%
%    DKff = GPCF_EXP_CFG(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2) with respect to th (cell array with matrix
%    elements).
%
%    DKff = GPCF_EXP_CFG(GPCF, X, [], MASK) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the diagonal of gradients of covariance matrix
%    Kff = k(X,X2) with respect to th (cell array with matrix
%    elements). This is needed for example with FIC sparse
%    approximation.
%
%  See also
%    GPCF_EXP_PAK, GPCF_EXP_UNPAK, GPCF_EXP_LP, GP_G

  gpp=gpcf.p;
  [n, m] =size(x);

  i1=0;i2=1;
  DKff = {};

  % Evaluate: DKff{1} = d Kff / d magnSigma2
  %           DKff{2} = d Kff / d lengthScale
  % NOTE! Here we have already taken into account that the parameters
  % are transformed through log() and thus dK/dlog(p) = p * dK/dp
  % evaluate the gradient for training covariance
  if nargin == 2
    Cdm = gpcf_exp_trcov(gpcf, x);

    ii1=0;
    if ~isempty(gpcf.p.magnSigma2)
      ii1 = ii1 +1;
      DKff{ii1} = Cdm;
    end

    if isfield(gpcf,'metric')
      dist = feval(gpcf.metric.fh.dist, gpcf.metric, x);
      distg = feval(gpcf.metric.fh.distg, gpcf.metric, x);
      gprior_dist = feval(gpcf.metric.fh.lpg, gpcf.metric);
      for i=1:length(distg)
        ii1 = ii1+1;
        DKff{ii1} = -Cdm.*distg{i};
      end
    else
      if ~isempty(gpcf.p.lengthScale)
        % loop over all the lengthScales
        if length(gpcf.lengthScale) == 1
          % In the case of isotropic EXP (no ARD)
          s = 1./gpcf.lengthScale;
          dist = 0;
          for i=1:m
            dist = dist + (bsxfun(@minus,x(:,i),x(:,i)')).^2;
          end
          D = Cdm.*s.*sqrt(dist);
          ii1 = ii1+1;
          DKff{ii1} = D;
        else
          % In the case ARD is used
          s = 1./gpcf.lengthScale.^2;
          dist = 0;
          dist2 = 0;
          for i=1:m
            dist = dist + s(i).*(bsxfun(@minus,x(:,i),x(:,i)')).^2;
          end
          dist = sqrt(dist);
          for i=1:m                      
            D = s(i).*Cdm.*(bsxfun(@minus,x(:,i),x(:,i)')).^2;
            D(dist~=0) = D(dist~=0)./dist(dist~=0);
            ii1 = ii1+1;
            DKff{ii1} = D;
          end
        end
      end
    end
    % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
  elseif nargin == 3
    if size(x,2) ~= size(x2,2)
      error('gpcf_exp -> _ghyper: The number of columns in x and x2 has to be the same. ')
    end
    
    ii1=0;
    K = feval(gpcf.fh.cov, gpcf, x, x2);
    if ~isempty(gpcf.p.magnSigma2)
      ii1 = ii1 +1;
      DKff{ii1} = K;
    end
    
    if isfield(gpcf,'metric')                
      dist = feval(gpcf.metric.fh.dist, gpcf.metric, x, x2);
      distg = feval(gpcf.metric.fh.distg, gpcf.metric, x, x2);
      gprior_dist = feval(gpcf.metric.fh.lpg, gpcf.metric);
      for i=1:length(distg)
        ii1 = ii1+1;                    
        DKff{ii1} = -K.*distg{i};                    
      end
    else
      if ~isempty(gpcf.p.lengthScale)
        % Evaluate help matrix for calculations of derivatives with respect
        % to the lengthScale
        if length(gpcf.lengthScale) == 1
          % In the case of an isotropic EXP
          s = 1./gpcf.lengthScale;
          dist = 0;
          for i=1:m
            dist = dist + (bsxfun(@minus,x(:,i),x2(:,i)')).^2;
          end
          DK_l = s.*K.*sqrt(dist);
          ii1=ii1+1;
          DKff{ii1} = DK_l;
        else
          % In the case ARD is used
          s = 1./gpcf.lengthScale.^2;        % set the length
          dist = 0; 
          for i=1:m
            dist = dist + s(i).*(bsxfun(@minus,x(:,i),x2(:,i)')).^2;
          end
          dist = sqrt(dist);
          for i=1:m
            D1 = s(i).*K.* bsxfun(@minus,x(:,i),x2(:,i)').^2;
            D1(dist~=0) = D1(dist~=0)./dist(dist~=0);
            ii1=ii1+1;
            DKff{ii1} = D1;
          end
        end
      end
    end
    % Evaluate: DKff{1}    = d mask(Kff,I) / d magnSigma2
    %           DKff{2...} = d mask(Kff,I) / d lengthScale
  elseif nargin == 4
    ii1=0;
    
    if ~isempty(gpcf.p.magnSigma2)
      ii1 = ii1+1;
      DKff{ii1} = feval(gpcf.fh.trvar, gpcf, x);   % d mask(Kff,I) / d magnSigma2
    end

    if isfield(gpcf,'metric')
      dist = 0;
      distg = feval(gpcf.metric.fh.distg, gpcf.metric, x, [], 1);
      gprior_dist = feval(gpcf.metric.fh.lpg, gpcf.metric);
      for i=1:length(distg)
        ii1 = ii1+1;
        DKff{ii1} = 0;
      end
    else
      if ~isempty(gpcf.p.lengthScale)
        for i2=1:length(gpcf.lengthScale)
          ii1 = ii1+1;
          DKff{ii1}  = 0; % d mask(Kff,I) / d lengthScale
        end
      end
    end
  end
end

function DKff = gpcf_exp_ginput(gpcf, x, x2)
%GPCF_EXP_GINPUT  Evaluate gradient of covariance function with 
%                 respect to x.
%
%  Description
%    DKff = GPCF_EXP_GINPUT(GPCF, X) takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns
%    DKff, the gradients of covariance matrix Kff = k(X,X) with
%    respect to X (cell array with matrix elements)
%
%    DKff = GPCF_EXP_GINPUT(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors
%    and returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2) with respect to X (cell array with matrix elements).
%
%  See also
%    GPCF_EXP_PAK, GPCF_EXP_UNPAK, GPCF_EXP_LP, GP_G
  
  [n, m] =size(x);
  ii1 = 0;
  if nargin == 2
    K = feval(gpcf.fh.trcov, gpcf, x);
    if isfield(gpcf,'metric')
      dist = feval(gpcf.metric.fh.dist, gpcf.metric, x);
      gdist = feval(gpcf.metric.fh.ginput, gpcf.metric, x);
      for i=1:length(gdist)
        ii1 = ii1+1;
        DKff{ii1} = -K.*gdist{ii1};
      end
    else
      if length(gpcf.lengthScale) == 1
        % In the case of an isotropic EXP
        s = repmat(1./gpcf.lengthScale.^2, 1, m);
      else
        s = 1./gpcf.lengthScale.^2;
      end
      dist=0;
      for i2=1:m
        dist = dist + s(i2).*(bsxfun(@minus,x(:,i2),x(:,i2)')).^2;
      end
      dist = sqrt(dist); 
      for i=1:m
        for j = 1:n
          D1 = zeros(n,n);
          D1(j,:) = -s(i).*bsxfun(@minus,x(j,i),x(:,i)');
          D1 = D1 + D1';
          
          D1(dist~=0) = D1(dist~=0)./dist(dist~=0);
          DK = D1.*K;
          ii1 = ii1 + 1;
          DKff{ii1} = DK;
        end
      end
    end
    
  elseif nargin == 3
    [n2, m2] =size(x2);
    K = feval(gpcf.fh.cov, gpcf, x, x2);

    if isfield(gpcf,'metric')
      dist = feval(gpcf.metric.fh.dist, gpcf.metric, x, x2);
      gdist = feval(gpcf.metric.fh.ginput, gpcf.metric, x, x2);
      for i=1:length(gdist)
        ii1 = ii1+1;
        DKff{ii1}   = -K.*gdist{ii1};
      end
    else 
      if length(gpcf.lengthScale) == 1
        % In the case of an isotropic EXP
        s = repmat(1./gpcf.lengthScale.^2, 1, m);
      else
        s = 1./gpcf.lengthScale.^2;
      end
      
      dist=0;
      for i2=1:m
        dist = dist + s(i2).*(bsxfun(@minus,x(:,i2),x2(:,i2)')).^2;
      end
      dist = sqrt(dist); 
      for i=1:m
        for j = 1:n
          D1 = zeros(n,n2);
          D1(j,:) = -s(i).*bsxfun(@minus,x(j,i),x2(:,i)');
          
          D1(dist~=0) = D1(dist~=0)./dist(dist~=0);
          DK = D1.*K;
          ii1 = ii1 + 1;
          DKff{ii1} = DK;
        end
      end
    end
  end
end

function C = gpcf_exp_cov(gpcf, x1, x2)
%GP_EXP_COV  Evaluate covariance matrix between two input vectors
%
%  Description        
%    C = GP_EXP_COV(GP, TX, X) takes in covariance function of a
%    Gaussian process GP and two matrixes TX and X that contain
%    input vectors to GP. Returns covariance matrix C. Every
%    element ij of C contains covariance between inputs i in TX
%    and j in X.
%
%  See also
%    GPCF_EXP_TRCOV, GPCF_EXP_TRVAR, GP_COV, GP_TRCOV
  
  if isempty(x2)
    x2=x1;
  end
  [n1,m1]=size(x1);
  [n2,m2]=size(x2);

  if m1~=m2
    error('the number of columns of X1 and X2 has to be same')
  end

  if isfield(gpcf,'metric')
    dist = feval(gpcf.metric.fh.dist, gpcf.metric, x1, x2);
    dist(dist<eps) = 0;
    C = gpcf.magnSigma2.*exp(-dist);
  else
    C=zeros(n1,n2);
    ma2 = gpcf.magnSigma2;
    
    % Evaluate the covariance
    if ~isempty(gpcf.lengthScale)  
      s2 = 1./gpcf.lengthScale.^2;
      % If ARD is not used make s a vector of 
      % equal elements 
      if size(s2)==1
        s2 = repmat(s2,1,m1);
      end
      dist=zeros(n1,n2);
      for j=1:m1
        dist = dist + s2(j).*(bsxfun(@minus,x1(:,j),x2(:,j)')).^2;
      end
      C = ma2.*exp(-sqrt(dist));
    end
    C(C<eps)=0;
  end
end

function C = gpcf_exp_trcov(gpcf, x)
%GP_EXP_TRCOV  Evaluate training covariance matrix of inputs
%
%  Description
%    C = GP_EXP_TRCOV(GP, TX) takes in covariance function of a
%    Gaussian process GP and matrix TX that contains training
%    input vectors. Returns covariance matrix C. Every element ij
%    of C contains covariance between inputs i and j in TX.
%
%
%  See also
%    GPCF_EXP_COV, GPCF_EXP_TRVAR, GP_COV, GP_TRCOV

  if isfield(gpcf,'metric')
    % If other than scaled euclidean metric
    dist = feval(gpcf.metric.fh.dist, gpcf.metric, x);
    dist(dist<eps) = 0;
    C = gpcf.magnSigma2.*exp(-dist);
  else
    % If scaled euclidean metric
    % Try to use the C-implementation            
    C = trcov(gpcf, x);
    if isnan(C)
      % If there wasn't C-implementation do here
      [n, m] =size(x);
      
      s = 1./(gpcf.lengthScale);
      s2 = s.^2;
      if size(s)==1
        s2 = repmat(s2,1,m);
      end
      ma2 = gpcf.magnSigma2;
      
      % Here we take advantage of the
      % symmetry of covariance matrix
      C=zeros(n,n);
      for i1=2:n
        i1n=(i1-1)*n;
        for i2=1:i1-1
          ii=i1+(i2-1)*n;
          for i3=1:m
            C(ii)=C(ii)+s2(i3).*(x(i1,i3)-x(i2,i3)).^2;       % the covariance function
          end
          C(i1n+i2)=C(ii);
        end
      end
      C = ma2.*exp(-sqrt(C));
      C(C<eps)=0;
    end
  end
end

function C = gpcf_exp_trvar(gpcf, x)
%GP_EXP_TRVAR  Evaluate training variance vector
%
%  Description
%    C = GP_EXP_TRVAR(GPCF, TX) takes in covariance function of a
%    Gaussian process GPCF and matrix TX that contains training
%    inputs. Returns variance vector C. Every element i of C
%    contains variance of input i in TX.
%
%  See also
%    GPCF_EXP_COV, GP_COV, GP_TRCOV

  [n, m] =size(x);

  C = ones(n,1).*gpcf.magnSigma2;
  C(C<eps)=0;
end

function reccf = gpcf_exp_recappend(reccf, ri, gpcf)
%RECAPPEND  Record append
%
%  Description
%    RECCF = GPCF_EXP_RECAPPEND(RECCF, RI, GPCF) takes a
%    covariance function record structure RECCF, record index RI
%    and covariance function structure GPCF with the current MCMC
%    samples of the parameters. Returns RECCF which contains
%    all the old samples and the current samples from GPCF .
%
%  See also
%    GP_MC and GP_MC -> RECAPPEND

% Initialize record
  if nargin == 2
    reccf.type = 'gpcf_exp';

    % Initialize parameters
    reccf.lengthScale= [];
    reccf.magnSigma2 = [];

    % Set the function handles
    reccf.fh.pak = @gpcf_exp_pak;
    reccf.fh.unpak = @gpcf_exp_unpak;
    reccf.fh.e = @gpcf_exp_lp;
    reccf.fh.lpg = @gpcf_exp_lpg;
    reccf.fh.cfg = @gpcf_exp_cfg;
    reccf.fh.cov = @gpcf_exp_cov;
    reccf.fh.trcov  = @gpcf_exp_trcov;
    reccf.fh.trvar  = @gpcf_exp_trvar;
    reccf.fh.recappend = @gpcf_exp_recappend;  
    reccf.p=[];
    reccf.p.lengthScale=[];
    reccf.p.magnSigma2=[];
    if isfield(ri.p,'lengthScale') && ~isempty(ri.p.lengthScale)
      reccf.p.lengthScale = ri.p.lengthScale;
    end
    if ~isempty(ri.p.magnSigma2)
      reccf.p.magnSigma2 = ri.p.magnSigma2;
    end
    return
  end

  gpp = gpcf.p;

  if ~isfield(gpcf,'metric')
    % record lengthScale
    if ~isempty(gpcf.lengthScale)
      reccf.lengthScale(ri,:)=gpcf.lengthScale;
      reccf.p.lengthScale = feval(gpp.lengthScale.fh.recappend, reccf.p.lengthScale, ri, gpcf.p.lengthScale);
    elseif ri==1
      reccf.lengthScale=[];
    end
  end
  % record magnSigma2
  if ~isempty(gpcf.magnSigma2)
    reccf.magnSigma2(ri,:)=gpcf.magnSigma2;
    reccf.p.magnSigma2 = feval(gpp.magnSigma2.fh.recappend, reccf.p.magnSigma2, ri, gpcf.p.magnSigma2);
  elseif ri==1
    reccf.magnSigma2=[];
  end
end
