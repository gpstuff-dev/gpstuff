function gpcf = gpcf_additive(varargin)
%GPCF_ADDITIVE  Create a mixture over products of kernels for each dimension
%
%  Description
%    GPCF = GPCF_ADDITIVE('PARAM1',VALUE1, 'PARAM2,VALUE2, ...)
%    creates a mixture over all possible product combinations of given
%    covariance functions for each input dimension separately.
%
%    Parameters [default]:
%      cf           - cell array {CF_1, CF_2, ... , CF_N} of covariance
%                     functions for each dimension [no default]
%      max_deg      - maximum order of interaction (must be less or equal
%                     to the number of covariance functions N) [2]
%      sigma2       - 1D array of length max_deg defining the variance for
%                     each order of interaction [0.1*ones]
%      sigma2_prior - prior for sigma2 [prior_logunif]
%
%    For example N = 3, max_deg = 2:
%      k1 = CF_1 + CF_2 + CF_3
%      k2 = CF_1*CF_2 + CF_1*CF_3 + CF_2*CF_3
%      k3 = CF_1*CF_2*CF_3
%      GPCF = sigam2(1)*k1 + sigam2(2)*k2,
%
%  See also
%    GP_SET, GPCF_*
%
%  References:
%    Duvenaud, D. K., Nickisch, H., & Rasmussen, C. E. (2011). Additive
%    Gaussian processes. In Advances in neural information processing
%    systems (pp. 226-234).

% Copyright (c) 2009-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari
% Copyright (c) 2014 Tuomas Sivula

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPCF_ADDITIVE';
  ip.addOptional('gpcf', [], @isstruct);
  ip.addParamValue('cf',[], @iscell);
  ip.addParamValue('max_deg', 2, @(x) isscalar(x) && isnumeric(x) ...
                                      && x>0 && mod(x,1) == 0);
  ip.addParamValue('sigma2', [], @(x) isnumeric(x) && isvector(x) && all(x>0));
  ip.addParamValue('sigma2_prior', prior_logunif(), ...
                   @(x) isstruct(x) || isempty(x))
  ip.parse(varargin{:});
  gpcf=ip.Results.gpcf;

  if isempty(gpcf)
    init=true;
    gpcf.type = 'gpcf_additive';
  else
    if ~isfield(gpcf,'type') && ~isequal(gpcf.type,'gpcf_additive')
      error('First argument does not seem to be a valid covariance function structure')
    end
    init=false;
  end
  
  % Initialize parameters
  
  % Kernels
  if init || ~ismember('cf',ip.UsingDefaults)
    if length(ip.Results.cf) < 2
      error('At least two covariance functions has to be given in cf');
    end
    gpcf.cf = ip.Results.cf;
  end
  ncf = length(gpcf.cf);
  
  % Max degree
  if init || ~ismember('max_deg',ip.UsingDefaults)
    if ip.Results.max_deg <= ncf
      gpcf.max_deg = ip.Results.max_deg;
    else
      warning('max_deg in additive kernel can not be greater than number of dimensions, max_deg truncated.');
      gpcf.max_deg = ncf;
    end
  end
  
  % Degree variances
  if init || ~ismember('sigma2',ip.UsingDefaults)
    if isempty(ip.Results.sigma2)
      gpcf.sigma2 = 0.1*ones(1,gpcf.max_deg);
    elseif length(ip.Results.sigma2) == gpcf.max_deg
      gpcf.sigma2 = ip.Results.sigma2;
    else
      error('Wrong number of elements in degree variance parameter vector sigma2')
    end
    % Ensure the right direction
    if size(gpcf.sigma2,1) ~= 1
      gpcf.sigma2 = gpcf.sigma2';
    end
  end
  
  % Degree variance priors
  if init || ~ismember('sigma2_prior',ip.UsingDefaults)
    gpcf.p.sigma2 = ip.Results.sigma2_prior;
  end
  
  % Ensure sigma2 matches max_deg
  if length(gpcf.sigma2) ~= gpcf.max_deg
    error('Parameters sigma2 and max_deg sizes does not match')
  end
  
  if init
    % Set the function handles to the subfunctions
    gpcf.fh.pak = @gpcf_additive_pak;
    gpcf.fh.unpak = @gpcf_additive_unpak;
    gpcf.fh.lp = @gpcf_additive_lp;
    gpcf.fh.lpg = @gpcf_additive_lpg;
    gpcf.fh.cfg = @gpcf_additive_cfg;
    gpcf.fh.ginput = @gpcf_additive_ginput;
    gpcf.fh.cov = @gpcf_additive_cov;
    gpcf.fh.trcov  = @gpcf_additive_trcov;
    gpcf.fh.trvar  = @gpcf_additive_trvar;
    gpcf.fh.recappend = @gpcf_additive_recappend;
  end

end

function [w, s, h] = gpcf_additive_pak(gpcf)
%GPCF_ADDITIVE_PAK  Combine GP covariance function parameters into one vector
%
%  Description
%    W = GPCF_ADDITIVE_PAK(GPCF, W) loops through all the covariance
%    functions and packs their parameters into one vector as
%    described in the respective functions. This is a mandatory 
%    subfunction used for example in energy and gradient computations.
%
%  See also
%    GPCF_ADDITIVE_UNPAK
  
  ncf = length(gpcf.cf);
  w = []; s = {}; h=[];
  
  if ~isempty(gpcf.p.sigma2)
    w = [w log(gpcf.sigma2)];
    s = [s; sprintf('log(additive.sigma2 x %d)',numel(gpcf.sigma2))];
    h = [h ones(1,numel(gpcf.sigma2))];
    % Hyperparameters of lengthScale
    [wh, sh, hh] = gpcf.p.sigma2.fh.pak(gpcf.p.sigma2);
    sh=strcat(repmat('prior-', size(sh,1),1),sh);
    w = [w wh];
    s = [s; sh];
    h = [h 1+hh];
  end
  
  for i=1:ncf
    cf = gpcf.cf{i};
    [wi, si, hi] = cf.fh.pak(cf);
    w = [w wi];
    s = [s; si];
    h = [h hi];
  end
end

function [gpcf, w] = gpcf_additive_unpak(gpcf, w)
%GPCF_ADDITIVE_UNPAK  Sets the covariance function parameters into
%                 the structures
%
%  Description
%    [GPCF, W] = GPCF_ADDITIVE_UNPAK(GPCF, W) loops through all the
%    covariance functions and unpacks their parameters from W to
%    each covariance function structure. This is a mandatory 
%    subfunction used for example in energy and gradient computations.
% 
%  See also
%    GPCF_ADDITIVE_PAK
%
  ncf = length(gpcf.cf);
  
  if isfield(gpcf.p,'sigma2') && ~isempty(gpcf.p.sigma2)
    i1=1;
    i2=length(gpcf.sigma2);
    gpcf.sigma2 = exp(w(i1:i2));
    w = w(i2+1:end);
    % Hyperparameters of lengthScale
    [p, w] = gpcf.p.sigma2.fh.unpak(gpcf.p.sigma2, w);
    gpcf.p.sigma2 = p;
  end
  
  for i=1:ncf
    cf = gpcf.cf{i};
    [cf, w] = cf.fh.unpak(cf, w);
    gpcf.cf{i} = cf;
  end

end

function lp = gpcf_additive_lp(gpcf)
%GPCF_ADDITIVE_LP  Evaluate the log prior of covariance function parameters
%
%  Description
%    LP = GPCF_ADDITIVE_LP(GPCF, X, T) takes a covariance function
%    structure GPCF and returns log(p(th)), where th collects the
%    parameters. This is a mandatory subfunction used for example 
%    in energy computations.
%
%  See also
%    GPCF_ADDITIVE_PAK, GPCF_ADDITIVE_UNPAK, GPCF_ADDITIVE_LPG, GP_E
  
  lp = 0;
  gpp = gpcf.p;

  if isfield(gpp,'sigma2') && ~isempty(gpp.sigma2)
    lp = lp +gpp.sigma2.fh.lp(gpcf.sigma2, gpp.sigma2) +sum(log(gpcf.sigma2));
  end
  
  ncf = length(gpcf.cf);
  for i=1:ncf
    cf = gpcf.cf{i};
    lp = lp + cf.fh.lp(cf);
  end
  
end

function lpg = gpcf_additive_lpg(gpcf)
%GPCF_ADDITIVE_LPG  Evaluate gradient of the log prior with respect
%               to the parameters.
%
%  Description
%    LPG = GPCF_ADDITIVE_LPG(GPCF) takes a covariance function
%    structure GPCF and returns LPG = d log (p(th))/dth, where th
%    is the vector of parameters. This is a mandatory subfunction 
%    used for example in gradient computations.
%
%  See also
%    GPCF_ADDITIVE_PAK, GPCF_ADDITIVE_UNPAK, GPCF_ADDITIVE_LP, GP_G
  
  % Evaluate the gradients
  lpg = [];
  gpp=gpcf.p;
  
  if isfield(gpp,'sigma2') && ~isempty(gpp.sigma2)
    lll = length(gpcf.sigma2);
    lpgs = gpp.sigma2.fh.lpg(gpcf.sigma2, gpp.sigma2);
    lpg = [lpg lpgs(1:lll).*gpcf.sigma2+1 lpgs(lll+1:end)];
  end
  
  ncf = length(gpcf.cf);
  for i=1:ncf
    cf = gpcf.cf{i};
    lpg=[lpg cf.fh.lpg(cf)];
  end

end



function es = degrees(r, zs, inds)
% DEGREEs - Internal function used to calculate the covariance degree
% components e_k(z_inds(1), ..., z_inds(n))
%
%   Parameters:
%     r    - max degree
%     zs   - all the covariances of the underlying kernels
%     inds - indexes of the used kernels (empty means all)
%
%   N.B. Here all Cs and inds are given instead using slicing because the
%   latter is memory inefficient. Compare the memory usage in the
%   following:
%
%     inds = 1:100;
%     zs = rand(1000,1000,100);
%     
%     % Memory inefficient
%     t = sum(zs(:,:,inds(inds~=13)),3);
%
%     % Memory efficient
%     t = zeros(1000,1000);
%     for i = inds(inds~=13)
%       t = t + zs(:,:,i);
%     end
%     
  
  if isempty(inds)
    
    ncf = size(zs,3);
    
    if r == 1
      % Only one degree (implemented here for completeness)
      es = sum(zs,3);

    elseif r == 2
      % Only two degrees (implemented here for completeness)
      es = zeros(size(zs,1),size(zs,2),r);
      es(:,:,1) = sum(zs,3);
      for i1 = 1:ncf-1
        for i2 = i1+1:ncf
          es(:,:,2) = es(:,:,2) + zs(:,:,i1).*zs(:,:,i2);
        end
      end

    elseif r > 2
      % Over two degrees ... use Newton-Girard formulae
      sk = zeros(size(zs,1),size(zs,2),r);
      sk(:,:,1) = sum(zs,3);
      for i = 2:r
        sk(:,:,i) = sum(zs.^i, 3);
      end
      es = zeros(size(zs,1),size(zs,2),r);
      es(:,:,1) = sk(:,:,1);
      for i = 2:r
        presign = true;
        for k = 1:i-1
          if presign
            es(:,:,i) = es(:,:,i) + es(:,:,i-k).*sk(:,:,k);
            presign = false;
          else
            es(:,:,i) = es(:,:,i) - es(:,:,i-k).*sk(:,:,k);
            presign = true;
          end
        end
        if presign
          es(:,:,i) = es(:,:,i) + sk(:,:,i);
        else
          es(:,:,i) = es(:,:,i) - sk(:,:,i);
        end
        es(:,:,i) = es(:,:,i)./i;
      end

    else
      error('Invalid max_deg parameter')
    end
  
  else
    % Use only selected kernels
    
    % Check direction
    if r > length(inds)
      error('Invalid max degree relative to inds')
    end
    if size(inds,1) ~= 1
      inds = inds';
      if size(inds,1) ~= 1
        error('Invalid parameter inds')
      end
    end
    
    if r == 1
      % Only one degree
      es = 0;
      for i = inds
        es = es + zs(:,:,i);
      end

    elseif r == 2
      % Only two degrees
      es = zeros(size(zs,1),size(zs,2),r);
      for i = inds
        es(:,:,1) = es(:,:,1) + zs(:,:,i);
      end
      for i1 = 1:length(inds)-1
        for i2 = i1+1:length(inds)
          es(:,:,2) = es(:,:,2) + zs(:,:,inds(i1)).*zs(:,:,inds(i2));
        end
      end

    elseif r > 2
      % Over two degrees ... use Newton-Girard formulae
      sk = zeros(size(zs,1),size(zs,2),r);
      for i = inds
        sk(:,:,1) = sk(:,:,1) + zs(:,:,i);
      end
      for i = 2:r
        for j = inds
          sk(:,:,i) = sk(:,:,i) + zs(:,:,j).^i;
        end
      end
      es = zeros(size(zs,1),size(zs,2),r);
      es(:,:,1) = sk(:,:,1);
      for i = 2:r
        presign = true;
        for k = 1:i-1
          if presign
            es(:,:,i) = es(:,:,i) + es(:,:,i-k).*sk(:,:,k);
            presign = false;
          else
            es(:,:,i) = es(:,:,i) - es(:,:,i-k).*sk(:,:,k);
            presign = true;
          end
        end
        if presign
          es(:,:,i) = es(:,:,i) + sk(:,:,i);
        else
          es(:,:,i) = es(:,:,i) - sk(:,:,i);
        end
        es(:,:,i) = es(:,:,i)./i;
      end

    else
      error('Invalid max_deg parameter')
    end
    
  end
  
end



function DKff = gpcf_additive_cfg(gpcf, x, x2, mask, i1)
%GPCF_ADDITIVE_CFG  Evaluate gradient of covariance function
%               with respect to the parameters.
%
%  Description
%    DKff = GPCF_ADDITIVE_CFG(GPCF, X) takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns
%    DKff, the gradients of covariance matrix Kff = k(X,X) with
%    respect to th (cell array with matrix elements). This is a 
%    mandatory subfunction used in gradient computations.
%
%    DKff = GPCF_ADDITIVE_CFG(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2) with respect to th (cell array with matrix
%    elements). This subfunction is needed when using sparse 
%    approximations (e.g. FIC).
%
%    DKff = GPCF_ADDITIVE_CFG(GPCF, X, [], MASK) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the diagonal of gradients of covariance matrix
%    Kff = k(X,X2) with respect to th (cell array with matrix
%    elements). This subfunction is needed when using sparse 
%    approximations (e.g. FIC).
%
%    DKff = GPCF_ADDITIVE_CFG(GPCF, X, X2, [], i) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2), or k(X,X) if X2 is empty, with respect to ith 
%    hyperparameter. This subfunction is needed when using 
%    memory save option in gp_set.
%
%  See also
%    GPCF_ADDITIVE_PAK, GPCF_ADDITIVE_UNPAK, GPCF_ADDITIVE_LP, GP_G

  [n, m] =size(x);
  ncf = length(gpcf.cf);
  r = gpcf.max_deg;

  DKff = {};

  if nargin==5
    % Use memory save option
    savememory=1;
    i3 = zeros(1,ncf);
    for k=1:ncf
      % Number of hyperparameters for each covariance function
      cf = gpcf.cf{k};
      i3(k) = cf.fh.cfg(cf,[],[],[],0);
    end
    if i1==0
      % Return number of hyperparameters
      DKff = sum(i3) + gpcf.max_deg;
      return
    end
    if i1 > gpcf.max_deg
      i1 = i1 - gpcf.max_deg;
      % The parameter belongs to one of the underlying kernels
      % Now i1 is [kernel_index, param_index_in_that_kernel]
      i3 = cumsum(i3);
      ind = find(i3 >= i1, 1);
      if ind > 1
        i1 = [ind, i1-i3(ind-1)];
      else
        i1 = [ind, i1];
      end
    end
  else
    savememory=0;
  end

  % Evaluate: DKff{1} = d Kff / d magnSigma2
  %           DKff{2} = d Kff / d lengthScale
  % NOTE! Here we have already taken into account that the parameters are transformed
  % through log() and thus dK/dlog(p) = p * dK/dp

  % evaluate the gradient for training covariance
  if nargin == 2 || (isempty(x2) && isempty(mask))
    
    % evaluate the individual covariance functions
    zs = zeros(n,n,ncf);
    for i=1:ncf
      cf = gpcf.cf{i};
      zs(:,:,i) = cf.fh.trcov(cf, x(:,i));
    end
    
    % Evaluate the gradients
    ind = 1:ncf;
    
    if ~savememory
      
      DKff = {};
      
      % Order variances sigma2
      es = degrees(r, zs, []);
      for i = 1:r
        % dlog(p) ... See NOTE above
        DKff{end+1} = gpcf.sigma2(i).*es(:,:,i);
      end
      
      % Subkernel hyperparameters
      for i=1:ncf
        cf = gpcf.cf{i};
        DK = cf.fh.cfg(cf, x(:,i));
        
        CC = gpcf.sigma2(1).*ones(n,n);
        if r > 1
          es = degrees(r-1, zs, ind(ind~=i));
          for j = 2:r
            CC = CC + gpcf.sigma2(j).*es(:,:,j-1);
          end
        end
        for j = 1:length(DK)
          DKff{end+1} = DK{j}.*CC;
        end
      end
      
    else
      if length(i1) == 1
        % Order variance sigma2
        es = degrees(i1, zs, []);
        % dlog(p) ... See NOTE above
        DKff = gpcf.sigma2(i1).*es(:,:,end);
      else
        
        cf = gpcf.cf{i1(1)};
        DK = cf.fh.cfg(cf,x(:,i1(1)),[],[],i1(2));

        CC = gpcf.sigma2(1).*ones(n,n);
        if r > 1
          es = degrees(r-1, zs, ind(ind~=i1(1)));
          for j = 2:r
            CC = CC + gpcf.sigma2(j).*es(:,:,j-1);
          end
        end
        DKff = DK.*CC;
        
      end
      
    end
    
    
    % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
  elseif nargin == 3 || isempty(mask)
    if size(x,2) ~= size(x2,2)
      error('gpcf_prod -> _ghyper: The number of columns in x and x2 has to be the same. ')
    end
        
    % evaluate the individual covariance functions
    zs = zeros(size(x,1),size(x2,1),ncf);
    for i=1:ncf
      cf = gpcf.cf{i};
      zs(:,:,i) = cf.fh.cov(cf, x(:,i), x2(:,i));
    end
    
    % Evaluate the gradients
    ind = 1:ncf;
    
    if ~savememory
      
      DKff = {};
      
      % Order variances sigma2
      es = degrees(r, zs, []);
      for i = 1:r
        % dlog(p) ... See NOTE above
        DKff{end+1} = gpcf.sigma2(i).*es(:,:,i);
      end
      
      % Subkernel hyperparameters
      for i=1:ncf
        cf = gpcf.cf{i};
        DK = cf.fh.cfg(cf, x(:,i), x2(:,i));
        
        CC = gpcf.sigma2(1).*ones(n,n);
        if r > 1
          es = degrees(r-1, zs, ind(ind~=i));
          for j = 2:r
            CC = CC + gpcf.sigma2(j).*es(:,:,j-1);
          end
        end
        for j = 1:length(DK)
          DKff{end+1} = DK{j}.*CC;
        end
      end
      
    else
      if length(i1) == 1
        % Order variance sigma2
        es = degrees(i1, zs, []);
        % dlog(p) ... See NOTE above
        DKff = gpcf.sigma2(i1).*es(:,:,end);
      else
        
        cf = gpcf.cf{i1(1)};
        DK = cf.fh.cfg(cf, x(:,i), x2(:,i), [], i1(2));

        CC = gpcf.sigma2(1).*ones(n,n);
        if r > 1
          es = degrees(r-1, zs, ind(ind~=i1(1)));
          for j = 2:r
            CC = CC + gpcf.sigma2(j).*es(:,:,j-1);
          end
        end
        DKff = DK.*CC;
        
      end
      
    end
    
    
    
    % Evaluate: DKff{1}    = d mask(Kff,I) / d magnSigma2
    %           DKff{2...} = d mask(Kff,I) / d lengthScale
  elseif nargin == 4 || nargin == 5
    
    % evaluate the individual covariance functions
    zs = zeros(size(x,1),ncf);
    for i=1:ncf
      cf = gpcf.cf{i};
      zs(:,i) = cf.fh.trvar(cf, x(:,i));
    end
    
    % Evaluate the gradients
    ind = 1:ncf;
    
    if ~savememory
      
      DKff = {};
      
      % Order variances sigma2
      es = degrees(r, zs, []);
      for i = 1:r
        % dlog(p) ... See NOTE above
        DKff{end+1} = gpcf.sigma2(i).*es(:,:,i);
      end
      
      % Subkernel hyperparameters
      for i=1:ncf
        cf = gpcf.cf{i};
        DK = cf.fh.cfg(cf, x(:,i), [], 1);
        
        CC = gpcf.sigma2(1).*ones(n,n);
        if r > 1
          es = degrees(r-1, zs, ind(ind~=i));
          for j = 2:r
            CC = CC + gpcf.sigma2(j).*es(:,:,j-1);
          end
        end
        for j = 1:length(DK)
          DKff{end+1} = DK{j}.*CC;
        end
      end
      
    else
      if length(i1) == 1
        % Order variance sigma2
        es = degrees(i1, zs, []);
        % dlog(p) ... See NOTE above
        DKff = gpcf.sigma2(i1).*es(:,:,end);
      else
        
        cf = gpcf.cf{i1(1)};
        DK = cf.fh.cfg(cf, x(:,i1(1)), [], 1, i1(2));

        CC = gpcf.sigma2(1).*ones(n,n);
        if r > 1
          es = degrees(r-1, zs, ind(ind~=i1(1)));
          for j = 2:r
            CC = CC + gpcf.sigma2(j).*es(:,:,j-1);
          end
        end
        DKff = DK.*CC;
        
      end
      
    end
    
    
  end
  
end


function DKff = gpcf_additive_ginput(gpcf, x, x2, i1)
%GPCF_ADDITIVE_GINPUT  Evaluate gradient of covariance function with 
%                  respect to x
%
%  Description
%    DKff = GPCF_ADDITIVE_GINPUT(GPCF, X) takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns
%    DKff, the gradients of covariance matrix Kff = k(X,X) with
%    respect to X (cell array with matrix elements). This subfunction 
%    is needed when computing gradients with respect to inducing 
%    inputs in sparse approximations.
%
%    DKff = GPCF_ADDITIVE_GINPUT(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2) with respect to X (cell array with matrix elements).
%    This subfunction is needed when computing gradients with 
%    respect to inducing inputs in sparse approximations.
%
%    DKff = GPCF_ADDITIVE_GINPUT(GPCF, X, X2, i) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2), or k(X,X) if X2 is empty, with respect to ith
%    covariate in X (cell array with matrix elements). This
%    subfunction is needed when using memory save option in
%    gp_set.
%
%  See also
%    GPCF_ADDITIVE_PAK, GPCF_ADDITIVE_UNPAK, GPCF_ADDITIVE_LP, GP_G
  
  [n, m] =size(x);
  r = gpcf.max_deg;
  
  if nargin==4
    % Use memory save option
    savememory=1;
    if i1==0
      % Return number of covariates
      if isfield(gpcf,'selectedVariables')
        DKff=length(gpcf.selectedVariables);
      else
        DKff=m;
      end
      return
    end
  else
    savememory=0;
  end
  % evaluate the gradient for training covariance
  if nargin == 2 || isempty(x2)
    
    ncf = length(gpcf.cf);
    
    % evaluate the individual covariance functions
    zs = zeros(n,n,ncf);
    for i=1:ncf
      cf = gpcf.cf{i};
      zs(:,:,i) = cf.fh.trcov(cf, x(:,i));
    end
    
    % Evaluate the gradients
    ind = 1:ncf;
    if ~savememory
      DKff=cellfun(@(a) zeros(n,n), cell(1,numel(x)), 'UniformOutput', 0);
    else
      DKff=cellfun(@(a) zeros(n,n), cell(1,n), 'UniformOutput', 0);
    end
    for i=1:ncf
      cf = gpcf.cf{i};
      if ~savememory
        DK = cf.fh.ginput(cf, x(:,i));
      else
        DK = cf.fh.ginput(cf, x(:,i), [], i1);
      end
      
      CC = gpcf.sigma2(1).*ones(n,n);
      if r > 1
        es = degrees(r-1, zs, ind(ind~=i));
        for j = 2:r
          CC = CC + gpcf.sigma2(j).*es(:,:,j-1);
        end
      end      
      for j = 1:length(DK)
        DKff{j} = DKff{j} + DK{j}.*CC;
      end
    end

    % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
  elseif nargin == 3 ||  nargin == 4
    if size(x,2) ~= size(x2,2)
      error('gpcf_prod -> _ghyper: The number of columns in x and x2 has to be the same. ')
    end
    
    ncf = length(gpcf.cf);
    
    % evaluate the individual covariance functions
    zs = zeros(n,n,ncf);
    for i=1:ncf
      cf = gpcf.cf{i};
      zs(:,:,i) = cf.fh.cov(cf, x(:,i), x2(:,i));
    end
    
    % Evaluate the gradients
    ind = 1:ncf;
    if ~savememory
      DKff=cellfun(@(a) zeros(n,n), cell(1,numel(x)), 'UniformOutput', 0);
    else
      DKff=cellfun(@(a) zeros(n,n), cell(1,n), 'UniformOutput', 0);
    end
    for i=1:ncf
      cf = gpcf.cf{i};
      if ~savememory
        DK = cf.fh.ginput(cf, x(:,i), x2(:,i));
      else
        DK = cf.fh.ginput(cf, x(:,i), x2(:,i), i1);
      end
      
      CC = gpcf.sigma2(1).*ones(n,n);
      if r > 1
        es = degrees(r-1, zs, ind(ind~=i));
        for j = 2:r
          CC = CC + gpcf.sigma2(j).*es(:,:,j-1);
        end
      end      
      for j = 1:length(DK)
        DKff{j} = DKff{j} + DK{j}.*CC;
      end
    end
  end
  
end


function C = gpcf_additive_cov(gpcf, x1, x2)
%GP_ADDITIVE_COV  Evaluate covariance matrix between two input vectors
%
%  Description         
%    C = GP_ADDITIVE_COV(GP, TX, X) takes in covariance function of a
%    Gaussian process GP and two matrixes TX and X that contain
%    input vectors to GP. Returns covariance matrix C. Every
%    element ij of C contains covariance between inputs i in TX
%    and j in X. This is a mandatory subfunction used for example in
%    prediction and energy computations.
%
%
%  See also
%    GPCF_ADDITIVE_TRCOV, GPCF_ADDITIVE_TRVAR, GP_COV, GP_TRCOV
  
  if isempty(x2)
    x2=x1;
  end
  [n1,m1]=size(x1);
  [n2,m2]=size(x2);

  if m1~=m2
    error('the number of columns of X1 and X2 has to be same')
  end

  ncf = length(gpcf.cf);
  
  if m1~=ncf 
    error('input dimension does not match with number of additive kernels')
  end
  
  r = gpcf.max_deg;
  
  if r == 1
    % Only one degree
    C = 0;
    for d = 1:ncf
      cf = gpcf.cf{d};
      C = C + cf.fh.cov(cf, x1(:,d), x2(:,d));
    end
    C = C.*gpcf.sigma2(1);
  
  elseif r == 2
    % Only two degrees
    zs = zeros(n1,n2,ncf);
    for d = 1:ncf
      cf = gpcf.cf{d};
      zs(:,:,d) = cf.fh.cov(cf, x1(:,d), x2(:,d));
    end
    C = 0;
    for i1 = 1:ncf-1
      for i2 = i1+1:ncf
        C = C + zs(:,:,i1).*zs(:,:,i2);
      end
    end
    C = C.*gpcf.sigma2(2);
    C = C + sum(zs,3).*gpcf.sigma2(1);
    
  elseif r > 2
    % Over two degrees ... use Newton-Girard formulae
    zs = zeros(n1,n2,ncf);
    for d = 1:ncf
      cf = gpcf.cf{d};
      zs(:,:,d) = cf.fh.cov(cf, x1(:,d), x2(:,d));
    end
    sk = zeros(n1,n2,r);
    sk(:,:,1) = sum(zs,3);
    for i = 2:r
      sk(:,:,i) = sum(zs.^i, 3);
    end
    clear zs
    es = zeros(n1,n2,r);
    es(:,:,1) = sk(:,:,1);
    for i = 2:r
      presign = true;
      for k = 1:i-1
        if presign
          es(:,:,i) = es(:,:,i) + es(:,:,i-k).*sk(:,:,k);
          presign = false;
        else
          es(:,:,i) = es(:,:,i) - es(:,:,i-k).*sk(:,:,k);
          presign = true;
        end
      end
      if presign
        es(:,:,i) = es(:,:,i) + sk(:,:,i);
      else
        es(:,:,i) = es(:,:,i) - sk(:,:,i);
      end
      es(:,:,i) = es(:,:,i)./i;
    end
    for i = 1:r
      es(:,:,i) = es(:,:,i).*gpcf.sigma2(i);
    end
    C = sum(es,3);
    
  else
    error('Invalid max_deg parameter')
  end
       
end

function C = gpcf_additive_trcov(gpcf, x)
%GP_ADDITIVE_TRCOV     Evaluate training covariance matrix of inputs
%
%  Description
%    C = GP_ADDITIVE_TRCOV(GP, TX) takes in covariance function of a
%    Gaussian process GP and matrix TX that contains training
%    input vectors. Returns covariance matrix C. Every element ij
%    of C contains covariance between inputs i and j in TX. This 
%    is a mandatory subfunction used for example in prediction and 
%    energy computations.
%
%  See also
%    GPCF_ADDITIVE_COV, GPCF_ADDITIVE_TRVAR, GP_COV, GP_TRCOV
  
  [n,m]=size(x);

  ncf = length(gpcf.cf);
  
  if m~=ncf 
    error('input dimension does not match with number of additive kernels')
  end
  
  r = gpcf.max_deg;
  
  if r == 1
    % Only one degree
    C = 0;
    for d = 1:ncf
      cf = gpcf.cf{d};
      C = C + cf.fh.trcov(cf, x(:,d));
    end
    C = C.*gpcf.sigma2(1);
  
  elseif r == 2
    % Only two degrees
    zs = zeros(n,n,ncf);
    for d = 1:ncf
      cf = gpcf.cf{d};
      zs(:,:,d) = cf.fh.trcov(cf, x(:,d));
    end
    C = 0;
    for i1 = 1:ncf-1
      for i2 = i1+1:ncf
        C = C + zs(:,:,i1).*zs(:,:,i2);
      end
    end
    C = C.*gpcf.sigma2(2);
    C = C + sum(zs,3).*gpcf.sigma2(1);
    
  elseif r > 2
    % Over two degrees ... use Newton-Girard formulae
    zs = zeros(n,n,ncf);
    for d = 1:ncf
      cf = gpcf.cf{d};
      zs(:,:,d) = cf.fh.trcov(cf, x(:,d));
    end
    sk = zeros(n,n,r);
    sk(:,:,1) = sum(zs,3);
    for i = 2:r
      sk(:,:,i) = sum(zs.^i, 3);
    end
    clear zs
    es = zeros(n,n,r);
    es(:,:,1) = sk(:,:,1);
    for i = 2:r
      presign = true;
      for k = 1:i-1
        if presign
          es(:,:,i) = es(:,:,i) + es(:,:,i-k).*sk(:,:,k);
          presign = false;
        else
          es(:,:,i) = es(:,:,i) - es(:,:,i-k).*sk(:,:,k);
          presign = true;
        end
      end
      if presign
        es(:,:,i) = es(:,:,i) + sk(:,:,i);
      else
        es(:,:,i) = es(:,:,i) - sk(:,:,i);
      end
      es(:,:,i) = es(:,:,i)./i;
    end
    for i = 1:r
      es(:,:,i) = es(:,:,i).*gpcf.sigma2(i);
    end
    C = sum(es,3);
    
  else
    error('Invalid max_deg parameter')
  end
end

function C = gpcf_additive_trvar(gpcf, x)
% GP_ADDITIVE_TRVAR     Evaluate training variance vector
%
%  Description
%    C = GP_ADDITIVE_TRVAR(GPCF, TX) takes in covariance function of
%    a Gaussian process GPCF and matrix TX that contains training
%    inputs. Returns variance vector C. Every element i of C
%    contains variance of input i in TX. This is a mandatory 
%    subfunction used for example in prediction and energy computations.
%
%  See also
%    GPCF_ADDITIVE_COV, GP_COV, GP_TRCOV


  [n,m]=size(x);

  ncf = length(gpcf.cf);
  
  if m~=ncf 
    error('input dimension does not match with number of additive kernels')
  end
  
  r = gpcf.max_deg;
  
  if r == 1
    % Only one degree
    C = 0;
    for d = 1:ncf
      cf = gpcf.cf{d};
      C = C + cf.fh.trvar(cf, x(:,d));
    end
    C = C.*gpcf.sigma2(1);
  
  elseif r == 2
    % Only two degrees
    zs = zeros(n,ncf);
    for d = 1:ncf
      cf = gpcf.cf{d};
      zs(:,d) = cf.fh.trvar(cf, x(:,d));
    end
    C = 0;
    for i1 = 1:ncf-1
      for i2 = i1+1:ncf
        C = C + zs(:,i1).*zs(:,i2);
      end
    end
    C = C.*gpcf.sigma2(2);
    C = C + sum(zs,2).*gpcf.sigma2(1);
    
  elseif r > 2
    % Over two degrees ... use Newton-Girard formulae
    zs = zeros(n,ncf);
    for d = 1:ncf
      cf = gpcf.cf{d};
      zs(:,d) = cf.fh.trvar(cf, x(:,d));
    end
    sk = zeros(n,r);
    sk(:,1) = sum(zs,2);
    for i = 2:r
      sk(:,i) = sum(zs.^i, 2);
    end
    clear zs
    es = zeros(n,r);
    es(:,1) = sk(:,1);
    for i = 2:r
      presign = true;
      for k = 1:i-1
        if presign
          es(:,i) = es(:,i) + es(:,i-k).*sk(:,k);
          presign = false;
        else
          es(:,i) = es(:,i) - es(:,i-k).*sk(:,k);
          presign = true;
        end
      end
      if presign
        es(:,i) = es(:,i) + sk(:,i);
      else
        es(:,i) = es(:,i) - sk(:,i);
      end
      es(:,i) = es(:,i)./i;
    end
    for i = 1:r
      es(:,i) = es(:,i).*gpcf.sigma2(i);
    end
    C = sum(es,2);
    
  else
    error('Invalid max_deg parameter')
  end
  
end

function reccf = gpcf_additive_recappend(reccf, ri, gpcf)
%RECAPPEND  Record append
%
%  Description
%    RECCF = GPCF_ADDITIVE_RECAPPEND(RECCF, RI, GPCF) takes a
%    covariance function record structure RECCF, record index RI
%    and covariance function structure GPCF with the current MCMC
%    samples of the parameters. Returns RECCF which contains all
%    the old samples and the current samples from GPCF. This 
%    subfunction is needed when using MCMC sampling (gp_mc).
%
%  See also
%    GP_MC, GP_MC->RECAPPEND
  
  if nargin == 2
    % Initialize the record
    reccf.type = 'gpcf_additive';

    % Initialize parameters
    reccf.sigma2=[];
    
    ncf = length(ri.cf);
    for i=1:ncf
      cf = ri.cf{i};
      reccf.cf{i} = cf.fh.recappend([], ri.cf{i});
    end
    
    % Set the function handles
    reccf.fh.pak = @gpcf_additive_pak;
    reccf.fh.unpak = @gpcf_additive_unpak;
    reccf.fh.lp = @gpcf_additive_lp;
    reccf.fh.lpg = @gpcf_additive_lpg;
    reccf.fh.cfg = @gpcf_additive_cfg;
    reccf.fh.cov = @gpcf_additive_cov;
    reccf.fh.trcov  = @gpcf_additive_trcov;
    reccf.fh.trvar  = @gpcf_additive_trvar;
    reccf.fh.recappend = @gpcf_additive_recappend;
    
    % Set other
    reccf.max_deg = ri.max_deg;
    reccf.p=[];
    reccf.p.sigma2=[];
    if isfield(ri.p,'sigma2') && ~isempty(ri.p.sigma2)
      reccf.p.sigma2 = ri.p.sigma2;
    end
    
  else
    % Append to the record
    gpp = gpcf.p;
    
    % record sigma2
    reccf.sigma2(ri,:) = gpcf.sigma2;
    if isfield(gpp,'sigma2') && ~isempty(gpp.sigma2)
      reccf.p.sigma2 = gpp.sigma2.fh.recappend(reccf.p.sigma2, ri, gpcf.p.sigma2);
    end
    
    % Loop over all of the covariance functions
    ncf = length(gpcf.cf);
    for i=1:ncf
      cf = gpcf.cf{i};
      reccf.cf{i} = cf.fh.recappend(reccf.cf{i}, ri, cf);
    end
  end
end

