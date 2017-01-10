function gpcf = gpcf_covar(varargin)
%   GPCF_COVAR creates a covariance matrix in the parameterization of
%   correlation and variances for the GP multiclass/categorical model
%   based on coregionalization models (LMC). See Gelfand et al. (2004) in
%   nonstationary multivariate process modelling through spatially varying
%   coregionalization.
%  
%   Description :
%
%   · GPCF = GPCF_COVAR('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%     creates correlation matrix structure in which
%     the named parameters have the specified values. Any
%     unspecified parameters are set to default values.
%
%   · GPCF = GPCF_COVAR(GPCF,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a covariance function structure with the named
%    parameters altered with the specified values.
%  
%   · Parameters for covariance matrix [default]
%     R_prior             - prior for correlation matrix  [prior_R]
%     V_prior             - prior for the variances. Must be a structure 
%                           with each component being also a structure (the
%                           priors). Otherwise, just use prior_fixed.
%     corrFun             - structure containing other covariance
%                           functions. More specifically, each element of 
%                           the structure must be a correlation function
%                           (σ²_f = 1). Each element of the structure can
%                           also be a combination of covariance functions, 
%                           which will result to a correlation function.
%                           You always need to follow this structure when
%                           using that specific type of covariance function
%                           in the gpstructure (gp_set).
%      
%     numberClass         - number of classes (categories, species, ...)
%                           being modelled (this is required to initialize 
%                           other parameters)
%
%     degreeFreedom_prior - prior for degree of freedoms 'nu' [prior_R]
%     classVariables      - value defining which column of x is used to
%                           identify the class variables. If this is not
%                           given returns an error. They have to be given
%                           increasing downwards, otherwise you get an
%                           error.
%
%   »» Note «« 
%   If the prior is 'prior_fixed' then the parameter in
%   question is considered fixed and it is not handled in
%   optimization, grid integration, MCMC etc.
%
% * See also
%   GP_SET, GPCF_*, PRIOR_* 
%
% Copyright (c) 2007-2010,2015 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari
% Copyright (c) 2014 Arno Solin and Jukka Koskenranta
% ̣̣̣────────────  2015 Marcelo Hartmann

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  if nargin > 0 && ischar(varargin{1}) && ismember(varargin{1}, {'init' 'set'})
    % remove init and set if isfield(gpcf,'metric')
    varargin(1) = [];
  end
  
  ip = inputParser;
  ip.FunctionName = 'GPCF_COVAR';
  ip.addOptional('gpcf', [], @isstruct);
  ip.addParamValue('R', 0.01, @(x) isvector(x) && ~any(abs(x) > 1));  
  ip.addParamValue('V', 1, @(x) isvector(x) && ~any(x < 0)); 
  ip.addParamValue('R_prior', prior_corrunif(), @(x) isstruct(x) || isempty(x));
  ip.addParamValue('V_prior', prior_t(), @(x) isstruct(x) || isempty(x));
  ip.addParamValue('corrFun', {}, @(x) ~isempty(x) && iscell(x));
  ip.addParamValue('classVariables', [], @(x) ~isempty(x) && mod(x, 1) == 0 && x > 0);
  ip.addParamValue('numberClass', [], @(x) mod(x, 1) == 0 && x > 1);
  ip.addParamValue('aValue', 1, @(x) isreal(x) &&  x > 0);
  
  ip.parse(varargin{:});
  
  gpcf = ip.Results.gpcf;
  
  if isempty(gpcf)
      init = true;
      gpcf.type = 'gpcf_covar';
  else
      if ~isfield(gpcf, 'type') && ~isequal(gpcf.type, 'gpcf_covar')
          error('First argument does not seem to be a valid correlation structure')
      end
      init = false;
  end
  
  if init
      % Set the function handles to the subfunctions
      gpcf.fh.RealToRho = @gpcf_covar_RealToRho;
      gpcf.fh.RhoToReal = @gpcf_covar_RhoToReal;
      gpcf.fh.pak = @gpcf_covar_pak;
      gpcf.fh.unpak = @gpcf_covar_unpak;
      gpcf.fh.lp = @gpcf_covar_lp;
      gpcf.fh.lpg = @gpcf_covar_lpg;
      gpcf.fh.cfg = @gpcf_covar_cfg;
      gpcf.fh.cov = @gpcf_covar_cov;
      gpcf.fh.sigma = @gpcf_covar_sigma;
      gpcf.fh.trcov = @gpcf_covar_trcov;
      gpcf.fh.trvar = @gpcf_covar_trvar;
      gpcf.fh.recappend = @gpcf_covar_recappend;
  end
  
  % initialize class variables and selected variables
  if isempty(ip.Results.classVariables) || isempty(ip.Results.numberClass)
      error('class variables and number of classes must be given')
  end
  gpcf.classVariables = ip.Results.classVariables;
  gpcf.numberClass = ip.Results.numberClass;
  gpcf.vectSize = (gpcf.numberClass^2 - gpcf.numberClass)/2;  
  gpcf.aValue = ip.Results.aValue;
  
  % Initialize correlations functions
  if init || ~ismember('corrFun', ip.UsingDefaults)
      gpcf.corrFun = ip.Results.corrFun;
  end

  if length(gpcf.corrFun) ~= gpcf.numberClass
      error('the number of correlation functions must match the number of classes')
  end

  % Initialize correlations parameters
  if init && ~ismember('R', ip.UsingDefaults)
      gpcf.R = ip.Results.R;
      if length(gpcf.R) ~= gpcf.vectSize
          error('length of the vector does not agree with the number of classes')
      end
  elseif init
      gpcf.R = (-0.99 + 1.98.*rand(1, gpcf.vectSize)) * 0.001;
  end
  
  % Initialize variance parameters
  if init && ~ismember('V', ip.UsingDefaults)
      gpcf.V = ip.Results.V;
      if length(gpcf.V) ~= gpcf.numberClass
          error('length of the variance vector does not agree with the number of classes')
      end
  elseif init
      % takes one if nothing is given.
      gpcf.V = ones(1, gpcf.numberClass);
  end
  
  % initialize prior structure
  if init
      gpcf.p = [];
  end
  
  % initialize correlations
  if init || ~ismember('R_prior', ip.UsingDefaults)
    gpcf.p.R = ip.Results.R_prior; 
    if ~isempty(gpcf.p.R)
        % corrects the prior
        gpcf.p.R.numberClass = gpcf.numberClass;
        gpcf.p.R.vectSize = gpcf.vectSize;
        gpcf.p.R.aValue = gpcf.aValue;
        if gpcf.p.R.numberClass > gpcf.p.R.nu
            error('degrees of freedom must be greater than number of classes')
        end
    end
  end
  
  % initialize variances
  if init && ~ismember('V_prior', ip.UsingDefaults)
      gpcf.p.V = ip.Results.V_prior; 
      if ~isempty(ip.Results.V_prior)
          gpcf.varfields = (fields(gpcf.p.V)');
          if length(structfun(@numel, gpcf.p.V)) ~= gpcf.numberClass
              error('prior missing for some variance')
          end
      end
  else
      var = [repmat('var', gpcf.numberClass , 1), num2str((1:gpcf.numberClass)')];
      for ind = 1:size(var, 1)
          Vpriors.(var(ind, :)).p = ip.Results.V_prior;
      end
      gpcf.p.V = Vpriors;
      % add var fields 
      % gpcf.varfields = fields(gpcf.p.V);
      gpcf.varfields = (fields(gpcf.p.V)');
  end
 
end   


function y = gpcf_covar_RealToRho(x, a, d)
% Description :
%   · Transforms the real line to the interval (-1, 1) using modified
%   logistic distribution function
%
%   · rho = 2/(1+exp(-ax)) - 1   
%   · x is the vector on the real line
%   · d equals to [], 1 or 2, and indicates whether the output corresponds
%     no derivatives, first or second derivative.

if isempty(d)
    % modified logistic distribution function
    y = 2 ./ (1 + exp(-a*x)) - 1;    
elseif d == 1
    % repeated terms
    eminax = exp(-a*x);    
    % 1st derivative, drho/dx
    y = 2 * a * (1 + eminax).^(-2) .* eminax;    
elseif d == 2
    % repeated terms
    a2 = a ^ 2;
    eminax = exp(-a*x); 
    % 2nd derivative, d2rho/dx2
    y = 4 * a2 * (1 + eminax).^(-3) .* eminax.^2 - 2 * a2 * (1 + eminax).^(-2) .* eminax;   
else
    error('Something is wrong. Maybe d should be [], 1 or 2')
end
end


function y = gpcf_covar_RhoToReal(x, a)
%  Description :
%   · Transforms the interval (-1, 1) to the real line, using inverse
%     modified logistic distribution function
%
%   · y = - 1/a * log(2/(x+1) - 1)   
%   · x in (-1, 1)^(dim(x))

if any(abs(x) > 1)
    error('domain error');
else
    y = -1/a * log(2./(x+1) - 1);
end

end


function [w, s, h] = gpcf_covar_pak(gpcf, w)
%  GPCF_COVAR_PAK Combine GP correlation matrix values into one vector 
%   keeping the order of input below.
%
%  Description :
%   · W = GPCF_COVAR_PAK(GPCF) takes the non-diagonal elements of 
%     the correlation matrix and put them into a single row vector w
%     This is a mandatory subfunction used for example 
%     in energy and gradient computations.
%
%   · w = [rho_(1, 2), ..., rho_(1, J), rho_(2, 3), ..., rho_(2, J), ..., rho_((J-1), J)]
%
%  * See also
%    GPCF_covar_UNPAK

  w = []; s = {}; h = [];

  % pak parameters of correlation functions
  for i = 1:gpcf.numberClass
      [wh, sh, hh] = gpcf.corrFun{i}.fh.pak(gpcf.corrFun{i});
      w = [w wh];
      s = [s; sh];
      h = [h 1+hh];
  end
  
  % pak correlations
  if ~isempty(gpcf.p.R)
      seq = 1:gpcf.vectSize;
      i = ceil(0.5 + 0.5 * sqrt(1 + 8 * seq));
      j = seq - (i - 2).*(i - 1)/2;
      
      w = [w gpcf.fh.RhoToReal(gpcf.R, gpcf.aValue)];
      S = [repmat('rhoreal.', gpcf.vectSize, 1), num2str(j'), num2str(i')];
      s = [s; cellstr(S)];
      h = [h 1];
      
      % pak hyperparameters of R
      [wh, sh, hh] = gpcf.p.R.fh.pak(gpcf.p.R);
      sh = strcat(repmat('prior-', size(sh, 1), 1), sh);
      w = [w wh];
      s = [s; sh];
      h = [h 1+hh];
  end
  
  % pak variances
  if ~isempty(gpcf.p.V)
    w = [w log(gpcf.V)];
    S = [repmat('log var.', gpcf.numberClass, 1), num2str((1:gpcf.numberClass)')];
    s = [s; cellstr(S)];
    h = [h 1];
    
    % pak hyperparameters of V
    for i = 1:gpcf.numberClass
        cfvf = gpcf.varfields{i};
        [wh, sh, hh] = gpcf.p.V.(cfvf).p.fh.pak(gpcf.p.V.(cfvf).p);
        sh = strcat(repmat('prior-', size(sh, 1), 1), sh);
        w = [w wh];
        s = [s; sh];
        h = [h 1+hh];
    end
   
   end        
end


function [gpcf, w] = gpcf_covar_unpak(gpcf, w)
%  GPCF_covar_UNPAK Sets the covariance function parameters into the 
%   structure
%
%  Description :
%   · [GPCF, W] = GPCF_COVAR_UNPAK(GPCF, W) takes a covariance
%     function structure GPCF and a hyperparameter vector W,
%     and returns a covariance function structure identical to
%     the input, except that the covariance hyperparameters have
%     been set to the values in W. Deletes the values set to GPCF
%     from W and returns the modified W. This is a mandatory 
%     subfunction used for example in energy and gradient computations.
%    
%  * See also
%    GPCF_covar_PAK

  % unpak parameters of correlations functions
  for i = 1:gpcf.numberClass
      cf = gpcf.corrFun{i};
      [gpcf.corrFun{i} w] = cf.fh.unpak(cf, w);
  end
  
  gpp = gpcf.p; 
  
  % unpak for Σ 
  if ~ isempty(gpp.R)
      i2 = gpcf.vectSize;
      gpcf.R = gpcf.fh.RealToRho(w(1:i2), gpcf.aValue, []);
      w = w((i2 + 1):end);
      
      % unpak hyperparameters of R
      [p, w] = gpcf.p.R.fh.unpak(gpcf.p.R, w);
      gpcf.p.R = p;
  end
  
  if ~ isempty(gpp.V)
      i3 = gpcf.numberClass;
      gpcf.V = exp(w(1:i3));
      w = w((i3 + 1):end);
      
      % unpak hyperparameters of V
      % loop for the hyperpriors
      for i = 1:gpcf.numberClass
          cfvf = gpcf.varfields{i};
          [p, w] = gpcf.p.V.(cfvf).p.fh.unpak(gpcf.p.V.(cfvf).p, w);
          gpcf.p.V.(cfvf).p = p;
      end
  end
end


function lp = gpcf_covar_lp(gpcf)
%  GPCF_COVAR_LP Evaluate the log prior of covariance function parameters
%
%  Description :
%   · LP = GPCF_COVAR_LP(GPCF, X, T) takes a correlation function
%     structure GPCF and evaluates log-prior(R)
%
%  * See also
%    GPCF_COVAR_PAK, GPCF_covar_UNPAK, GPCF_covar_LPG, GP_E

  lp = 0;
  
  % add log-prior for correlations functions parameters
  for i = 1:gpcf.numberClass
      lp = lp + gpcf.corrFun{i}.fh.lp(gpcf.corrFun{i});
  end
  
  gpp = gpcf.p;
  
  % log-prior for correlations
  if ~isempty(gpp.R)
     % mapping to the real line
     x = gpcf.fh.RhoToReal(gpcf.R, gpcf.aValue);
     drho = gpcf.fh.RealToRho(x, gpcf.aValue, 1);
     
     % add log-prior on rho parameterization
     lp = lp + gpp.R.fh.lp(gpcf.R, gpp.R);
     
     % add log(|det J|) on the real line parameterization
     lp = lp + sum(log(drho));
  end
  
  % log-prior for variances
  if ~isempty(gpp.V)
      for i = 1:gpcf.numberClass
          cfvf = gpcf.varfields{i};
          
          lp = lp + gpp.V.(cfvf).p.fh.lp(gpcf.V(i), ...
              gpp.V.(cfvf).p) + log(gpcf.V(i));
      end
  end
end


function lpg = gpcf_covar_lpg(gpcf)
%  GPCF_COVAR_LPG Evaluate gradient of the log prior with respect
%   to the parameters.
%
%  Description :
%   · LPG = GPCF_COVAR_LPG(GPCF) takes a covariance function
%     structure GPCF and returns LPG = d log (p(th))/dth, where th
%     is the parametric vector of correlations and variances.
%     This is a mandatory subfunction used for example in gradient 
%     computations.
%
%  * See also
%    GPCF_covar_PAK, GPCF_covar_UNPAK, GPCF_covar_LP, GP_G

  lpg = [];
  
  % stack grad vector of the correlation functions parameters
  for i = 1:gpcf.numberClass
      lpg = [lpg gpcf.corrFun{i}.fh.lpg(gpcf.corrFun{i})];
  end
  
  gpp = gpcf.p;
  
  % for correlations
  if ~isempty(gpcf.p.R)    
      % mapping to the real line
      x = gpcf.fh.RhoToReal(gpcf.R, gpcf.aValue);
      
      % mapping to the first derivative of rho
      drho = gpcf.fh.RealToRho(x, gpcf.aValue, 1);
      
      % mapping to the second derivative of rho
      d2rho = gpcf.fh.RealToRho(x, gpcf.aValue, 2);
      
      % building the grad vector on the real line parameterization
      lpgs = gpp.R.fh.lpg(gpcf.R, gpp.R) .* drho + d2rho./drho;
      lpg = [lpg lpgs];
  end
  
  % for variances
  if ~isempty(gpp.V)
      % building the grad vector on the real line parameterization
      for i = 1:gpcf.numberClass
          cfvf = gpcf.varfields{i};
          lpgs = gpp.V.(cfvf).p.fh.lpg(gpcf.V(i), gpp.V.(cfvf).p);
          lpg = [lpg lpgs(1).*gpcf.V(i)+1 lpgs(2:end)];
      end
  end
end


function C = gpcf_covar_sigma(gpcf, type)
%  GP_COVAR_COV  Evaluate the covariance matrix Σ of the linear model of
%  coregionalization and its Cholesky decomposition.
%
%  Description :
%   · C = GP_COVAR_COV(GP, i1) takes the gpcf structure and
%     returns the covariance matrix Σ and its Cholesky decomposition.
%     Every element (i, j) of Σ contains the covariance (parameterised by
%     correlation and variances) between class i and class j. This is a
%     mandatory subfunction. 
%
%  * See also
%    GPCF_COVAR_TRCOV, GPCF_COVAR_TRVAR, GP_COV, GP_TRCOV

% building covariance matrix in correlation parameterization
seq = 1 : gpcf.vectSize;

% linear indexes
i = ceil(0.5 + 0.5 * sqrt(1 + 8 * seq));
j = seq - (i - 2) .* (i - 1) / 2;
ind1 = (j - 1) * gpcf.numberClass + i;
ind2 = (i - 1) * gpcf.numberClass + j;
    
% corr matrix
R = eye(gpcf.numberClass);  

% filling elements in lower and upper part
R([ind1, ind2]) = [gpcf.R'; gpcf.R'];  
S = diag(sqrt(gpcf.V));

% matrix Σ 
Sig = S * R * S;

% other options
if nargin == 2
    % empty matrix
    C{1} = Sig;
    
    % calcule cholesk decomposition
    [L, notpositivedefinite] = chol(Sig, 'lower');
    
    % test if it is whether is positive-definite or not
    if notpositivedefinite
        L = NaN(gpcf.numberClass, gpcf.numberClass);
        L(ind2) = 0;
    end
    
    % take cholesky decomposition
    C{2} = L;
    
else
    C = Sig;
    
end
end


function C = gpcf_covar_cov(gpcf, x1, x2)
%  GP_COVAR_COV  Evaluate the covariance matrix between two input vectors
%
%  Description
%   · C = GP_COVAR_COV(GP, TX, X) takes in correlation structure
%     and two matrixes TX and X that contain input vectors to GP. 
%     Returns covariance matrix C. 
%     Every element ij of C contains correlation between inputs i
%     in TX and j in X. This is a mandatory subfunction used for 
%     example in prediction and energy computations.
%
%
%  * See also
%    GPCF_COVAR_TRCOV, GPCF_COVAR_TRVAR, GP_COV, GP_TRCOV
 
 if isempty(x2)
     x2 = x1;
 end
 
 if size(x1, 2) ~= size(x2, 2)
     error('the number of columns of X1 and X2 has to be same')
 end
 
 xClass1 = x1(:, gpcf.classVariables);
 xClass2 = x2(:, gpcf.classVariables);
 
 if ~issorted(xClass1) || ~issorted(xClass2)
     error('you need to give the class variable increasing downwards');
 end
 
 if max(xClass1) > gpcf.numberClass || max(xClass2) > gpcf.numberClass
     error('more classes than the given number of classes');
 end 
 
 n1 = size(xClass1, 1);
 n2 = size(xClass2, 1);
 
 % information of the classes in the data
 ind1 = unique(xClass1)'; nind1 = size(ind1, 2);
 ind2 = unique(xClass2)'; nind2 = size(ind2, 2);
 
 % positions of observations in each vector
 nb1 = find(diff([-inf xClass1' inf]));
 nb2 = find(diff([-inf xClass2' inf]));
 
%  if ~all([na1 na2] == gpcf.numberClass) 
%      error('more/less classes than given to the model');
%  end
 
 % chol(Σ)
 M = gpcf.fh.sigma(gpcf, 'chol');
 L = M{2};
 
 C = zeros(n1, n2);
 for k = 1:gpcf.numberClass
     
     % kth column of the cholesk decomposition of Σ = SRS
     T = L(:, k) * L(:, k)';
     
     for i = 1:nind1
         for j = 1:nind2
             inb1 = nb1(i) : nb1(i + 1) - 1;
             inb2 = nb2(j) : nb2(j + 1) - 1;
             K(inb1, inb2) = T(ind1(i), ind2(j));
         end
     end
     
     cf = gpcf.corrFun{k};
     C = C + cf.fh.cov(cf, x1, x2) .* K;
end
 
 C(abs(C) < eps) = 0;
end


function dCRV = gpcf_covar_cfg(gpcf, x, x2, mask, i1)
%  GPCF_COVAR_CFG  Evaluate the derivarive of the covar matrix
%
%  Description :
%   · dRff = GPCF_COVAR_CFG(GPCF, X) takes a
%     covariance structure GPCF, a matrix of inputs
%     vectors and returns dSRS/drho and/or dSRS/dV, the gradients of 
%     SRS matrix
%
%   · dRff = GPCF_COVAR_CFG(GPCF, X, X2) takes a
%     covariance structure GPCF, a matrix X of input
%     vectors and returns  dSRS/drho and/or dSRS/dV, the gradients of 
%     SRS matrix with respect to rho (cell array with matrix
%     elements). This subfunction is needed when using sparse 
%     approximations (e.g. FIC).
%
%  * See also
%    GPCF_COVAR_PAK, GPCF_COVAR_UNPAK, GPCF_COVAR_LP, GP_G

dCRV = {}; 

% covariance matrix Σ and chol(Σ)
M = gpcf_covar_sigma(gpcf, 1);
Sig = M{1}; L = M{2};

% auxiliar indexes
seq = 1 : gpcf.vectSize;
ii = ceil(0.5 + 0.5 * sqrt(1 + 8 * seq));
jj = seq - (ii - 2) .* (ii - 1) / 2;
 
% for correlations and variances;
ind3 = [jj' ii' seq'];

% take class variables
xC = x(:, gpcf.classVariables);

% positions of the observations
nb = diff([0 xC' xC(end) + 1]) .* (1:(length(xC) + 1));
nb = nb(nb ~= 0);

% number of observation in each class 
diffnb = diff(nb);

% number of classes
nc = gpcf.numberClass;

% number of observations
n = length(xC);

if nargin == 5
    % Use memory save option
    savememory = 1;
%     if i1 == 0
%         i = 0; j = 0;
%         % Return number of hyperparameters
%         if ~isempty(gpcf.p.R)
%             i = length(gpcf.R);
%         end
%         if ~isempty(gpcf.p.V)
%             j = length(gpcf.V);
%         end
%         dCRV = i + j;
%         return
%     end
else
    savememory = 0;
end
  
% Evaluate the values (dSRS/drho_12, dSRS/drho_13, ..., dSRS/drho_1J, dSRS/
% drho_23, ...,  ..., dSRS/drho_2J, ..., dSRS/drho_(J-1)J and/or dSRS/dV_1
% ...  dSRS/dV_J

if nargin == 2 || (isempty(x2) && isempty(mask)) 
    
    % build the big Tj's
    T = cell(1, gpcf.numberClass);

    for  k = 1:gpcf.numberClass
        Taux = L(:, k) * L(:, k)';
        
        for i = 1:gpcf.numberClass
            inb1 = nb(i) : nb(i + 1) - 1;
            T{k}(inb1, inb1) = Taux(i, i);
            
            for j = (i + 1):gpcf.numberClass
                inb2 = nb(j):nb(j + 1) - 1;
                
                T{k}(inb1, inb2) = Taux(i, j);
                T{k}(inb2, inb1) = Taux(j, i);
            end
        end
    end    
    
    % correlations matrices
    dC = {};
    
    for i1 = 1:nc
        dKK = gpcf.corrFun{i1}.fh.cfg(gpcf.corrFun{i1}, x);
        
        if ~isempty(dKK)
            % number of derivatives of the correlation function
            i2 = length(dKK);
            
            % derivatives of the specific correlation function
            for i3 = 1:i2
                dC{end + 1} = dKK{i3} .* T{i1}; 
            end
        end
    end
        
    % correlations and variances in Σ = SRS
    dR = {}; 
    dV = {};
    
    if ~isempty(gpcf.p.R) || ~isempty(gpcf.p.V);
        
        dA = zeros(gpcf.numberClass, gpcf.numberClass);
    
        % for correlations in Σ        
        if ~isempty(gpcf.p.R)
            % creating cell of matrices
            dR = cell(1, gpcf.vectSize);
            
            % building the matrices          
            for k = 1:gpcf.vectSize
                % transforming to the real line
                real = gpcf.fh.RhoToReal(gpcf.R(k), gpcf.aValue);
                
                % first derivatives w.r.t correlations x variances
                drhoV = gpcf.fh.RealToRho(real, gpcf.aValue, 1) * ...
                        prod(sqrt(gpcf.V(ind3(k, 1:2))));
                
                % indexes
                u = ind3(k, 1:2);

                % derivatives
                dA(u(1), u(2)) = drhoV;
                dA(u(2), u(1)) = drhoV;

                % derivative matrix and put zeros back to dA
                dR{k} = dA;
                dA(:, :) = 0;
            end
        end
        
        % for variances in Σ
        if ~isempty(gpcf.p.V)    
            % creating cell of matrices
            dV = cell(1, nc);
            
            % building the matrices
            for k = 1:gpcf.numberClass
                % takes shared correlations
                aux = ind3(logical((ind3(:, 1) == k) + (ind3(:, 2) == k)), 1:2);
                aux = [k k; aux];
                
                % get linear indexes
                ind = (aux(:, 1) - 1)*nc + aux(:, 2);
                
                % take the elements of Sig
                u = Sig(ind);
                
                % derivative w.r.t variances
                dA(aux(1, 1), aux(1, 2)) = u(1);
                
                % loop for building the matrix
                for kk = 2:nc
                    dA(aux(kk, 1), aux(kk, 2)) = 0.5 * u(kk);
                    dA(aux(kk, 2), aux(kk, 1)) = 0.5 * u(kk);
                end
                
                % derivative matrix and put zeros back to dA
                dV{k} = dA;
                dA(:, :) = 0;
            end
        end
        
        % derivatives w.r.t to correlations or/and variances;
        dSig = [dR dV];
        
        if ~isempty(dSig)
            % evaluate derivatives of cholesky decomposition
            % see Bayesian filtering and smoothing by Simo Särkkã page 211 
            
            % indexes for the upper triangular matrix
            ind = (ii - 1) * nc + jj;
            
            nSig = length(dSig);
            dL = cell(1, nSig);
            % dCorr = zeros(n, n);
            
            for k = 1:nSig
                % L^-1  x  dΣ/dθ  x  L^-t
                M = (L \ dSig{k}) / L';
                
                % apply function ø(·)
                M(1 : nc + 1 : nc^2) = M(1 : nc + 1 : nc^2) * 0.5;
                M(ind) = 0;
                
                % compute the derivatives of cholesky decomposition 
                dA = L * M;
                dA(abs(dA) < eps) = 0;
                dLs = dA;
                
                % sum_j R_j kron (daj/dΘ * aj' + aj * daj/dΘ')
                dL{k} = zeros(n, n);
                
                for j = 1:nc
                    aj  = repelem(L(:, j).', diffnb);
                    daj = repelem(dLs(:, j), diffnb);
                    dd = daj * aj; 
                    
                    cf = gpcf.corrFun{j};
                    dCorr = cf.fh.trcov(cf, x) .* (dd + dd.');
                    dL{k} = dL{k} + dCorr;
                end
            end
        end
    end
    
    % return matrices
    dCRV = [dC dL];    
    
% related to sparse approximations
elseif nargin == 3 || isempty(mask)  
    error('nargin == 3 || isempty(mask) not implemented yet');   
    
elseif nargin == 4 || nargin == 5
    error('nargin == 4 || nargin == 5 not implemented yet');
    
end

end


function C = gpcf_covar_trcov(gpcf, x)
%  GP_COVAR_TRCOV  Evaluate training covariance matrix of inputs
%
%  Description :
%   · C = GP_COVAR_TRCOV(GP, TX) takes in correlation structure
%     matrix TX that contains training input vectors.
%     Returns covariance matrix C. Every element ij of C contains
%     the correlation between inputs i and j in TX.
%     This is a mandatory subfunction used for example in
%     prediction and energy computations.
%
%  * See also
%    GPCF_COVAR_COV, GPCF_COVAR_TRVAR, GP_COV, GP_TRCOV
%
% HERE IS TO TAKE ADVANTAGE OF THE SYMMETRIC MATRIX 

% takes the class variables
xC = x(:, gpcf.classVariables);

% checking
if ~issorted(xC) 
    error('class variable should increasing downwards');
end

% number of observations
n = size(xC, 1);

% getting the information of the classes in the data
a = unique(xC); % na = size(a, 1);

% checking
if max(a) > gpcf.numberClass
% if na ~= gpcf.numberClass || max(a) ~= gpcf.numberClass
    error('more or less classes than given to the model');
end

% getting the information of the classes in the data
ind = unique(xC)'; nind = size(ind, 2);

% number of observations in each class
nb = find(diff([-inf xC' inf]));
% nb = diff([0 xC' xC(end) + 1]) .* (1:(n + 1));
% nb = nb(nb ~= 0);

% Try to use the C implementation
% C = trcov_corr(gpcf, x1); 

C = NaN;

if isnan(C)
    % take chol(Σ)
    M = gpcf.fh.sigma(gpcf, 'chol');
    L = M{2};
       
    % full matrices
    C = zeros(n, n); K = zeros(n, n);  
    
    for k = 1:gpcf.numberClass
        % T_k = a_k * a_k' matrices,
        T =  L(:, k) * L(:, k)';
        
        for i = 1:nind
            inb1 = nb(i) : nb(i + 1) - 1;
            K(inb1, inb1) = T(ind(i), ind(i));
            
            for j = (i + 1):nind
                inb2 = nb(j) : nb(j + 1) - 1;
                
                K(inb1, inb2) = T(ind(i), ind(j));
                K(inb2, inb1) = T(ind(j), ind(i));
            end
        end
        
        % corrFun kronecker T_k
        cf = gpcf.corrFun{k};
        C = C + cf.fh.trcov(cf, x) .* K;
    end
    
    C(abs(C) < eps) = 0;
end
end


function C = gpcf_covar_trvar(gpcf, x)
%  GP_COVAR_VECTOR  Evaluate training variance vector
%
%  Description:
%   · C = GP_COVAR_TRVAR(GPCF, TX) takes in correlation structure
%     of and matrix TX that contains training inputs. 
%     Returns correlation vector C, which are ones. 
%     Every element i of C contains the correlation of the input in TX. 
%     This is a mandatory subfunction used for example in prediction and 
%     energy computations.
%
%  * See also:
%    GPCF_COVAR_COV, GP_COV, GP_TRCOV  

 % take indicator class
 x = x(:, gpcf.classVariables);
  
 % information of the classes in the data
 a = unique(x);
 
 % locate each class
 nb = find(diff([-inf x' inf]));
 
 % number of observation in each class
 diffnb = diff(nb);
 
 C = repelem(gpcf.V(a), diffnb)';
 end


function reccf = gpcf_covar_recappend(reccf, ri, gpcf)
%  RECAPPEND  Record append
%
%  Description:
%   · RECCF = GPCF_COVAR_RECAPPEND(RECCF, RI, GPCF) takes a
%     correlation structure record structure RECCF, record index RI
%     and correlation structure GPCF with the current MCMC
%     samples of the parameters. Returns RECCF which contains
%     all the old samples and the current samples from GPCF.
%     This subfunction is needed when using MCMC sampling (gp_mc).
%
%  * See also:
%    GP_MC and GP_MC -> RECAPPEND

  if nargin == 2
      % initialize the record
      reccf.type = 'gpcf_covar';
      
      % initialize parameters for correlations functions
      for i = 1:ri.numberClass
          reccf.corrFun{i} = ri.corrFun{i}.fh.recappend([], ri.corrFun{i});
      end
      
      % Initialize parameters
      reccf.R = [];
      reccf.V = [];
      
      % Set the function handles
      reccf.fh.RealToRho = @gpcf_covar_RealToRho;
      reccf.fh.RhoToReal = @gpcf_covar_RhoToReal;
      reccf.fh.pak = @gpcf_covar_pak;
      reccf.fh.unpak = @gpcf_covar_unpak;
      reccf.fh.lp = @gpcf_covar_lp;
      reccf.fh.lpg = @gpcf_covar_lpg;
      reccf.fh.cfg = @gpcf_covar_cfg;
      reccf.fh.cov = @gpcf_covar_cov;
      reccf.fh.sigma = @gpcf_covar_sigma;
      reccf.fh.trcov = @gpcf_covar_trcov;
      reccf.fh.trvar = @gpcf_covar_trvar;
      reccf.fh.recappend = @gpcf_covar_recappend;
      reccf.p = [];
      reccf.p.R = [];
      reccf.p.V = [];
      
      if isfield(ri.p, 'R') && ~isempty(ri.p.R)
          reccf.p.R = ri.p.R;
      end
            
      if isfield(ri.p, 'V') && ~isempty(ri.p.V)
          for i = 1:ri.numberClass
              reccf.p.V.(ri.varfields{i}).p = ri.p.V.(ri.varfields{i}).p;
          end
      end
     
      if isfield(ri, 'classVariables') 
          reccf.classVariables = ri.classVariables;
          reccf.numberClass = ri.numberClass;
          reccf.vectSize = ri.vectSize;
          reccf.aValue = ri.aValue;
          reccf.varfields = ri.varfields;
      end
      
  else
      % Append to the record
      gpp = gpcf.p;
      
      if ~ isfield(gpcf,'metric')
          % record correlation function elements
          for i = 1:gpcf.numberClass
              cf = gpcf.corrFun{i};
              reccf.corrFun{i} = cf.fh.recappend(reccf.corrFun{i}, ri, cf);
          end
          
          % record R elements
          reccf.R(ri, :) = gpcf.R;
          if isfield(gpp, 'R') && ~isempty(gpp.R)
              reccf.p.R = gpp.R.fh.recappend(reccf.p.R, ri, gpcf.p.R);
          end
          
          % record V elements
          reccf.V(ri, :) = gpcf.V;
          if isfield(gpp, 'V') && ~isempty(gpp.V)
              for i = 1:gpcf.numberClass
                  cfvf = gpcf.varfields{i};
                  reccf.p.V.(cfvf).p = ... 
                      gpp.V.(cfvf).p.fh.recappend(reccf.p.V.(cfvf).p, ... 
                      ri, gpcf.p.V.(cfvf).p); 
              end
          end
      end
      
  end
end

