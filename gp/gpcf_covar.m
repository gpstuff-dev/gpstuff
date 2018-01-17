function gpcf = gpcf_covar(varargin)
%   GPCF_COVAR creates a gpstuff covariance structure for the multivariate 
%   Gaussian Process model based on the coregionalization models (LMC). 
%   See Gelfand et al. (2004) in nonstationary multivariate process modelling
%   through spatially varying coregionalization. Sociedad de Estadistica e
%   Investigacion Operativa, Test.
%
%   The coregionalization matrix is a covariance matrix and its
%   parametrization is given via variances and the correlation matrix. The
%   correlation matrix is parametrizated with the transformation proposed by 
%   Lewandowski et al. (2009) in generating correlation matrices based on
%   vines and extended onion method. Journal of multivariate analysis.
%  
%   Description :
%
%   ─ GPCF = GPCF_COVAR('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%     creates correlation matrix structure in which
%     the named parameters have the specified values. Any
%     unspecified parameters are set to default values.
%
%   ─ GPCF = GPCF_COVAR(GPCF,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a covariance function structure with the named
%    parameters altered with the specified values.
%  
%   ─ Parameters (of the gpstuff structure) for the covariance structure [default]
%     R_prior             - prior for correlation matrix  [prior_corrunif()]
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
%     numberClass         - number of classes (GPs, categories, species, ...)
%                           being modelled (this is required to initialize 
%                           other parameters)
%
%     degreeFreedom_prior - prior for degree of freedoms 'nu' [prior_corrunif()]
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
% ─────────────── 2018 Marcelo Hartmann.

% If you use this file. Learn to recognize its author. Copyright is good but
% misleading.

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
  ip.addParamValue('V_prior', prior_t('s2', 2), @(x) isstruct(x) || isempty(x));
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
      gpcf.fh.realtoz = @gpcf_covar_realtoz;
      gpcf.fh.ztoreal = @gpcf_covar_ztoreal;
      gpcf.fh.rhotoreal = @gpcf_covar_rhotoreal;
      gpcf.fh.realtorho = @gpcf_covar_realtorho;
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
  
  % create indexes which will easy our lives from now on
  seq = 1:gpcf.vectSize;
  i = ceil(0.5 + 0.5 * sqrt(1 + 8 * seq)); % ith column
  j = seq - (i - 2).*(i - 1)/2;            % jth column
  ind1 = (j - 1) * gpcf.numberClass + i;   % linear indexes for lower triangular
  ind2 = (i - 1) * gpcf.numberClass + j;   % linear indexes for upper triangular
  gpcf.index = [i; j; ind1; ind2; seq]';
  
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
          var_aux = var(ind, :); 
          var_aux = var_aux(~isspace(var_aux));
          Vpriors.(var_aux).p = ip.Results.V_prior;
          %Vpriors.(var(ind, :)).p = ip.Results.V_prior;
      end
      gpcf.p.V = Vpriors;
      gpcf.varfields = (fields(gpcf.p.V)');
  end
 
end   


function y = gpcf_covar_realtoz(x, a, d)
% Description :
%   · Transforms the real line to the interval (-1, 1) using the shifted
%   logistic function
%
%   · rho = 2 ./ (1 + exp(-a*x)) - 1;
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


function y = gpcf_covar_ztoreal(x, a)
%  Description :
%   · Transforms the interval (-1, 1) to the real line, using the inverse
%   hyperbolic tangent function
%
%   · y = - 1/a * log(2/(x+1) - 1)   
%   · x in (-1, 1)^(dim(x))

if any(abs(x) >= 1)
    error('domain error');
else
    y = -1/a * log(2./(x+1) - 1);
end
end


function y = gpcf_covar_rhotoreal(gpcf)
%  Description :
%   · Transforms the matrix R to the matrix Z (composed by the elements
%   gamma on the real line)

ind1 = gpcf.index(:, 3);
ind2 = gpcf.index(:, 4);

Z = zeros(gpcf.numberClass, gpcf.numberClass);
R = eye(gpcf.numberClass);  
R([ind1, ind2]) = [gpcf.R'; gpcf.R'];  

W = chol(R);
Z(1, 2:gpcf.numberClass) = W(1, 2:gpcf.numberClass);
for i1 = 2:(gpcf.numberClass-1)
    for j1 = (i1+1):gpcf.numberClass
        Z(i1, j1) = W(i1, j1) .* exp(-0.5*sum(log(1-Z(1:(i1-1), j1).^2)));
        % Z(i1, j1) = W(i1, j1) .* 1/sqrt(prod(1-Z(1:(i1-1), j1).^2));
    end
end

y = gpcf.fh.ztoreal(Z(ind2), gpcf.aValue);
end


function y = gpcf_covar_realtorho(gpcf, w, d)
%  Description :
%   · Transforms the vector gamma on the real line to the matrix R (in a vector);

%   · z = 2 ./ (1 + exp(-a*x)) - 1;
%   · x is the vector on the real line
%   · d equals to [], 1 or 2, and indicates whether the output corresponds
%     no derivatives, first or second derivative.

i = gpcf.index(:, 1);
j = gpcf.index(:, 2);
ind2 = gpcf.index(:, 4);
seq = gpcf.index(:, end);
i2 = gpcf.vectSize;

Z = zeros(gpcf.numberClass, gpcf.numberClass);
W = eye(gpcf.numberClass, gpcf.numberClass);

Z(ind2) = gpcf.fh.realtoz(w(1:i2), gpcf.aValue, []);
W(1, 2:gpcf.numberClass) = Z(1, 2:gpcf.numberClass);

for i1 = 2:gpcf.numberClass
    for j1 = i1:gpcf.numberClass
        ztmp = 0.5.*sum(log(1 - Z(1:(i1-1), j1).^2));
        W(i1, j1) = ((i1 == j1) + (i1 ~= j1)*Z(i1, j1)) * exp(ztmp);
        % ztmp = sqrt(1-Z(1:(i1-1), j1).^2);
        % W(i1, j1) = ((i1 == j1) + ~(i1 == j1)*Z(i1, j1)) * prod(ztmp);
    end
end

if isempty(d)
    % correlation matrix
    R = W'*W;
    y = R(ind2);    
    
elseif d == 1
    % indexes
    ind3 = [j i seq];
    
    % derivatives of the cholesk decomposition w.r.t values on the real line
    dW = zeros(gpcf.numberClass, gpcf.numberClass, i2); 
    dL = zeros(gpcf.numberClass, gpcf.numberClass, i2); 
    z = Z(ind2);
    dz = gpcf.fh.realtoz(w, gpcf.aValue, 1);

    fz = sqrt(1 - z.^2); 
    dfz = - 1./sqrt(1 - z.^2) .*  z .* dz;
    
    for k = 1:i2
        % take the column
        j1 = ind3(ind3(:, 3) == k, 2);

        % take the indexes for that column
        ind4 = ind3(ind3(:, 2) == j1, end);
        
        fzaux = fz(ind4);
        dfzaux = dfz(ind4);
        
        for i1 = 1:j1
            if i1 == 1
                dW(i1, j1, k) = (k == ind4(1))*dz(k);
                
            elseif i1 ~= j1
                kaux = ind4(i1);
             
                if kaux < k
                    continue
                else
                    ind5 = ind4(1:(i1-1));
                    fzaux2 = fz(ind5);
                    dfzaux2 = dfz(ind5);
                    e = (k == ind5);
                    
                    if  kaux == k
                        %dW(i1, j1, k) = dz(k).*prod(fzaux2(~e));
                        dW(i1, j1, k) = dz(k).*exp(sum(log(fzaux2(~e))));
                    end
                                                           
                    if kaux > k
                        %dW(i1, j1, k) = z(kaux).*dfzaux2(e)*prod(fzaux2(~e));
                        dW(i1, j1, k) = z(kaux).*dfzaux2(e)*exp(sum(log(fzaux2(~e))));
                    end
                end
                
            elseif i1 == j1
                e = (k == ind4);
                %dW(i1, j1, k) = dfzaux(e)*prod(fzaux(~e));
                dW(i1, j1, k) = dfzaux(e)*exp(sum(log(fzaux(~e))));
                
            end
        end
        dL(:, :, k) = dW(:, :, k)';
        
    end
    y = dL;
    
elseif d == 2
    % lower cholesk decompostion of R
    y = W';

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
      i = gpcf.index(:, 1);
      j = gpcf.index(:, 2);
      waux = gpcf.fh.rhotoreal(gpcf);
      w = [w waux'];
      S = [repmat('realcorr.', gpcf.vectSize, 1), num2str(j), num2str(i)];
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
      [gpcf.corrFun{i}, w] = cf.fh.unpak(cf, w);
  end
  
  gpp = gpcf.p; 
  
  % unpak for Σ 
  if ~ isempty(gpp.R)
      gpcf.R = gpcf.fh.realtorho(gpcf, w(1:gpcf.vectSize), [])';       
      w = w((gpcf.vectSize + 1):end);
      
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
     x  = gpcf.fh.rhotoreal(gpcf);
     z  = gpcf.fh.realtoz(x, gpcf.aValue, []);
     dz = gpcf.fh.realtoz(x, gpcf.aValue, 1);
     
     % add log-prior on rho parameterization
     lp = lp + gpp.R.fh.lp(gpcf.R, gpp.R);
     
     % add log(|det J|) on the real line parameterization. Equation 11 in
     % Lewandowski et al (2009).
     pwr = gpcf.numberClass - gpcf.index(:, 2) - 1;
     lp = lp + 0.5 .* sum(pwr .* log(1 - z.^2)) + sum(log(dz));

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
      real = gpcf.fh.rhotoreal(gpcf);
      z   = gpcf.fh.realtoz(real, gpcf.aValue, [])';
      dz  = gpcf.fh.realtoz(real, gpcf.aValue, 1)';
      d2z = gpcf.fh.realtoz(real, gpcf.aValue, 2)';
      pwr = (gpcf.numberClass - gpcf.index(:, 2) - 1)';
           
      % cholesk decomposition
      L = gpcf.fh.realtorho(gpcf, real, 2);
      
      % transforming to first derivative of L w.r.t gammas
      dL = gpcf.fh.realtorho(gpcf, real, 1);
      
      b = zeros(gpcf.vectSize, gpcf.vectSize);
      for k = 1:size(dL, 3)
          dRaux = dL(:, :, k)*L' + L*dL(:, :, k)';
          b(:, k) = dRaux(gpcf.index(:, 4));
      end
      
      % dlogp(R(gamma))/dgamma
      dlpdgm = gpp.R.fh.lpg(gpcf.R, gpp.R) * b;
      
      lpgs = dlpdgm - pwr.*(z.*dz./(1 - z.^2)) + d2z./dz;
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

% linear indexes
ind1 = gpcf.index(:, 3);
ind2 = gpcf.index(:, 4);
    
% corr matrix
R = eye(gpcf.numberClass);  

% filling elements in lower and upper part
R([ind1, ind2]) = [gpcf.R, gpcf.R];  

% matrix Σ 
Sig = bsxfun(@times, bsxfun(@times, sqrt(gpcf.V)', R), sqrt(gpcf.V));
% Sig = S * R * S;

% other options
if nargin == 2
    switch type
        case 'cov'
            % empty matrix
            C{1} = Sig;
  
            % calcule cholesk decomposition of Sigma
            [L, notpositivedefinite] = chol(Sig, 'lower');
            % test whether is positive-definite or not
            if notpositivedefinite
                L = NaN(gpcf.numberClass, gpcf.numberClass);
                L(ind2) = 0;
            end
            
            % take cholesky decomposition
            C{2} = L;
            
        case 'corr'
            x = gpcf.fh.rhotoreal(gpcf);
            L = gpcf.fh.realtorho(gpcf, x, 2);

            C{1} = L*L';
            C{2} = L;
    end
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
 
 % chol(Σ)
 M = gpcf.fh.sigma(gpcf, 'cov');
 L = M{2};
 
 % M = gpcf.fh.sigma(gpcf, 'corr');
 % L = bsxfun(@times, sqrt(gpcf.V)', M{2});
 
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
M = gpcf.fh.sigma(gpcf, 'corr'); % R = M{1}; 
L = M{2};

% covariance matrix
%Sig = sVL * sVL';

% \sqrt(diag(var)) times L
sVL = bsxfun(@times, sqrt(gpcf.V)', L);

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

% use savememory option
savememory = nargin == 5;

if nargin == 2 || (isempty(x2) && isempty(mask)) 
    
    % Evaluate the values (dSRS/dgamma_12, dSRS/dgamma_13, ..., dSRS/dgamma_(J-1)J
    % and/or dSRS/dV_1 ...  dSRS/dV_J
    
    % build the big Tj's
    T = cell(1, gpcf.numberClass);

    for  k = 1:gpcf.numberClass
        Taux = sVL(:, k) * sVL(:, k)';
        
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
        
    % correlations and variances in Σ = sqrt(S) L L sqrt(S)
    sVdL = []; 
    dsVL = [];
    
    if ~isempty(gpcf.p.R) || ~isempty(gpcf.p.V)
     
        % for gammas in Σ        
        if ~isempty(gpcf.p.R)
            % transforming to the real line
            real = gpcf.fh.rhotoreal(gpcf);
            
            % transforming to first derivative of L w.r.t gammas
            dL = gpcf.fh.realtorho(gpcf, real, 1);
            
            % variances times first derivatives w.r.t correlations 
            % \sqrt(diag(var)) (dLj/dΘ) 
            sVdL = bsxfun(@times, sqrt(gpcf.V)', dL);
        end
        
        % for variances in Σ
        if ~isempty(gpcf.p.V) 
            dV = 0.5 .* sqrt(gpcf.V);  
            dsVLm = bsxfun(@times, dV', L); % take the lines ...
            dsVL = zeros(nc, nc, nc);

            for k = 1:nc
                dsVL(k, :, k) = dsVLm(k, :);
            end
        end
        
        % derivatives w.r.t to correlations or/and variances
        dSig = cat(3, sVdL, dsVL);
        
        if ~isempty(dSig)

            nSig = length(dSig);
            dL = cell(1, nSig);
            % dCorr = zeros(n, n);
            
            for k = 1:nSig
                % sum_j R_j kron (daj/dΘ * aj' + aj * daj/dΘ')
                dL{k} = zeros(n, n);
                
                for j = 1:nc
                    aj  = repelem(sVL(:, j).', diffnb);
                    daj = repelem(dSig(:, j, k), diffnb);
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

% Try to use the C implementation
% C = trcov_corr(gpcf, x1); 

C = NaN;

if isnan(C)
    % take chol(Σ)
    M = gpcf.fh.sigma(gpcf, 'cov');
    L = M{2};
       
    % full matrices
    C = zeros(n, n);  K = zeros(n, n);
    
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
      reccf.fh.realtoz = @gpcf_covar_realtoz;
      reccf.fh.ztoreal = @gpcf_covar_ztoreal;
      reccf.fh.rhotoreal = @gpcf_covar_rhotoreal;
      reccf.fh.realtorho = @gpcf_covar_realtorho;
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
          reccf.index = ri.index;
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

