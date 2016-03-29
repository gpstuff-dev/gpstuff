function gpcf = gpcf_linearLogistic(varargin)
%GPCF_LINEARLOGISTIC  Create a covariance function corresponding to
%                     logistic mean function 
%
%  Description
%    GPCF = GPCF_LINEARLOGISTIC('PARAM1',VALUE1,'PARAM2,VALUE2,...) creates
%    a covariance function structure corresponding to logistic mean
%    function in which the named parameters have the specified values. Any
%    unspecified parameters are set to default values.
%
%    GPCF = GPCF_LINEARLOGISTIC(GPCF,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a covariance function structure with the named
%    parameters altered with the specified values.
%
%    The logistic functional form is given by
%         h(x) = w* (logitinv(a.*x + b) - 0.5);
%    By giving a zero mean Gaussian prior for weight, 
%         w ~ N(0, coeffSigma2)
%    the prior for h(x) is
%         h(x) ~ N(0, H(x)*H(x)'*coeffSigma2) )
%    where H(x) = [h(x(1)), ... , h(x(n))]' and, hence,
%    H(x)*H(x)'*coeffSigma2) is the covariance function related to the 
%    logistic mean function.
%  
%    Parameters for linearLogistic (dot product) covariance function
%      a                 - regression coefficient of linear part [1].
%                          For identifiability a is restricted to positive
%                          values by log transformation
%      b                 - intercept of the linear part [0].
%                          b can be positive or negative.
%      a_prior           - prior for a [prior_gaussian('s2',10)]
%      b_prior           - prior for b [prior_gaussian('s2',10)]
%      coeffSigma2       - prior variance for regressor coefficients [10]
%                          This can be either scalar corresponding
%                          to a common prior variance or vector
%                          defining own prior variance for each
%                          coefficient.
%      coeffSigma2_prior - prior structure for coeffSigma2 [prior_logunif]
%      selectedVariables - vector defining which inputs are used [all]
%
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%  Example:
%   a=0.2;
%   b = -10;
%   x = linspace(0,100,100)';
%   y = 3.*(logitinv(a.*x + b) - 0.5) + 0.1*randn(100,1);
%   cf = gpcf_linearLogistic('a', a, 'b', b, 'selectedVariables', 1)  ;
%   gp = gp_set('cf', cf);
%   Ef = gp_pred(gp,x,y,x);
%   figure,plot(x,y,'.'), hold on, plot(x,Ef,'k')
%
%  See also
%    GP_SET, GPCF_*, PRIOR_*, MEAN_*
%
% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2008-2010 Jaakko Riihimäki
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPCF_LINEARLOGISTIC';
  ip.addOptional('gpcf', [], @isstruct);
  ip.addParamValue('coeffSigma2',10, @(x) isvector(x) && all(x>0));
  ip.addParamValue('a',1, @(x) isvector(x) && all(x>0));
  ip.addParamValue('b',0, @(x) isvector(x) );
  ip.addParamValue('coeffSigma2_prior',prior_logunif, @(x) isstruct(x) || isempty(x));
  ip.addParamValue('a_prior',prior_gaussian('s2',10), @(x) isstruct(x) || isempty(x));
  ip.addParamValue('b_prior',prior_gaussian('s2',10), @(x) isstruct(x) || isempty(x));
  ip.addParamValue('selectedVariables',[], @(x) isvector(x) && all(x>0));
  ip.parse(varargin{:});
  gpcf=ip.Results.gpcf;

  if isempty(gpcf)
    init=true;
    gpcf.type = 'gpcf_linearLogistic';
  else
    if ~isfield(gpcf,'type') && ~isequal(gpcf.type,'gpcf_linearLogistic')
      error('First argument does not seem to be a valid covariance function structure')
    end
    init=false;
  end
  
  % Initialize parameter
  if init || ~ismember('coeffSigma2',ip.UsingDefaults)
    gpcf.coeffSigma2=ip.Results.coeffSigma2;
  end
  if init || ~ismember('a',ip.UsingDefaults)
    gpcf.a=ip.Results.a;
  end
  if init || ~ismember('b',ip.UsingDefaults)
    gpcf.b=ip.Results.b;
  end

  % Initialize prior structure
  if init
    gpcf.p=[];
  end
  if init || ~ismember('coeffSigma2_prior',ip.UsingDefaults)
    gpcf.p.coeffSigma2=ip.Results.coeffSigma2_prior;
  end
  if init || ~ismember('a_prior',ip.UsingDefaults)
      gpcf.p.a=ip.Results.a_prior;
  end
  if init || ~ismember('b_prior',ip.UsingDefaults)
      gpcf.p.b=ip.Results.b_prior;
  end

  if ~ismember('selectedVariables',ip.UsingDefaults)
    selectedVariables=ip.Results.selectedVariables;
    if ~isempty(selectedVariables)
      gpcf.selectedVariables = selectedVariables;
    end
  end
  
  if init
    % Set the function handles to the subfunctions
    gpcf.fh.pak = @gpcf_linearLogistic_pak;
    gpcf.fh.unpak = @gpcf_linearLogistic_unpak;
    gpcf.fh.lp = @gpcf_linearLogistic_lp;
    gpcf.fh.lpg = @gpcf_linearLogistic_lpg;
    gpcf.fh.cfg = @gpcf_linearLogistic_cfg;
    gpcf.fh.cfdg = @gpcf_linearLogistic_cfdg;
    gpcf.fh.cfdg2 = @gpcf_linearLogistic_cfdg2;
    gpcf.fh.ginput = @gpcf_linearLogistic_ginput;
    gpcf.fh.ginput2 = @gpcf_linearLogistic_ginput2;
    gpcf.fh.ginput3 = @gpcf_linearLogistic_ginput3;
    gpcf.fh.ginput4 = @gpcf_linearLogistic_ginput4;
    gpcf.fh.cov = @gpcf_linearLogistic_cov;
    gpcf.fh.trcov  = @gpcf_linearLogistic_trcov;
    gpcf.fh.trvar  = @gpcf_linearLogistic_trvar;
    gpcf.fh.recappend = @gpcf_linearLogistic_recappend;
  end        

end

function [w, s, h] = gpcf_linearLogistic_pak(gpcf, w)
%GPCF_GPCF_LINEARLOGISTIC_PAK  Combine GP covariance function parameters into one vector
%
%  Description
%    W = GPCF_GPCF_LINEARLOGISTIC_PAK(GPCF) takes a covariance function
%    structure GPCF and combines the covariance function
%    parameters and their hyperparameters into a single row
%    vector W. This is a mandatory subfunction used for 
%    example in energy and gradient computations.
%
%       w = [ log(gpcf.coeffSigma2)
%             (hyperparameters of gpcf.coeffSigma2)
%              log(gpcf.a)
%             (hyperparameters of gpcf.a)
%              gpcf.b
%             (hyperparameters of gpcf.b)]'
%
%  See also
%    GPCF_GPCF_LINEARLOGISTIC_UNPAK
  
  w = []; s = {}; h =[];
  if ~isempty(gpcf.p.coeffSigma2)
    w = log(gpcf.coeffSigma2);
    if numel(gpcf.coeffSigma2)>1
      s = [s; sprintf('log(linearLogistic.coeffSigma2 x %d)',numel(gpcf.coeffSigma2))];
    else
      s = [s; 'log(linearLogistic.coeffSigma2)'];
    end
    h = [h ones(1, numel(gpcf.coeffSigma2))];
    % Hyperparameters of coeffSigma2
    [wh, sh, hh] = gpcf.p.coeffSigma2.fh.pak(gpcf.p.coeffSigma2);
    sh=strcat(repmat('prior-', size(sh,1),1),sh);
    w = [w wh];
    s = [s; sh];
    h = [h 1+hh];
  end
  
  if ~isempty(gpcf.p.a)
      w = [w log(gpcf.a)];
      if numel(gpcf.a)>1
          s = [s; sprintf('log(linearLogistic.a x %d)',numel(gpcf.a))];
      else
          s = [s; 'log(linearLogistic.a)'];
      end
      h = [h ones(1, numel(gpcf.a))];
      % Hyperparameters of a
      [wh, sh, hh] = gpcf.p.a.fh.pak(gpcf.p.a);
      sh=strcat(repmat('prior-', size(sh,1),1),sh);
      w = [w wh];
      s = [s; sh];
      h = [h 1+hh];
  end
  
  if ~isempty(gpcf.p.b)
      w = [w gpcf.b];
      if numel(gpcf.b)>1
          s = [s; sprintf('linearLogistic.b x %d',numel(gpcf.b))];
      else
          s = [s; 'linearLogistic.b'];
      end
      h = [h ones(1, numel(gpcf.b))];
      % Hyperparameters of b
      [wh, sh, hh] = gpcf.p.b.fh.pak(gpcf.p.b);
      sh=strcat(repmat('prior-', size(sh,1),1),sh);
      w = [w wh];
      s = [s; sh];
      h = [h 1+hh];
  end

  
end

function [gpcf, w] = gpcf_linearLogistic_unpak(gpcf, w)
%GPCF_GPCF_LINEARLOGISTIC_UNPAK  Sets the covariance function parameters 
%                   into the structure
%
%  Description
%    [GPCF, W] = GPCF_GPCF_LINEARLOGISTIC_UNPAK(GPCF, W) takes a covariance
%    function structure GPCF and a hyper-parameter vector W, and
%    returns a covariance function structure identical to the
%    input, except that the covariance hyper-parameters have been
%    set to the values in W. Deletes the values set to GPCF from
%    W and returns the modified W. This is a mandatory subfunction 
%    used for example in energy and gradient computations.
%
%    Assignment is inverse of  
%       w = [ log(gpcf.coeffSigma2)
%             (hyperparameters of gpcf.coeffSigma2)
%              log(gpcf.a)
%             (hyperparameters of gpcf.a)
%              log(gpcf.b)
%             (hyperparameters of gpcf.b)]'
%
%  See also
%   GPCF_GPCF_LINEARLOGISTIC_PAK
  
  gpp=gpcf.p;

  if ~isempty(gpp.coeffSigma2)
    i2=length(gpcf.coeffSigma2);
    i1=1;
    gpcf.coeffSigma2 = exp(w(i1:i2));
    w = w(i2+1:end);
    
    % Hyperparameters of coeffSigma2
    [p, w] = gpcf.p.coeffSigma2.fh.unpak(gpcf.p.coeffSigma2, w);
    gpcf.p.coeffSigma2 = p;
  end
  
  if ~isempty(gpp.a)
      i2=length(gpcf.a);
      i1=1;
      gpcf.a = exp(w(i1:i2));
      w = w(i2+1:end);
      
      % Hyperparameters of a
      [p, w] = gpcf.p.a.fh.unpak(gpcf.p.a, w);
      gpcf.p.a = p;
  end
  if ~isempty(gpp.b)
      i2=length(gpcf.b);
      i1=1;
      gpcf.b = w(i1:i2);
      w = w(i2+1:end);
      
      % Hyperparameters of b
      [p, w] = gpcf.p.b.fh.unpak(gpcf.p.b, w);
      gpcf.p.b = p;
  end
end

function lp = gpcf_linearLogistic_lp(gpcf)
%GPCF_GPCF_LINEARLOGISTIC_LP  Evaluate the log prior of covariance function
%                             parameters 
%
%  Description
%    LP = GPCF_GPCF_LINEARLOGISTIC_LP(GPCF) takes a covariance function
%    structure GPCF and returns log(p(th)), where th collects the
%    parameters. This is a mandatory subfunction used for example 
%    in energy computations.
%
%  See also
%   GPCF_GPCF_LINEARLOGISTIC_PAK, GPCF_GPCF_LINEARLOGISTIC_UNPAK,
%   GPCF_GPCF_LINEARLOGISTIC_LPG, GP_E 

% Evaluate the prior contribution to the error. The parameters that
% are sampled are from space W = log(w) where w is all the "real" samples.
% On the other hand errors are evaluated in the W-space so we need take
% into account also the  Jacobian of transformation W -> w = exp(W).
% See Gelman et al. (2013), Bayesian Data Analysis, third edition, p. 21.
  lp = 0;
  gpp=gpcf.p;

  if ~isempty(gpp.coeffSigma2)
    lp = gpp.coeffSigma2.fh.lp(gpcf.coeffSigma2, gpp.coeffSigma2) + sum(log(gpcf.coeffSigma2));
  end
  if ~isempty(gpp.a)
    lp = lp + gpp.a.fh.lp(gpcf.a, gpp.a) + sum(log(gpcf.a));
  end
  if ~isempty(gpp.b)
    lp = lp + gpp.b.fh.lp(gpcf.b, gpp.b);
  end
end

function lpg = gpcf_linearLogistic_lpg(gpcf)
%GPCF_GPCF_LINEARLOGISTIC_LPG  Evaluate gradient of the log prior with respect
%                 to the parameters.
%
%  Description
%    LPG = GPCF_GPCF_LINEARLOGISTIC_LPG(GPCF) takes a covariance function
%    structure GPCF and returns LPG = d log (p(th))/dth, where th
%    is the vector of parameters. This is a mandatory subfunction 
%    used for example in gradient computations.
%
%  See also
%    GPCF_GPCF_LINEARLOGISTIC_PAK, GPCF_GPCF_LINEARLOGISTIC_UNPAK,
%    GPCF_GPCF_LINEARLOGISTIC_LP, GP_G 

  lpg = [];
  gpp=gpcf.p;
  
  if ~isempty(gpcf.p.coeffSigma2)
      lll=length(gpcf.coeffSigma2);
      lpgs = gpp.coeffSigma2.fh.lpg(gpcf.coeffSigma2, gpp.coeffSigma2);
      lpg = [lpg lpgs(1:lll).*gpcf.coeffSigma2+1 lpgs(lll+1:end)];
  end
  if ~isempty(gpcf.p.a)
      lll=length(gpcf.a);
      lpgs = gpp.a.fh.lpg(gpcf.a, gpp.a);
      lpg = [lpg lpgs(1:lll).*gpcf.a+1 lpgs(lll+1:end)];
  end
  if ~isempty(gpcf.p.b)
      lll=length(gpcf.b);
      lpgs = gpp.b.fh.lpg(gpcf.b, gpp.b);
      lpg = [lpg lpgs(1:lll) lpgs(lll+1:end)];
  end

end

function DKff = gpcf_linearLogistic_cfg(gpcf, x, x2, mask, i1)
%GPCF_GPCF_LINEARLOGISTIC_CFG  Evaluate gradient of covariance function
%                 with respect to the parameters
%
%  Description
%    DKff = GPCF_GPCF_LINEARLOGISTIC_CFG(GPCF, X) takes a covariance
%    function structure GPCF, a matrix X of input vectors and returns 
%    DKff, the gradients of covariance matrix Kff = k(X,X) with
%    respect to th (cell array with matrix elements). This is a 
%    mandatory subfunction used in gradient computations.
%
%    DKff = GPCF_GPCF_LINEARLOGISTIC_CFG(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2) with respect to th (cell array with matrix
%    elements). This subfunction is needed when using sparse 
%    approximations (e.g. FIC).
%
%    DKff = GPCF_GPCF_LINEARLOGISTIC_CFG(GPCF, X, [], MASK) takes a
%    covariance function structure GPCF, a matrix X of input vectors and
%    returns DKff, the diagonal of gradients of covariance matrix
%    Kff = k(X,X2) with respect to th (cell array with matrix
%    elements). This subfunction is needed when using sparse 
%    approximations (e.g. FIC).
%
%    DKff = GPCF_GPCF_LINEARLOGISTIC_CFG(GPCF,X,X2,MASK,i) takes a
%    covariance function structure GPCF, a matrix X of input vectors and 
%    returns DKff, the gradient of covariance matrix Kff = 
%    k(X,X2), or k(X,X) if X2 is empty, with respect to ith 
%    hyperparameter. This subfunction is needed when using
%    memory save option in gp_set.
%
%  See also
%   GPCF_GPCF_LINEARLOGISTIC_PAK, GPCF_GPCF_LINEARLOGISTIC_UNPAK,
%   GPCF_GPCF_LINEARLOGISTIC_LP, GP_G 

  [n, m] =size(x);

  DKff = {};
  
  if nargin==5
    % Use memory save option
    savememory=1;
    if i1==0
      % Return number of hyperparameters
      DKff=0;
      if ~isempty(gpcf.p.coeffSigma2)
        DKff=length(gpcf.coeffSigma2);
      end
      if ~isempty(gpcf.p.a)
          DKff=DKff+length(gpcf.a);
      end
      if ~isempty(gpcf.p.b)
          DKff=DKff+length(gpcf.b);
      end
      return
    end
  else
    savememory=0;
  end
  
  % Evaluate: DKff{1} = d Kff / d coeffSigma2
  % NOTE! Here we have already taken into account that the parameters are
  % transformed through log() and thus dK/dlog(p) = p * dK/dp

  h = logitinv(gpcf.a.*x + gpcf.b) - 0.5;
  
  % evaluate the gradient for training covariance
  if nargin == 2 || (isempty(x2) && isempty(mask))
    
    if isfield(gpcf, 'selectedVariables')
      if ~isempty(gpcf.p.coeffSigma2)
        if length(gpcf.coeffSigma2) == 1
          DKff{1}=gpcf.coeffSigma2*h(:,gpcf.selectedVariables)*(h(:,gpcf.selectedVariables)');
        else
          if ~savememory
            i1=1:length(gpcf.coeffSigma2);
          end
          for ii1=i1
            DD = gpcf.coeffSigma2(ii1)*h(:,gpcf.selectedVariables(ii1))*(h(:,gpcf.selectedVariables(ii1))');
            DD(abs(DD)<=eps) = 0;
            DKff{ii1}= (DD+DD')./2;
          end
        end
      end
      if ~isempty(gpcf.p.a)
          ii1= length(DKff) +1 ;
          hh = logitinv(gpcf.a.*x(:,gpcf.selectedVariables) + gpcf.b).^2.*...
              gpcf.a.*x(:,gpcf.selectedVariables).*exp(-gpcf.a.*x(:,gpcf.selectedVariables)-gpcf.b);
          DKff{ii1} = gpcf.coeffSigma2.* (hh*h(:,gpcf.selectedVariables)' + h(:,gpcf.selectedVariables)*hh');
      end
      if ~isempty(gpcf.p.b)
          ii1= length(DKff) +1 ;
          hh = logitinv(gpcf.a.*x(:,gpcf.selectedVariables) + gpcf.b).^2.*exp(-gpcf.a.*x(:,gpcf.selectedVariables)-gpcf.b);
          DKff{ii1} = gpcf.coeffSigma2* (hh*h(:,gpcf.selectedVariables)' + h(:,gpcf.selectedVariables)*hh');
      end
    else
      if ~isempty(gpcf.p.coeffSigma2)
        if length(gpcf.coeffSigma2) == 1
          DKff{1}=gpcf.coeffSigma2*h*(h');
        else
          if isa(gpcf.coeffSigma2,'single')
            epsi=eps('single');
          else
            epsi=eps;
          end
          if ~savememory
            i1=1:length(gpcf.coeffSigma2);
          end
          DKff=cell(1,length(i1));
          for ii1=i1
            DD = gpcf.coeffSigma2(ii1)*h(:,ii1)*(h(:,ii1)');
            DD(abs(DD)<=epsi) = 0;
            DKff{ii1}= (DD+DD')./2;
          end
        end
      end
      if ~isempty(gpcf.p.a)
          ii1= length(DKff) +1 ;
          hh = logitinv(gpcf.a.*x(:,gpcf.selectedVariables) + gpcf.b).^2.*gpcf.a.*x.*exp(gpcf.a.*x+gpcf.b);
          DKff{ii1} = gpcf.coeffSigma2* (hh*h' + h*hh');
      end
      if ~isempty(gpcf.p.b)
          ii1= length(DKff) +1 ;
          hh = logitinv(gpcf.a.*x(:,gpcf.selectedVariables) + gpcf.b).^2.*exp(gpcf.a.*x+gpcf.b);
          DKff{ii1} = gpcf.coeffSigma2* (hh*h' + h*hh');
      end
    end
    
    
    % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
  elseif nargin == 3 || isempty(mask)
    if size(x,2) ~= size(x2,2)
      error('gpcf_linearLogistic -> _ghyper: The number of columns in x and x2 has to be the same. ')
    end
    error('gpcf_linearLogistic -> _ghyper: "nargin == 3 || isempty(mask)" not implemented')
    if isfield(gpcf, 'selectedVariables')
      if ~isempty(gpcf.p.coeffSigma2)
        if length(gpcf.coeffSigma2) == 1
          DKff{1}=gpcf.coeffSigma2*x(:,gpcf.selectedVariables)*(x2(:,gpcf.selectedVariables)');
        else
          if ~savememory
            i1=1:length(gpcf.coeffSigma2);
          end
          for ii1=i1
            DKff{ii1}=gpcf.coeffSigma2(ii1)*x(:,gpcf.selectedVariables(ii1))*(x2(:,gpcf.selectedVariables(ii1))');
          end
        end
      end
    else
      if ~isempty(gpcf.p.coeffSigma2)
        if length(gpcf.coeffSigma2) == 1
          DKff{1}=gpcf.coeffSigma2*x*(x2');
        else
          if ~savememory
            i1=1:m;
          end            
          for ii1=i1
            DKff{ii1}=gpcf.coeffSigma2(ii1)*x(:,ii1)*(x2(:,ii1)');
          end
        end
      end
    end
    % Evaluate: DKff{1}    = d mask(Kff,I) / d coeffSigma2
    %           DKff{2...} = d mask(Kff,I) / d coeffSigma2
  elseif nargin == 4 || nargin == 5
    error('gpcf_linearLogistic -> _ghyper: "nargin == 4 || nargin == 5" not implemented')
    if isfield(gpcf, 'selectedVariables')
      if ~isempty(gpcf.p.coeffSigma2)
        if length(gpcf.coeffSigma2) == 1
          DKff{1}=gpcf.coeffSigma2*sum(x(:,gpcf.selectedVariables).^2,2); % d mask(Kff,I) / d coeffSigma2
        else
          if ~savememory
            i1=1:length(gpcf.coeffSigma2);
          end
          for ii1=i1
            DKff{ii1}=gpcf.coeffSigma2(ii1)*(x(:,gpcf.selectedVariables(ii1)).^2); % d mask(Kff,I) / d coeffSigma2
          end
        end
      end
    else
      if ~isempty(gpcf.p.coeffSigma2)
        if length(gpcf.coeffSigma2) == 1
          DKff{1}=gpcf.coeffSigma2*sum(x.^2,2); % d mask(Kff,I) / d coeffSigma2
        else
          if ~savememory
            i1=1:m;
          end
          for ii1=i1
            DKff{ii1}=gpcf.coeffSigma2(ii1)*(x(:,ii1).^2); % d mask(Kff,I) / d coeffSigma2
          end
        end
      end
    end
  end
  if savememory
    DKff=DKff{i1};
  end
end

function C = gpcf_linearLogistic_cov(gpcf, x1, x2, varargin)
%GP_GPCF_LINEARLOGISTIC_COV  Evaluate covariance matrix between two input
%                            vectors 
%
%  Description         
%    C = GP_GPCF_LINEARLOGISTIC_COV(GP, TX, X) takes in covariance function of
%    a Gaussian process GP and two matrixes TX and X that contain
%    input vectors to GP. Returns covariance matrix C. Every
%    element ij of C contains covariance between inputs i in TX
%    and j in X. This is a mandatory subfunction used for example in
%    prediction and energy computations.
%
%  See also
%    GPCF_GPCF_LINEARLOGISTIC_TRCOV, GPCF_GPCF_LINEARLOGISTIC_TRVAR,
%    GP_COV, GP_TRCOV 
  
  if isempty(x2)
    x2=x1;
  end
  [n1,m1]=size(x1);
  [n2,m2]=size(x2);

  if m1~=m2
    error('the number of columns of X1 and X2 has to be same')
  end
  
  if isfield(gpcf, 'selectedVariables')
      h1 = logitinv(gpcf.a.*x1(:,gpcf.selectedVariables) + gpcf.b) - 0.5;
      h2 = logitinv(gpcf.a.*x2(:,gpcf.selectedVariables) + gpcf.b) - 0.5;        
      C = h1*diag(gpcf.coeffSigma2)*(h2');
  else
      h1 = logitinv(gpcf.a.*x1 + gpcf.b) - 0.5;
      h2 = logitinv(gpcf.a.*x2 + gpcf.b) - 0.5;
      C = h1*diag(gpcf.coeffSigma2)*(h2');
  end
  C(abs(C)<=eps) = 0;
end

function C = gpcf_linearLogistic_trcov(gpcf, x)
%GP_GPCF_LINEARLOGISTIC_TRCOV  Evaluate training covariance matrix of
%                              inputs 
%
%  Description
%    C = GP_GPCF_LINEARLOGISTIC_TRCOV(GP, TX) takes in covariance function
%    of a Gaussian process GP and matrix TX that contains training
%    input vectors. Returns covariance matrix C. Every element ij
%    of C contains covariance between inputs i and j in TX. This 
%    is a mandatory subfunction used for example in prediction and 
%    energy computations.
%
%  See also
%    GPCF_GPCF_LINEARLOGISTIC_COV, GPCF_GPCF_LINEARLOGISTIC_TRVAR, GP_COV,
%    GP_TRCOV 

  if isfield(gpcf, 'selectedVariables')
      h = logitinv(gpcf.a.*x(:,gpcf.selectedVariables) + gpcf.b) - 0.5;
      C = h*diag(gpcf.coeffSigma2)*(h');
  else
      h = logitinv(gpcf.a.*x + gpcf.b) - 0.5;
      C = h*diag(gpcf.coeffSigma2)*(h');
  end
  C(abs(C)<=eps) = 0;
  C = (C+C')./2;

end


function C = gpcf_linearLogistic_trvar(gpcf, x)
%GP_GPCF_LINEARLOGISTIC_TRVAR  Evaluate training variance vector
%
%  Description
%    C = GP_GPCF_LINEARLOGISTIC_TRVAR(GPCF, TX) takes in covariance
%    function of a Gaussian process GPCF and matrix TX that contains
%    training inputs. Returns variance vector C. Every element i
%    of C contains variance of input i in TX. This is a mandatory 
%    subfunction used for example in prediction and energy computations.
%
%
%  See also
%    GPCF_GPCF_LINEARLOGISTIC_COV, GP_COV, GP_TRCOV

  if length(gpcf.coeffSigma2) == 1
    if isfield(gpcf, 'selectedVariables')
        h = logitinv(gpcf.a.*x(:,gpcf.selectedVariables) + gpcf.b) - 0.5;
      C=gpcf.coeffSigma2.*sum(h.^2,2);
    else
        h = logitinv(gpcf.a.*x + gpcf.b) - 0.5;
      C=gpcf.coeffSigma2.*sum(h.^2,2);
    end
  else
    if isfield(gpcf, 'selectedVariables')
        h = logitinv(gpcf.a.*x(:,gpcf.selectedVariables) + gpcf.b) - 0.5;
      C=sum(repmat(gpcf.coeffSigma2, size(x,1), 1).*h.^2,2);
    else
        h = logitinv(gpcf.a.*x + gpcf.b) - 0.5;
      C=sum(repmat(gpcf.coeffSigma2, size(h,1), 1).*h.^2,2);
    end
  end
  C(abs(C)<eps)=0;
  
end

function reccf = gpcf_linearLogistic_recappend(reccf, ri, gpcf)
%RECAPPEND Record append
%
%  Description
%    RECCF = GPCF_GPCF_LINEARLOGISTIC_RECAPPEND(RECCF, RI, GPCF) takes a
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
    reccf.type = 'gpcf_linearLogistic';

    % Initialize parameters
    reccf.coeffSigma2= [];
    reccf.a= [];
    reccf.b= [];

    % Set the function handles
    reccf.fh.pak = @gpcf_linearLogistic_pak;
    reccf.fh.unpak = @gpcf_linearLogistic_unpak;
    reccf.fh.lp = @gpcf_linearLogistic_lp;
    reccf.fh.lpg = @gpcf_linearLogistic_lpg;
    reccf.fh.cfg = @gpcf_linearLogistic_cfg;
    reccf.fh.cfdg = @gpcf_linearLogistic_cfdg;
    reccf.fh.cfdg2 = @gpcf_linearLogistic_cfdg2;
    reccf.fh.ginput = @gpcf_linearLogistic_ginput;
    reccf.fh.ginput2 = @gpcf_linearLogistic_ginput2;
    reccf.fh.ginput3 = @gpcf_linearLogistic_ginput3;
    reccf.fh.ginput4 = @gpcf_linearLogistic_ginput4;
    reccf.fh.cov = @gpcf_linearLogistic_cov;
    reccf.fh.trcov  = @gpcf_linearLogistic_trcov;
    reccf.fh.trvar  = @gpcf_linearLogistic_trvar;
    reccf.fh.recappend = @gpcf_linearLogistic_recappend;
    reccf.p=[];
    reccf.p.coeffSigma2=[];
    if ~isempty(ri.p.coeffSigma2)
      reccf.p.coeffSigma2 = ri.p.coeffSigma2;
    end
    if ~isempty(ri.p.a)
      reccf.p.a = ri.p.a;
    end
    if ~isempty(ri.p.b)
      reccf.p.b = ri.p.b;
    end

  else
    % Append to the record
    gpp = gpcf.p;
    
    % record coeffSigma2
    reccf.coeffSigma2(ri,:)=gpcf.coeffSigma2;
    if isfield(gpp,'coeffSigma2') && ~isempty(gpp.coeffSigma2)
      reccf.p.coeffSigma2 = gpp.coeffSigma2.fh.recappend(reccf.p.coeffSigma2, ri, gpcf.p.coeffSigma2);
    end

    reccf.a(ri,:)=gpcf.a;
    if isfield(gpp,'a') && ~isempty(gpp.a)
      reccf.p.a = gpp.a.fh.recappend(reccf.p.a, ri, gpcf.p.a);
    end

    reccf.b(ri,:)=gpcf.b;
    if isfield(gpp,'b') && ~isempty(gpp.b)
      reccf.p.b = gpp.b.fh.recappend(reccf.p.b, ri, gpcf.p.b);
    end

    
    if isfield(gpcf, 'selectedVariables')
      reccf.selectedVariables = gpcf.selectedVariables;
    end
  end
end
