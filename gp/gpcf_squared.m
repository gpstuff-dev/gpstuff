function gpcf = gpcf_squared(varargin)
%GPCF_SQUARED  Create a squared (dot product) covariance function
%
%  Description
%    GPCF = GPCF_SQUARED('PARAM1',VALUE1,'PARAM2,VALUE2,...) creates
%    a squared (dot product) covariance function structure in which
%    the named parameters have the specified values. Any unspecified
%    parameters are set to default values. The squared covariance function
%    corresponds to x.^2 mean function and the respective covariance matrix
%    is given as C = x.^2*diag(gpcf.coeffSigma2)*(x'.^2);
%
%    GPCF = GPCF_SQUARED(GPCF,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a covariance function structure with the named
%    parameters altered with the specified values.
%  
%    Parameters for squared (dot product) covariance function
%      interactions      - twoway interactions (default off)
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
%  See also
%    GP_SET, GPCF_*, PRIOR_*, MEAN_*
%
% Copyright (c) 2007-2016 Jarno Vanhatalo
% Copyright (c) 2008-2010 Jaakko Riihimäki
% Copyright (c) 2010 Aki Vehtari
% Copyright (c) 2014 Arno Solin

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPCF_SQUARED';
  ip.addOptional('gpcf', [], @isstruct);
  ip.addParamValue('interactions', 'off', @(x) ismember(x,{'on' 'off'}))
  ip.addParamValue('coeffSigma2',10, @(x) isvector(x) && all(x>0));
  ip.addParamValue('coeffSigma2_prior',prior_logunif, @(x) isstruct(x) || isempty(x));
  ip.addParamValue('selectedVariables',[], @(x) isvector(x) && all(x>0));
  ip.parse(varargin{:});
  gpcf=ip.Results.gpcf;

  if isempty(gpcf)
    init=true;
    gpcf.type = 'gpcf_squared';
  else
    if ~isfield(gpcf,'type') && ~isequal(gpcf.type,'gpcf_squared')
      error('First argument does not seem to be a valid covariance function structure')
    end
    init=false;
  end
  
  if init || ~ismember('interactions',ip.UsingDefaults)
    gpcf.interactions=ip.Results.interactions;
  end
  
  % Initialize parameter
  if init || ~ismember('coeffSigma2',ip.UsingDefaults)
    gpcf.coeffSigma2=ip.Results.coeffSigma2;
  end

  % Initialize prior structure
  if init
    gpcf.p=[];
  end
  if init || ~ismember('coeffSigma2_prior',ip.UsingDefaults)
    gpcf.p.coeffSigma2=ip.Results.coeffSigma2_prior;
  end
  if ~ismember('selectedVariables',ip.UsingDefaults)
    selectedVariables=ip.Results.selectedVariables;
    if ~isempty(selectedVariables)
      gpcf.selectedVariables = selectedVariables;
    end
  end
  
  if init
    % Set the function handles to the subfunctions
    gpcf.fh.pak = @gpcf_squared_pak;
    gpcf.fh.unpak = @gpcf_squared_unpak;
    gpcf.fh.lp = @gpcf_squared_lp;
    gpcf.fh.lpg = @gpcf_squared_lpg;
    gpcf.fh.cfg = @gpcf_squared_cfg;
    gpcf.fh.cfdg = @gpcf_squared_cfdg;
    gpcf.fh.cfdg2 = @gpcf_squared_cfdg2;
    gpcf.fh.ginput = @gpcf_squared_ginput;
    gpcf.fh.ginput2 = @gpcf_squared_ginput2;
    gpcf.fh.ginput3 = @gpcf_squared_ginput3;
    gpcf.fh.ginput4 = @gpcf_squared_ginput4;
    gpcf.fh.cov = @gpcf_squared_cov;
    gpcf.fh.trcov  = @gpcf_squared_trcov;
    gpcf.fh.trvar  = @gpcf_squared_trvar;
    gpcf.fh.recappend = @gpcf_squared_recappend;
    gpcf.fh.cf2ss = @gpcf_squared_cf2ss;
  end        

end

function [w, s, h] = gpcf_squared_pak(gpcf, w)
%GPCF_squared_PAK  Combine GP covariance function parameters into one vector
%
%  Description
%    W = GPCF_squared_PAK(GPCF) takes a covariance function
%    structure GPCF and combines the covariance function
%    parameters and their hyperparameters into a single row
%    vector W. This is a mandatory subfunction used for 
%    example in energy and gradient computations.
%
%       w = [ log(gpcf.coeffSigma2)
%             (hyperparameters of gpcf.coeffSigma2)]'
%
%  See also
%    GPCF_squared_UNPAK
  
  w = []; s = {}; h =[];
  if ~isempty(gpcf.p.coeffSigma2)
    w = log(gpcf.coeffSigma2);
    if numel(gpcf.coeffSigma2)>1
      s = [s; sprintf('log(squared.coeffSigma2 x %d)',numel(gpcf.coeffSigma2))];
    else
      s = [s; 'log(squared.coeffSigma2)'];
    end
    h = [h ones(1, numel(gpcf.coeffSigma2))];
    % Hyperparameters of coeffSigma2
    [wh, sh, hh] = gpcf.p.coeffSigma2.fh.pak(gpcf.p.coeffSigma2);
    sh=strcat(repmat('prior-', size(sh,1),1),sh);
    w = [w wh];
    s = [s; sh];
    h = [h 1+hh];
  end
end

function [gpcf, w] = gpcf_squared_unpak(gpcf, w)
%GPCF_squared_UNPAK  Sets the covariance function parameters 
%                   into the structure
%
%  Description
%    [GPCF, W] = GPCF_squared_UNPAK(GPCF, W) takes a covariance
%    function structure GPCF and a hyper-parameter vector W, and
%    returns a covariance function structure identical to the
%    input, except that the covariance hyper-parameters have been
%    set to the values in W. Deletes the values set to GPCF from
%    W and returns the modified W. This is a mandatory subfunction 
%    used for example in energy and gradient computations.
%
%    Assignment is inverse of  
%       w = [ log(gpcf.coeffSigma2)
%             (hyperparameters of gpcf.coeffSigma2)]'
%
%  See also
%   GPCF_squared_PAK
  
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
end

function lp = gpcf_squared_lp(gpcf)
%GPCF_squared_LP  Evaluate the log prior of covariance function parameters
%
%  Description
%    LP = GPCF_squared_LP(GPCF) takes a covariance function
%    structure GPCF and returns log(p(th)), where th collects the
%    parameters. This is a mandatory subfunction used for example 
%    in energy computations.
%
%  See also
%   GPCF_squared_PAK, GPCF_squared_UNPAK, GPCF_squared_LPG, GP_E

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
end

function lpg = gpcf_squared_lpg(gpcf)
%GPCF_squared_LPG  Evaluate gradient of the log prior with respect
%                 to the parameters.
%
%  Description
%    LPG = GPCF_squared_LPG(GPCF) takes a covariance function
%    structure GPCF and returns LPG = d log (p(th))/dth, where th
%    is the vector of parameters. This is a mandatory subfunction 
%    used for example in gradient computations.
%
%  See also
%    GPCF_squared_PAK, GPCF_SQUARED_UNPAK, GPCF_SQUARED_LP, GP_G

  lpg = [];
  gpp=gpcf.p;
  
  if ~isempty(gpcf.p.coeffSigma2)            
    lll=length(gpcf.coeffSigma2);
    lpgs = gpp.coeffSigma2.fh.lpg(gpcf.coeffSigma2, gpp.coeffSigma2);
    lpg = [lpg lpgs(1:lll).*gpcf.coeffSigma2+1 lpgs(lll+1:end)];
  end
end

function DKff = gpcf_squared_cfg(gpcf, x, x2, mask, i1)
%GPCF_SQUARED_CFG  Evaluate gradient of covariance function
%                 with respect to the parameters
%
%  Description
%    DKff = GPCF_SQUARED_CFG(GPCF, X) takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns
%    DKff, the gradients of covariance matrix Kff = k(X,X) with
%    respect to th (cell array with matrix elements). This is a 
%    mandatory subfunction used in gradient computations.
%
%    DKff = GPCF_SQUARED_CFG(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2) with respect to th (cell array with matrix
%    elements). This subfunction is needed when using sparse 
%    approximations (e.g. FIC).
%
%    DKff = GPCF_SQUARED_CFG(GPCF, X, [], MASK) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the diagonal of gradients of covariance matrix
%    Kff = k(X,X2) with respect to th (cell array with matrix
%    elements). This subfunction is needed when using sparse 
%    approximations (e.g. FIC).
%
%    DKff = GPCF_SQUARED_CFG(GPCF,X,X2,MASK,i) takes a covariance 
%    function structure GPCF, a matrix X of input vectors and 
%    returns DKff, the gradient of covariance matrix Kff = 
%    k(X,X2), or k(X,X) if X2 is empty, with respect to ith 
%    hyperparameter. This subfunction is needed when using
%    memory save option in gp_set.
%
%  See also
%   GPCF_SQUARED_PAK, GPCF_SQUARED_UNPAK, GPCF_SQUARED_LP, GP_G

  if nargin>2 && ~isempty(x2)
      if size(x,2) ~= size(x2,2)
          error('gpcf_squared -> _cfg: the number of columns of X1 and X2 has to be same')
      end
  end

  if isfield(gpcf, 'selectedVariables')
      x = x(:,gpcf.selectedVariables);
  end
  [n, m] =size(x);
  h = x.^2;
  if isequal(gpcf.interactions,'on')
      for xi1=1:m
          for xi2=xi1+1:m
              h = [h x(:,xi1).*x(:,xi2)];
          end
      end
  end
  
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
      return
    end
  else
    savememory=0;
  end
  
  % Evaluate: DKff{1} = d Kff / d coeffSigma2
  % NOTE! Here we have already taken into account that the parameters are transformed
  % through log() and thus dK/dlog(p) = p * dK/dp

  
  % evaluate the gradient for training covariance
  if nargin == 2 || (isempty(x2) && isempty(mask))
      
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
    
    
    % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
  elseif nargin == 3 || isempty(mask)
      
      error('this part of the subfunction has not been tested')
      
      if isfield(gpcf, 'selectedVariables')
          x2 = x2(:,gpcf.selectedVariables);
      end
      h2 = x2.^2;
      if isequal(gpcf.interactions,'on')
          for xi1=1:m
              for xi2=xi1+1:m
                  h2 = [h2 x2(:,xi1).*x2(:,xi2)];
              end
          end
      end
      
      if ~isempty(gpcf.p.coeffSigma2)
          if length(gpcf.coeffSigma2) == 1
              DKff{1}=gpcf.coeffSigma2*h*(h2');
          else
              if ~savememory
                  i1=1:m;
              end
              for ii1=i1
                  DKff{ii1}=gpcf.coeffSigma2(ii1)*h(:,ii1)*(h2(:,ii1)');
              end
          end
      end
      % Evaluate: DKff{1}    = d mask(Kff,I) / d coeffSigma2
    %           DKff{2...} = d mask(Kff,I) / d coeffSigma2
  elseif nargin == 4 || nargin == 5
    
      error('this part of the subfunction has not been tested')
    
      if ~isempty(gpcf.p.coeffSigma2)
        if length(gpcf.coeffSigma2) == 1
          DKff{1}=gpcf.coeffSigma2*sum(h.^2,2); % d mask(Kff,I) / d coeffSigma2
        else
          if ~savememory
            i1=1:m;
          end
          for ii1=i1
            DKff{ii1}=gpcf.coeffSigma2(ii1)*(h(:,ii1).^2); % d mask(Kff,I) / d coeffSigma2
          end
        end
      end
  end
  if savememory
    DKff=DKff{i1};
  end
end

function DKff = gpcf_squared_cfdg(gpcf, x, x2, dims)
%GPCF_SQUARED_CFDG  Evaluate gradient of covariance function, of
%                  which has been taken partial derivative with
%                  respect to x, with respect to parameters.
%
%  Description
%    DKff = GPCF_SQUARED_CFDG(GPCF, X) takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns
%    DKff, the gradients of derivatived covariance matrix
%    dK(df,f)/dhyp = d(d k(X,X)/dx)/dhyp, with respect to the
%    parameters
%
%    Evaluate: DKff{1:m} = d Kff / d coeffSigma2
%    m is the dimension of inputs. This subfunction is needed when using
%    derivative observations.
%
%    Note! When coding the derivatives of the covariance function, remember
%    to double check them. See gp_cov for lines of code to check the
%    matrices
%
%  See also
%    GPCF_SQUARED_GINPUT

[~,m]=size(x);
if nargin<3
    x2=x;
end
if nargin < 4 || isempty(dims)
    dims = 1:m;
end
ii1=0;
DKff={};
if isfield(gpcf,'selectedVariables')
    selVars = gpcf.selectedVariables;
else
    selVars = 1:m;
end
c = gpcf.coeffSigma2;

h = x.^2;
h2 = x2.^2;

ii1=0;
DKff={};
if ~isempty(gpcf.p.coeffSigma2)
  if length(gpcf.coeffSigma2)==1
    % One coeffSigma2
    for i1=dims
      if isfield(gpcf, 'selectedVariables') && sum(gpcf.selectedVariables==i1)==0
        DK{i1}=zeros(size(x,1),size(x2,1));
      else
        DK{i1}=c(1).*2*x(:,i1)*h2(:,i1)';
        if isequal(gpcf.interactions,'on')
            for xi2=selVars
                if xi2~=i1
                    DK{i1} = DK{i1} + c(1)*x(:,xi2)*(x2(:,i1).*x2(:,xi2))';
                end
            end
        end
      end
    end
    ii1=ii1+1;
    DKff{ii1}=cat(1,DK{1:end});
  else
      % vector of coeffSigma2s
      for i1=1:length(selVars)
          for j=dims
              if selVars(i1)~=j %|| (isfield(gpcf, 'selectedVariables') && sum(gpcf.selectedVariables==i1)==0)
                  DK{j}=zeros(size(x,1),size(x2,1));
              else
                  DK{j}=c(i1).*2*x(:,j)*h2(:,j)';
              end
          end
          ii1=ii1+1;
          DKff{ii1}=cat(1,DK{1:end});
      end
      if isequal(gpcf.interactions,'on')
          for xi1=1:length(selVars)
              for xi2=xi1+1:length(selVars)
                  i1=i1+1;
                  for j=dims
                      if j==xi1 
                          DK{j} = c(i1)*x(:,xi2)*(x2(:,xi1).*x2(:,xi2))';
                      elseif j==xi2
                          DK{j} = c(i1)*x(:,xi1)*(x2(:,xi1).*x2(:,xi2))';
                      else
                          DK{j}=zeros(size(x,1),size(x2,1));
                      end
                  end
                  ii1=ii1+1;
                  DKff{ii1}=cat(1,DK{1:end});
              end
          end
      end
  end
end

end

function DKff = gpcf_squared_cfdg2(gpcf, x, x2, dims1, dims2)
%GPCF_SQUARED_CFDG2  Evaluate gradient of covariance function, of which has
%                    been taken partial derivatives with respect to both
%                    input variables x, with respect to parameters.
%
%  Description
%    DKff = GPCF_SQUARED_CFDG2(GPCF, X) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of derivative covariance matrix
%    dK(df,df)/dhyp = d(d^2 k(X1,X2)/dX1dX2)/dhyp with respect to
%    the parameters
%
%    Evaluate: DKff{1:m} = d K(df,df) / d coeffSigma
%    m is the dimension of inputs.  This subfunction is needed when using
%    derivative observations.
%
%    Note! When coding the derivatives of the covariance function, remember
%    to double check them. See gp_cov for lines of code to check the
%    matrices
%
%  See also
%   GPCF_SQUARED_GINPUT, GPCF_SQUARED_GINPUT2


[~, m] =size(x);
if nargin <3 || isempty(x2)
    x2=x;
end
if nargin < 4 || isempty(dims1)
    %dims1 = 1:m;
    error('dims1 needs to be given')
end
if nargin < 5 || isempty(dims2)
    %dims2 = 1:m;
    error('dims2 needs to be given')
end
if isfield(gpcf,'selectedVariables')
    selVars = gpcf.selectedVariables;
else
    selVars = 1:m;
end
% NOTICE. AS OF NOW we assume that dims1 and dims2 are scalars

ii1=0;
if length(gpcf.coeffSigma2)==1
    c=repmat(gpcf.coeffSigma2,1,(1+length(selVars))*length(selVars)/2);
else
    c=gpcf.coeffSigma2;
end
if length(gpcf.coeffSigma2)==1
    % One coeffSigma2
    if dims1~=dims2 %|| (isfield(gpcf, 'selectedVariables') && (sum(gpcf.selectedVariables==j)==0 || sum(gpcf.selectedVariables==k)==0))
        DK=zeros(size(x,1),size(x2,1));
        if isequal(gpcf.interactions,'on') && (sum(selVars==dims1)>0 || sum(selVars==dims2)>0)
            DK = c(1)*x(:,dims1)*x2(:,dims2)';
        end
    else
        if sum(selVars==dims1)==0
            DK=zeros(size(x,1),size(x2,1));
        else
            DK=c(1).*4.*x(:,dims1)*x2(:,dims2)';
        end
        if isequal(gpcf.interactions,'on') && sum(selVars==dims1)>0
            for xi2=selVars
                if xi2~=dims1
                    DK = DK + c(1)*x(:,xi2)*x2(:,xi2)';
                end
            end
        end
    end
    ii1=ii1+1;
    DKff{ii1}=DK;
else
    % vector of coeffSigma2s
    for i1=1:length(selVars)
        if dims1~=dims2 || dims2~=selVars(i1)
            DK=zeros(size(x,1),size(x2,1));
        else
            DK=c(i1).*4.*x(:,dims1)*x2(:,dims2)';
        end
        ii1=ii1+1;
        DKff{ii1}=DK;
    end
    if isequal(gpcf.interactions,'on')
        for xi1=1:length(selVars)
            for xi2=xi1+1:length(selVars)
                i1=i1+1;
                %if k==xi1 && j==xi2
                if dims1==xi1 && dims2==xi2
                    DK = c(i1)*x(:,xi2)*x2(:,xi1)';
                %elseif k==xi2 && j==xi1
                elseif dims1==xi2 && dims2==xi1
                    DK = c(i1)*x(:,xi1)*x2(:,xi2)';
                %elseif k==j && k==xi1
                elseif dims1==dims2 && dims1==xi1    
                    DK = c(i1)*x(:,xi2)*x2(:,xi2)';
                %elseif k==j && k==xi2
                elseif dims1==dims2 && dims1==xi2
                    DK = c(i1)*x(:,xi1)*x2(:,xi1)';
                else
                    DK=zeros(size(x,1),size(x2,1));
                end
                ii1=ii1+1;
                DKff{ii1}=DK;
            end
        end
    end
end

end


function DKff = gpcf_squared_ginput(gpcf, x, x2, i1)
%GPCF_SQUARED_GINPUT  Evaluate gradient of covariance function with 
%                    respect to x.
%
%  Description
%    DKff = GPCF_SQUARED_GINPUT(GPCF, X, X2) takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns DKff, the
%    gradients of covariance matrix Kff = k(X,X2) with respect to X (cell
%    array with matrix elements). If called with only two inputs
%    GPCF_SQUARED_GINPUT(GPCF, X), X2=X. This subfunction is needed when
%    computing gradients with respect to inducing inputs in sparse
%    approximations. 
%
%    DKff = GPCF_SQUARED_GINPUT(GPCF, X, X2, i) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2) with respect to ith covariate in X (matrix).
%    This subfunction is needed when using memory save option
%    in gp_set.
%
%    Note! When coding the derivatives of the covariance function, remember
%    to double check them. See gp_cov for lines of code to check the
%    matrices
%
%  See also
%   GPCF_SQUARED_PAK, GPCF_SQUARED_UNPAK, GPCF_SQUARED_LP, GP_G        
  
if isfield(gpcf, 'selectedVariables') 
    error('The selectedVariables option has not yet been implemented for gpcf_squared with derivobs=''on'' ')
    % notice, some parts of the code already take into account the
    % selectedVariables but the code has not been checked
end

[n, m] =size(x);

if nargin==4
    % Use memory save option
    savememory=1;
    if i1==0
        % Return number of covariates
        DKff=m;
        return
    end
else
    savememory=0;
end

if nargin == 2 || isempty(x2)
    
    xx = x.^2;
    if length(gpcf.coeffSigma2)==1
        s=gpcf.coeffSigma2.*ones(1,(1+m)*m/2);
    else
        s=gpcf.coeffSigma2;
    end
    ii1 = 0;
    if nargin<4
        i1=1:m;
    end
    for j = 1:n
        for i=i1
            DK = zeros(n);
            DK(j,:)=2*s(i)*x(j,i)*xx(:,i)';
            if isequal(gpcf.interactions,'on')
                for xi2=i1
                    if xi2~=i
                        DK(j,:) = DK(j,:) + s(i)*x(j,xi2).*(x(:,i).*x(:,xi2))';
                    end
                end
            end
            DK = DK + DK';
            ii1 = ii1 + 1;
            DKff{ii1} = DK;
        end
    end
    
elseif nargin == 3 || nargin == 4
    
    xx2 = x2.^2;
    if length(gpcf.coeffSigma2)==1
        s=gpcf.coeffSigma2.*ones(1,(1+m)*m/2);
    else
        s=gpcf.coeffSigma2;
    end
    ii1 = 0;
    if ~savememory
        i1=1:m;
    end
    for j = 1:n
        for i=i1
            DK = zeros(n, size(x2,1));
            DK(j,:)=2*s(i)*x(j,i)*xx2(:,i)';
            if isequal(gpcf.interactions,'on')
                for xi2=i1
                    if xi2~=i
                        DK(j,:) = DK(j,:) + s(i)*x(j,xi2).*(x2(:,i).*x2(:,xi2))';
                    end
                end
            end
            ii1 = ii1 + 1;
            DKff{ii1} = DK;
        end
    end
end

end

function DKff = gpcf_squared_ginput2(gpcf, x, x2, dims,takeOnlyDiag)
%GPCF_SQUARED_GINPUT2  Evaluate gradient of covariance function with
%                   respect to both input variables x and x2 in
%                   same dimension.
%
%  Description
%    DKff = GPCF_SQUARED_GINPUT2(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of twice derivatived covariance
%    matrix K(df,df) = dk(X1,X2)/dX1dX2 (cell array with matrix
%    elements). Input variable's dimensions are expected to be
%    same.  This subfunction is needed when using derivative 
%    observations.
%   
%    Note! When coding the derivatives of the covariance function, remember
%    to double check them. See gp_cov for lines of code to check the
%    matrices
%
%  See also
%    GPCF_SQUARED_GINPUT, GPCF_SQUARED_GINPUT2, GPCF_SQUARED_CFDG2       

[~,m]=size(x);
ii1=0;
if nargin<4 || isempty(dims)
    dims=1:m;
end
if length(gpcf.coeffSigma2)==1
    c=repmat(gpcf.coeffSigma2,1,(1+m)*m/2);
else
    c=gpcf.coeffSigma2;
end
if isfield(gpcf, 'selectedVariables')
    sv = gpcf.selectedVariables;
else 
    sv = 1:m;
end

if nargin==5 && isequal(takeOnlyDiag,'takeOnlyDiag')
    for i1=dims
        if ~any(sv==i1)%isfield(gpcf, 'selectedVariables') && sum(gpcf.selectedVariables==i1)==0
            DK=zeros(size(x,1),1);
        else
            DK=c(i1).*4.*(x(:,i1).*x2(:,i1));
            if isequal(gpcf.interactions,'on')
                i2=length(sv);
                for xi1=1:length(sv)
                    for xi2=xi1+1:length(sv)
                        if any(sv==xi1) && any(sv==xi2)
                            i2=i2+1;
                            if i1==xi1
                                DK = DK + c(i2)*x(:,xi2).*x2(:,xi2);
                            elseif i1==xi2
                                DK = DK + c(i2)*x(:,xi1).*x2(:,xi1);
                            end
                        end
                    end
                end
            end
        end
        ii1=ii1+1;
        DKff{ii1}=DK;
    end
else
    for i1=dims
        if ~any(sv==i1)%isfield(gpcf, 'selectedVariables') && sum(gpcf.selectedVariables==i1)==0
            DK=zeros(size(x,1),size(x2,1));
        else
            DK=c(i1).*4.*(x(:,i1)*x2(:,i1)');
            if isequal(gpcf.interactions,'on')
                i2=length(sv);
                for xi1=1:length(sv)
                    for xi2=xi1+1:length(sv)
                        if any(sv==xi1) && any(sv==xi2)
                            i2=i2+1;
                            if i1==xi1
                                DK = DK + c(i2)*x(:,xi2)*x2(:,xi2)';
                            elseif i1==xi2
                                DK = DK + c(i2)*x(:,xi1)*x2(:,xi1)';
                            end
                        end
                    end
                end
            end
        end
        ii1=ii1+1;
        DKff{ii1}=DK;
    end
end
end

function DKff = gpcf_squared_ginput3(gpcf, x, x2, dims1, dims2)
%GPCF_SQUARED_GINPUT3  Evaluate gradient of covariance function with
%                   respect to both input variables x and x2 (in
%                   different dimensions).
%
%  Description
%    DKff = GPCF_SQUARED_GINPUT3(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of twice derivatived covariance
%    matrix K(df,df) = dk(X1,X2)/dX1dX2 (cell array with matrix
%    elements). The derivative is calculated in multidimensional
%    problem between input's observation dimensions which are not
%    same. This subfunction is needed when using derivative 
%    observations.
%
%    DKff is a cell array with the following elements:
%      DKff{1} = dk(X1,X2)/dX1_1dX2_2
%      DKff{2} = dk(X1,X2)/dX1_1dX2_3
%       ... 
%      DKff{m-1} = dk(X1,X2)/dX1_1dX2_m
%      DKff{m} = dk(X1,X2)/dX1_2dX2_3
%       ...
%      DKff{m} = dk(X1,X2)/dX1_(m-1)dX2_m
%    where _m denotes the input dimension with respect to which the
%    gradient is calculated.
%
%    Note! When coding the derivatives of the covariance function, remember
%    to double check them. See gp_cov for lines of code to check the
%    matrices
%   
%  See also
%    GPCF_SQUARED_GINPUT, GPCF_SQUARED_GINPUT2, GPCF_SQUARED_CFDG2        


[~,m]=size(x);
if nargin < 3
    error('Needs at least 3 input arguments')
end
if nargin<4 || isempty(dims1)
    dims1=1:m;
end
if nargin<5 || isempty(dims2)
    dims2=1:m;
end
if isfield(gpcf, 'selectedVariables')
    sv = gpcf.selectedVariables;
else 
    sv = 1:m;
end

ii1=0;
if length(gpcf.coeffSigma2)==1
  c=repmat(gpcf.coeffSigma2,1,(1+m)*m/2);
else
  c=gpcf.coeffSigma2;
end
DK=zeros(size(x,1),size(x2,1));
i2=m;
for i=dims1
    for j=dims2
        ii1=ii1+1;
        if isequal(gpcf.interactions,'on') && any(sv==i) && any(sv==j)
            i2 = min(i,j)*length(sv) - 0.5*min(i,j)*(min(i,j)-1) + max(i,j)-min(i,j);
            DKff{ii1} = c(i2)*x(:,j)*x2(:,i)';
        else
            DKff{ii1}=DK;
        end
    end
end

% i2=m;
% for i=1:m-1
%     for j=i+1:m
%         i2=i2+1;
%         if any(dims1==i) && any(dims2==j)
%             ii1=ii1+1;
%             if isequal(gpcf.interactions,'on') && any(sv==i) && any(sv==j)
% %             if isequal(gpcf.interactions,'on') &&...
% %                     ~(isfield(gpcf, 'selectedVariables') && (sum(gpcf.selectedVariables==i)==0 || sum(gpcf.selectedVariables==j)==0))
%                 DKff{ii1} = c(i2)*x(:,j)*x2(:,i)';
%             else
%                 DKff{ii1}=DK;
%             end
%         end
%     end
% end

end

function DKff = gpcf_squared_ginput4(gpcf, x, x2, dims)
%GPCF_SQUARED_GINPUT  Evaluate gradient of covariance function with respect
%                    to x. Simplified and faster version of squared_ginput,
%                    returns full matrices. 
%
%  Description
%    DKff = GPCF_SQUARED_GINPUT4(GPCF, X, X2) takes a covariance function
%    structure GPCF, matrices X and X2 of input vectors and returns DKff,
%    the gradients of covariance matrix Kff = k(X,X2) with respect to X
%    (whole matrix); that is d k(X,X2)/dX. If called with only two inputs
%    GPCF_SQUARED_GINPUT4(GPCF, X), X2=X.
%
%    This subfunction is needed when using derivative observations. 
%
%    Note! When coding the derivatives of the covariance function, remember
%    to double check them. See gp_cov for lines of code to check the
%    matrices
%
%  See also
%    GPCF_SQUARED_PAK, GPCF_SQUARED_UNPAK, GPCF_SQUARED_LP, GP_G


% if isfield(gpcf, 'selectedVariables') 
%     error('The selectedVariables option has not yet been implemented for gpcf_squared with derivobs=''on'' ')
%     % notice, some parts of the code already take into account the
%     % selectedVariables but the code has not been checked
% end
% 
% if isfield(gpcf, 'selectedVariables')
%     m = length(gpcf.selectedVariables);
%     sv = gpcf.selectedVariables;
% else 
%     [~,m]=size(x);
%     sv = 1:m;
% end
% if nargin==2
%     x2=x;
% end
% if nargin<4
%     dims=1:size(x,2);
% end
% if length(gpcf.coeffSigma2)==1    
%     c=repmat(gpcf.coeffSigma2,1,(1+m)*m/2);
% else
%     % If coeffSigma is vector, we trust it is of rigth length
%     c=gpcf.coeffSigma2;
% end
% h2=x2.^2;
% ii1=0;
% for i=dims
%     if ~any(sv==i) %isfield(gpcf, 'selectedVariables') && sum(gpcf.selectedVariables==i)==0
%         DK=zeros(size(x,1),size(x2,1));
%     else
%         DK=c(i).*2.*x(:,i)*h2(:,i)';
%         if isequal(gpcf.interactions,'on')
%             i2=m;
%             for xi1=sv
%                 sv2 = sv(sv>xi1);
%                 for xi2=sv2 %xi1+1:m
%                     i2 = i2+1;
%                     if i==xi1
%                         DK = DK + c(i2)*x(:,xi2)*(x2(:,i).*x2(:,xi2))';
%                     elseif i==xi2
%                         DK = DK + c(i2)*x(:,xi1)*(x2(:,i).*x2(:,xi1))';
%                     end
%                 end
%             end
%         end
%     end
%     ii1=ii1+1;
%     DKff{ii1}=DK;
% end

[n,m]=size(x);
if nargin<4
    dims=1:m;
end
if isfield(gpcf, 'selectedVariables')
    sv = gpcf.selectedVariables;
else 
    sv = 1:m;
end
ii1=0;
if nargin==2
    x2=x;
end
h2=x2.^2;
if length(gpcf.coeffSigma2)==1
    c=repmat(gpcf.coeffSigma2,1,(1+m)*m/2);
else
    c=gpcf.coeffSigma2;
end
for i=dims
    if ~any(sv==i) % isfield(gpcf, 'selectedVariables') && sum(gpcf.selectedVariables==i)==0
        DK=zeros(size(x,1),size(x2,1));
    else
        DK=c(i).*2.*x(:,i)*h2(:,i)';
        if isequal(gpcf.interactions,'on')
            i2=length(sv);
            for xi1=1:length(sv)
                for xi2=xi1+1:length(sv)
                    if any(sv==xi1) && any(sv==xi2)
                        i2=i2+1;
%                        i2 = min(i,j)*length(sv) - 0.5*min(i,j)*(min(i,j)-1) + max(i,j)-min(i,j)
                        if i==xi1
                            DK = DK + c(i2)*x(:,xi2)*(x2(:,i).*x2(:,xi2))';
                        elseif i==xi2
                            DK = DK + c(i2)*x(:,xi1)*(x2(:,i).*x2(:,xi1))';
                        end
                    end
                end
            end
        end
    end
    ii1=ii1+1;
    DKff{ii1}=DK;
end
end

function C = gpcf_squared_cov(gpcf, x1, x2, varargin)
%GP_SQUARED_COV  Evaluate covariance matrix between two input vectors
%
%  Description         
%    C = GP_SQUARED_COV(GP, TX, X) takes in covariance function of
%    a Gaussian process GP and two matrixes TX and X that contain
%    input vectors to GP. Returns covariance matrix C. Every
%    element ij of C contains covariance between inputs i in TX
%    and j in X. This is a mandatory subfunction used for example in
%    prediction and energy computations.
%
%  See also
%    GPCF_SQUARED_TRCOV, GPCF_SQUARED_TRVAR, GP_COV, GP_TRCOV
  
  if isempty(x2)
    x2=x1;
  end
  [n1,m1]=size(x1);
  [n2,m2]=size(x2);
  if m1~=m2
    error('the number of columns of X1 and X2 has to be same')
  end
  if isfield(gpcf, 'selectedVariables')
      x1 = x1(:,gpcf.selectedVariables);
      x2 = x2(:,gpcf.selectedVariables);
  end
  h1 = x1.^2;
  h2 = x2.^2;
  if isequal(gpcf.interactions,'on')
      m=size(x1,2);
      for xi1=1:m
          for xi2=xi1+1:m
              h1 = [h1 x1(:,xi1).*x1(:,xi2)];
              h2 = [h2 x2(:,xi1).*x2(:,xi2)];
          end
      end
  end
  C = h1*diag(gpcf.coeffSigma2)*(h2');
  C(abs(C)<=eps) = 0;
  
end

function C = gpcf_squared_trcov(gpcf, x)
%GP_SQUARED_TRCOV  Evaluate training covariance matrix of inputs
%
%  Description
%    C = GP_SQUARED_TRCOV(GP, TX) takes in covariance function of
%    a Gaussian process GP and matrix TX that contains training
%    input vectors. Returns covariance matrix C. Every element ij
%    of C contains covariance between inputs i and j in TX. This 
%    is a mandatory subfunction used for example in prediction and 
%    energy computations.
%
%  See also
%    GPCF_SQUARED_COV, GPCF_SQUARED_TRVAR, GP_COV, GP_TRCOV

  
  if isfield(gpcf, 'selectedVariables')
      x = x(:,gpcf.selectedVariables);
  end
  h = x.^2;
  if isequal(gpcf.interactions,'on')
      m=size(x,2);
      for xi1=1:m
          for xi2=xi1+1:m
              h = [h x(:,xi1).*x(:,xi2)];
          end
      end
  end
  C = h*diag(gpcf.coeffSigma2)*(h');
  C(abs(C)<=eps) = 0;
  C = (C+C')./2;
 
end


function C = gpcf_squared_trvar(gpcf, x)
%GP_SQUARED_TRVAR  Evaluate training variance vector
%
%  Description
%    C = GP_SQUARED_TRVAR(GPCF, TX) takes in covariance function
%    of a Gaussian process GPCF and matrix TX that contains
%    training inputs. Returns variance vector C. Every element i
%    of C contains variance of input i in TX. This is a mandatory 
%    subfunction used for example in prediction and energy computations.
%
%
%  See also
%    GPCF_SQUARED_COV, GP_COV, GP_TRCOV

  
  if isfield(gpcf, 'selectedVariables')
      x = x(:,gpcf.selectedVariables);
  end
  h = x.^2;
  if isequal(gpcf.interactions,'on')
      m=size(x,2);
      for xi1=1:m
          for xi2=xi1+1:m
              h = [h x(:,xi1).*x(:,xi2)];
          end
      end
  end
  if length(gpcf.coeffSigma2) == 1
      C=gpcf.coeffSigma2.*sum(h.^2,2);
  else
      C=sum(repmat(gpcf.coeffSigma2, size(x,1), 1).*h.^2,2);
  end
  C(abs(C)<eps)=0;
    
  
end

function reccf = gpcf_squared_recappend(reccf, ri, gpcf)
%RECAPPEND Record append
%
%  Description
%    RECCF = GPCF_SQUARED_RECAPPEND(RECCF, RI, GPCF) takes a
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
    reccf.type = 'gpcf_squared';
    
    % Initialize parameters
    reccf.coeffSigma2= [];

    % Set the function handles
    reccf.fh.pak = @gpcf_squared_pak;
    reccf.fh.unpak = @gpcf_squared_unpak;
    reccf.fh.lp = @gpcf_squared_lp;
    reccf.fh.lpg = @gpcf_squared_lpg;
    reccf.fh.cfg = @gpcf_squared_cfg;
    reccf.fh.cfdg = @gpcf_squared_cfdg;
    reccf.fh.cfdg2 = @gpcf_squared_cfdg2;
    reccf.fh.ginput = @gpcf_squared_ginput;
    reccf.fh.ginput2 = @gpcf_squared_ginput2;
    reccf.fh.ginput3 = @gpcf_squared_ginput3;
    reccf.fh.ginput4 = @gpcf_squared_ginput4;
    reccf.fh.cov = @gpcf_squared_cov;
    reccf.fh.trcov  = @gpcf_squared_trcov;
    reccf.fh.trvar  = @gpcf_squared_trvar;
    reccf.fh.recappend = @gpcf_squared_recappend;
    reccf.p=[];
    reccf.p.coeffSigma2=[];
    if ~isempty(ri.p.coeffSigma2)
      reccf.p.coeffSigma2 = ri.p.coeffSigma2;
    end

  else
    % Append to the record
    gpp = gpcf.p;
    
    reccf.interactions = gpcf.interactions;
    
    % record coeffSigma2
    reccf.coeffSigma2(ri,:)=gpcf.coeffSigma2;
    if isfield(gpp,'coeffSigma2') && ~isempty(gpp.coeffSigma2)
      reccf.p.coeffSigma2 = gpp.coeffSigma2.fh.recappend(reccf.p.coeffSigma2, ri, gpcf.p.coeffSigma2);
    end
  
    if isfield(gpcf, 'selectedVariables')
      reccf.selectedVariables = gpcf.selectedVariables;
    end
  end
end

function [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = gpcf_squared_cf2ss(gpcf,x)
%GPCF_SQUARED_CF2SS Convert the covariance function to state space form
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

  % Check arguments
  if nargin < 2 || isempty(x), x = 0; end

  % Scaling
  x0 = min(x);
  
  % Define the model
  F      = [0 1; 0 0]; 
  L      = [0; 1]; 
  Qc     = 0; 
  H      = [1 0];
  Pinf   = [x0^2 x0; x0 1]*gpcf.coeffSigma2;
  dF     = zeros(2,2,1);
  dQc    = zeros(1,1,1);
  dPinf  = [x0^2 x0; x0 1];
  params = {};

  % Set params
  params.stationary = false;
  
  % Check which parameters are optimized
  if isempty(gpcf.p.coeffSigma2), ind(1) = false; else ind(1) = true; end
  
  % Return only those derivatives that are needed
  dF    = dF(:,:,ind);
  dQc   = dQc(:,:,ind);
  dPinf = dPinf(:,:,ind);
  
end