function metric = metric_euclidean(varargin)
%METRIC_EUCLIDEAN An euclidean metric function
%
%  Description
%    METRIC = METRIC_EUCLIDEAN('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates a an euclidean metric function structure in which the
%    named parameters have the specified values. Either
%    'components' or 'deltaflag' has to be specified. Any
%    unspecified parameters are set to default values.
%   
%    METRIC = METRIC_EUCLIDEAN(METRIC,'PARAM1',VALUE1,'PARAM2,VALUE2,...)
%    modify a metric function structure with the named parameters
%    altered with the specified values.
%
%    Parameters for Euclidean metric function [default]
%	components        - cell array of vectors specifying which 
%                           inputs are grouped together with a same
%                           scaling parameter. For example, the
%                           component specification {[1 2] [3]}
%                           means that distance between 3
%                           dimensional vectors computed as 
%                           r = (r_1^2 + r_2^2 )/l_1 + r_3^2/l_2,
%                           where r_i are distance along component
%                           i, and l_1 and l_2 are lengthscales for
%                           corresponding component sets. If
%                           'components' is not specified, but
%                           'deltaflag' is specified, then default
%                           is {1 ... length(deltaflag)}
%       deltaflag         - indicator vector telling which component sets
%                           are handled using the delta distance 
%                           (0 if x=x', and 1 otherwise). Default is
%                           false for all component sets.
%       lengthScale       - lengthscales for each input component set
%                           Default is 1 for each set
%       lengthScale_prior - prior for lengthScales [prior_unif]
%
%  See also
%    GP_SET, GPCF_SEXP
  
% Copyright (c) 2008 Jouni Hartikainen 
% Copyright (c) 2008 Jarno Vanhatalo     
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'METRIC_EUCLIDEAN';
  ip.addOptional('metric', [], @isstruct);
  ip.addParamValue('components',[], @(x) isempty(x) || iscell(x));
  ip.addParamValue('deltaflag',[], @(x) isvector(x));
  ip.addParamValue('lengthScale',[], @(x) isvector(x) && all(x>0));
  ip.addParamValue('lengthScale_prior',prior_unif, ...
                   @(x) isstruct(x) || isempty(x));
  ip.parse(varargin{:});
  metric=ip.Results.metric;

  if isempty(metric)
    % Initialize a Gaussian process
    init=true;
  else
    % Modify a Gaussian process
    if ~isfield(metric,'cf')
      error('First argument does not seem to be a metric structure')
    end
    init=false;
  end

  if init
    % Type
    metric.type = 'metric_euclidean';
  end
  
  % Components
  if init | ~ismember('components',ip.UsingDefaults)
    metric.components = ip.Results.components;
  end
  % Deltaflag
  if init | ~ismember('deltaflag',ip.UsingDefaults)
    metric.deltaflag = ip.Results.deltaflag;
  end
  % Components+Deltaflag check and defaults
  if isempty(metric.components) && isempty(metric.deltaflag)
    error('Either ''components'' or ''deltaflag'' has to be specified')
  elseif isempty(metric.components)
    metric.components=num2cell(1:length(metric.components));
  elseif isempty(metric.deltaflag)
    metric.deltaflag = false(1,length(metric.components));
  end
  % Lengthscale
  if init | ~ismember('lengthScale',ip.UsingDefaults)
    metric.lengthScale = ip.Results.lengthScale;
    if isempty(metric.lengthScale)
      metric.lengthScale = repmat(1,1,length(metric.components));
    end
  end
  % Prior for lengthscale
  if init | ~ismember('lengthScale_prior',ip.UsingDefaults)
    metric.p=[];
    metric.p.lengthScale = ip.Results.lengthScale_prior;
  end
  
  if init
    % Set the function handles to the nested functions
    metric.pak        = @metric_euclidean_pak;
    metric.unpak      = @metric_euclidean_unpak;
    metric.e          = @metric_euclidean_e;
    metric.ghyper     = @metric_euclidean_ghyper;
    metric.ginput     = @metric_euclidean_ginput;
    metric.distance   = @metric_euclidean_distance;
    metric.recappend  = @metric_euclidean_recappend;
  end
  
  function w = metric_euclidean_pak(metric)
  %METRIC_EUCLIDEAN_PAK	 Combine GP covariance function hyper-parameters into one vector.
  %
  %	Description
  %   W = METRIC_EUCLIDEAN_PAK(GPCF) takes a covariance function data
  %   structure GPCF and combines the covariance function parameters
  %   and their hyperparameters into a single row vector W and takes
  %   a logarithm of the covariance function parameters.
  %
  %       w = [ log(gpcf.lengthScale(:))
  %             (hyperparameters of gpcf.lengthScale)]'
  %	  
  %
  %	See also
  %	GPCF_SEXP_UNPAK
    
    
    if ~isempty(metric.p.lengthScale)
      w = log(metric.lengthScale);
      
      % Hyperparameters of lengthScale
      w = [w feval(metric.p.lengthScale.fh.pak, metric.p.lengthScale)];
    else
      w = [];
    end
  end

  function [metric, w] = metric_euclidean_unpak(metric, w)
  %METRIC_EUCLIDEAN_UNPAK  Separate metric parameter vector into components.
  %
  %   Description
  %   METRIC, W] = METRIC_EUCLIDEAN_UNPAK(METRIC, W) takes a metric data
  %   structure GPCF and a hyper-parameter vector W, and returns a
  %   covariance function data structure identical to the input, except
  %   that the covariance hyper-parameters have been set to the values
  %   in W. Deletes the values set to GPCF from W and returns the
  %   modeified W.
  %
  %   The covariance function parameters are transformed via exp
  %   before setting them into the structure.
  %
  %	See also
  %	METRIC_EUCLIDEAN_PAK
  %
    
    if ~isempty(metric.p.lengthScale)
      i2=length(metric.lengthScale);
      i1=1;
      metric.lengthScale = exp(w(i1:i2));
      w = w(i2+1:end);
      
      % Hyperparameters of lengthScale
      [p, w] = feval(metric.p.lengthScale.fh.unpak, metric.p.lengthScale, w);
      metric.p.lengthScale = p;
    end
  end

  function eprior = metric_euclidean_e(metric, x, t)
  %METRIC_EUCLIDEAN_E     Evaluate the energy of prior of metric parameters
  %
  %   Description
  %   E = METRIC_EUCLIDEAN_E(METRIC, X, T) takes a metric data structure
  %   GPCF together with a matrix X of input vectors and a vector T of
  %   target vectors and evaluates log p(th) x J, where th is a vector
  %   of SEXP parameters and J is the Jacobian of transformation exp(w)
  %   = th. (Note that the parameters are log transformed, when packed.)
  %
  %   Also the log prior of the hyperparameters of the covariance
  %   function parameters is added to E if hyper-hyperprior is
  %   defined.
  %
  %   See also
  %   METRIC_EUCLIDEAN_PAK, METRIC_EUCLIDEAN_UNPAK, METRIC_EUCLIDEAN_G, GP_E
  %
    [n, m] = size(x);

    % Evaluate the prior contribution to the error. The parameters that
    % are sampled are from space W = log(w) where w is all the "real" samples.
    % On the other hand errors are evaluated in the W-space so we need take
    % into account also the  Jakobian of transformation W -> w = exp(W).
    % See Gelman et.all., 2004, Bayesian data Analysis, second edition, p24.
    if ~isempty(metric.p.lengthScale)
      eprior = feval(metric.p.lengthScale.fh.e, metric.lengthScale, metric.p.lengthScale) - sum(log(metric.lengthScale));
    else
      eprior=0;
    end
    
  end

  function [gdist, gprior]  = metric_euclidean_ghyper(metric, x, x2, mask) 
  %METRIC_EUCLIDEAN_GHYPER Evaluate the gradient of the metric function
  %                    and hyperprior w.r.t to it's hyperparameters.
  %
  %    Description
  %     [GDIST, GPRIOR_DIST] = METRIC_EUCLIDEAN_GHYPER(METRIC, X) takes a
  %     metric data structure METRIC together with a matrix X of
  %     input vectors and return the gradient matrices GDIST and
  %     GPRIOR_DIST for each hyperparameter.
  %
  %     [GDIST, GPRIOR_DIST] = METRIC_EUCLIDEAN_GHYPER(METRIC, X, X2)
  %     forms the gradient matrices between two input vectors X and
  %     X2.
  %     
  %     [GDIST, GPRIOR_DIST] = METRIC_EUCLIDEAN_GHYPER(METRIC, X, X2,
  %     MASK) forms the gradients for masked covariances matrices
  %     used in sparse approximations.
  %
  %	See also
  %	METRIC_EUCLIDEAN_PAK, METRIC_EUCLIDEAN_UNPAK, METRIC_EUCLIDEAN, GP_E
  %

    gdist=[];gprior=[];
    components = metric.components;
    
    n = size(x,1);
    m = length(components);
    i1=0;i2=1;

    % NOTE! Here we have already taken into account that the parameters
    % are transformed through log() and thus dK/dlog(p) = p * dK/dp
    
    if ~isempty(metric.p.lengthScale)
      if nargin <= 3
        if nargin == 2
          x2 = x;
        end
        ii1=0;            

        dist  =  0;
        distc = cell(1,m);
        % Compute the distances for each component set
        for i=1:m
          s = 1./metric.lengthScale(i).^2;
          distc{i} = 0;
          for j = 1:length(components{i})
            if metric.deltaflag(i)
              distc{i} = distc{i} + double(bsxfun(@ne,x(:,components{i}(j)),x2(:,components{i}(j))'));
            else
              distc{i} = distc{i} + bsxfun(@minus,x(:,components{i}(j)),x2(:,components{i}(j))').^2;
            end
          end
          distc{i} = distc{i}.*s;
          % Accumulate to the total distance
          dist = dist + distc{i};
        end
        dist = sqrt(dist);
        % Loop through component sets 
        for i=1:m
          D = -distc{i};
          D(dist~=0) = D(dist~=0)./dist(dist~=0);
          ii1 = ii1+1;
          gdist{ii1} = D;
        end
  % $$$         elseif nargin == 3
  % $$$             if size(x,2) ~= size(x2,2)
  % $$$                 error('metric_euclidean -> _ghyper: The number of columns in x and x2 has to be the same. ')
  % $$$             end
      elseif nargin == 4
        gdist = cell(1,length(metric.lengthScale));
      end

      % Evaluate the prior contribution of gradient with respect to lengthScale
      if ~isempty(metric.p.lengthScale)
        i1=1; 
        lll = length(metric.lengthScale);
        gg = feval(metric.p.lengthScale.fh.g, metric.lengthScale, metric.p.lengthScale);
        gprior(i1:i1-1+lll) = gg(1:lll).*metric.lengthScale - 1;
        gprior = [gprior gg(lll+1:end)];
      end
    end
  end


  function [dist]  = metric_euclidean_distance(metric, x1, x2)         
  %METRIC_EUCLIDEAN_DISTANCE   Compute the euclidean distence between
  %                            one or two matrices.
  %
  %	Description
  %	[DIST] = METRIC_EUCLIDEAN_DISTANCE(METRIC, X) takes a metric data
  %   structure METRIC together with a matrix X of input vectors and 
  %   calculates the euclidean distance matrix DIST.
  %
  %	[DIST] = METRIC_EUCLIDEAN_DISTANCE(METRIC, X1, X2) takes a metric data
  %   structure METRIC together with a matrices X1 and X2 of input vectors and 
  %   calculates the euclidean distance matrix DIST.
  %
  %	See also
  %	METRIC_EUCLIDEAN_PAK, METRIC_EUCLIDEAN_UNPAK, METRIC_EUCLIDEAN, GP_E
  %
    if nargin == 2 || isempty(x2)
      x2=x1;
    end
    
    [n1,m1]=size(x1);
    [n2,m2]=size(x2);
    
    if m1~=m2
      error('the number of columns of X1 and X2 has to be same')
    end
    
    components = metric.components;
    m = length(components);
    dist  =  0;        
    
    for i=1:m
      s = 1./metric.lengthScale(i).^2;
      for j = 1:length(components{i})
        if metric.deltaflag(i)
          dist = dist + s.*double(bsxfun(@ne,x1(:,components{i}(j)),x2(:,components{i}(j))'));
        else
          dist = dist + s.*bsxfun(@minus,x1(:,components{i}(j)),x2(:,components{i}(j))').^2;
        end
      end
    end
    dist=sqrt(dist); % euclidean distance
    
  end

  function [ginput, gprior_input]  = metric_euclidean_ginput(metric, x1, x2)         
  %METRIC_EUCLIDEAN_GINPUT  Compute the gradient of the
  %  euclidean distance function with respect to input. [n, m]=size(x);
    ii1 = 0;
    components = metric.components;
    
    if nargin == 2 || isempty(x2)
      x2=x1;
    end
    
    [n1,m1]=size(x1);
    [n2,m2]=size(x2);
    
    if m1~=m2
      error('the number of columns of X1 and X2 has to be same')
    end
    
    s = 1./metric.lengthScale.^2;
    dist = 0;
    for i=1:length(components)
      for j = 1:length(components{i})
        if metric.deltaflag(i)
          dist = dist + s(i).*double(bsxfun(@ne,x1(:,components{i}(j)),x2(:,components{i}(j))'));
        else
          dist = dist + s(i).*bsxfun(@minus,x1(:,components{i}(j)),x2(:,components{i}(j))').^2;
        end
      end
    end
    dist = sqrt(dist);
    
    for i=1:m1
      for j = 1:n1
        DK = zeros(n1,n2);                
        for k = 1:length(components)
          if ismember(i,components{k})
            if metric.deltaflag(i)
              DK(j,:) = DK(j,:)+s(k).*double(bsxfun(@ne,x1(j,i),x2(:,i)'));
            else
              DK(j,:) = DK(j,:)+s(k).*bsxfun(@minus,x1(j,i),x2(:,i)');
            end
          end
        end
        if nargin == 2
          DK = DK + DK';
        end
        DK(dist~=0) = DK(dist~=0)./dist(dist~=0);
        
        ii1 = ii1 + 1;
        ginput{ii1} = DK;
        gprior_input(ii1) = 0; 
      end
    end
    %size(ginput)
    %ginput
    
  end


  function recmetric = metric_euclidean_recappend(recmetric, ri, metric)
  % RECAPPEND - Record append
  %   Description
  %     RECMETRIC = METRIC_EUCLIDEAN_RECAPPEND(RECMETRIC, RI, METRIC)
  %     takes old metric function record RECMETRIC, record index
  %     RI and metric function structure. Appends the parameters
  %     of METRIC to the RECMETRIC in the ri'th place.
  %
  %          See also
  %          GP_MC and GP_MC -> RECAPPEND

  % Initialize record
    if nargin == 2
      recmetric.type = 'metric_euclidean';
      metric.components = recmetric.components;
      
      % Initialize parameters
      recmetric.lengthScale = [];

      % Set the function handles
      recmetric.pak       = @metric_euclidean_pak;
      recmetric.unpak     = @metric_euclidean_unpak;
      recmetric.e         = @metric_euclidean_e;
      recmetric.ghyper    = @metric_euclidean_ghyper;
      recmetric.ginput    = @metric_euclidean_ginput;            
      recmetric.distance  = @metric_euclidean_distance;
      recmetric.recappend = @metric_euclidean_recappend;
      return
    end
    mp = metric.p;

    % record parameters
    if ~isempty(metric.lengthScale)
      recmetric.lengthScale(ri,:)=metric.lengthScale;
      recmetric.p.lengthScale = feval(metric.p.lengthScale.fh.recappend, recmetric.p.lengthScale, ri, metric.p.lengthScale);
    elseif ri==1
      recmetric.lengthScale=[];
    end

  end
  
end
