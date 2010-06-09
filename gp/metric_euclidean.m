function metric = metric_euclidean(do, varargin)
%METRIC_EUCLIDEAN An Euclidean distance for Gaussian process models.
%
%	Description
%	METRIC = METRIC_EUCLIDEAN('INIT', COMPONENTS, OPTIONS) Constructs data
%       structure for an euclidean metric used in covariance function
%       of a GP model. OPTIONS is optional parameter-value pair used
%       as described below by GPCF_SEXP('set',...
%
%       METRIC = METRIC_EUCLIDEAN('SET', METRIC, OPTIONS) Set the
%       fields of GPCF as described by the parameter-value pairs
%       ('FIELD', VALUE) in the OPTIONS. The fields that can be
%       modified are:
%
%	components          = Cell array of vectors specifying which 
%                             inputs are grouped together with a same
%                             scaling parameter. For example, the 
%                             component specification {[1 2] [3]} means 
%                             that distance between 3 dimensional vectors 
%                             x and z is computed as 
%                             r = sqrt( ( (x_1-z_1)^2+(x_2-z_2)^2 )/l_1 + (x_3-z_3)/l_2), 
%                             where l_1 and l_2 are lengthscales for 
%                             corresponding component sets.
%       lengthScales        = Hyperparameters of the metric, which in
%                             this case are lengthscales for each 
%                             input component set. 
%       lengthScale_prior   = prior structure for lengthScales
%
%       See also
%       GPCF_SEXP
    
% Copyright (c) 2008 Jouni Hartikainen 
% Copyright (c) 2008 Jarno Vanhatalo     

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end

    if strcmp(do, 'init')
        metric.type = 'metric_euclidean';
        metric.components = varargin{1};
        
        metric.lengthScales = repmat(1,1,length(metric.components));

        % Initialize prior structure
        metric.p=[];
        metric.p.lengthScales = prior_unif('init');
        
        % Set the function handles to the nested functions
        metric.pak        = @metric_euclidean_pak;
        metric.unpak      = @metric_euclidean_unpak;
        metric.e          = @metric_euclidean_e;
        metric.ghyper     = @metric_euclidean_ghyper;
        metric.ginput     = @metric_euclidean_ginput;
        metric.distance   = @metric_euclidean_distance;
        metric.recappend  = @metric_euclidean_recappend;
        
        if length(varargin) > 1
            if mod(nargin,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=2:2:length(varargin)-1
                switch varargin{i}
                  case 'lengthScales'
                    if size(varargin{i+1}) ~= size(metric.lengthScales)
                        error('Incorrect number of parameters given.');
                    end
                    metric.lengthScales = varargin{i+1};
                  case 'lengthScales_prior'
                    metric.p.lengthScales = varargin{i+1};
                  otherwise
                    error('Wrong parameter name!')
                end
            end
        end

    end

    % Set the parameter values of covariance function
    if strcmp(do, 'set')
        if mod(nargin,2) ~=0
            error('Wrong number of arguments')
        end
        metric = varargin{1};
        % Loop through all the parameter values that are changed
        for i=2:2:length(varargin)-1
            switch varargin{i}
              case 'lengthScales'
                if size(varargin{i+1}) ~= size(metric.lengthScales)
                    error('Incorrect number of parameters given.');
                end                
                metric.lengthScales = varargin{i+1};
              case 'lengthScales_prior'
                metric.p.lengthScales = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end
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
    
    if ~isempty(metric.p.lengthScales)
        w = log(metric.lengthScales);
        
        % Hyperparameters of lengthScale
        w = [w feval(metric.p.lengthScales.fh_pak, metric.p.lengthScales)];
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
    
    if ~isempty(metric.p.lengthScales)
        i2=length(metric.lengthScales);
        i1=1;
        metric.lengthScales = exp(w(i1:i2));
        w = w(i2+1:end);
        
        % Hyperparameters of lengthScale
        [p, w] = feval(metric.p.lengthScales.fh_unpak, metric.p.lengthScales, w);
        metric.p.lengthScales = p;
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

    eprior = feval(metric.p.lengthScales.fh_e, metric.lengthScales, metric.p.lengthScales) - sum(log(metric.lengthScales));
    
end

function [gdist, gprior]  = metric_euclidean_ghyper(metric, x, x2, mask) 
%METRIC_EUCLIDEAN_GHYPER     Evaluate the gradient of the metric function and hyperprior 
%                            w.r.t to it's hyperparameters.
%
%	Description
%	[GDIST, GPRIOR_DIST] = METRIC_EUCLIDEAN_GHYPER(METRIC, X) takes a
%   metric data structure METRIC together with a matrix X of input vectors and 
%   return the gradient matrices GDIST and GPRIOR_DIST for each hyperparameter.
%
%	[GDIST, GPRIOR_DIST] = METRIC_EUCLIDEAN_GHYPER(METRIC, X, X2) forms the gradient
%   matrices between two input vectors X and X2.
%     
%	[GDIST, GPRIOR_DIST] = METRIC_EUCLIDEAN_GHYPER(METRIC, X, X2, MASK) forms
%   the gradients for masked covariances matrices used in sparse approximations.
%
%	See also
%	METRIC_EUCLIDEAN_PAK, METRIC_EUCLIDEAN_UNPAK, METRIC_EUCLIDEAN, GP_E
%

    mp=metric.p;
    components = metric.components;
    
    n = size(x,1);
    m = length(components);
    i1=0;i2=1;

    % NOTE! Here we have already taken into account that the parameters are transformed
    % through log() and thus dK/dlog(p) = p * dK/dp
   
    if ~isempty(metric.p.lengthScales)
        if nargin <= 3
            if nargin == 2
                x2 = x;
            end
            ii1=0;            

            dist  =  0;
            distc = cell(1,m);
            % Compute the distances for each component set
            for i=1:m
                s = 1./metric.lengthScales(i).^2;
                distc{i} = 0;
                for j = 1:length(components{i})
                    distc{i} = distc{i} + bsxfun(@minus,x(:,components{i}(j)),x2(:,components{i}(j))').^2;
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
            gdist = cell(1,length(metric.lengthScales));
        end

        % Evaluate the prior contribution of gradient with respect to lengthScale
        if ~isempty(metric.p.lengthScales)
            i1=1; 
            lll = length(metric.lengthScales);
            gg = feval(metric.p.lengthScales.fh_g, metric.lengthScales, metric.p.lengthScales);
            gprior(i1:i1-1+lll) = gg(1:lll).*metric.lengthScales - 1;
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
        s = 1./metric.lengthScales(i).^2;
        for j = 1:length(components{i})
            dist = dist + s.*bsxfun(@minus,x1(:,components{i}(j)),x2(:,components{i}(j))').^2;
        end
    end
    dist = sqrt(dist);

end

function [ginput, gprior_input]  = metric_euclidean_ginput(metric, x1, x2)         
%METRIC_EUCLIDEAN_GINPUT   Compute the gradient of the euclidean distance
%                          function with respect to input.
%[n, m] =size(x);
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
    
    s = 1./metric.lengthScales.^2;
    dist = 0;
    for i=1:length(components)
        for j = 1:length(components{i})
            dist = dist + s(i).*bsxfun(@minus,x1(:,components{i}(j)),x2(:,components{i}(j))').^2;
        end
    end
    dist = sqrt(dist);
    
    for i=1:m1
        for j = 1:n1
            DK = zeros(n1,n2);                
            for k = 1:length(components)
                if ismember(i,components{k})
                    DK(j,:) = DK(j,:)+s(k).*bsxfun(@minus,x1(j,i),x2(:,i)');
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
%          Description
%          RECMETRIC = METRIC_EUCLIDEAN_RECAPPEND(RECMETRIC, RI, METRIC) takes old covariance
%          function record RECMETRIC, record index RI and covariance function structure. 
%          Appends the parameters of METRIC to the RECMETRIC in the ri'th place.
%
%          See also
%          GP_MC and GP_MC -> RECAPPEND

% Initialize record
    if nargin == 2
        recmetric.type = 'metric_euclidean';
        metric.components = recmetric.components;
        
        % Initialize parameters
        recmetric.lengthScales = [];

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
    if ~isempty(metric.lengthScales)
        recmetric.lengthScales(ri,:)=metric.lengthScales;
        recmetric.p.lengthScales = feval(metric.p.lengthScales.fh_recappend, recmetric.p.lengthScales, ri, metric.p.lengthScales);
    elseif ri==1
        recmetric.lengthScales=[];
    end

end