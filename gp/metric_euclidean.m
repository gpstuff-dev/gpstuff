function metric = metric_euclidean(do, varargin)
%METRIC_EUCLIDEAN An Euclidean distance metric for Gaussian process models.
%
%	Description
%
%	METRIC = METRIC_EUCLIDEAN('INIT', NIN, COMPONENTS) Constructs an data
%       structure for an euclidean metric used in covariance function of a GP model.
%
%	The fields and (default values) in METRIC_EUCLIDEAN are:
%	  type        = 'metric_euclidean'
%	  nin         = Number of inputs in the data. (NIN)
%	  components  = Cell array of vectors specifying which inputs are grouped together
%                       with a same scaling parameter. 
%                       For example, the component specification {[1 2] [3]} means that
%                       distance between 3 dimensional vectors x and z is computed as
%                       r = sqrt(((x_1-z_1)^2+(x_2-z_2)^2)/l_1 + (x_3-z_3)/l_2),
%                       where l_1 and l_2 are lengthscales for corresponding component sets.
%         params      = Hyperparameters of the metric, which in this case are 
%                       lengthscales for each input component set. 
%         p           = Prior structure for metric parameters. 
%                       (e.g. p.params.)
%         pak         = function handle to pack function
%                       (@metric_euclidean_pak)
%         unpak       = function handle to unpack function
%                       (@metric_euclidean_unpak)
%         e           = function handle to energy function
%                       (@metric_euclidean_e)
%         ghyper      = function handle to gradient of energy with respect to hyperparameters
%                       (@metric_euclidean_ghyper)
%         ginput      = function handle to gradient of function with respect to inducing inputs
%                       (@metric_euclidean_ginput)
%         distance    = function handle to distance function of the metric.
%                       (@metric_euclidean_distance)
%         fh_recappend   = function handle to append the record function 
%                          (metric_euclidean_recappend)
%
%	METRIC = METRIC_EUCLIDEAN('SET', METRIC, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in METRIC.
%
    
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
        metric.nin = varargin{1};
        metric.components = varargin{2};
        
        metric.params = repmat(1,1,length(metric.components));

        % Initialize prior structure
        metric.p=[];
        metric.p.params=[];
        
        % Set the function handles to the nested functions
        metric.pak    = @metric_euclidean_pak;
        metric.unpak  = @metric_euclidean_unpak;
        metric.e      = @metric_euclidean_e;
        metric.ghyper = @metric_euclidean_ghyper;
        metric.ginput = @metric_euclidean_ginput;
        metric.distance   = @metric_euclidean_distance;
        
        if length(varargin) > 2
            if mod(nargin,2) ~=1
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=3:2:length(varargin)-1
                switch varargin{i}
                  case 'params'
                    if size(varargin{i+1}) ~= size(metric.params)
                        error('Incorrect number of parameters given.');
                    end
                    metric.params = varargin{i+1};
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
              case 'params'
                if size(varargin{i+1}) ~= size(metric.params)
                    error('Incorrect number of parameters given.');
                end                
                metric.params = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end
end
    
    function w = metric_euclidean_pak(metric, w)
    %METRIC_EUCLIDEAN_PAK	 Combine the metric parameters into one vector.
    %
    %	Description
    %	W = METRIC_EUCLIDEAN_PAK(METRIC, W) takes a metric data structure METRIC and
    %	combines the parameters into a single row vector W.
    %
    %	See also
    %	METRIC_EUCLIDEAN_UNPAK
        mp=metric.p;

        i1=0;i2=1;
        if ~isempty(w)
            i1 = length(w);
        end
        i2=i1+length(metric.params);
        i1=i1+1;
        w(i1:i2)=metric.params;
        i1=i2;
        
        % Parameters
        if isfield(mp.params, 'p') && ~isempty(mp.params.p)
            i1=i1+1;
            w(i1)=mp.params.a.s;
            if any(strcmp(fieldnames(mp.params.p),'nu'))
                i1=i1+1;
                w(i1)=mp.params.a.nu;
            end
        end
    end
    
    function [metric, w] = metric_euclidean_unpak(metric, w)
    %METRIC_EUCLIDEAN_UNPAK  Separate metric parameter vector into components.
    %
    %	Description
    %	[METRIC, W] = METRIC_EUCLIDEAN_UNPAK(METRIC, W) takes a metric data structure
    %   METRIC parameter vector W, and returns a metric data structure  identical to the
    %   input, except that the parameters has been set to the values in W. Deletes the values
    %   set to METRIC from W and returns the modified W. 
    %
    %	See also
    %	METRIC_EUCLIDEAN_PAK
    %
        mp=metric.p;
        i1=0;i2=1;
        i2=i1+length(metric.params);
        i1=i1+1;
        metric.params=w(i1:i2);
        i1=i2;
        % Parameters
        if isfield(mp.params, 'p') && ~isempty(mp.params.p)
            i1=i1+1;
            metric.p.params.a.s=w(i1);
            if any(strcmp(fieldnames(mp.params.p),'nu'))
                i1=i1+1;
                metric.p.params.a.nu=w(i1);
            end
        end        
        w = w(i1+1:end);
    end

    function eprior = metric_euclidean_e(metric, x, t)
    %METRIC_EUCLIDEAN_E     Evaluate the energy of prior of metric parameters
    %
    %	Description
    %	E = METRIC_EUCLIDEAN_E(METRIC, X, T) takes a metric data structure 
    %   METRIC together with a matrix X of input vectors and a matrix T of target 
    %   vectors and evaluates log p(th) x J, where th is a vector of metric parameters 
    %   and J is the Jakobian of transformation exp(w) = th. (Note that the parameters 
    %   are log transformed, when packed.)
    %
    %	See also
    %	METRIC_EUCLIDEAN_PAK, METRIC_EUCLIDEAN_UNPAK, METRIC_EUCLIDEAN_G, GP_E
    %
        [n, m] = size(x);

        % Evaluate the prior contribution to the error. The parameters that
        % are sampled are from space W = log(w) where w is all the "real" samples.
        % On the other hand errors are evaluated in the W-space so we need take
        % into account also the  Jakobian of transformation W -> w = exp(W).
        % See Gelman et.all., 2004, Bayesian data Analysis, second edition, p24.
        eprior = 0;
        mp=metric.p;

        if isfield(mp.params, 'p') && ~isempty(mp.params.p)
            eprior=eprior...
                   +feval(mp.params.p.s.fe, ...
                          gpp.params.a.s, mp.params.p.s.a)...
                   -log(mp.params.a.s);
            if any(strcmp(fieldnames(mp.params.p),'nu'))
                eprior=eprior...
                       +feval(mp.p.params.nu.fe, ...
                              mp.params.a.nu, mp.params.p.nu.a)...
                       -log(mp.params.a.nu);
            end
        end
        eprior=eprior...
               +feval(mp.params.fe, ...
                      metric.params, mp.params.a)...
               -sum(log(metric.params));

    end
    
    function [gdist, gprior_dist]  = metric_euclidean_ghyper(metric, x, x2, mask) 
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
        
        if nargin <= 3
            if nargin == 2
                x2 = x;
            end
            ii1=0;            

            dist  =  0;
            distc = cell(1,m);
            % Compute the distances for each component set
            for i=1:m
                s = 1./metric.params(i).^2;
                distc{i} = 0;
                for j = 1:length(components{i})
                    distc{i} = distc{i} + gminus(x(:,components{i}(j)),x2(:,components{i}(j))').^2;
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
            gdist = cell(1,length(metric.params));
        end
        
        % Evaluate the data contribution of gradient with respect to lengthScale
        for i2=1:m
            i1=i1+1;
            gprior_dist(i1)=feval(mp.params.fg, ...
                             metric.params(i2), ...
                             mp.params.a, 'x').*metric.params(i2) - 1;
        end

        % Evaluate the prior contribution of gradient with respect to lengthScale.p.s (and lengthScale.p.nu)
        if isfield(mp.params, 'p') && ~isempty(mp.params.p)
            i1=i1+1;
            gprior_dist(i1)=...
                feval(mp.params.p.s.fg, ...
                      mp.params.a.s,...
                      mp.params.p.s.a, 'x').*mp.params.a.s - 1 ...
                +feval(mp.params.fg, ...
                       metric.params, ...
                       mp.params.a, 's').*mp.params.a.s;
            if any(strcmp(fieldnames(mp.params.p),'nu'))
                i1=i1+1;
                gprior_dist(i1)=...
                    feval(mp.params.p.nu.fg, ...
                          mp.params.a.nu,...
                          mp.params.p.nu.a, 'x').*mp.params.a.nu -1 ...
                    +feval(mp.params.fg, ...
                           metric.params, ...
                           mp.params.a, 'nu').*mp.params.a.nu;
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
            s = 1./metric.params(i).^2;
            for j = 1:length(components{i})
                dist = dist + s.*gminus(x1(:,components{i}(j)),x2(:,components{i}(j))').^2;
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
        
        s = 1./metric.params.^2;
        dist = 0;
        for i=1:length(components)
            for j = 1:length(components{i})
                dist = dist + s(i).*gminus(x1(:,components{i}(j)),x2(:,components{i}(j))').^2;
            end
        end
        dist = sqrt(dist);
        
        for i=1:m1
            for j = 1:n1
                DK = zeros(n1,n2);                
                for k = 1:length(components)
                    if ismember(i,components{k})
                        DK(j,:) = DK(j,:)+s(k).*gminus(x1(j,i),x2(:,i)');
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