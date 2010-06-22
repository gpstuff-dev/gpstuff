function metric = metric_ibs_gxe(do, varargin)
%METRIC_IBS_GXE    An Euclidean distance metric for Gaussian process models.
%
%	Description
%	METRIC = METRIC_ibs_gxe('INIT', NIN) Constructs an data
%       structure for using both a distancematrix and a euclidean
%       metric used in covariance function of a GP model.
%
%       Uses metric_distancematrix and metric_euclidean to combine
%       genetic (ibs distance matrix) and environmental factors. These
%       have to be set up beforehand and added to this using 'init' or
%       'set'. The input to the GP should a matrix of size n x n+d
%       where n is the number of individuals and d the number of
%       additional covariates.
%
%         pak         = function handle to pack function
%                       (@metric_ibs_gxe_pak)
%         unpak       = function handle to unpack function
%                       (@metric_ibs_gxe_unpak)
%         e           = function handle to energy function
%                       (@metric_ibs_gxe_e)
%         ghyper      = function handle to gradient of energy with respect to hyperparameters
%                       (@metric_ibs_gxe_ghyper)
%         ginput      = function handle to gradient of function with respect to inducing inputs
%                       (@metric_ibs_gxe_ginput)
%         distance    = function handle to distance function of the metric.
%                       (@metric_ibs_gxe_distance)
%         fh_recappend   = function handle to append the record function 
%                          (metric_ibs_gxe_recappend)
%
%	METRIC = METRIC_ibs_gxe('SET', METRIC, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in METRIC.
%
    
% Copyright (c) 2009-2010 Heikki Peura

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end

    if strcmp(do, 'init')
        metric.type = 'metric_ibs_gxe';
        n=min(size(varargin{1}));
        metric.nin = varargin{1};
        n1=varargin{1}-n;
        %metric.components = varargin{2};
        %metric.params=varargin{3};
        %metric.x1x2matrix=[];
        
        %metric.x1matrix=[];
        %metric.x2matrix=[];
%         metric.x1=[];
%         metric.x2=[];
        %metric.n1=[];
        %metric.n2=[];
        %metric.gdist=[];
        %metric.params = repmat(1,1,length(metric.components));
        %metric.Kfu=[];
        %metric.Kffstar=[];
        
        %metric.metric_gene=metric_distancematrix('init', n,[],'lengthScales',[2]);
        %metric.metric_gene.p.params = gamma_p({3 1});
        %metric.metric_env=metric_euclidean('init', n1, varargin{2},'lengthScales',varargin{4});
        %metric.metric_env.p.params = gamma_p({3 1});
        
        
        % Initialize prior structure
        %metric.p=[];
        %metric.p.params=[];
        
        % Set the function handles to the nested functions
        metric.pak        = @metric_ibs_gxe_pak;
        metric.unpak      = @metric_ibs_gxe_unpak;
        metric.e          = @metric_ibs_gxe_e;
        metric.ghyper     = @metric_ibs_gxe_ghyper;
        metric.ginput     = @metric_ibs_gxe_ginput;
        metric.distance   = @metric_ibs_gxe_distance;
        metric.recappend  = @metric_ibs_gxe_recappend;
        %metric.matrix     = @metric_ibs_gxe_matrix;
        %metric.initmatrices = @metric_ibs_gxe_initmatrices;
        
        if length(varargin) > 2
            if mod(nargin,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=2:2:length(varargin)-1
                switch varargin{i}
%                   case 'params'
%                     if size(varargin{i+1}) ~= size(metric.params)
%                         error('Incorrect number of parameters given.');
%                     end
%                     metric.params = varargin{i+1};
                 case 'metric_gene'
                    metric.metric_gene = varargin{i+1};
                 case 'metric_env'
                    metric.metric_env = varargin{i+1};
%                   case 'x1'
%                     metric.x1=varargin{i+1};
%                     metric.n1=size(varargin{i+1},1);
%                     [metric.x1matrix,metric.gdist]=metric.matrix(varargin{i+1});
%                   case 'x2'
%                     metric.x2=varargin{i+1};
%                     metric.n2=size(varargin{i+1},1);
%                     metric.x2matrix=metric.matrix(varargin{i+1});
%                   case 'x1matrix'
%                     metric.x1matrix=varargin{i+1};
%                     metric.x1matrix=reshape(metric.x1matrix,[1,size(metric.x1matrix)]);
%                     metric.n1=size(varargin{i+1},1);
%                     for i=1:length(metric.components)
%                         metric.gdist{1}=-metric.x1matrix;
%                     end
%                   case 'x2matrix'
%                     metric.x2matrix=varargin{i+1};
%                     metric.x2matrix=reshape(metric.x2matrix,[1,size(metric.x2matrix)]);
%                     metric.n2=size(varargin{i+1},1);
%                   case 'x1x2matrix'
%                     metric.x1x2matrix=varargin{i+1};
%                     metric.x1x2matrix=reshape(metric.x1x2matrix,[1,size(metric.x1x2matrix)]);
                  otherwise
                    error('Wrong parameter name!')
                end
            end
%         if isempty(metric.x1x2matrix)
%             metric.x1x2matrix = metric.matrix(metric.x1,metric,x2);
%         end
        
 

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
%                 case 'params'
%                     if size(varargin{i+1}) ~= size(metric.params)
%                         error('Incorrect number of parameters given.');
%                     end
%                     metric.params = varargin{i+1};
                case 'metric_gene'
                    metric.metric_gene = varargin{i+1};
                case 'metric_env'
                    metric.metric_env = varargin{i+1};
%                 case 'x1'
%                     metric.x1=varargin{i+1};
%                     metric.n1=size(varargin{i+1},1);
%                     [metric.x1matrix,metric.gdist]=metric.matrix(varargin{i+1});
%                 case 'x2'
%                     metric.x2=varargin{i+1};
%                     metric.n2=size(varargin{i+1},1);
%                     metric.x2matrix=metric.matrix(varargin{i+1});
%                 case 'x1matrix'
%                     metric.x1matrix=varargin{i+1};
%                     metric.x1matrix=reshape(metric.x1matrix,[1,size(metric.x1matrix)]);
%                     metric.n1=size(varargin{i+1},1);
%                     for i=1:length(metric.components)
%                         metric.gdist{1}=-metric.x1matrix;
%                     end
%                 case 'x2matrix'
%                     metric.x2matrix=varargin{i+1};
%                     metric.x2matrix=reshape(metric.x2matrix,[1,size(metric.x2matrix)]);
%                     metric.n2=size(varargin{i+1},1);
%                 case 'x1x2matrix'
%                     metric.x1x2matrix=varargin{i+1};
%                     metric.x1x2matrix=reshape(metric.x1x2matrix,[1,size(metric.x1x2matrix)]);
                otherwise
                    error('Wrong parameter name!')
            end
        end
%         if isempty(metric.x1x2matrix)
%             metric.x1x2matrix = metric.matrix(metric.x1,metric,x2);
%         end
    end
    end

    
    function w = metric_ibs_gxe_pak(metric)
    %METRIC_ibs_PAK	 Combine the metric parameters into one vector.
    %
    %	Description
    %	W = METRIC_ibs_PAK(METRIC, W) takes a metric data structure METRIC and
    %	combines the parameters into a single row vector W.
    %
    %	See also
    %	METRIC_ibs_UNPAK
       
%         i1=0;i2=1;
%         if ~isempty(w)
%             i1 = length(w);
%         end
%         i1=i1+1;
        w=[feval(metric.metric_gene.pak,metric.metric_gene) feval(metric.metric_env.pak,metric.metric_env)];
%         i1=i1+1;
       % w=;
    end
    
    function [metric, w] = metric_ibs_gxe_unpak(metric, w)
    %METRIC_ibs_UNPAK  Separate metric parameter vector into components.
    %
    %	Description
    %	[METRIC, W] = METRIC_ibs_UNPAK(METRIC, W) takes a metric data structure
    %   METRIC parameter vector W, and returns a metric data structure  identical to the
    %   input, except that the parameters has been set to the values in W. Deletes the values
    %   set to METRIC from W and returns the modified W. 
    %
    %	See also
    %	METRIC_ibs_PAK
    %
%        i1=1+length(metric.metric_env.components);
%        w1=w(2:2+length(metric.metric_env.components));
%        w=w(i1+1:end);

       [metric_gene, w] = feval(metric.metric_gene.unpak, metric.metric_gene, w);
       metric.metric_gene=metric_gene;
       [metric_env, w] = feval(metric.metric_env.unpak, metric.metric_env, w);
       metric.metric_env=metric_env;
    end

    function eprior = metric_ibs_gxe_e(metric, x, t)
    %METRIC_ibs_E     Evaluate the energy of prior of metric parameters
    %
    %	Description
    %	E = METRIC_ibs_E(METRIC, X, T) takes a metric data structure 
    %   METRIC together with a matrix X of input vectors and a matrix T of target 
    %   vectors and evaluates log p(th) x J, where th is a vector of metric parameters 
    %   and J is the Jakobian of transformation exp(w) = th. (Note that the parameters 
    %   are log transformed, when packed.)
    %
    %	See also
    %	METRIC_ibs_PAK, METRIC_ibs_UNPAK, METRIC_ibs_G, GP_E
    %
        %[n, m] = size(x);

        % Evaluate the prior contribution to the error. The parameters that
        % are sampled are from space W = log(w) where w is all the "real" samples.
        % On the other hand errors are evaluated in the W-space so we need take
        % into account also the  Jakobian of transformation W -> w = exp(W).
        % See Gelman et al., 2004, Bayesian data Analysis, second edition, p24.
        eprior = 0;
        eprior=eprior+metric.metric_gene.e(metric.metric_gene,x,t);
        eprior=eprior+metric.metric_env.e(metric.metric_env,x,t);

%         mp=metric.p;
% 
%         if isfield(mp.params, 'p') && ~isempty(mp.params.p)
%             eprior=eprior...
%                    +feval(mp.params.p.s.fe, ...
%                           gpp.params.a.s, mp.params.p.s.a)...
%                    -log(mp.params.a.s);
%             if any(strcmp(fieldnames(mp.params.p),'nu'))
%                 eprior=eprior...
%                        +feval(mp.p.params.nu.fe, ...
%                               mp.params.a.nu, mp.params.p.nu.a)...
%                        -log(mp.params.a.nu);
%             end
%         end
%         eprior=eprior...
%                +feval(mp.params.fe, ...
%                       metric.params, mp.params.a)...
%                -sum(log(metric.params));

    end
    
    function [gdist, gprior_dist]  = metric_ibs_gxe_ghyper(metric, x1, x2, mask) 
    %METRIC_ibs_GHYPER     Evaluate the gradient of the metric function and hyperprior 
    %                            w.r.t to it's hyperparameters.
    %
    %	Description
    %	[GDIST, GPRIOR_DIST] = METRIC_ibs_GHYPER(METRIC, X) takes a
    %   metric data structure METRIC together with a matrix X of input vectors and 
    %   return the gradient matrices GDIST and GPRIOR_DIST for each hyperparameter.
    %
    %	[GDIST, GPRIOR_DIST] = METRIC_ibs_GHYPER(METRIC, X, X2) forms the gradient
    %   matrices between two input vectors X and X2.
    %     
    %	[GDIST, GPRIOR_DIST] = METRIC_ibs_GHYPER(METRIC, X, X2, MASK) forms
    %   the gradients for masked covariances matrices used in sparse approximations.
    %
    %	See also
    %	METRIC_ibs_PAK, METRIC_ibs_UNPAK, METRIC_ibs, GP_E
    %

        %mp=metric.p;
        %components = metric.components;
        
        %n = size(x1,1);
        m = length(metric.metric_env.lengthScales);
        i1=0;i2=1;
        %[n1,m1]=size(x1);
        
%         if nargin < 3
%             n2=n1;
%         end


        gdist=cell(1,m+1);
        n1=min(size(x1));
        n=n1;
        g1=x1(:,1);
        e1=x1(:,2:end);
           
        %n1=size(x1matrix,1);
        dist=0;
%        s = 1./metric.params;
        if (nargin == 3)
            g2=x2(:,1);
            e2=x2(:,2:end);
            n2=min(size(x2));

            [gdist_gene,gprior_dist_gene]=metric.metric_gene.ghyper(metric.metric_gene,g1,g2);

            [gdist_env,gprior_dist_env]=metric.metric_env.ghyper(metric.metric_env,e1,e2);
        else
            [gdist_gene,gprior_dist_gene]=metric.metric_gene.ghyper(metric.metric_gene,g1);

            [gdist_env,gprior_dist_env]=metric.metric_env.ghyper(metric.metric_env,e1);
            
        end
        gdist{1}=gdist_gene{1};
        for i=2:m+1
            
            gdist{i}=gdist_env{i-1};
        end
            
            
            gprior_dist=[gprior_dist_gene gprior_dist_env];
%             s = 1./metric.params;
% %             if nargin < 3
% %                 dist=s.*metric.x1x2matrix;
% %             elseif n1==size(metric.x1,1)
% %                 dist=s.*metric.x1matrix;
% %                 
% %             else
% %                 dist=s.*metric.x2matrix;
% %             end
%             
%             for i=1:m 
%                 gdist{i}=-s.*sqrt(squeeze(x1matrix));
%             end
%             
% %             ii1=0;            
% % 
%             dist  =  0;
%             distc = cell(1,m);
%             % Compute the distances for each component set
%             for i=1:m
%                 s = 1./metric.params(i).^2;
%                 distc{i} = 0;
%                 for j = 1:length(components{i})
%                     distc{i} = distc{i} + bsxfun(@minus,x(:,components{i}(j)),x2(:,components{i}(j))').^2;
%                 end
%                 distc{i} = distc{i}.*s;
%                 % Accumulate to the total distance
%                 dist = dist + distc{i};
%             end
%             dist = sqrt(dist);
%             % Loop through component sets 
%             for i=1:m
%                 D = -distc{i};
%                 D(dist~=0) = D(dist~=0)./dist(dist~=0);
%                 ii1 = ii1+1;
%                 gdist{ii1} = D;
%             end

% $$$         elseif nargin == 3
% $$$             if size(x,2) ~= size(x2,2)
% $$$                 error('metric_ibs -> _ghyper: The number of columns in x and x2 has to be the same. ')
% $$$             end
%         elseif nargin == 4
%             gdist = cell(1,length(metric.params)+1);
%         end
%         
%         % Evaluate the prior contribution of gradient with respect to lengthScale
%         for i2=1:m
%             i1=i1+1;
%             gprior_dist(i1)=feval(mp.params.fg, ...
%                              metric.params(i2), ...
%                              mp.params.a, 'x').*metric.params(i2) - 1;
%         end
% 
%         % Evaluate the prior contribution of gradient with respect to lengthScale.p.s (and lengthScale.p.nu)
%         if isfield(mp.params, 'p') && ~isempty(mp.params.p)
%             i1=i1+1;
%             gprior_dist(i1)=...
%                 feval(mp.params.p.s.fg, ...
%                       mp.params.a.s,...
%                       mp.params.p.s.a, 'x').*mp.params.a.s - 1 ...
%                 +feval(mp.params.fg, ...
%                        metric.params, ...
%                        mp.params.a, 's').*mp.params.a.s;
%             if any(strcmp(fieldnames(mp.params.p),'nu'))
%                 i1=i1+1;
%                 gprior_dist(i1)=...
%                     feval(mp.params.p.nu.fg, ...
%                           mp.params.a.nu,...
%                           mp.params.p.nu.a, 'x').*mp.params.a.nu -1 ...
%                     +feval(mp.params.fg, ...
%                            metric.params, ...
%                            mp.params.a, 'nu').*mp.params.a.nu;
%             end
%         end
    end
    
%     function [matrix,gdist]  = metric_ibs_gxe_matrix(metric, x1, x2)         
%     %METRIC_ibs_matrix   Compute the ibs distence between
%     %                            one or two matrices.
%     %
%     %	Description
%     %	[DIST] = METRIC_ibs_matrix(METRIC, X) takes a metric data
%     %   structure METRIC together with a matrix X of input vectors and 
%     %   calculates the ibs distance matrix DIST.
%     %
%     %	[DIST] = METRIC_ibs_matrix(METRIC, X1, X2) takes a metric data
%     %   structure METRIC together with a matrices X1 and X2 of input vectors and 
%     %   calculates the ibs distance matrix DIST.
%     %
%     %	See also
%     %	METRIC_ibs_PAK, METRIC_ibs_UNPAK, METRIC_ibs, GP_E
%     %
%         if nargin == 2 || isempty(x2)
%             x2=x1;
%         end
%         
%         [n1,m1]=size(x1);
%         [n2,m2]=size(x2);
%         
%         if m1~=m2
%             error('the number of columns of X1 and X2 has to be same')
%         end
%         mp=metric.p;
%         components = metric.components;
%         m = length(components);
%         dist  =  0;        
%         i1=0;
%         if nargout>1
%             
%             ii1  =  0;
%             distc = cell(1,m);
%             % Compute the distances for each component set
%             for i=1:m
%                 %s = 1./metric.params(i).^2;
%                 distc{i} = 0;
%                 for j = 1:length(components{i})
%                     distc{i} = distc{i} + bsxfun(@minus,x1(:,components{i}(j)),x2(:,components{i}(j))').^2;
%                 end
%                 distc{i} = distc{i};
%                 % Accumulate to the total distance
%                 dist = dist + distc{i};
%             end
%             dist = sqrt(dist);
%             % Loop through component sets 
%             for i=1:m
%                 D = -distc{i};
%                 D(dist~=0) = D(dist~=0)./dist(dist~=0);
%                 ii1 = ii1+1;
%                 gdist{ii1} = D;
%             end
%            
%         else
%             
%         for i=1:m
%             for j = 1:length(components{i})
%                 dist = dist + bsxfun(@minus,x1(:,components{i}(j)),x2(:,components{i}(j))').^2;
%             end
%         end
%         dist=sqrt(dist);
%         
%         end
%         
%         matrix = dist;
%        
%         
%        
%         
% 
%     end
    
    
    function [dist]  = metric_ibs_gxe_distance(metric, x1, x2)         
    %METRIC_ibs_DISTANCE   Compute the ibs distence between
    %                            one or two matrices.
    %
    %	Description
    %	[DIST] = METRIC_ibs_DISTANCE(METRIC, X) takes a metric data
    %   structure METRIC together with a matrix X of input vectors and 
    %   calculates the ibs distance matrix DIST.
    %
    %	[DIST] = METRIC_ibs_DISTANCE(METRIC, X1, X2) takes a metric data
    %   structure METRIC together with a matrices X1 and X2 of input vectors and 
    %   calculates the ibs distance matrix DIST.
    %
    %	See also
    %	METRIC_ibs_PAK, METRIC_ibs_UNPAK, METRIC_ibs, GP_E
    %
    
    
%         if nargin == 2 || isempty(n2)
%             n2=n1;
%         end
        
%         [n1,m1]=size(x1);
%         [n2,m2]=size(x2);
        
%         if m1~=m2
%             error('the number of columns of X1 and X2 has to be same')
%         end
        


        [n1,m1]=size(x1);
        
        if nargin == 3
           [n2,m2]=size(x2); 
           g1=x1(:,1);
           g2=x2(:,1);
           dist_g=metric.metric_gene.distance(metric.metric_gene,g1,g2);
           e1=x1(:,2:end);
           e2=x2(:,2:end);
           dist_e=metric.metric_env.distance(metric.metric_env,e1,e2);
        else
           g1=x1(:,1);
           dist_g=metric.metric_gene.distance(metric.metric_gene,g1);
           e1=x1(:,2:end);
           dist_e=metric.metric_env.distance(metric.metric_env,e1);
        end
        
        dist=dist_e+dist_g;

%         n1min=min(size(x1matrix));
%         n1max=max(size(x1matrix));
%         %n1=size(x1matrix,1);
%         dist=0;
% %        s = 1./metric.params;
%         jep=0;
%         if (nargin == 3)
%             n2min=min(size(x2matrix));
%             n2max=max(size(x2matrix));
%             if n1max<=n2max
%                 
%                 gene_matrix1=x1matrix(1:n1min,1:n1min);
%                 env_matrix1=x1matrix(1:end,n1min+1:end);
%                 if size(x2matrix,1)==n2max
%                     gene_matrix2=x2matrix(1:n2min,1:n1min);
%                     env_matrix2=x2matrix(n2min+1:end,1:end)';
%                 else
%                     gene_matrix2=x2matrix(1:n1min,1:n2min)';
%                     env_matrix2=x2matrix(1:end,n2min+1:end);
%                 end
%             else
%                 jep=1;
%                 gene_matrix1=x2matrix(1:n2min,1:n2min);
%                 env_matrix1=x2matrix(1:end,n2min+1:end);
%                 if size(x1matrix,1)==n1max
%                     gene_matrix2=x1matrix(1:n1min,1:n2min);
%                     env_matrix2=x1matrix(n1min+1:end,1:end)';
%                 else
%                     gene_matrix2=x1matrix(1:n2min,1:n1min)';
%                     env_matrix2=x1matrix(1:end,n1min+1:end);
%                 end
%             end
%             dist_gene=metric.metric_gene.distance(metric.metric_gene,gene_matrix1,gene_matrix2);
%             dist_env=metric.metric_env.distance(metric.metric_env,env_matrix1,env_matrix2);
%         else
%             gene_matrix1=x1matrix(1:n1min,1:n1min);
%             env_matrix1=x1matrix(1:end,n1min+1:end);
%             dist_gene=metric.metric_gene.distance(metric.metric_gene,gene_matrix1);
%             dist_env=metric.metric_env.distance(metric.metric_env,env_matrix1);
%         end
%         
%         dist=dist_gene+dist_env;
%         
%         if jep==1
%             dist=dist';
%         end
        
%         % gene effect: square root of manhattan metric
%         if (nargin == 3)
%             n2=min(size(x2matrix));
%             gene_matrix2=x2matrix(1:n,1:n);
%             env_matrix2=x2matrix(n+1:end,n+1:end);
%             %n1=size(genematrix1,1);
%             %n2=size(genematrix2,1);
%             if size(metric.Kfu)==[n1,n2]
%                 dist=s(1).*sqrt(squeeze(Kfu));
%             elseif size(metric.Kfu)==[n2,n1]
%                 dist=s(1).*sqrt(squeeze(Kfu))';
%             elseif size(metric.Kffstar)==[n1,n2]
%                 dist=s(1).*sqrt(squeeze(Kffstar));
%             elseif size(metric.Kffstar)==[n2,n1]
%                 dist=s(1).*sqrt(squeeze(Kffstar))';
%             else
%                 dist=s(1).*sqrt(squeeze(genematrix1));
%             end
% %         elseif n1~=metric.n1
% %             dist=s.*squeeze(metric.x2matrix);
%         else
%             dist=s(1).*sqrt(squeeze(genematrix1));
%             env_matrix2=env_matrix1;
%         end
% 
%         % environmental effect: euclidean metric
%         components = metric.components;
%         m = length(components);
%         dist_e  =  0;        
%         
%         if isempty(metric.dmatrix)
%             for i=1:m
%                 s = 1./metric.params(i+1).^2;
%                 for j = 1:length(components{i})
%                     dist_e = dist_e + s.*bsxfun(@minus,env_matrix1(:,components{i}(j)),env_matrix2(:,components{i}(j))').^2;
%                 end
%             end
%         dist_e = sqrt(dist_e);
%         end
%         
%         dist=dist+dist_e;
%         
%         %dist=metric.dmatrix;
        

    end
    
    function [ginput, gprior_input]  = metric_ibs_gxe_ginput(metric, x1, x2)         
    %METRIC_ibs_GINPUT   Compute the gradient of the ibs distance
    %                          function with respect to input.
    %[n, m] =size(x);
        ii1 = 0;
        components = metric.components;
%                
%         if nargin == 2 || isempty(x2)
%             x2=x1;
%         end
%         
%         [n1,m1]=size(x1);
%         [n2,m2]=size(x2);
%         
%         if m1~=m2
%             error('the number of columns of X1 and X2 has to be same')
%         end

%         if n1~=n2
%             dist=metric.x1x2matrix;
%         elseif n1==length(metric.x1)
%             dist=metric.x1matrix;
%         else
%             dist=metric.x2matrix;
%         end
        dist=metric.distance(metric,x1,x2);
        
%         s = 1./metric.params.^2;
%         dist = 0;
%         for i=1:length(components)
%             for j = 1:length(components{i})
%                 dist = dist + s(i).*bsxfun(@minus,x1(:,components{i}(j)),x2(:,components{i}(j))').^2;
%             end
%         end
%         dist = sqrt(dist);


        [ginput_g, gprior_input_g]  = metric.metric_gene.ginput(metric.metric_gene, x1, x2);
        [ginput_e, gprior_input_e]  = metric.metric_env.ginput(metric.metric_env, x1, x2);

        for i=1:length(ginput_g)
            ginput{i}=ginput_g{i}+ginput_e{i};
            gprior_input=[]; %KORJAA
        end

%         
%         for i=1:m1
%             for j = 1:n1
%                 DK = zeros(n1,n2);                
%                 for k = 1:length(components)
%                     if ismember(i,components{k})
%                         DK(j,:) = DK(j,:)+s(k).*bsxfun(@minus,x1(j,i),x2(:,i)');
%                     end
%                 end
%                 if nargin == 2
%                     DK = DK + DK';
%                 end
%                 DK(dist~=0) = DK(dist~=0)./dist(dist~=0);
%                                         
%                 ii1 = ii1 + 1;
%                 ginput{ii1} = DK;
%                 gprior_input(ii1) = 0; 
%             end
%         end
        %size(ginput)
        %ginput
        
    end
    
    
    function recmetric = metric_ibs_gxe_recappend(recmetric, ri, metric)
    % RECAPPEND - Record append
    %          Description
    %          RECMETRIC = METRIC_ibs_RECAPPEND(RECMETRIC, RI, METRIC) takes old covariance
    %          function record RECMETRIC, record index RI and covariance function structure. 
    %          Appends the parameters of METRIC to the RECMETRIC in the ri'th place.
    %
    %          RECAPPEND returns a structure RECMETRIC containing following record fields:
    %          lengthHyper    
    %          lengthHyperNu  
    %          lengthScale    
    %
    %          See also
    %          GP_MC and GP_MC -> RECAPPEND

    % Initialize record
        if nargin == 2
            recmetric.type = 'metric_ibs_gxe';
            recmetric.nin = ri;
            metric.components = recmetric.components;
            
            % Initialize parameters
            recmetric.params = [];

            % Set the function handles
            recmetric.pak       = @metric_ibs_gxe_pak;
            recmetric.unpak     = @metric_ibs_gxe_unpak;
            recmetric.e         = @metric_ibs_gxe_e;
            recmetric.ghyper    = @metric_ibs_gxe_ghyper;
            recmetric.ginput    = @metric_ibs_gxe_ginput;            
            recmetric.distance  = @metric_ibs_gxe_distance;
            recmetric.recappend = @metric_ibs_gxe_recappend;
            recmetric.matrix    = @metric_ibs_gxe_matrix;
            return
        end
        mp = metric.p;

        % record parameters
        if ~isempty(metric.params)
            if ~isempty(mp.params)
                recmetric.lengthHyper(ri,:)=mp.params.a.s;
                if isfield(mp.params,'p')
                    if isfield(mp.params.p,'nu')
                        recmetric.lengthHyperNu(ri,:)=mp.params.a.nu;
                    end
                end
            elseif ri==1
                recmetric.lengthHyper=[];
            end
            recmetric.params(ri,:)=metric.params;
        elseif ri==1
            recmetric.params=[];
        end
    end
