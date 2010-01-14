function gp = gp_init(do, varargin)
%GP_INIT	Create a Gaussian Process.
%
%	Description
%
%	GP = GP_INIT('INIT', 'TYPE', NIN, 'LIKELIH', GPCF, NOISE, VARARGIN) 
%       Creates a Gaussian Process model with a single output. Takes the number 
%	of inputs  NIN together with string/structure 'LIKELIH', which spesifies 
%       likelihood function used, GPCF array specifying the covariance functions 
%       and NOISE array, which specify the noise covariance functions used for
%       Gaussian process. At minimum one covariance function has to be given. 
%       
%       The GPCF and NOISE arrays consist of covariance function structures 
%       (see, for example, gpcf_sexp).
%   
%       The LIKELIH is a string  'regr' for a regression model with additive Gaussian 
%       noise. Other likelihood models require a likelihood structure for LIKELIH 
%       parameter (see, for example, likelih_probit).
%
%       TYPE defines the type of GP, possible types are:
%        'FULL'        (full GP), 
%        'FIC'         (fully independent conditional), 
%        'PIC'         (partially independent condional), 
%        'CS+FIC'      (Compact support + FIC model)
%
%       With VARAGIN the fields of the GP structure can be set into different values 
%       VARARGIN = 'FIELD1', VALUE1, 'FIELD2', VALUE2, ... 
%
%	GP = GPINIT('SET', GP, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of the fields FIELD1... to the values VALUE1... in GP.
%
%	The minimum number of fields (in case of full GP regression model) and 
%       their default values are:
%         type           = 'FULL'
%	  nin            = number of inputs
%	  nout           = number of outputs: always 1
%         cf             = struct of covariance functions
%         noise          = struct of noise functions
%	  jitterSigmas   = jitter term for covariance function
%                          (initialized to 0)
%         p.r            = Prior Structure for residual parameters
%                          (defined only in case likelih == 'regr')
%         likelih        = a string or structure defining the likelihood
%    
%       The additional fields needed in sparse approximations are:
%         X_u            = Inducing inputs in FIC, PIC and CS+FIC models
%         blocks         = Initializes the blocks for the PIC model
%                          The value for blocks has to be a cell array of type 
%                          {'method', matrix of training inputs, param}. 
%
%                          The possible methods are:
%                            'manual', which takes in the place of param a structure of the index vectors
%                                appointing the data points into blocks. For example, if x is a matrix of data inputs
%                                then x(param{i},:) are the inputs belonging to the ith block.
%
%       The additional fields when the model is not for regression (likelih ~='regr') is:
%         latent_method  = Defines a method for marginalizing over latent values. Possible 
%                          methods are 'MCMC', 'Laplace' and 'EP'. The fields for them are
%                         
%                          In case of MCMC:
%                            fh_latentmc    = Function handle to function which samples the latent values
%                            latentValues   = Vector of latent values 
%                          and they are set as following
%                            gp_init('SET', GP, 'latent_method', {'MCMC', @fh_latentmc Z});
%                          where Z is a (1xn) vector of latent values 
%
%                          In case of EP:
%                            fh_e       = function handle to an energy function
%                          and they are set as following
%                            gp_init('SET', GP, 'latent_method', {'Laplace', x, y, 'param'});
%                          where x is a matrix of inputs, y vector/matrix of outputs and 'param' a 
%                          string defining wich parameters are sampled/optimized (see gp_pak).
% 
%                          In case of EP:
%                            fh_e       = function handle to an energy function
%                            site_tau   = vector (size 1xn) of tau site parameters 
%                            site_mu    = vector (size 1xn) of mu site parameters 
%                          and they are set as following
%                            gp_init('SET', GP, 'latent_method', {'EP', x, y, 'param'});
%                          where x is a matrix of inputs, y vector/matrix of outputs and 'param' a 
%                          string defining wich parameters are sampled/optimized (see gp_pak).
%
%         
%
%	See also
%	GPINIT, GP2PAK, GP2UNPAK
%
%

% Copyright (c) 2006      Helsinki University of Technology (author Jarno Vanhatalo)
% Copyright (c) 2007-2008 Jarno Vanhatalo
    
% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    if nargin < 4
        error('Not enough arguments')
    end

    % Initialize the Gaussian process
    if strcmp(do, 'init')
        
        gp.type = varargin{1};
        gp.nin = varargin{2};
        gp.nout = 1;
        
        % Set likelihood. 
        gp.likelih = varargin{3};   % Remember to set the latent_method.

        % Set covariance functions into gpcf
        gpcf = varargin{4};
        for i = 1:length(gpcf)
            gp.cf{i} = gpcf{i};
        end
        
        % Set noise functions into noise
        if length(varargin) > 4
            gp.noise = [];
            gpnoise = varargin{5};
            for i = 1:length(gpnoise)
                gp.noise{i} = gpnoise{i};
            end
        else
            gp.noise = [];
        end

        % Initialize parameters
        gp.jitterSigmas=0;
        gp.p=[];
        gp.p.jitterSigmas=[];
        
        switch gp.type
          case 'FIC' 
            gp.X_u = [];
            gp.nind = [];
          case {'PIC' 'PIC_BLOCK'}
            gp.X_u = [];
            gp.nind = [];
            gp.blocktype = [];
            gp.tr_index = {};            
        end
                
        if length(varargin) > 5
            if mod(length(varargin),2) ==0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=6:2:length(varargin)-1
                switch varargin{i}
                  case 'jitterSigmas'
                    gp.jitterSigmas = varargin{i+1};
                  case 'likelih'
                    gp.likelih = varargin{i+1};
                    if strcmp(gp.likelih_e, 'regr')
                        gp.p.r=[];
                    end
                  case 'likelih_e'
                    gp.likelih_e = varargin{i+1};
                  case 'likelih_g'
                    gp.likelih_g = varargin{i+1};
                  case 'type'
                    gp.type = varargin{i+1};
                  case 'X_u'
                    if size(varargin{i+1},2)~=gp.nin
                        error('The size of X_u has to be u x nin.')
                    else
                        gp.X_u = varargin{i+1};
                        gp.nind = size(varargin{i+1},1);
                    end
                  case 'blocks'
                    init_blocks(varargin{i+1})
                  case 'truncated'
                    init_truncated(varargin{i+1})
                  case 'latent_method'
                    gp.latent_method = varargin{i+1}{1};
                    switch varargin{i+1}{1}
                      case 'MCMC'
                        gp.latentValues = varargin{i+1}{2};
                        gp.fh_mc = varargin{i+1}{3};
                      case 'EP'
                        % Note in the case of EP, you have to give varargin{i+1} = {x, y, param}
                        gp.ep_opt.maxiter = 20;
                        gp.ep_opt.tol = 1e-6;
                        gp = gpep_e('init', gp, varargin{i+1}{2}, varargin{i+1}{3}, varargin{i+1}{4});
                        w = gp_pak(gp, varargin{i+1}{4});
                        [e, edata, eprior, site_tau, site_nu] = gpep_e(w, gp, varargin{i+1}{2}, varargin{i+1}{3}, varargin{i+1}{4});
                        gp.site_tau = site_tau';
                        gp.site_nu = site_nu';
                      case 'Laplace'
                        gp.laplace_opt.maxiter = 20;
                        gp.laplace_opt.tol = 1e-12;
                        gp.laplace_opt.optim_method = 'newton';
                        gp = gpla_e('init', gp, varargin{i+1}{2}, varargin{i+1}{3}, varargin{i+1}{4});
                        w = gp_pak(gp, varargin{i+1}{4});
                        [e, edata, eprior, f] = gpla_e(w, gp, varargin{i+1}{2}, varargin{i+1}{3}, varargin{i+1}{4});
% $$$                         gp.f = f;
                      otherwise
                        error('Unknown type of latent_method!')
                    end
                  case 'compact_support'
                    % Note: Add the possibility for more than one compactly supported cf later.
                    gp.cs = varargin{i+1};      
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
        gp = varargin{1};
        % Loop through all the parameter values that are changed
        for i=2:2:length(varargin)-1
            switch varargin{i}
              case 'jitterSigmas'
                gp.jitterSigmas = varargin{i+1};
              case 'likelih'
                gp.likelih = varargin{i+1};
                if strcmp(gp.likelih, 'regr')
                    gp.p.r=[];
                end
              case 'likelih_e'
                gp.likelih_e = varargin{i+1};
              case 'likelih_g'
                gp.likelih_g = varargin{i+1};
              case 'type'
                gp.type = varargin{i+1};
              case 'X_u'
                if size(varargin{i+1},2)~=gp.nin
                    error('The size of X_u has to be u x nin.')
                else
                    gp.X_u = varargin{i+1};
                    gp.nind = size(varargin{i+1},1);
                end
              case 'blocks'
                init_blocks(varargin{i+1})
              case 'truncated'
                init_truncated(varargin{i+1})
              case 'latent_method'
                gp.latent_method = varargin{i+1}{1};
                switch varargin{i+1}{1}
                  case 'MCMC'
                    gp.latentValues = varargin{i+1}{2};
                    gp.fh_mc = varargin{i+1}{3};
                  case 'EP'
                    % Note in the case of EP, you have to give varargin{i+1} = {x, y, param}
                    gp.ep_opt.maxiter = 20;
                    gp.ep_opt.tol = 1e-6;
                    gp = gpep_e('init', gp, varargin{i+1}{2}, varargin{i+1}{3}, varargin{i+1}{4});
                    w = gp_pak(gp, varargin{i+1}{4});
                    [e, edata, eprior, site_tau, site_nu] = gpep_e(w, gp, varargin{i+1}{2}, varargin{i+1}{3}, varargin{i+1}{4});
                    gp.site_tau = site_tau';
                    gp.site_nu = site_nu';
                  case 'Laplace'
                    gp.laplace_opt.maxiter = 20;
                    gp.laplace_opt.tol = 1e-6;
                    gp.laplace_opt.optim_method = 'newton';
                    gp = gpla_e('init', gp, varargin{i+1}{2}, varargin{i+1}{3}, varargin{i+1}{4});
                    w = gp_pak(gp, varargin{i+1}{4});
                    [e, edata, eprior, f] = gpla_e(w, gp, varargin{i+1}{2}, varargin{i+1}{3}, varargin{i+1}{4});
% $$$                     gp.f = f;
                  otherwise
                    error('Unknown type of latent_method!')
                end
              case 'compact_support'
                % Note: Add the possibility for more than one compactly supported cf later.
                gp.cs = varargin{i+1};                
              otherwise
                error('Wrong parameter name!')
            end    
        end
    end


    % Add new covariance function
    if strcmp(do, 'add_cf')
        gp = varargin{1};
        for i = 2:length(varargin)
            gp.cf{end+1} = varargin{i};
        end
    end


    function init_blocks(var)
        
        if length(var) ~= 3
            error('Wrong kind of value for the clustering type! See help gp_init!')
        else
            x = var{2};
            switch var{1}
              case 'manual'
                gp.blocktype = 'manual';
                gp.tr_index = var{3};
              case 'farthest_point'
                gp.blocktype = 'farthest_point';
                gp.blockcent = var{3};
              otherwise
                error('Wrong value for the clustering type! See help gp_init!')
            end
        end
    end
    
    function init_truncated(var)
        if length(var) < 2
            error('Wrong kind of value for the truncated type! See help gp_init!')
        end

        x= var{1};
        R= var{2};
        gp.truncated_R = R;
        n = size(x,1);
        
        if size(x,2)~=gp.nin
            error('The size of x for "truncated" has to be n x nin!')
        end
                
        C = sparse([],[],[],n,n,0);
        for i1=2:n
            i1n=(i1-1)*n;
            for i2=1:i1-1
                ii=i1+(i2-1)*n;
                D = 0;
                for i3=1:gp.nin
                    D =D+(x(i1,i3)-x(i2,i3)).^2;       % the covariance function
                end
                if sqrt(D) < R
                    C(ii)=1;
                    C(i1n+i2)=C(ii); 
                end
            end
        end
        C= C + speye(n,n);
        if length(var) == 3
            if var{3} == 1
                spy(C)
                title('the sparsity structure of the covariance matrix')
                fprintf('The density of the sparse correlation matrix is %f \n',nnz(C)/prod(size(C)))
            end
        end
                
        [I,J,s] = find(C);
        gp.tr_index = [I(:) J(:)];
    end
end