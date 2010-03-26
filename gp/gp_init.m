function gp = gp_init(do, varargin)
%GP_INIT	Create a Gaussian Process.
%
%     Description
%
%	GP = GP_INIT(DO, TYPE, 'LIKELIH', GPCF, NOISE, VARARGIN) 
%
%        Creates a Gaussian Process model with a single output. 
%        Takes a string/structure 'LIKELIH', which spesifies
%        likelihood function used, GPCF array specifying the
%        covariance functions and NOISE array, which specify the
%        noise covariance functions used for Gaussian process. At
%        minimum one covariance function has to be given.
%       
%       TYPE defines the type of GP, possible types are:
%        'FULL'        (full GP), 
%        'FIC'         (fully independent conditional), 
%        'PIC'         (partially independent condional), 
%        'CS+FIC'      (Compact support + FIC model)
%   
%       LIKELIH is a string 'regr' for a regression model with
%        additive Gaussian noise. Other likelihood models require a
%        likelihood structure for LIKELIH parameter (see, for
%        example, likelih_probit).
%
%        The GPCF and NOISE arrays consist of covariance function
%        structures (see, for example, gpcf_sexp, gpcf_noiset).
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
%         cf             = struct of covariance functions
%         noise          = struct of noise functions
%	  jitterSigma2   = jitter term for covariance function
%                          (initialized to 0)
%         p.r            = Prior Structure for residual parameters
%                          (defined only in case likelih == 'regr')
%         likelih        = a string or structure defining the likelihood
%    
%       The additional fields needed in sparse approximations are:
%         X_u            = Inducing inputs in FIC, PIC and CS+FIC models
%         blocks         = Initializes the blocks for the PIC model
%                          The value for blocks has to be a cell
%                          array of the index vectors appointing
%                          the data points into blocks. For
%                          example, if x is a matrix of data inputs
%                          then x(param{i},:) are the inputs
%                          belonging to the ith block.
%
%       The additional fields when the model is not for regression
%       (likelih ~='regr') are:
%         latent_method  = Defines a method for marginalizing over latent 
%                          values. Possible methods are 'MCMC',
%                          'Laplace' and 'EP'. The fields for them
%                          are
%                         
%        In case of MCMC:
%         fh_latentmc    = Function handle to function which samples the 
%                          latent values
%         latentValues   = Vector of latent values and they are set as 
%                          following
%         gp_init('SET', GP, 'latent_method', {'MCMC', @fh_latentmc Z});
%                          where Z is a (1xn) vector of latent values 
%
%        In case of Laplace:
%         fh_e           = Function handle to an energy function and they 
%                          are set as following
%         gp_init('SET', GP, 'latent_method', {'Laplace', x, y, 'param'});
%                          where x is a matrix of inputs, y vector/matrix 
%                          of outputs and 'param' a string defining which 
%                          parameters are inferred (see gp_pak).
% 
%        In case of EP:
%         fh_e           = function handle to an energy function
%         site_tau       = vector (size 1xn) of tau site parameters 
%         site_mu        = vector (size 1xn) of mu site parameters 
%                          and they are set as following
%         gp_init('SET', GP, 'latent_method', {'EP', x, y, 'param'});
%                          where x is a matrix of inputs, y vector/matrix 
%                          of outputs and 'param' a string defining which 
%                          parameters are sampled/optimized (see gp_pak).
%
%	See also
%	GPINIT, GP2PAK, GP2UNPAK

% Copyright (c) 2006-2010 Jarno Vanhatalo
    
% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    if nargin < 4
        error('Not enough arguments')
    end

    % Initialize the Gaussian process
    if strcmp(do, 'init')        
        gp.type = varargin{1};
        
        % Set likelihood. 
        gp.likelih = varargin{2};   % Remember to set the latent_method.

        % Set covariance functions into gpcf
        gpcf = varargin{3};
        for i = 1:length(gpcf)
            gp.cf{i} = gpcf{i};
        end
        
        % Set noise functions into noise
        if length(varargin) > 3
            gp.noise = [];
            gpnoise = varargin{4};
            for i = 1:length(gpnoise)
                gp.noise{i} = gpnoise{i};
            end
        else
            gp.noise = [];
        end

        % Initialize parameters
        gp.jitterSigma2=0;
        gp.p=[];
        
        switch gp.type
          case 'FIC' 
            gp.X_u = [];
            gp.nind = [];
            gp.p.X_u = [];
          case {'PIC' 'PIC_BLOCK'}
            gp.X_u = [];
            gp.nind = [];
            gp.tr_index = {};            
        end
                
        if length(varargin) > 4
            if mod(length(varargin),2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=5:2:length(varargin)-1
                switch varargin{i}
                  case 'jitterSigma2'
                    gp.jitterSigma2 = varargin{i+1};
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
                    gp.X_u = varargin{i+1};
                    gp.nind = size(varargin{i+1},1);
                    gp.p.X_u = prior_unif('init');
                  case 'Xu_prior'
                    gp.p.X_u = varargin{i+1};                    
                  case 'blocks'
                    gp.tr_index = varargin{i+1};
                  case 'infer_params'
                    gp.infer_params = varargin{i+1};
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
                        gp = gpep_e('init', gp, varargin{i+1}{2:end});
                        w = gp_pak(gp, varargin{i+1}{4});
                        [e, edata, eprior, site_tau, site_nu] = gpep_e(w, gp, varargin{i+1}{2:end});
                        gp.site_tau = site_tau';
                        gp.site_nu = site_nu';
                      case 'Laplace'
                        gp.laplace_opt.maxiter = 20;
                        gp.laplace_opt.tol = 1e-12;
                        gp.laplace_opt.optim_method = 'newton';
                        gp = gpla_e('init', gp, varargin{i+1}{2:end});
                        w = gp_pak(gp, varargin{i+1}{4});
                        [e, edata, eprior, f] = gpla_e(w, gp, varargin{i+1}{2:end});
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
              case 'jitterSigma2'
                gp.jitterSigma2 = varargin{i+1};
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
                gp.X_u = varargin{i+1};
                gp.nind = size(varargin{i+1},1);
                gp.p.X_u = prior_unif('init');
              case 'Xu_prior'
                gp.p.X_u = varargin{i+1};                
              case 'blocks'
                gp.tr_index = varargin{i+1};
              case 'infer_params'
                gp.infer_params = varargin{i+1};
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
                    gp = gpep_e('init', gp, varargin{i+1}{2:end});
                    w = gp_pak(gp, varargin{i+1}{4});
                    [e, edata, eprior, site_tau, site_nu] = gpep_e(w, gp, varargin{i+1}{2:end});
                    gp.site_tau = site_tau';
                    gp.site_nu = site_nu';
                  case 'Laplace'
                    gp.laplace_opt.maxiter = 20;
                    gp.laplace_opt.tol = 1e-12;
                    gp.laplace_opt.optim_method = 'newton';
                    gp = gpla_e('init', gp, varargin{i+1}{2:end});
                    w = gp_pak(gp, varargin{i+1}{4});
                    [e, edata, eprior, f] = gpla_e(w, gp, varargin{i+1}{2:end});
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
