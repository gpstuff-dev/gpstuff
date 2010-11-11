function gp = gp_set(varargin)
%GP_INIT  Create a Gaussian process model data structure. 
%
%  Description
%    GP = GP_SET('PARAM1',VALUE1,'PARAM2',VALUE2,...)
%    creates a Gaussian process structure in which the named
%    parameters have the specified values. Any unspecified
%    parameters are set to default values. Either 'cf' or 
%    'meanf' parameter has to be specified. 
%  
%    GP = GP_SET(GP,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a Gaussian process structure with the named
%    parameters altered with the specified values.
%
%    Parameters for Gaussian process [default]
%      cf           - single covariance structure or cell array of 
%                     covariance function structures created by
%                     gpcf_* functions [{}]
%                     This or meanf has to be defined as non-empty. 
%      type         - type of Gaussian process ['FULL']
%                      'FULL'   full GP
%                      'FIC'    fully independent conditional sparse
%                               approximation
%                      'PIC'    partially independent conditional  
%                               sparse approximation
%                      'CS+FIC' compact support + FIC model sparse 
%                               approximation
%                      'DTC'    deterministic training conditional 
%                               sparse approximation
%                      'SOR'    subset of regressors sparse
%                               approximation
%                      'VAR'    variational sparse approximation
%      lik          - likelihood structure created by one of the 
%                     likelihood functions lik_* ['lik_gaussian']
%                     If non-Gaussian, see latent_method below.  
%      jitterSigma2 - positive jitter to be added in the diagonal of 
%                     covariance matrix [0]
%      infer_params - String defining which hyperparameters are inferred.
%                     ['covariance+likelihood']  
%                      'covariance'     = infer hyperparameters of 
%                                         covariance function
%                      'likelihood'     = infer parameters of likelihood
%                      'inducing'       = infer inducing inputs (in sparse
%                                         approximations): W = gp.X_u(:)    
%                       By combining the strings one can infer more than 
%                       one group of parameters. For example:
%                      'covariance+likelih' = infer covariance function
%                                             and likelihood parameters
%                      'covariance+inducing' = infer covariance function
%                                              parameters and inducing 
%                                              inputs
%
%    The additional fields when the likelihood is not Gaussian
%    (lik ~=lik_gaussian or lik_smt) are:
%      latent_method - Defines a method for marginalizing over 
%                      latent values. Possible methods are 'MCMC',
%                      'Laplace' and 'EP'. If just the latent
%                      method is given, default options values are
%                      used. Optionally it is possible to give
%                      latent method options in a cell array, e.g.,
%                      {'MCMC', 'method', @scaled_hmc}.
%      The options which can be set for each method are
%        MCMC:
%          method    - Function handle to function which samples the
%                      latent values @scaled_mh (default) or @scaled_hmc
%          f         - 1xn vector of latent values
%        Laplace:
%          optim_method    - String telling which optimization method is used
%                      for estimating parameters of the Laplace approximation
%                      'newton' (default except for lik_t)
%                      'stabilized-newton', 'fminuc_large', or
%                      'lik_specific' (applicable and default for lik_t)
%        EP: currently no options to set  
%  
%     The additional fields needed with mean functions
%      meanf        - single mean function structure or cell array of 
%                     mean function structures created by
%                     gpmf_* functions [{}]
%
%     The additional fields needed in sparse approximations are:
%      X_u          - inducing inputs, no default, has to be set when
%                     FIC, PIC, PIC_BLOCK, VAR, DTC, or SOR is used.
%      Xu_prior     - prior for inducing inputs [prior_unif]
%
%     The additional field required by PIC sparse approximation is:
%       tr_index    - The blocks for the PIC model [{}]. The value has to
%                     be a cell array of the index vectors appointing
%                     the data points into blocks. For example, if x  
%                     is a matrix of data inputs then x(tr_index{i},:) 
%                     are the inputs belonging to the i'th block.
%
%    The additional fields needed with derivative observations
%      derivobs     - tells whether derivative observations are
%                     used: 'on' or 'off' ['off']
%
%       See also
%       GPCF_*, LIK_*, PRIOR_*, GP_PAK, GP_UNPAK, GP_E, GP_G, GP_EG,
%       GP_PRED, GP_MC, GP_IA, ...
%
%   References:
%    Qui�onero-Candela, J. and Rasmussen, C. E. (2005). A unifying
%    view of sparse approximate Gaussian process regression. 
%    Journal of Machine Learning Research, 6(3):1939-1959.
%
%    Rasmussen, C. E. and Williams, C. K. I. (2006). Gaussian
%    Processes for Machine Learning. The MIT Press.
%
%    Snelson, E. and Ghahramani, Z. (2006). Sparse Gaussian process
%    using pseudo-inputs. In Weiss, Y., Sch�lkopf, B., and Platt,
%    J. (eds) Advances in Neural Information Processing Systems 18,
%    pp. 1257-1264.
%
%    Titsias, M. K. (2009). Variational Model Selection for Sparse
%    Gaussian Process Regression. Technical Report, University of
%    Manchester.
%
%    Vanhatalo, J. and Vehtari, A. (2008). Modelling local and
%    global phenomena with sparse Gaussian processes. Proceedings
%    of the 24th Conference on Uncertainty in Artificial
%    Intelligence,

% Copyright (c) 2006-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari
  
% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GP_SET';
  ip.addOptional('gp', [], @isstruct);
  ip.addParamValue('cf',[], @(x) isempty(x) || isstruct(x) || iscell(x));
  ip.addParamValue('meanf',[], @(x) isempty(x) || isstruct(x) || iscell(x));
  ip.addParamValue('type','FULL', ...
                   @(x) ismember(x,{'FULL' 'FIC' 'PIC' 'PIC_BLOCK' 'VAR' ...
                      'DTC' 'SOR' 'CS+FIC'}));
  ip.addParamValue('lik',lik_gaussian(), @(x) isstruct(x));
  ip.addParamValue('jitterSigma2',0, @(x) isscalar(x) && x>=0);
  ip.addParamValue('infer_params','covariance+likelihood', @(x) ischar(x));
  ip.addParamValue('latent_method','NA', @(x) ischar(x) || iscell(x));
  ip.addParamValue('X_u',[],  @(x) isreal(x) && all(isfinite(x(:))));
  ip.addParamValue('Xu_prior',prior_unif,  @(x) isstruct(x) || isempty(x));
  ip.addParamValue('tr_index', [], @(x) ~isempty(x) || iscell(x))    
  ip.addParamValue('derivobs',false, @(x) islogical(x) || isscalar(x) || ...
                   (ischar(x) && ismember(x,{'on' 'off'})));
  ip.parse(varargin{:});
  gp=ip.Results.gp;

  if isempty(gp)
    % Initialize a Gaussian process
    init=true;
  else
    % Modify a Gaussian process
    if ~isfield(gp,'cf')
      error('First argument does not seem to be a Gaussian process structure')
    end
    init=false;
  end

  % FULL or sparse
  if init || ~ismember('type',ip.UsingDefaults)
    gp.type=ip.Results.type;
  end
  % Likelihood
  if init || ~ismember('lik',ip.UsingDefaults)
    gp.lik = ip.Results.lik;
  end
  % Covariance function(s)
  if init || ~ismember('cf',ip.UsingDefaults)
    gp.cf=ip.Results.cf;
    if isstruct(gp.cf)
      % store single structure in a cell array, too
      gp.cf={gp.cf};
    end
  end
  % Mean function(s)
  if init || ~ismember('meanf',ip.UsingDefaults)
      if ~isempty(ip.Results.meanf)
        gp.meanf=ip.Results.meanf;
        if isstruct(gp.meanf)
          % store single structure in a cell array, too
          gp.meanf={gp.meanf};
        end
      end
  end
  if isempty(gp.cf) && isempty(gp.meanf)
    error('At least one covariance or mean function has to defined')
  end
  % Inference for which parameters 
  if init || ~ismember('infer_params',ip.UsingDefaults)
    gp.infer_params=ip.Results.infer_params;
  end
  % Jitter
  if init || ~ismember('jitterSigma2',ip.UsingDefaults)
    gp.jitterSigma2=ip.Results.jitterSigma2;
  end
   % Gradient observation
  if init || ~ismember('derivobs',ip.UsingDefaults)
    derivobs=ip.Results.derivobs;
    if ~ischar(derivobs)
      if derivobs
        derivobs='on';
      else
        derivobs='off';
      end
      switch derivobs
        case 'on'
          gp.derivobs=true;
        case 'off'
          if isfield(gp,'derivobs')
            gp=rmfield(gp,'derivobs')
          end
      end
    end
  end

  % Inducing inputs
  if ismember(gp.type,{'FIC' 'CS+FIC' 'DTC' 'VAR' 'SOR' 'PIC' 'PIC_BLOCK'})
    if init || ~ismember('X_u',ip.UsingDefaults)
      gp.X_u = ip.Results.X_u;
      gp.nind = size(gp.X_u,1);
    end
    if init || ~ismember('Xu_prior',ip.UsingDefaults) || ~isfield(gp,'p')
      gp.p.X_u = ip.Results.Xu_prior;
    end
    if ismember(gp.type, {'PIC' 'PIC_BLOCK'})
      % + PIC block indexes
      if init || ~ismember('tr_index',ip.UsingDefaults)
        gp.tr_index = ip.Results.tr_index;
      end
    end
  end
  if ismember(gp.type,{'FIC' 'PIC' 'PIC_BLOCK' 'VAR' 'DTC' 'SOR'}) ...
      && isempty(gp.X_u)
    error(sprintf('Need to set X_u when using %s',gp.type))
  end
  % Latent method
  if init || ~ismember('latent_method',ip.UsingDefaults)
    latent_method=ip.Results.latent_method;
    latent_method_opt={};
    if iscell(latent_method)
      latent_method_opt=latent_method(2:end);
      latent_method=latent_method{1};
    end
    if isempty(latent_method)
      latent_method='NA'
    end
    switch latent_method
      case 'MCMC'
        gp.latent_method=latent_method;
        ipmc=inputParser;
        ipmc.FunctionName = 'GP_SET - latent method MCMC options';
        ipmc.addParamValue('f',[],  @(x) isreal(x) && all(isfinite(x(:))));
        ipmc.addParamValue('method',@scaled_mh, @(x) isa(x,'function_handle'));
        ipmc.parse(latent_method_opt{:});
        gp.latentValues = ipmc.Results.f;
      case 'EP'
        if isfield(gp,'latentValues'); gp=rmfield(gp,'latentValues'); end
        gp.latent_method=latent_method;
        gp.latent_opt.maxiter = 20;
        gp.latent_opt.tol = 1e-6;
        % following sets gp.fh.e = @ep_algorithm;
        gp = gpep_e('init', gp);
      case 'Laplace'
        if isfield(gp,'latentValues'); gp=rmfield(gp,'latentValues'); end
        gp.latent_method=latent_method;
        gp.latent_opt.maxiter = 20;
        gp.latent_opt.tol = 1e-12;
        ipla=inputParser;
        ipla.FunctionName = 'GP_SET - latent method Laplace options';
        ipla.addParamValue('method',[], @(x) ischar(x));
        ipla.parse(latent_method_opt{:});
        method=ipla.Results.method;
        if ~isempty(method)
          gp.latent_opt.optim_method;
        else
          switch gp.lik.type
            case 'Student-t'
              % slower than newton but more robust
              gp.latent_opt.optim_method='lik_specific'; 
            otherwise
              gp.latent_opt.optim_method='newton';
          end
        end
        switch gp.lik.type
          % following sets gp.fh.e = @laplace_algorithm;
          case 'Softmax'
            gp = gpla_softmax_e('init', gp);
          otherwise
            gp = gpla_e('init', gp);
        end
      case 'NA'
        % no latent method set
        if isfield(gp,'latent_method'); gp=rmfield(gp,'latent_method'); end
        if isfield(gp,'latent_opt'); gp=rmfield(gp,'latent_opt'); end
      otherwise
        error('Unknown type of latent_method!')
    end % switch latent_method
  end % if init || ~ismember('latent_method',ip.UsingDefaults)
  
end
